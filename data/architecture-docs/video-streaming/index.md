# Video Streaming Platform: Design Document


## Overview

A scalable video streaming service that handles large file uploads, transcodes videos into adaptive streaming formats, and delivers HLS streams with quality switching. The key architectural challenge is orchestrating the entire pipeline from upload through transcoding to delivery while managing concurrent processing and optimizing for both upload experience and playback performance.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** All milestones (1-4) - Understanding these fundamentals is essential before implementing any component of the video streaming platform.

Building a video streaming platform presents unique technical challenges that go far beyond simple file storage and download. While serving a static image or document requires only basic HTTP file serving, video streaming demands sophisticated infrastructure to handle massive files, multiple format variations, and real-time adaptive delivery based on network conditions.

Think of the difference between mailing a physical book versus running a television broadcast network. Mailing a book requires packaging, addressing, and delivery - straightforward logistics. But television broadcasting requires content acquisition, format conversion for different transmission standards, signal encoding for various reception qualities, real-time transmission scheduling, and adaptive signal strength management based on atmospheric conditions. Video streaming platforms face analogous technical complexity in the digital realm.

The core challenges that make video streaming architecturally complex include **large file handling** (videos can be gigabytes in size, requiring chunked uploads and resumable transfers), **format compatibility** (browsers, mobile devices, and smart TVs each support different codecs and streaming protocols), and **adaptive delivery** (network bandwidth fluctuates constantly, requiring real-time quality switching to maintain smooth playback without buffering).

Modern video platforms like YouTube, Netflix, and Vimeo have each developed sophisticated solutions to these challenges, but their approaches differ significantly based on their scale, audience, and technical constraints. Understanding these existing solutions provides crucial context for the architectural decisions we'll make in our own streaming platform implementation.

### Video Streaming Fundamentals

**Progressive download** represents the simplest approach to video delivery - essentially treating video files like any other downloadable content. When a user clicks play, the browser begins downloading the entire video file from the beginning while simultaneously starting playback of the already-downloaded portions. This approach works adequately for short videos with reliable, fast connections, but fails catastrophically under real-world conditions.

Consider a user attempting to watch a 2GB movie file over a mobile connection. Progressive download forces them to wait for the entire 2GB file to transfer before they can seek to any position near the end. If their connection drops halfway through download, they must restart the entire process. Worse, if the user only wants to watch the first 10 minutes, they've still consumed bandwidth downloading the remaining 90 minutes of unwatched content.

**Adaptive streaming** solves these fundamental problems by completely restructuring how video content is encoded, stored, and delivered. Instead of treating a video as a single monolithic file, adaptive streaming platforms pre-process each video into multiple **quality renditions** (the same content encoded at different resolutions and bitrates) and segment each rendition into small, independently downloadable chunks typically lasting 2-10 seconds each.

This segmentation creates a fundamentally different playback model. The video player requests segments sequentially, maintaining a small buffer of upcoming segments. When network conditions change, the player can immediately switch to requesting segments from a different quality rendition without interrupting playback. A user experiencing network congestion might automatically drop from 1080p to 720p for a few segments, then seamlessly return to higher quality when bandwidth improves.

The **manifest file** serves as the coordination mechanism that makes adaptive streaming possible. This small metadata file (typically in M3U8 format for HLS streaming) contains references to all available quality renditions and lists every segment URL for each rendition. The video player downloads the manifest first, uses it to understand what quality options are available, then begins requesting segments starting from the most appropriate quality level based on current network conditions.

**HTTP Live Streaming (HLS)** has emerged as the dominant adaptive streaming protocol, originally developed by Apple but now supported across virtually all devices and browsers. HLS uses standard HTTP requests for all communication, making it compatible with existing web infrastructure including CDNs, load balancers, and caching systems. This HTTP-based approach contrasts with older streaming protocols that required specialized server software and often struggled with firewalls and NAT configurations.

The **transcoding pipeline** represents perhaps the most computationally intensive aspect of video streaming platforms. Raw uploaded videos arrive in dozens of different formats, codecs, resolutions, and quality levels. The platform must convert each upload into the standardized format and quality ladder required for adaptive streaming. This process involves:

1. **Source analysis** to determine the optimal quality ladder (no point generating a 4K rendition from a 480p source)
2. **Codec conversion** to ensure browser compatibility (older formats like WMV must be converted to modern codecs like H.264)
3. **Multiple rendition encoding** to create the quality variants needed for adaptive streaming
4. **Segmentation** to split each rendition into the small chunks required for HLS delivery
5. **Manifest generation** to create the M3U8 playlist files that coordinate playback

This entire pipeline can take anywhere from 1x to 10x the video's runtime to complete, depending on the complexity of the source material and the number of output renditions required.

| Streaming Approach | File Structure | Bandwidth Usage | Seeking Performance | Network Adaptation | Implementation Complexity |
|-------------------|----------------|-----------------|-------------------|-------------------|-------------------------|
| Progressive Download | Single large file | High (downloads entire file) | Poor (must download to seek point) | None | Low |
| Adaptive Streaming | Many small segments + manifest | Optimized (only downloads watched content) | Excellent (seeks between segments) | Real-time quality switching | High |

### Existing Approaches Comparison

**YouTube's architecture** prioritizes scale and global reach over premium video quality. With over 500 hours of content uploaded every minute, YouTube's transcoding pipeline emphasizes speed and efficiency over perfect quality. Their approach uses aggressive compression algorithms that can process uploads quickly, accepting some quality degradation in favor of faster time-to-publish.

YouTube generates a relatively conservative quality ladder, typically offering 360p, 720p, and 1080p renditions for most content, with 4K reserved for channels that meet specific criteria. This limitation helps control storage and bandwidth costs across their massive content library. Their segmentation strategy uses longer segment durations (typically 10 seconds) to reduce the number of HTTP requests and improve caching efficiency at their global CDN scale.

Their player implementation heavily emphasizes bandwidth detection and conservative quality switching. YouTube's algorithm tends to err on the side of lower quality to avoid buffering, based on their data showing that users abandon videos more frequently due to buffering interruptions than quality complaints. The player also pre-loads the next segment at multiple quality levels, allowing for instant switching when network conditions change.

> **Key YouTube Insight:** At massive scale, optimizing for consistency and availability often trumps optimizing for peak quality. Their architecture prioritizes ensuring every video plays smoothly over ensuring every video looks perfect.

**Netflix's architecture** takes the opposite approach, optimizing for premium viewing experiences over upload scale. Since Netflix controls their entire content library and ingests a relatively small number of high-quality source materials compared to user-generated platforms, they can afford extensive transcoding processing to maximize visual quality.

Netflix pioneered **per-title encoding**, where their transcoding pipeline analyzes each piece of content individually to determine the optimal quality ladder. A fast-action movie might require higher bitrates to maintain quality during motion scenes, while a dialogue-heavy drama can achieve excellent quality at lower bitrates. This analysis can take hours per video but results in significantly better quality-to-bandwidth ratios.

Their segmentation strategy uses shorter segments (typically 4 seconds) to enable more responsive quality switching. This approach generates more HTTP requests but allows the player to adapt more quickly to changing network conditions - crucial for mobile viewers whose connectivity varies dramatically as they move between cell towers and WiFi networks.

Netflix's player includes sophisticated bandwidth prediction algorithms that consider not just current network speed but also time-of-day patterns, device capabilities, and content-specific requirements. Their quality switching tends to be more aggressive than YouTube's, prioritizing visual quality for users with sufficient bandwidth while still maintaining smooth playback.

**Vimeo's architecture** targets creative professionals and premium content creators, emphasizing quality and creator control over massive scale. Their transcoding pipeline includes options for higher bitrate encoding and preserves more detail in the original uploaded content, accepting longer processing times in exchange for better visual fidelity.

Vimeo offers creators more control over their quality ladder, allowing manual specification of target bitrates and enabling features like download options for original files. Their player includes more granular quality controls, letting viewers manually select specific resolutions rather than just "high/medium/low" options.

Their infrastructure operates at a smaller scale than YouTube or Netflix, allowing for more specialized optimizations. Vimeo can afford to maintain higher per-video storage costs because their business model relies on subscription revenue from creators rather than advertising revenue that scales with view volume.

| Platform | Primary Optimization | Quality Ladder Strategy | Segment Duration | Transcoding Speed | Target Audience |
|----------|---------------------|------------------------|------------------|-------------------|----------------|
| YouTube | Scale & availability | Conservative, standardized | 10 seconds | Fast processing | General public |
| Netflix | Premium quality | Per-title optimization | 4 seconds | Extensive processing | Premium viewers |
| Vimeo | Creator control | Configurable bitrates | 6 seconds | Quality-focused | Creative professionals |

The **content delivery network (CDN) strategies** employed by these platforms also differ significantly. YouTube leverages Google's global infrastructure with edge servers positioned close to major internet exchange points. Their caching strategy aggressively caches popular content but relies on origin servers for long-tail content that may only be viewed occasionally.

Netflix operates their own CDN called Open Connect, placing storage appliances directly within internet service provider networks. This approach minimizes the network hops between Netflix content and end users, crucial for their high-bitrate 4K content that would otherwise consume enormous bandwidth crossing multiple network boundaries.

Vimeo relies primarily on third-party CDN providers, optimizing for cost-effectiveness rather than building custom infrastructure. This approach works well for their smaller scale but limits their ability to implement specialized optimizations for video delivery.

**Error handling and resilience** strategies also vary based on each platform's priorities. YouTube's approach focuses on graceful degradation - if high-quality segments fail to load, the player immediately falls back to lower quality rather than interrupting playback. Their massive scale means they encounter virtually every possible network condition and device configuration, so their error handling tends to be comprehensive but conservative.

Netflix implements more sophisticated retry logic, attempting multiple CDN endpoints and quality levels before giving up on a segment. Their business model depends on premium user experience, so they invest more heavily in ensuring playback succeeds even under challenging network conditions.

> **Decision: Architectural Philosophy for Our Platform**
> - **Context**: We need to choose which existing approach most closely aligns with our goals and constraints
> - **Options Considered**: YouTube's scale-optimized approach, Netflix's quality-optimized approach, Vimeo's creator-focused approach
> - **Decision**: Adopt a hybrid approach closer to Vimeo's model with selective Netflix-inspired optimizations
> - **Rationale**: Our platform serves an intermediate scale where we can afford higher per-video processing costs than YouTube but cannot match Netflix's infrastructure investment. Vimeo's balance provides a realistic target for implementation complexity while still delivering professional-quality results
> - **Consequences**: We'll implement per-video quality analysis (simplified version of Netflix's approach) but use standardized transcoding parameters (borrowing from YouTube's efficiency). This enables good quality without requiring extensive custom optimization infrastructure

Understanding these existing approaches provides the foundation for making informed architectural decisions in our own implementation. We'll borrow proven strategies from each platform while adapting them to our specific scale and requirements. The key insight is that video streaming architecture must be designed holistically - decisions about transcoding affect storage requirements, which impact CDN strategy, which influences player behavior, which determines user experience quality.

### Implementation Guidance

The implementation of our video streaming platform requires careful selection of technologies that balance functionality, performance, and development complexity. This section provides concrete recommendations for building each component using Node.js as our primary language.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Upload Service | Express.js + Multer for multipart uploads | Express.js + custom chunked upload middleware |
| File Storage | Local filesystem with organized directories | AWS S3 or Google Cloud Storage with signed URLs |
| Transcoding | FFmpeg subprocess calls with progress parsing | Bull queue + Redis for job management |
| Streaming Service | Express.js static file serving with custom headers | Express.js + CDN integration (CloudFront/CloudFlare) |
| Database | SQLite for development, PostgreSQL for production | PostgreSQL with Redis caching layer |
| Video Player | Video.js with HLS.js plugin | Custom HLS.js implementation with analytics |

**B. Recommended Project Structure:**

```
video-streaming-platform/
├── src/
│   ├── services/
│   │   ├── upload/
│   │   │   ├── uploadService.js       ← handles chunked uploads
│   │   │   ├── fileValidator.js       ← validates video files
│   │   │   └── metadataExtractor.js   ← extracts video metadata
│   │   ├── transcoding/
│   │   │   ├── transcodingService.js  ← manages FFmpeg jobs
│   │   │   ├── qualityLadder.js       ← determines output qualities
│   │   │   └── hlsGenerator.js        ← creates HLS manifests
│   │   ├── streaming/
│   │   │   ├── streamingService.js    ← serves HLS content
│   │   │   ├── manifestGenerator.js   ← creates M3U8 files
│   │   │   └── segmentServer.js       ← serves TS segments
│   │   └── shared/
│   │       ├── database.js            ← database connection
│   │       ├── storage.js             ← file storage abstraction
│   │       └── logger.js              ← structured logging
│   ├── models/
│   │   ├── Video.js                   ← video entity model
│   │   ├── TranscodingJob.js          ← transcoding job model
│   │   └── HLSManifest.js             ← manifest metadata model
│   ├── routes/
│   │   ├── upload.js                  ← upload API endpoints
│   │   ├── streaming.js               ← streaming API endpoints
│   │   └── admin.js                   ← management endpoints
│   └── public/
│       ├── player.html                ← video player interface
│       ├── player.js                  ← HLS.js integration
│       └── style.css                  ← player styling
├── storage/
│   ├── uploads/                       ← original uploaded files
│   ├── transcoded/                    ← HLS output files
│   └── temp/                          ← temporary processing files
├── scripts/
│   ├── setup-database.js              ← database initialization
│   └── cleanup-storage.js             ← storage maintenance
└── tests/
    ├── integration/
    └── unit/
```

**C. Infrastructure Starter Code:**

The following storage abstraction provides a unified interface for both local filesystem and cloud storage, allowing easy switching between development and production environments:

```javascript
// src/services/shared/storage.js
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class StorageService {
  constructor(config = {}) {
    this.basePath = config.basePath || './storage';
    this.ensureDirectories();
  }

  async ensureDirectories() {
    const dirs = ['uploads', 'transcoded', 'temp'];
    for (const dir of dirs) {
      await fs.mkdir(path.join(this.basePath, dir), { recursive: true });
    }
  }

  generateUniqueFilename(originalName) {
    const timestamp = Date.now();
    const randomBytes = crypto.randomBytes(8).toString('hex');
    const extension = path.extname(originalName);
    return `${timestamp}_${randomBytes}${extension}`;
  }

  async saveUploadedFile(buffer, originalName, category = 'uploads') {
    const filename = this.generateUniqueFilename(originalName);
    const filepath = path.join(this.basePath, category, filename);
    
    await fs.writeFile(filepath, buffer);
    
    return {
      filename,
      filepath,
      url: `/storage/${category}/${filename}`,
      size: buffer.length
    };
  }

  async getFileStats(filepath) {
    try {
      const stats = await fs.stat(filepath);
      return {
        exists: true,
        size: stats.size,
        created: stats.birthtime,
        modified: stats.mtime
      };
    } catch (error) {
      return { exists: false };
    }
  }

  async deleteFile(filepath) {
    try {
      await fs.unlink(filepath);
      return true;
    } catch (error) {
      return false;
    }
  }

  getStoragePath(category, filename) {
    return path.join(this.basePath, category, filename);
  }
}

module.exports = StorageService;
```

Database connection and model base class:

```javascript
// src/services/shared/database.js
const sqlite3 = require('sqlite3').verbose();
const { promisify } = require('util');

class DatabaseService {
  constructor(dbPath = './video_platform.db') {
    this.db = new sqlite3.Database(dbPath);
    this.run = promisify(this.db.run.bind(this.db));
    this.get = promisify(this.db.get.bind(this.db));
    this.all = promisify(this.db.all.bind(this.db));
    this.initializeTables();
  }

  async initializeTables() {
    // Videos table
    await this.run(`
      CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        duration REAL,
        width INTEGER,
        height INTEGER,
        codec TEXT,
        bitrate INTEGER,
        status TEXT DEFAULT 'uploaded',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Transcoding jobs table
    await this.run(`
      CREATE TABLE IF NOT EXISTS transcoding_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        status TEXT DEFAULT 'pending',
        progress REAL DEFAULT 0,
        error_message TEXT,
        started_at DATETIME,
        completed_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos (id)
      )
    `);

    // HLS manifests table
    await this.run(`
      CREATE TABLE IF NOT EXISTS hls_manifests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        quality TEXT NOT NULL,
        manifest_path TEXT NOT NULL,
        segment_duration REAL DEFAULT 6.0,
        segment_count INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos (id)
      )
    `);
  }

  async close() {
    return new Promise((resolve) => {
      this.db.close(resolve);
    });
  }
}

module.exports = DatabaseService;
```

**D. Core Logic Skeleton Code:**

```javascript
// src/services/upload/uploadService.js
const multer = require('multer');
const StorageService = require('../shared/storage');
const DatabaseService = require('../shared/database');

class UploadService {
  constructor() {
    this.storage = new StorageService();
    this.db = new DatabaseService();
    this.setupMulter();
  }

  setupMulter() {
    // TODO 1: Configure multer for memory storage to handle chunks
    // TODO 2: Set file size limits and allowed MIME types
    // TODO 3: Create multer middleware for single file uploads
    // Hint: Use memoryStorage() to keep files in memory for processing
  }

  async handleChunkedUpload(req, res) {
    // TODO 1: Extract chunk information from headers (chunk index, total chunks, file hash)
    // TODO 2: Validate chunk size and sequence
    // TODO 3: Save chunk to temporary storage with unique identifier
    // TODO 4: Check if all chunks received, if so trigger assembly
    // TODO 5: Return progress information to client
    // Hint: Use req.headers['x-chunk-index'] and req.headers['x-total-chunks']
  }

  async assembleFileChunks(fileHash, totalChunks) {
    // TODO 1: Read all chunk files in correct order
    // TODO 2: Concatenate chunks into complete file buffer
    // TODO 3: Verify assembled file integrity using hash comparison
    // TODO 4: Save complete file using storage service
    // TODO 5: Clean up temporary chunk files
    // TODO 6: Extract metadata and save video record to database
  }

  async validateVideoFile(buffer, originalName) {
    // TODO 1: Check file extension against allowed types (mp4, avi, mov, etc)
    // TODO 2: Verify file size is within limits
    // TODO 3: Use FFprobe to validate file is actually a video
    // TODO 4: Extract basic metadata (duration, resolution, codec)
    // TODO 5: Return validation result with metadata or error details
  }
}

module.exports = UploadService;
```

```javascript
// src/services/transcoding/transcodingService.js
const { spawn } = require('child_process');
const path = require('path');

class TranscodingService {
  constructor() {
    this.activeJobs = new Map();
    this.qualityLadder = [
      { name: '360p', width: 640, height: 360, bitrate: '800k' },
      { name: '720p', width: 1280, height: 720, bitrate: '2500k' },
      { name: '1080p', width: 1920, height: 1080, bitrate: '5000k' }
    ];
  }

  async startTranscoding(videoId, inputPath, outputDir) {
    // TODO 1: Create transcoding job record in database with 'pending' status
    // TODO 2: Analyze input video to determine appropriate quality levels
    // TODO 3: Create output directory structure for HLS files
    // TODO 4: For each quality level, spawn FFmpeg process for HLS transcoding
    // TODO 5: Monitor FFmpeg progress and update job status
    // TODO 6: Generate master playlist when all qualities complete
    // Hint: Use FFmpeg -hls_time 6 -hls_list_size 0 for HLS output
  }

  spawnFFmpegProcess(inputPath, outputPath, quality) {
    // TODO 1: Build FFmpeg command arguments for HLS transcoding
    // TODO 2: Include quality-specific settings (resolution, bitrate)
    // TODO 3: Set HLS segment duration and naming pattern
    // TODO 4: Spawn child process with proper error handling
    // TODO 5: Set up progress monitoring by parsing FFmpeg stderr output
    // TODO 6: Return promise that resolves when transcoding completes
  }

  parseFFmpegProgress(progressLine, duration) {
    // TODO 1: Extract timestamp from FFmpeg progress line using regex
    // TODO 2: Convert time format (HH:MM:SS.mmm) to seconds
    // TODO 3: Calculate percentage: (current_time / total_duration) * 100
    // TODO 4: Update job progress in database
    // TODO 5: Emit progress event for real-time updates
  }
}

module.exports = TranscodingService;
```

**E. Language-Specific Hints:**

- **FFmpeg Integration**: Use `child_process.spawn()` rather than `exec()` for better control over long-running transcoding processes
- **Progress Monitoring**: Parse FFmpeg stderr output using regex patterns like `/time=(\d{2}):(\d{2}):(\d{2}\.\d{2})/`
- **File Streaming**: Use `fs.createReadStream()` with proper range header support for serving video segments
- **Async Processing**: Implement job queues using Bull + Redis for production, or simple in-memory queues for development
- **Error Handling**: Wrap FFmpeg processes with timeout handling - transcoding jobs can hang indefinitely on corrupt input
- **Memory Management**: Use streams instead of loading entire video files into memory, especially for large uploads

**F. Milestone Checkpoints:**

After implementing the upload service (Milestone 1):
- **Test Command**: `curl -X POST -F "video=@test.mp4" http://localhost:3000/api/upload`
- **Expected Response**: JSON with upload confirmation, file ID, and metadata
- **Verification**: Check that uploaded file appears in `./storage/uploads/` directory
- **Database Check**: Query videos table should show new record with extracted metadata

After implementing transcoding pipeline (Milestone 2):
- **Test Command**: Start transcoding via API endpoint or direct service call
- **Expected Behavior**: FFmpeg processes spawn and create HLS files in `./storage/transcoded/[videoId]/`
- **File Verification**: Should see `.m3u8` playlist files and `.ts` segment files for each quality level
- **Progress Monitoring**: Job status should update from 'pending' → 'processing' → 'completed'

After implementing streaming service (Milestone 3):
- **Test Command**: `curl http://localhost:3000/stream/[videoId]/master.m3u8`
- **Expected Response**: Valid M3U8 master playlist with quality variant references
- **Player Test**: Load master playlist URL in HLS.js test player
- **Verification**: Video should play with quality switching capabilities

**G. Common Implementation Pitfalls:**

⚠️ **Pitfall: Loading Large Files into Memory**
- **Problem**: Using `fs.readFileSync()` or storing entire uploaded videos in memory causes crashes with large files
- **Diagnosis**: Node.js process memory usage spikes during upload processing
- **Fix**: Use streams (`fs.createReadStream()`) and process files in chunks rather than loading entirely into memory

⚠️ **Pitfall: FFmpeg Process Orphaning**
- **Problem**: FFmpeg child processes continue running after Node.js application crashes or restarts
- **Diagnosis**: Check running processes with `ps aux | grep ffmpeg` after application shutdown
- **Fix**: Implement proper cleanup in process exit handlers and store process PIDs for cleanup on startup

⚠️ **Pitfall: CORS Issues with Video Streaming**
- **Problem**: Browser blocks HLS manifest or segment requests due to CORS policy
- **Diagnosis**: Browser console shows CORS errors when loading video player
- **Fix**: Configure Express to send proper CORS headers for all streaming endpoints

⚠️ **Pitfall: Incomplete Chunk Upload Handling**
- **Problem**: Network interruptions leave partial chunk files that never get cleaned up
- **Diagnosis**: Temporary storage directory accumulates orphaned chunk files over time
- **Fix**: Implement chunk timeout cleanup and startup cleanup scan for orphaned files


## Goals and Non-Goals

> **Milestone(s):** All milestones (1-4) - These goals and constraints guide every architectural decision throughout the video streaming platform development.

Building a video streaming platform requires careful scoping to balance feature completeness with implementation complexity. Think of this goals definition as drawing the blueprint boundaries for a house - we need to be crystal clear about what rooms we're building, what we're explicitly leaving out, and why those decisions matter for the overall architecture.

The streaming platform's core mission centers on the fundamental workflow that millions of users experience daily: uploading a video file, having the system automatically convert it into a format optimized for streaming, and then playing that video back with smooth quality adaptation based on network conditions. This seemingly simple user journey involves intricate technical challenges around large file handling, media processing pipelines, HTTP streaming protocols, and browser compatibility.

### Primary Goals

The platform must accomplish four essential capabilities that define a complete streaming solution. Each goal directly maps to one of our implementation milestones and represents a non-negotiable requirement for a functional video streaming service.

**Large File Upload Handling with Resumability**

The platform must accept video file uploads ranging from small clips to multi-gigabyte productions. Think of this like building a robust cargo loading system at a shipping port - we need to handle everything from small packages to massive containers, with the ability to resume loading if a crane breaks down mid-operation. The system must support **chunked upload** protocols that allow clients to transmit large files in small sequential pieces, enabling recovery from network interruptions without starting over.

This capability requires implementing progress tracking that reports both bytes uploaded and estimated completion percentage, file validation that rejects unsupported formats and oversized files before processing begins, and storage management that organizes uploaded content with unique identifiers and logical directory structures. The upload service must gracefully handle partial failures, cleanup incomplete uploads, and avoid memory exhaustion when processing files that exceed available RAM.

**Automated Video Transcoding Pipeline**

Every uploaded video must be automatically converted into streaming-optimized formats regardless of the original codec, resolution, or container format. This transcoding pipeline functions like an assembly line in a manufacturing plant - raw materials (source videos) enter one end, pass through multiple processing stations (FFmpeg operations), and emerge as finished products (HLS segments and manifests) ready for distribution.

The transcoding system must generate multiple **quality renditions** from each source file, typically including 360p, 720p, and 1080p variants to support adaptive streaming. Background job processing ensures that transcoding operations run asynchronously without blocking new uploads, while progress monitoring provides real-time feedback on completion percentage and estimated time remaining. The pipeline must handle codec compatibility issues, manage FFmpeg memory usage, and recover gracefully from interrupted transcoding jobs.

**HTTP Live Streaming (HLS) Delivery**

The platform must serve transcoded videos using industry-standard HLS protocol, which breaks videos into small segments and provides manifest files that describe available quality levels. Think of HLS serving like operating a sophisticated restaurant menu system - the master playlist acts as the main menu showing available options, quality variant playlists are like detailed descriptions of each dish, and individual video segments are the actual food portions delivered to customers.

HLS delivery requires generating valid M3U8 manifest files that reference all video segments in proper sequence, serving TS (Transport Stream) files with correct MIME type headers and caching directives, supporting HTTP range requests for efficient seeking within videos, and implementing CDN-friendly headers that enable edge caching for global content distribution. The streaming service must handle CORS requirements for cross-origin players and maintain manifest consistency during ongoing transcoding operations.

**Adaptive Quality Video Player**

The platform must provide a web-based video player that automatically adjusts video quality based on available bandwidth while allowing manual quality selection. This player functions like an intelligent television that constantly monitors your internet connection and adjusts picture quality to prevent buffering, while also providing manual controls when you want to override the automatic decisions.

Player integration requires incorporating HLS.js library for browsers without native HLS support, implementing quality switching UI that displays available renditions and current selection, providing standard playback controls for play/pause/seek/volume operations, and tracking analytics events including view duration, quality switches, and buffering occurrences. The player must handle browser compatibility issues, prevent memory leaks during component unmounting, and provide accurate bandwidth estimation for adaptive decisions.

### Performance and Scale Requirements

The platform must handle concurrent operations efficiently while maintaining responsive user experience across all components. Upload performance should support multiple simultaneous file transfers without degrading individual upload speeds, with each chunked upload session maintaining independent progress tracking and resumability state.

Transcoding throughput must scale with available system resources, supporting configurable concurrency limits that prevent resource exhaustion while maximizing parallel job processing. The **transcoding pipeline** should complete standard definition videos within 2x real-time duration (a 10-minute video transcoded in under 20 minutes) and high definition content within 4x real-time duration.

Streaming delivery must support hundreds of concurrent video playback sessions with sub-second manifest request response times and efficient segment serving that leverages HTTP caching headers. The HLS serving infrastructure should integrate seamlessly with content delivery networks to enable global distribution without requiring application-level geographic awareness.

Player performance requires smooth playback initialization within 3 seconds of user request, seamless quality switching without playback interruption, and accurate seeking that lands within 1 second of the requested position. Analytics collection must operate without impacting playback performance or introducing privacy concerns for user viewing behavior.

### Technical Quality Standards

The streaming platform must implement robust error handling and recovery mechanisms throughout the entire video processing pipeline. Upload failures should preserve partial progress and enable resumption from the last completed chunk, while providing clear error messages that distinguish between temporary network issues and permanent validation failures.

Transcoding errors require comprehensive logging that captures FFmpeg output, input file characteristics, and system resource state at failure time. The job queue must implement retry logic with exponential backoff for transient failures while marking jobs as permanently failed after repeated unsuccessful attempts. Failed transcoding jobs should not prevent successful uploads from proceeding through the pipeline.

Streaming delivery must handle missing segments gracefully, serve appropriate HTTP status codes for various error conditions, and maintain manifest consistency even when transcoding operations are incomplete. The player should implement buffering strategies that prevent playback interruption during temporary network issues while providing user feedback during extended connectivity problems.

Security considerations include file upload validation that prevents malicious content injection, input sanitization for all user-provided metadata, and proper HTTP headers that prevent cross-site scripting attacks. The platform should implement reasonable rate limiting to prevent abuse while avoiding impact on legitimate usage patterns.

### Non-Goals and Explicit Exclusions

Clearly defining what the platform will NOT implement is crucial for maintaining focus and preventing scope creep during development. These exclusions represent conscious architectural decisions that simplify implementation while still delivering a complete streaming solution.

**Live Streaming Capabilities**

The platform explicitly excludes real-time video streaming where content is broadcast as it's being recorded. Live streaming requires fundamentally different architecture patterns including real-time ingest protocols (RTMP/WebRTC), ultra-low latency delivery mechanisms, and immediate transcoding pipelines that process video frames as they arrive rather than operating on complete files.

This exclusion eliminates the need for complex buffering strategies, real-time encoder configuration, and the sophisticated load balancing required to handle thousands of simultaneous live streams. The platform focuses exclusively on video-on-demand (VOD) scenarios where complete files are uploaded before any streaming begins.

**Digital Rights Management (DRM)**

Content protection and encryption capabilities are explicitly excluded from the initial platform design. DRM implementation requires integration with third-party licensing servers, encrypted segment generation, player modifications for license acquisition, and complex key rotation mechanisms that significantly increase architectural complexity.

This exclusion allows the streaming service to use standard HLS segments and manifests without encryption overhead, simplifies player implementation by avoiding license management, and eliminates the need for secure key storage and distribution infrastructure.

**Advanced Video Processing**

The platform excludes sophisticated video editing capabilities including thumbnail generation, automated content analysis, closed caption extraction, multi-language audio track handling, and video enhancement filters. These features require additional FFmpeg expertise, expanded storage for derivative content, and complex UI for managing multiple content variants.

The transcoding pipeline focuses solely on format conversion and quality adaptation, using standard FFmpeg presets without custom video processing logic. This constraint ensures reliable transcoding performance while avoiding the complexity of managing multiple processing workflows per uploaded video.

**User Management and Authentication**

The platform excludes comprehensive user account systems, authentication mechanisms, authorization controls, and content access restrictions. This exclusion eliminates the need for user databases, session management, password handling, and the complex permission systems required for multi-tenant video platforms.

The streaming service operates as a single-tenant system where all uploaded content is publicly accessible through direct URL access. This simplification allows focus on the core streaming functionality without the substantial complexity of user identity management.

**Content Management Features**

Advanced content organization capabilities including video categorization, search functionality, recommendation engines, and content moderation tools are explicitly excluded. These features require additional database schemas, search indexing infrastructure, and machine learning capabilities that extend far beyond core streaming functionality.

The platform provides basic video storage and retrieval through unique identifiers without sophisticated content discovery mechanisms. Users must maintain their own catalogs of uploaded video URLs and metadata outside the streaming platform.

**Analytics and Monitoring Dashboards**

While the platform collects basic playback analytics, it excludes comprehensive reporting dashboards, usage analytics visualization, and operational monitoring interfaces. Advanced analytics require additional storage systems, data processing pipelines, and web interface development that extends beyond the core streaming functionality.

Analytics collection focuses on essential playback events needed for debugging and basic usage understanding, but visualization and reporting remain outside the platform's scope.

### Architecture Decision Records

> **Decision: Video-on-Demand Focus Over Live Streaming**
> - **Context**: Video platforms can support either pre-recorded content (VOD) or real-time broadcasts (live streaming), with fundamentally different technical requirements and complexity levels.
> - **Options Considered**: VOD-only platform, live streaming-only platform, hybrid platform supporting both modes
> - **Decision**: Implement video-on-demand exclusively, excluding all live streaming capabilities
> - **Rationale**: VOD allows complete files to be processed offline with quality optimization, eliminates real-time processing constraints, and reduces infrastructure complexity by removing the need for immediate transcoding and ultra-low latency delivery. Live streaming requires different ingest protocols (RTMP), real-time transcoding, and specialized CDN configurations that would triple implementation complexity.
> - **Consequences**: Simplifies transcoding pipeline design since complete files are always available, enables high-quality multi-pass encoding optimization, and allows background job processing without time pressure. However, excludes real-time broadcasting use cases and requires users to upload complete videos before any streaming can begin.

| Option | Pros | Cons |
|--------|------|------|
| VOD-only | Complete file processing, offline optimization, simpler architecture | No real-time broadcasting, upload-before-stream requirement |
| Live streaming-only | Real-time interaction, immediate broadcast capability | Complex real-time processing, limited quality optimization |
| Hybrid platform | Supports all use cases, maximum flexibility | Doubled architecture complexity, resource management challenges |

> **Decision: HLS Protocol Over DASH or Progressive Download**
> - **Context**: Video delivery requires choosing between progressive download (entire file), HTTP Live Streaming (HLS), or Dynamic Adaptive Streaming over HTTP (DASH) protocols.
> - **Options Considered**: Progressive MP4 download, HLS with M3U8 manifests, DASH with MPD manifests
> - **Decision**: Implement HLS exclusively for all video delivery
> - **Rationale**: HLS provides excellent browser support through HLS.js, enables adaptive bitrate streaming for varying network conditions, and uses simple HTTP infrastructure without requiring specialized streaming servers. DASH offers similar capabilities but requires more complex manifest generation and has less universal browser support. Progressive download provides no quality adaptation and requires users to download entire files.
> - **Consequences**: Enables adaptive quality streaming for optimal user experience, simplifies CDN integration through standard HTTP serving, and provides broad device compatibility. However, requires video segmentation during transcoding and adds manifest serving complexity compared to direct file serving.

| Option | Pros | Cons |
|--------|------|------|
| HLS | Broad browser support, adaptive streaming, HTTP-based serving | Requires segmentation, manifest complexity |
| DASH | Industry standard, advanced features, codec flexibility | Limited browser support, complex manifests |
| Progressive download | Simple implementation, direct file serving | No quality adaptation, large bandwidth requirements |

> **Decision: Background Job Queue Over Real-time Transcoding**
> - **Context**: Video transcoding can be performed immediately during upload (synchronous) or queued for background processing (asynchronous).
> - **Options Considered**: Synchronous transcoding blocking upload response, asynchronous background job queue, hybrid approach with small previews
> - **Decision**: Implement asynchronous background transcoding with job queue management
> - **Rationale**: Video transcoding often takes longer than typical HTTP request timeouts, especially for high-resolution content. Background processing allows users to upload multiple files concurrently, provides better resource utilization through controlled concurrency, and enables retry logic for failed transcoding attempts. Synchronous processing would create poor user experience with long-running upload requests.
> - **Consequences**: Enables concurrent uploads without blocking on transcoding completion, provides better system resource management through job prioritization, and allows comprehensive error handling with retry mechanisms. However, introduces complexity of job state tracking and requires users to poll for transcoding completion status.

| Option | Pros | Cons |
|--------|------|------|
| Background queue | Concurrent processing, resource control, retry capability | State tracking complexity, polling required |
| Synchronous transcoding | Immediate results, simpler state management | Request timeouts, blocking behavior |
| Hybrid approach | Quick previews, eventual full processing | Doubled transcoding work, complex coordination |

### Common Pitfalls

⚠️ **Pitfall: Scope Creep Through Feature Addition**

Teams frequently expand platform scope by adding "simple" features like user authentication, thumbnail generation, or content categorization without recognizing their architectural impact. Each additional feature introduces new failure modes, increases testing complexity, and creates dependencies that complicate the core streaming functionality.

This scope expansion typically occurs when developers encounter limitations during implementation and attempt to solve them through feature addition rather than focusing on the core streaming pipeline. For example, discovering that videos need identification might lead to implementing a full content management system rather than using simple unique identifiers.

To avoid scope creep, maintain strict adherence to the defined goals and explicitly reject feature requests that extend beyond video upload, transcoding, streaming, and playback. Document rejected features in a future enhancements list rather than implementing them immediately, and evaluate any proposed additions against the impact on core streaming functionality reliability and performance.

⚠️ **Pitfall: Underestimating Non-Functional Requirements**

Developers often focus exclusively on functional capabilities (upload works, transcoding produces output, player shows video) while neglecting performance, reliability, and error handling requirements that distinguish prototype code from production systems.

This oversight typically manifests as systems that work perfectly with small test files but fail catastrophically with realistic video sizes, concurrent usage, or network instability. The upload service might handle 10MB test videos flawlessly but run out of memory with 2GB production files, or the transcoding pipeline might work reliably until multiple jobs run simultaneously and exhaust system resources.

Address non-functional requirements from the beginning by testing with realistic file sizes, implementing proper error handling and recovery mechanisms, and designing for concurrent usage patterns. Use production-like test scenarios including large files, network interruptions, and resource constraints to validate system behavior under stress.

⚠️ **Pitfall: Ignoring Browser Compatibility and Edge Cases**

Video streaming involves complex interactions between server-side processing and client-side playback across diverse browsers, devices, and network conditions. Teams often test exclusively in a single browser with perfect network conditions, missing compatibility issues and edge cases that affect real users.

Browser-specific behaviors around HLS support, seeking accuracy, quality switching responsiveness, and error handling can vary significantly between Chrome, Firefox, Safari, and mobile browsers. Network conditions including high latency, packet loss, and bandwidth fluctuations create additional complexity that doesn't appear in local development environments.

Validate streaming functionality across multiple browsers and devices throughout development, test with simulated network conditions including high latency and packet loss, and implement comprehensive error handling for playback edge cases. Use browser developer tools to simulate various network conditions and device capabilities during player integration testing.

### Implementation Guidance

The goals and constraints defined above directly influence technology selection and architectural patterns throughout the streaming platform implementation. Understanding these boundaries helps make informed decisions about frameworks, libraries, and infrastructure components.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Upload Handling | Express.js + Multer for multipart uploads | Custom Node.js streams with resumable-js |
| File Storage | Local filesystem with organized directories | AWS S3 or Google Cloud Storage |
| Transcoding | Direct FFmpeg subprocess execution | Bull queue + Redis for job management |
| Database | SQLite for development simplicity | PostgreSQL for production reliability |
| Video Serving | Express.js static file serving | Nginx with optimized caching headers |
| Player Integration | HLS.js with basic HTML5 video element | Video.js with advanced plugin ecosystem |

#### Recommended Project Structure

The goals-driven architecture supports a clean separation of concerns across the video processing pipeline:

```
video-streaming-platform/
├── src/
│   ├── upload/
│   │   ├── upload-service.js      ← handles chunked uploads and validation
│   │   ├── storage-service.js     ← file system abstraction
│   │   └── metadata-extractor.js  ← FFprobe integration
│   ├── transcoding/
│   │   ├── transcoding-service.js ← FFmpeg process management
│   │   ├── job-queue.js          ← background job processing
│   │   └── quality-profiles.js   ← encoding presets
│   ├── streaming/
│   │   ├── hls-service.js        ← manifest and segment serving
│   │   ├── manifest-generator.js ← M3U8 file creation
│   │   └── segment-server.js     ← TS file delivery
│   ├── player/
│   │   ├── video-player.js       ← HLS.js integration
│   │   ├── quality-selector.js   ← manual quality switching
│   │   └── analytics-tracker.js  ← playback event collection
│   ├── shared/
│   │   ├── database.js           ← connection and query helpers
│   │   ├── config.js             ← environment-specific settings
│   │   └── logger.js             ← structured logging utility
│   └── server.js                 ← Express.js application entry point
├── public/
│   ├── player.html               ← video player interface
│   └── upload.html               ← file upload interface
├── storage/
│   ├── uploads/                  ← original video files
│   ├── transcoded/               ← HLS output directories
│   └── manifests/                ← M3U8 playlist files
└── package.json
```

#### Infrastructure Starter Code

**Configuration Management**

```javascript
// src/shared/config.js
const path = require('path');

const config = {
  // Upload constraints based on goals
  upload: {
    maxFileSize: 5 * 1024 * 1024 * 1024, // 5GB limit
    chunkSize: 10 * 1024 * 1024,         // 10MB chunks for resumability
    allowedFormats: ['mp4', 'avi', 'mov', 'mkv', 'webm'],
    uploadDir: path.join(__dirname, '../../storage/uploads')
  },

  // Transcoding settings aligned with quality goals
  transcoding: {
    outputDir: path.join(__dirname, '../../storage/transcoded'),
    segmentDuration: 6, // seconds per HLS segment
    qualityLadder: [
      { name: '360p', width: 640, height: 360, bitrate: '800k' },
      { name: '720p', width: 1280, height: 720, bitrate: '2500k' },
      { name: '1080p', width: 1920, height: 1080, bitrate: '5000k' }
    ],
    concurrency: 2 // parallel FFmpeg processes
  },

  // Streaming delivery configuration
  streaming: {
    manifestDir: path.join(__dirname, '../../storage/manifests'),
    segmentCacheTTL: 3600, // 1 hour cache for TS files
    manifestCacheTTL: 30,  // 30 second cache for M3U8
    corsOrigins: ['http://localhost:3000'] // adjust for production
  },

  // Database connection (SQLite for simplicity)
  database: {
    filename: path.join(__dirname, '../../storage/videos.db'),
    options: {
      // Enable WAL mode for concurrent reads during transcoding
      pragma: {
        journal_mode: 'WAL',
        foreign_keys: 'ON'
      }
    }
  }
};

module.exports = config;
```

**Database Schema Setup**

```javascript
// src/shared/database.js
const sqlite3 = require('sqlite3').verbose();
const { open } = require('sqlite');
const config = require('./config');

class DatabaseService {
  constructor() {
    this.db = null;
  }

  async connect() {
    this.db = await open({
      filename: config.database.filename,
      driver: sqlite3.Database
    });

    // Apply pragmas for performance and consistency
    await this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA foreign_keys = ON;
      PRAGMA synchronous = NORMAL;
    `);

    await this.createTables();
  }

  async createTables() {
    // Video metadata table supporting goals
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS videos (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        original_path TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        duration REAL,
        width INTEGER,
        height INTEGER,
        codec TEXT,
        upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'uploaded'
      )
    `);

    // Transcoding job tracking for background processing
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS transcoding_jobs (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        progress INTEGER DEFAULT 0,
        started_at DATETIME,
        completed_at DATETIME,
        error_message TEXT,
        FOREIGN KEY (video_id) REFERENCES videos (id)
      )
    `);

    // HLS manifest tracking for streaming delivery
    await this.db.exec(`
      CREATE TABLE IF NOT EXISTS hls_manifests (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        quality TEXT NOT NULL,
        manifest_path TEXT NOT NULL,
        segment_count INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos (id)
      )
    `);
  }

  // Promise-based query methods for easier async handling
  async get(sql, params = []) {
    return await this.db.get(sql, params);
  }

  async all(sql, params = []) {
    return await this.db.all(sql, params);
  }

  async run(sql, params = []) {
    return await this.db.run(sql, params);
  }
}

module.exports = DatabaseService;
```

#### Core Logic Skeleton

**Goals Validation Service**

```javascript
// src/shared/goals-validator.js
const config = require('./config');

class GoalsValidator {
  
  // Validates uploaded files against platform goals and constraints
  static validateUploadFile(file, metadata) {
    // TODO 1: Check file size against maxFileSize limit from goals
    // TODO 2: Verify file format is in allowedFormats list
    // TODO 3: Validate video metadata has required properties (duration, dimensions)
    // TODO 4: Ensure file is not corrupted based on metadata extraction success
    // Hint: Return { valid: boolean, errors: string[] } for detailed feedback
  }

  // Ensures transcoding configuration meets quality goals
  static validateTranscodingConfig(inputMetadata, outputConfig) {
    // TODO 1: Check if input resolution supports requested output qualities
    // TODO 2: Validate bitrate settings don't exceed reasonable limits
    // TODO 3: Ensure segment duration meets streaming goals (4-10 seconds optimal)
    // TODO 4: Verify output directory structure exists and is writable
    // Hint: Don't upscale videos - filter quality ladder to input resolution or lower
  }

  // Validates HLS output meets streaming delivery goals
  static validateHLSOutput(manifestPath, segmentPaths) {
    // TODO 1: Verify manifest file is valid M3U8 format
    // TODO 2: Check all referenced segments actually exist on disk
    // TODO 3: Validate segment durations match configured SEGMENT_DURATION
    // TODO 4: Ensure total manifest duration matches original video duration
    // Hint: Use m3u8-parser library for manifest validation
  }

  // Checks if platform performance goals are being met
  static validatePerformanceMetrics(metrics) {
    // TODO 1: Verify upload chunk processing stays under timeout limits
    // TODO 2: Check transcoding time doesn't exceed 4x real-time for HD content  
    // TODO 3: Validate manifest serving responds within sub-second requirements
    // TODO 4: Ensure concurrent job processing doesn't exceed resource limits
    // Hint: metrics should include timestamps, resource usage, and success rates
  }
}

module.exports = GoalsValidator;
```

#### Milestone Checkpoints

**Milestone 1 Checkpoint: Upload Goals Verification**
- Run upload service with test files of various sizes (1MB, 100MB, 1GB)
- Verify chunked upload support by interrupting network mid-upload and resuming
- Check file validation rejects unsupported formats (.txt, .exe) and oversized files
- Confirm progress tracking reports accurate percentages during large file uploads
- Expected behavior: Uploads complete successfully, progress updates smoothly, invalid files rejected with clear error messages

**Milestone 2 Checkpoint: Transcoding Goals Verification**
- Upload test videos and verify background transcoding jobs are created
- Check that multiple quality levels (360p, 720p, 1080p) are generated for each video
- Monitor transcoding progress reporting during FFmpeg processing
- Confirm transcoding doesn't block new uploads (test concurrent operations)
- Expected behavior: Quality ladder output generated, progress updates correctly, system remains responsive

**Milestone 3 Checkpoint: Streaming Goals Verification**
- Request HLS manifests and verify valid M3U8 format with correct segment references
- Test video segment serving with proper MIME types and caching headers
- Verify byte-range requests work for seeking within video segments
- Check CORS headers allow cross-origin player requests
- Expected behavior: Manifests load correctly, segments stream smoothly, seeking works accurately

**Milestone 4 Checkpoint: Player Goals Verification**
- Load player with HLS stream and verify automatic quality adaptation
- Test manual quality switching between available renditions
- Confirm playback controls (play/pause/seek/volume) function correctly
- Verify analytics tracking captures playback events without performance impact
- Expected behavior: Smooth playback experience, responsive quality switching, accurate seeking


## High-Level Architecture

> **Milestone(s):** All milestones (1-4) - The high-level architecture provides the foundation for implementing upload, transcoding, streaming, and player components.

Building a video streaming platform requires orchestrating multiple complex processes that traditionally operate independently. Think of it like a modern automobile assembly line - raw materials (uploaded videos) enter at one end, pass through specialized stations (upload validation, transcoding, segmentation), and emerge as finished products (streamable content) that customers can consume immediately. Each station has specific responsibilities and must coordinate seamlessly with adjacent stations while handling failures gracefully.

The fundamental challenge in video streaming architecture is managing the **impedance mismatch** between different operational characteristics. Upload operations are bursty and user-driven, transcoding is CPU-intensive and time-consuming, streaming requires low-latency response, and players need adaptive behavior. A monolithic architecture would create bottlenecks and single points of failure, while a poorly designed distributed architecture would suffer from coordination overhead and consistency problems.

Our architecture addresses these challenges by decomposing the system into four specialized components that communicate through well-defined interfaces and shared storage. Each component can scale independently, fail independently, and be developed by separate teams while maintaining overall system coherence.

![System Architecture Overview](./diagrams/system-architecture.svg)

### Component Overview

The video streaming platform architecture centers on **separation of concerns** - each component owns a specific phase of the video lifecycle and provides clear interfaces to adjacent components. Think of this like a restaurant kitchen where prep cooks, line cooks, and expeditors each have distinct responsibilities but coordinate through standardized communication protocols (tickets, plating standards, timing cues).

#### Upload Service Component

The **Upload Service** acts as the system's front door, handling all incoming video content and user interactions. Its primary responsibility is transforming unreliable, variable-speed internet uploads into reliably stored, validated video assets that the rest of the system can process confidently.

| Responsibility | Description | Boundary Conditions |
|----------------|-------------|-------------------|
| Chunked Upload Management | Receives large video files in small, resumable chunks | Handles chunks up to `CHUNK_SIZE` limit, coordinates chunk ordering |
| File Validation | Ensures uploaded content meets platform requirements | Rejects files exceeding size limits or unsupported formats |
| Metadata Extraction | Reads technical properties from uploaded videos | Extracts duration, resolution, codec, bitrate using FFprobe |
| Storage Coordination | Persists validated files to permanent storage | Manages unique identifiers and directory structure |
| Progress Tracking | Reports upload progress to clients | Calculates completion percentage and estimated time remaining |

The Upload Service maintains **upload session state** to support resumability - if a client's connection drops during upload, they can resume from the last successfully received chunk rather than starting over. This state includes chunk sequence numbers, received byte ranges, and session timeouts.

> **Key Design Insight**: Upload Service treats the network as fundamentally unreliable. Every operation is designed to be idempotent and resumable, with client-provided identifiers ensuring duplicate chunks are handled gracefully.

#### Transcoding Pipeline Component

The **Transcoding Pipeline** transforms uploaded videos into streaming-friendly formats through background processing. Unlike the Upload Service which must respond to users in real-time, the transcoding pipeline optimizes for throughput and resource utilization over latency.

| Responsibility | Description | Boundary Conditions |
|----------------|-------------|-------------------|
| Job Queue Management | Schedules transcoding work across available resources | Manages concurrency limits and job prioritization |
| FFmpeg Process Orchestration | Spawns and monitors FFmpeg child processes | Handles process failures and resource constraints |
| Quality Ladder Generation | Creates multiple renditions from source video | Produces 360p, 720p, 1080p variants based on source quality |
| HLS Segmentation | Splits transcoded videos into streamable segments | Creates segments of `SEGMENT_DURATION` length with proper boundaries |
| Progress Monitoring | Tracks transcoding completion across multiple jobs | Parses FFmpeg output and updates job status in real-time |

The transcoding pipeline implements a **job queue pattern** where transcoding requests are queued for background processing. This decouples upload completion from transcoding completion - users can see their upload succeed immediately while transcoding happens asynchronously. The pipeline manages **transcoding job state** through a finite state machine with states: pending, processing, completed, and failed.

> **Key Design Insight**: Transcoding is inherently CPU-bound and time-consuming. The pipeline prioritizes reliable completion over speed, with comprehensive error handling and job recovery mechanisms.

#### Streaming Service Component

The **Streaming Service** serves transcoded content using the HLS (HTTP Live Streaming) protocol. Its architecture optimizes for low-latency delivery and CDN compatibility, treating each request as independent to maximize cacheability.

| Responsibility | Description | Boundary Conditions |
|----------------|-------------|-------------------|
| Manifest Generation | Creates M3U8 playlists referencing video segments | Generates master playlists and quality variant playlists |
| Segment Serving | Delivers video segments with appropriate HTTP headers | Serves TS files with correct MIME types and cache directives |
| Adaptive Bitrate Support | Provides multiple quality levels for client switching | Maintains quality ladder consistency across all videos |
| CDN Integration | Optimizes responses for edge caching | Includes cache-control headers and supports HTTP range requests |
| CORS Configuration | Enables cross-origin player access | Configures appropriate CORS headers for web player compatibility |

The Streaming Service is **stateless by design** - each request contains all necessary information to generate the appropriate response. This enables horizontal scaling and CDN caching while simplifying deployment and failure recovery.

> **Key Design Insight**: Streaming Service treats every request as potentially originating from a CDN edge server. All responses include appropriate caching headers and are designed to be cacheable for extended periods.

#### Player Client Component

The **Player Client** runs in users' browsers and coordinates with the Streaming Service to provide adaptive video playback. Unlike server components, the player must handle unpredictable network conditions and diverse browser capabilities while maintaining smooth playback experience.

| Responsibility | Description | Boundary Conditions |
|----------------|-------------|-------------------|
| HLS Protocol Implementation | Fetches manifests and segments according to HLS specification | Uses HLS.js library for browsers without native HLS support |
| Adaptive Quality Switching | Adjusts video quality based on network conditions | Monitors buffer health and available bandwidth |
| Playback Controls | Provides standard video player interface | Implements play, pause, seek, volume, and quality selector |
| Analytics Collection | Records playback metrics and user behavior | Tracks view duration, quality switches, and buffering events |
| Error Recovery | Handles network failures and playback issues | Implements retry logic and graceful degradation |

The Player Client maintains **playback state** including current position, buffer levels, available quality levels, and network bandwidth estimates. This state drives adaptive bitrate decisions and user interface updates.

> **Key Design Insight**: Player Client assumes network conditions will change frequently during playback. All quality switching decisions are based on recent measurements rather than initial conditions.

#### Component Interaction Patterns

Components communicate through three primary patterns: **request-response** for synchronous operations, **event publishing** for status updates, and **shared storage** for data exchange.

| Interaction Type | Participants | Purpose | Mechanism |
|------------------|-------------|---------|-----------|
| Upload Completion Notification | Upload Service → Transcoding Pipeline | Trigger transcoding job creation | Database record insertion |
| Transcoding Progress Updates | Transcoding Pipeline → Upload Service | Report job status to users | Database status field updates |
| Content Availability Check | Streaming Service → Storage | Verify segments exist before serving | File system or object storage queries |
| Player Manifest Requests | Player Client → Streaming Service | Fetch playlist and segment information | HTTP GET requests |
| Analytics Reporting | Player Client → Upload Service | Record playback metrics | HTTP POST requests |

**Database-mediated communication** handles most inter-component coordination. Components write status updates to shared tables, and other components poll or receive notifications of relevant changes. This pattern provides durability and simplifies error handling compared to direct messaging.

**Shared storage** serves as the primary data exchange mechanism. Upload Service writes validated videos to storage, Transcoding Pipeline reads source videos and writes transcoded output, and Streaming Service reads transcoded content to serve clients. Storage paths follow consistent naming conventions that all components understand.

> **Architecture Decision: Database-Mediated Communication vs. Message Queues**
> - **Context**: Components need to coordinate state changes and trigger downstream processing
> - **Options Considered**: 
>   1. Direct HTTP API calls between services
>   2. Message queue system (Redis, RabbitMQ)
>   3. Database polling with status fields
> - **Decision**: Database-mediated communication with polling
> - **Rationale**: Simpler deployment (no additional infrastructure), natural durability (state persisted automatically), easier debugging (all state visible in database), sufficient performance for video processing workloads
> - **Consequences**: Enables easy system introspection and debugging but requires careful database design to avoid polling overhead

### Recommended File Structure

The codebase organization reflects the component architecture while providing shared utilities and clear separation of concerns. Think of this structure like organizing a library - related books are grouped together, but there's also a reference section that everyone can access, and clear navigation between different sections.

```
video-streaming-platform/
├── services/                           # Main service implementations
│   ├── upload-service/                 # Upload Service component
│   │   ├── src/
│   │   │   ├── controllers/
│   │   │   │   ├── upload-controller.js      # Chunked upload endpoints
│   │   │   │   └── progress-controller.js    # Upload progress tracking
│   │   │   ├── middleware/
│   │   │   │   ├── chunk-handler.js          # Multipart chunk processing
│   │   │   │   ├── file-validator.js         # Format and size validation
│   │   │   │   └── upload-session.js         # Resumable upload state
│   │   │   ├── services/
│   │   │   │   ├── metadata-extractor.js     # FFprobe integration
│   │   │   │   └── storage-manager.js        # File system operations
│   │   │   └── server.js                     # Express.js application setup
│   │   ├── uploads/                          # Temporary upload storage
│   │   ├── package.json
│   │   └── README.md
│   │
│   ├── transcoding-service/            # Transcoding Pipeline component
│   │   ├── src/
│   │   │   ├── queue/
│   │   │   │   ├── job-queue.js              # Background job management
│   │   │   │   └── worker-pool.js            # Concurrent transcoding workers
│   │   │   ├── transcoder/
│   │   │   │   ├── ffmpeg-wrapper.js         # FFmpeg process management
│   │   │   │   ├── quality-ladder.js         # Output rendition configuration
│   │   │   │   └── progress-parser.js        # FFmpeg output parsing
│   │   │   ├── hls/
│   │   │   │   ├── segmenter.js              # HLS segment generation
│   │   │   │   └── manifest-generator.js     # M3U8 playlist creation
│   │   │   └── server.js                     # Job queue service
│   │   ├── package.json
│   │   └── README.md
│   │
│   ├── streaming-service/              # Streaming Service component
│   │   ├── src/
│   │   │   ├── controllers/
│   │   │   │   ├── manifest-controller.js    # M3U8 playlist serving
│   │   │   │   └── segment-controller.js     # TS segment serving
│   │   │   ├── middleware/
│   │   │   │   ├── cors-handler.js           # Cross-origin configuration
│   │   │   │   ├── cache-headers.js          # CDN-friendly headers
│   │   │   │   └── range-request.js          # HTTP range support
│   │   │   ├── services/
│   │   │   │   └── content-locator.js        # HLS content path resolution
│   │   │   └── server.js                     # Express.js streaming server
│   │   ├── package.json
│   │   └── README.md
│   │
│   └── player-client/                  # Player Client component
│       ├── src/
│       │   ├── components/
│       │   │   ├── video-player.js           # HLS.js integration
│       │   │   ├── player-controls.js        # Play/pause/seek controls
│       │   │   ├── quality-selector.js       # Manual quality switching
│       │   │   └── progress-bar.js           # Seek bar and position display
│       │   ├── services/
│       │   │   ├── analytics-tracker.js      # Playback event collection
│       │   │   └── bandwidth-estimator.js    # Network condition monitoring
│       │   ├── styles/
│       │   │   └── player.css                # Player UI styling
│       │   └── index.html                    # Player demo page
│       ├── package.json
│       └── README.md
│
├── shared/                             # Shared utilities and libraries
│   ├── database/
│   │   ├── models/
│   │   │   ├── video.js                      # Video entity definition
│   │   │   ├── transcoding-job.js            # Job status tracking
│   │   │   └── hls-manifest.js               # Manifest metadata
│   │   ├── migrations/
│   │   │   ├── 001-create-videos.sql         # Video table schema
│   │   │   ├── 002-create-transcoding-jobs.sql
│   │   │   └── 003-create-hls-manifests.sql
│   │   └── connection.js                     # Database connection pooling
│   │
│   ├── storage/
│   │   ├── storage-service.js                # Unified storage interface
│   │   ├── local-storage.js                  # Local filesystem implementation
│   │   └── s3-storage.js                     # AWS S3 implementation
│   │
│   ├── config/
│   │   ├── constants.js                      # System-wide constants
│   │   ├── quality-profiles.js               # Transcoding quality ladder
│   │   └── environment.js                    # Environment configuration
│   │
│   └── utils/
│       ├── validation.js                     # Input validation utilities
│       ├── file-utils.js                     # File operation helpers
│       └── logger.js                         # Structured logging
│
├── storage/                            # File storage directories
│   ├── uploads/                        # Original uploaded videos
│   ├── transcoded/                     # Processed video outputs
│   │   └── {video-id}/                 # Per-video transcoding results
│   │       ├── 360p/                   # Quality-specific directories
│   │       ├── 720p/
│   │       └── 1080p/
│   └── hls/                           # HLS segments and manifests
│       └── {video-id}/                 # Per-video streaming assets
│           ├── master.m3u8             # Master playlist
│           ├── 360p.m3u8              # Quality variant playlists
│           ├── 720p.m3u8
│           ├── 1080p.m3u8
│           └── segments/               # TS segment files
│
├── docs/                              # Documentation and diagrams
│   ├── api/                           # API documentation
│   ├── architecture/                  # Architecture decision records
│   └── deployment/                    # Deployment guides
│
├── scripts/                           # Development and deployment scripts
│   ├── start-all-services.sh         # Development environment startup
│   ├── setup-database.sh             # Database initialization
│   └── cleanup-storage.sh            # Storage maintenance
│
├── docker-compose.yml                 # Local development environment
├── package.json                       # Root package configuration
└── README.md                          # Project overview and setup
```

#### File Structure Design Rationale

**Service Isolation**: Each service lives in its own directory with independent `package.json` files, enabling separate deployment and dependency management. Services can evolve independently and be maintained by different team members without coordination overhead.

**Shared Dependencies**: The `shared/` directory contains code used by multiple services - database models, storage abstractions, and common utilities. This prevents code duplication while maintaining clear ownership boundaries. Shared modules use well-defined interfaces that services can depend on without tight coupling.

**Storage Organization**: The `storage/` directory structure mirrors the data flow through the system. Original uploads are preserved in `uploads/`, transcoded outputs are organized by video ID and quality in `transcoded/`, and HLS streaming assets are structured according to the HLS specification in `hls/`. This organization makes debugging easier and supports efficient cleanup operations.

**Configuration Management**: The `shared/config/` directory centralizes system-wide configuration including quality profiles, constants like `CHUNK_SIZE` and `SEGMENT_DURATION`, and environment-specific settings. This ensures consistent behavior across all services while supporting different deployment environments.

> **Architecture Decision: Monorepo vs. Multiple Repositories**
> - **Context**: Need to organize code for multiple services with shared dependencies
> - **Options Considered**:
>   1. Separate repository per service
>   2. Monorepo with service directories  
>   3. Single repository with mixed code
> - **Decision**: Monorepo with service directories
> - **Rationale**: Easier coordination during development, shared utilities can evolve with services, simplified deployment pipeline, better for learning environment where students work on all components
> - **Consequences**: Enables rapid iteration and cross-service refactoring but requires discipline to maintain service boundaries

#### Development Workflow Organization

The file structure supports a **service-oriented development workflow** where developers can work on individual components while running the complete system locally. The `docker-compose.yml` file defines the development environment with all services, databases, and storage volumes.

**Local Development Pattern**:
1. Start shared infrastructure (database, storage) using Docker Compose
2. Run individual services in development mode with file watching
3. Use the player client to test end-to-end functionality
4. Run service-specific tests in isolation

**Deployment Strategy**: Each service directory contains its own `Dockerfile` and deployment configuration. The monorepo structure simplifies CI/CD pipelines while maintaining the ability to deploy services independently in production.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|--------------|-----------------|
| Upload Service | Express.js + Multer for multipart uploads | Fastify + custom streaming parser |
| Transcoding Service | Node.js child_process with FFmpeg | Bull queue + Redis for job management |
| Streaming Service | Express.js + static file serving | Nginx proxy + Node.js for manifest generation |
| Player Client | HLS.js + vanilla JavaScript | Video.js + HLS.js plugin |
| Database | SQLite for development | PostgreSQL for production |
| Storage | Local filesystem | AWS S3 or Google Cloud Storage |

#### Complete Infrastructure Starter Code

**Database Connection Service** (`shared/database/connection.js`):
```javascript
const sqlite3 = require('sqlite3').verbose();
const { promisify } = require('util');
const path = require('path');

class DatabaseService {
  constructor(dbPath = './video_platform.db') {
    this.db = new sqlite3.Database(dbPath);
    
    // Promisify database methods for async/await usage
    this.get = promisify(this.db.get.bind(this.db));
    this.all = promisify(this.db.all.bind(this.db));
    this.run = promisify(this.db.run.bind(this.db));
  }

  async initialize() {
    // Create tables if they don't exist
    await this.run(`
      CREATE TABLE IF NOT EXISTS videos (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        original_path TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        duration REAL,
        width INTEGER,
        height INTEGER,
        codec TEXT,
        bitrate INTEGER,
        upload_completed_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    await this.run(`
      CREATE TABLE IF NOT EXISTS transcoding_jobs (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
        progress INTEGER DEFAULT 0,
        quality_level TEXT NOT NULL,
        output_path TEXT,
        error_message TEXT,
        started_at DATETIME,
        completed_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos (id)
      )
    `);

    await this.run(`
      CREATE TABLE IF NOT EXISTS hls_manifests (
        id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        quality_level TEXT NOT NULL,
        manifest_path TEXT NOT NULL,
        segment_count INTEGER DEFAULT 0,
        total_duration REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos (id)
      )
    `);
  }

  close() {
    this.db.close();
  }
}

module.exports = { DatabaseService };
```

**Storage Service Abstraction** (`shared/storage/storage-service.js`):
```javascript
const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');

class StorageService {
  constructor(basePath = './storage') {
    this.basePath = basePath;
  }

  async initialize() {
    // Create directory structure
    const dirs = ['uploads', 'transcoded', 'hls'];
    for (const dir of dirs) {
      await fs.mkdir(path.join(this.basePath, dir), { recursive: true });
    }
  }

  async saveUploadedFile(buffer, originalName, category = 'uploads') {
    const fileId = uuidv4();
    const extension = path.extname(originalName);
    const filename = `${fileId}${extension}`;
    const filePath = path.join(this.basePath, category, filename);
    
    await fs.writeFile(filePath, buffer);
    
    return {
      id: fileId,
      filename: filename,
      path: filePath,
      size: buffer.length
    };
  }

  async createVideoDirectory(videoId, category = 'transcoded') {
    const dirPath = path.join(this.basePath, category, videoId);
    await fs.mkdir(dirPath, { recursive: true });
    return dirPath;
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  async deleteFile(filePath) {
    try {
      await fs.unlink(filePath);
      return true;
    } catch {
      return false;
    }
  }

  getPath(...pathSegments) {
    return path.join(this.basePath, ...pathSegments);
  }
}

module.exports = { StorageService };
```

**System Constants** (`shared/config/constants.js`):
```javascript
const CHUNK_SIZE = 1024 * 1024; // 1MB chunks for resumable uploads
const SEGMENT_DURATION = 10; // 10-second HLS segments
const MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024; // 2GB max upload
const SUPPORTED_FORMATS = ['.mp4', '.mov', '.avi', '.mkv'];

const QUALITY_LADDER = [
  {
    name: '360p',
    width: 640,
    height: 360,
    videoBitrate: '800k',
    audioBitrate: '96k'
  },
  {
    name: '720p',
    width: 1280,
    height: 720,
    videoBitrate: '2500k',
    audioBitrate: '128k'
  },
  {
    name: '1080p',
    width: 1920,
    height: 1080,
    videoBitrate: '5000k',
    audioBitrate: '192k'
  }
];

const TRANSCODING_STATES = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed'
};

module.exports = {
  CHUNK_SIZE,
  SEGMENT_DURATION,
  MAX_FILE_SIZE,
  SUPPORTED_FORMATS,
  QUALITY_LADDER,
  TRANSCODING_STATES
};
```

#### Core Component Skeletons

**Upload Service Core Logic** (`services/upload-service/src/services/upload-service.js`):
```javascript
const { DatabaseService } = require('../../../../shared/database/connection');
const { StorageService } = require('../../../../shared/storage/storage-service');
const { CHUNK_SIZE, MAX_FILE_SIZE, SUPPORTED_FORMATS } = require('../../../../shared/config/constants');

class UploadService {
  constructor() {
    this.db = new DatabaseService();
    this.storage = new StorageService();
    this.uploadSessions = new Map(); // Track active upload sessions
  }

  // Handle individual chunk uploads with resumability support
  async handleChunkedUpload(req, res) {
    // TODO 1: Extract chunk metadata from headers (chunk index, total chunks, session ID)
    // TODO 2: Validate chunk size doesn't exceed CHUNK_SIZE limit
    // TODO 3: Check if session exists, create new session if first chunk
    // TODO 4: Validate chunk sequence number to detect missing chunks
    // TODO 5: Store chunk data and update session progress
    // TODO 6: If last chunk received, assemble complete file and validate
    // TODO 7: Return progress response with bytes received and completion percentage
    // Hint: Use req.headers['content-range'] for chunk position information
  }

  async validateUploadFile(fileBuffer, metadata) {
    // TODO 1: Check file size against MAX_FILE_SIZE limit
    // TODO 2: Verify file extension is in SUPPORTED_FORMATS array
    // TODO 3: Use FFprobe to extract video metadata (duration, resolution, codec)
    // TODO 4: Validate video has required properties (duration > 0, valid dimensions)
    // TODO 5: Return validation result with extracted metadata or error details
    // Hint: Use child_process.spawn with 'ffprobe' command for metadata extraction
  }

  async saveVideoRecord(videoData, metadata) {
    // TODO 1: Generate unique video ID using UUID
    // TODO 2: Save file to permanent storage using StorageService.saveUploadedFile
    // TODO 3: Insert video record into database with metadata
    // TODO 4: Create initial transcoding jobs for each quality level
    // TODO 5: Return video record with ID and processing status
  }
}

module.exports = { UploadService };
```

**Transcoding Service Core Logic** (`services/transcoding-service/src/services/transcoding-service.js`):
```javascript
const { spawn } = require('child_process');
const { DatabaseService } = require('../../../../shared/database/connection');
const { QUALITY_LADDER, SEGMENT_DURATION, TRANSCODING_STATES } = require('../../../../shared/config/constants');

class TranscodingService {
  constructor() {
    this.db = new DatabaseService();
    this.activeJobs = new Map(); // Track running FFmpeg processes
  }

  async startTranscoding(videoId, inputPath, outputDir) {
    // TODO 1: Query database for video metadata to determine source properties
    // TODO 2: Create transcoding jobs for each applicable quality level
    // TODO 3: For each quality level, spawn FFmpeg process with HLS output
    // TODO 4: Set up progress monitoring by parsing FFmpeg stderr output
    // TODO 5: Update job status in database as transcoding progresses
    // TODO 6: Handle FFmpeg process completion and error conditions
    // TODO 7: Generate master M3U8 playlist when all qualities complete
    // Hint: Use QUALITY_LADDER to determine output resolutions and bitrates
  }

  spawnFFmpegProcess(inputPath, outputPath, qualityConfig) {
    // TODO 1: Build FFmpeg command arguments for HLS transcoding
    // TODO 2: Include quality-specific video and audio encoding parameters
    // TODO 3: Set HLS segment duration using SEGMENT_DURATION constant
    // TODO 4: Configure output format for browser compatibility
    // TODO 5: Spawn child process and return process handle
    // Hint: FFmpeg HLS arguments: -f hls -hls_time SEGMENT_DURATION -hls_playlist_type vod
  }

  parseFFmpegProgress(progressLine, totalDurationSeconds) {
    // TODO 1: Parse FFmpeg progress output for current time position
    // TODO 2: Calculate completion percentage based on total duration
    // TODO 3: Extract bitrate and speed information if available
    // TODO 4: Return progress object with percentage and estimated time remaining
    // Hint: Look for 'time=' in FFmpeg stderr output for current position
  }
}

module.exports = { TranscodingService };
```

#### Language-Specific Development Tips

**Node.js Streaming and File Handling**:
- Use `fs.createReadStream()` and `fs.createWriteStream()` for large file operations to avoid memory issues
- Implement proper error handling with try-catch blocks around async operations
- Use `child_process.spawn()` instead of `exec()` for FFmpeg to handle long-running processes
- Set up proper cleanup handlers to kill child processes when the Node.js process exits

**Express.js Middleware Patterns**:
- Create custom middleware for upload session management and chunk validation
- Use `multer` with custom storage engines for chunked upload handling
- Implement CORS middleware that supports preflight requests for browser compatibility
- Add request logging middleware to track upload progress and errors

**Error Handling Strategy**:
- Use structured error objects with error codes for different failure types
- Implement retry logic for transient failures (network timeouts, temporary file locks)
- Set up proper logging with correlation IDs to track requests across services
- Create health check endpoints that verify database connectivity and storage availability

#### Development Environment Setup

**Initial Project Setup Script** (`scripts/start-all-services.sh`):
```bash
#!/bin/bash

# Initialize database and storage
echo "Setting up database..."
cd shared/database && node -e "
  const { DatabaseService } = require('./connection');
  const db = new DatabaseService();
  db.initialize().then(() => {
    console.log('Database initialized');
    db.close();
  });
"

echo "Creating storage directories..."
mkdir -p storage/{uploads,transcoded,hls}

# Start services in development mode
echo "Starting upload service..."
cd services/upload-service && npm run dev &

echo "Starting transcoding service..."
cd services/transcoding-service && npm run dev &

echo "Starting streaming service..."
cd services/streaming-service && npm run dev &

echo "All services started. Upload: http://localhost:3001, Streaming: http://localhost:3003"
```

#### Milestone Checkpoints

**After Component Setup**:
1. Run `npm install` in project root and each service directory
2. Execute `scripts/start-all-services.sh` to verify all services start without errors
3. Check that database tables are created correctly by inspecting `video_platform.db`
4. Verify storage directories are created with proper permissions
5. Test service health endpoints return 200 status codes

**Component Integration Verification**:
1. Upload a small test video file through the upload service
2. Verify video record appears in database with correct metadata
3. Check that transcoding jobs are created in pending state
4. Confirm uploaded file is stored in correct directory structure
5. Test that streaming service can serve a simple manifest file (even if empty)


## Data Model

> **Milestone(s):** All milestones (1-4) - The data model forms the foundation for video upload (M1), transcoding coordination (M2), streaming delivery (M3), and player integration (M4).

Think of the data model as the **blueprint and filing system** for your video streaming platform. Just like a library needs a catalog system to track books, their locations, checkout status, and metadata, our streaming platform needs a structured way to track videos, their processing states, and all the different formats we create for streaming. The data model is the shared language that allows our upload service, transcoding pipeline, streaming service, and player to coordinate around the same video assets.

The core insight of our data model design is that **video processing is inherently stateful and asynchronous**. Unlike a simple file storage system where you upload a file and it's immediately available, video streaming requires a complex pipeline where a single uploaded video becomes dozens of processed artifacts (segments, manifests, quality renditions) over time. Our data model must capture not just the final state, but the entire journey from raw upload through transcoding completion.

![Data Model Relationships](./diagrams/data-model.svg)

Our data model centers around three primary entities that represent different stages of the video lifecycle. The `Video` entity represents the original uploaded file and its extracted metadata. The `TranscodingJob` entity tracks the asynchronous processing that converts the uploaded video into streaming-ready formats. The `HLSManifest` entity represents the final streaming artifacts that clients consume during playback. This separation allows each component to operate independently while maintaining consistency through database-mediated communication.

### Video Metadata Schema

The `Video` entity serves as the **central registry** for all video assets in our platform. Think of it as the master record card in our library catalog - it contains all the essential information about a video file, from basic identification to detailed technical specifications. This entity bridges the gap between the physical file stored on disk and the logical video concept that users and other services interact with.

The video metadata schema must capture three distinct types of information: **identification data** for uniquely tracking videos across the system, **technical metadata** extracted from the actual video file for transcoding decisions, and **operational metadata** for managing the video's lifecycle and processing state. This comprehensive approach ensures that any component in our system can make informed decisions about how to handle a particular video.

| Field Name | Type | Description | Source | Required |
|------------|------|-------------|---------|----------|
| `id` | UUID String | Unique identifier for the video across all system components | Generated on upload | Yes |
| `filename` | String | Original filename as provided by the user during upload | Upload request | Yes |
| `file_path` | String | Relative path to the stored video file from storage base directory | Storage service | Yes |
| `file_size` | Long Integer | Size of the original video file in bytes | File system | Yes |
| `mime_type` | String | MIME type of the uploaded file (e.g., "video/mp4", "video/quicktime") | File validation | Yes |
| `duration_seconds` | Decimal | Total playback duration extracted from video metadata | FFprobe analysis | Yes |
| `width` | Integer | Video frame width in pixels | FFprobe analysis | Yes |
| `height` | Integer | Video frame height in pixels | FFprobe analysis | Yes |
| `frame_rate` | Decimal | Video frame rate in frames per second | FFprobe analysis | Yes |
| `bitrate` | Integer | Average bitrate of the original video in bits per second | FFprobe analysis | Yes |
| `codec` | String | Video codec identifier (e.g., "h264", "hevc", "vp9") | FFprobe analysis | Yes |
| `audio_codec` | String | Audio codec identifier (e.g., "aac", "mp3", "opus") | FFprobe analysis | No |
| `upload_date` | Timestamp | When the video upload was completed and file validation passed | Upload completion | Yes |
| `processing_status` | Enum | Current state of video processing pipeline | System state | Yes |
| `created_by` | String | User or system identifier that initiated the upload | Upload context | No |
| `metadata_json` | JSON Text | Additional metadata extracted by FFprobe for advanced processing | FFprobe output | No |

The `processing_status` field deserves special attention as it coordinates the handoff between different pipeline stages. This field uses an enum with values: `UPLOADED` (file stored but not yet processed), `TRANSCODING` (currently being processed by FFmpeg), `READY` (transcoding completed successfully), and `FAILED` (processing encountered unrecoverable errors). This status acts as a state machine controller that determines which operations are valid for a particular video.

> **Key Design Insight**: The video schema intentionally separates **immutable metadata** (extracted from the original file) from **mutable processing state**. The technical metadata fields like duration, resolution, and codec never change after extraction, while the processing_status field evolves as the video moves through the transcoding pipeline. This separation prevents race conditions and makes the system more predictable.

**Architecture Decision: Metadata Extraction Strategy**

> **Decision: Extract All Video Metadata During Upload**
> - **Context**: We need video metadata for transcoding decisions, quality ladder selection, and player initialization. We could extract metadata during upload, lazily during transcoding, or on-demand when requested.
> - **Options Considered**: 
>   1. Extract during upload using FFprobe before transcoding
>   2. Extract during transcoding as part of FFmpeg processing
>   3. Extract on-demand when first requested by streaming service
> - **Decision**: Extract all essential metadata during upload using FFprobe
> - **Rationale**: Upload-time extraction enables immediate transcoding job configuration, allows early rejection of invalid files, and provides metadata for client applications before transcoding completes. The overhead of running FFprobe is minimal compared to full transcoding.
> - **Consequences**: Upload processing time increases by 1-3 seconds for metadata extraction, but transcoding jobs can start immediately with optimal configuration. Failed extractions can reject invalid uploads before consuming transcoding resources.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Extract during upload | Immediate availability, early validation, enables transcoding optimization | Slightly slower uploads, requires FFprobe installation | ✓ Yes |
| Extract during transcoding | Faster uploads, single FFmpeg process | Delayed availability, cannot validate before transcoding | No |
| Extract on-demand | Fastest uploads, minimal storage | Complex caching, potential delays during playback | No |

The metadata extraction process integrates with our upload workflow through a dedicated `extractVideoMetadata(filePath)` function that invokes FFprobe as a subprocess. This function parses the JSON output from FFprobe and populates the video record fields. If metadata extraction fails (corrupted file, unsupported format, etc.), the upload is rejected and the stored file is cleaned up to prevent orphaned data.

> ⚠️ **Pitfall: Incomplete Metadata Extraction**
> Many developers skip validation of the FFprobe output and assume all metadata fields will be present. However, some video files have incomplete metadata, missing audio tracks, or unusual encoding parameters. Always check for null values in the metadata response and have fallback strategies. For example, if frame_rate extraction fails, use a default value like 30fps rather than leaving it null, which would break transcoding configuration.

### Transcoding State Management

The `TranscodingJob` entity manages the **asynchronous journey** from uploaded video to streaming-ready content. Think of it as a **work order in a video processing factory** - it tracks what needs to be done, who's working on it, how much progress has been made, and whether the job completed successfully. This entity is crucial because transcoding is the most complex and time-consuming operation in our pipeline, often taking minutes or hours for large videos.

Transcoding state management addresses a fundamental challenge in video streaming platforms: **how do you coordinate long-running, resource-intensive processes across multiple system components?** The job entity provides database-mediated communication between the upload service (which creates jobs), the transcoding service (which processes them), and the streaming service (which waits for completion). Without proper state management, you get lost jobs, duplicate processing, and inconsistent system state.

| Field Name | Type | Description | Updates | Required |
|------------|------|-------------|---------|----------|
| `id` | UUID String | Unique identifier for this specific transcoding job | Never | Yes |
| `video_id` | UUID String | Foreign key reference to the Video being processed | Never | Yes |
| `input_path` | String | Path to the source video file for transcoding | Never | Yes |
| `output_directory` | String | Directory where transcoded segments and manifests will be written | Never | Yes |
| `quality_config` | JSON | Target quality settings including resolution, bitrate, and codec parameters | Never | Yes |
| `status` | Enum | Current processing state from TRANSCODING_STATES enum | Frequently | Yes |
| `progress_percent` | Integer | Completion percentage from 0-100 based on FFmpeg progress output | Every few seconds | Yes |
| `started_at` | Timestamp | When transcoding process began execution | Once on start | No |
| `completed_at` | Timestamp | When transcoding finished (success or failure) | Once on completion | No |
| `error_message` | String | Detailed error description if transcoding failed | On failure only | No |
| `worker_id` | String | Identifier of the transcoding worker processing this job | On assignment | No |
| `retry_count` | Integer | Number of times this job has been retried after failures | On retry | Yes |
| `estimated_duration` | Integer | Expected processing time in seconds based on video length and quality | Once calculated | No |
| `output_manifest_path` | String | Path to the generated HLS master manifest file | On completion | No |
| `segment_count` | Integer | Number of video segments generated during transcoding | On completion | No |

The heart of transcoding state management is the **job lifecycle state machine** defined by the `TRANSCODING_STATES` enum. This state machine ensures that jobs progress through predictable stages and prevents invalid state transitions that could corrupt the processing pipeline.

| Current State | Valid Events | Next State | Actions Taken | Rollback Strategy |
|---------------|--------------|------------|---------------|-------------------|
| `PENDING` | Worker claims job | `PROCESSING` | Set worker_id, started_at timestamp | Clear worker_id if worker crashes |
| `PROCESSING` | FFmpeg progress update | `PROCESSING` | Update progress_percent field | None (progress updates are idempotent) |
| `PROCESSING` | Transcoding completes | `COMPLETED` | Set completed_at, output_manifest_path, segment_count | Mark as FAILED if output validation fails |
| `PROCESSING` | FFmpeg error/crash | `FAILED` | Set error_message, completed_at, increment retry_count | None (error state is terminal for this attempt) |
| `FAILED` | Manual retry trigger | `PENDING` | Clear error fields, reset progress_percent | None (retry creates fresh attempt) |
| `COMPLETED` | Output validation fails | `FAILED` | Set error about invalid output | Clean up partial output files |

The progress tracking mechanism deserves special attention because it's the primary way our system provides user feedback during long-running transcoding operations. The `progress_percent` field is updated by parsing FFmpeg's stderr output using the `parseFFmpegProgress(line, duration)` function. FFmpeg outputs progress information in a structured format that includes timestamps and frame counts, which we convert to percentage completion.

> **Key Design Insight**: Transcoding progress is **inherently imprecise** because FFmpeg's progress reporting depends on frame processing speed, which varies based on video complexity, system load, and encoding parameters. Our progress tracking provides directional feedback ("about halfway done") rather than precise estimates. The `estimated_duration` field helps set user expectations, but actual completion times can vary by 50% or more.

**Architecture Decision: Job Queue vs. Database Polling**

> **Decision: Use Database-Mediated Job Queue with Polling**
> - **Context**: Transcoding jobs need to be distributed across multiple worker processes, with failure recovery and progress tracking. We could use a dedicated message queue (Redis, RabbitMQ), database polling, or hybrid approach.
> - **Options Considered**:
>   1. Dedicated message queue with Redis/RabbitMQ for job distribution
>   2. Database polling where workers query for PENDING jobs periodically
>   3. Hybrid with message queue for distribution and database for state persistence
> - **Decision**: Database polling with exponential backoff for job claiming
> - **Rationale**: Simplifies deployment (no additional queue infrastructure), provides ACID guarantees for job state transitions, and enables complex queries for job monitoring and retry logic. Performance is adequate for video transcoding workloads which are naturally batch-oriented.
> - **Consequences**: Workers must poll database regularly (every 5-10 seconds), creating moderate database load. Job claiming requires careful locking to prevent duplicate processing. Benefits include simplified error recovery and administrative queries.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Dedicated message queue | Low latency, built-in retry, horizontal scaling | Additional infrastructure, separate failure modes | No |
| Database polling | Simple deployment, ACID guarantees, rich queries | Higher database load, polling latency | ✓ Yes |
| Hybrid approach | Best of both worlds | Complex architecture, more failure modes | No |

The job claiming mechanism uses database transactions to ensure exclusive job assignment. When a transcoding worker starts, it executes a transaction that finds the oldest `PENDING` job, updates its status to `PROCESSING`, sets the `worker_id`, and commits. If multiple workers attempt to claim the same job simultaneously, only one transaction succeeds due to database isolation guarantees.

```sql
-- Example job claiming query (for reference, not implementation)
UPDATE transcoding_jobs 
SET status = 'PROCESSING', worker_id = ?, started_at = NOW() 
WHERE id = (
  SELECT id FROM transcoding_jobs 
  WHERE status = 'PENDING' 
  ORDER BY created_at ASC 
  LIMIT 1
) AND status = 'PENDING'  -- Double-check to prevent race conditions
```

Error handling and retry logic are critical components of transcoding state management because video processing involves many potential failure points: corrupted input files, insufficient disk space, FFmpeg crashes, worker process failures, and network interruptions during output writing. Our retry strategy implements **exponential backoff with jitter** to prevent thundering herd problems when multiple jobs fail simultaneously.

| Failure Type | Detection Method | Recovery Strategy | Retry Policy |
|--------------|------------------|-------------------|---------------|
| FFmpeg crash | Process exit code != 0 | Mark job as FAILED, capture stderr | Retry up to 3 times with exponential backoff |
| Worker crash | Heartbeat timeout (job in PROCESSING > 1 hour) | Reset job to PENDING, clear worker_id | Immediate retry (different worker) |
| Disk space full | FFmpeg stderr contains "No space left" | Mark job as FAILED with specific error | No retry until disk space resolved |
| Invalid input | FFmpeg cannot read file | Mark job as FAILED, do not retry | No retry (requires manual intervention) |
| Network interruption | Output writing fails | Mark job as FAILED, cleanup partial files | Retry up to 2 times after 5 minute delay |

> ⚠️ **Pitfall: Zombie Job Detection**
> Transcoding workers can crash or lose network connectivity, leaving jobs stuck in PROCESSING state forever. Implement a cleanup process that periodically scans for jobs in PROCESSING state with `started_at` timestamps older than your maximum expected transcoding time (e.g., 2 hours). Reset these zombie jobs to PENDING status so they can be retried by healthy workers.

> ⚠️ **Pitfall: Progress Parsing Errors**
> FFmpeg's progress output format can vary between versions and may include unexpected characters or malformed lines. Wrap the `parseFFmpegProgress()` function in robust error handling that gracefully ignores unparseable lines rather than crashing the transcoding worker. A transcoding job with inaccurate progress is much better than a crashed worker that loses all progress.

The completion phase of transcoding jobs requires careful validation to ensure the generated HLS output is correct and playable. This includes verifying that the master manifest file exists, all referenced segment files are present, and the manifest syntax is valid according to HLS specification. The `output_manifest_path` and `segment_count` fields are populated only after this validation passes, providing downstream services confidence that the transcoded content is ready for streaming.

### Implementation Guidance

The data model implementation requires careful attention to database schema design, data access patterns, and consistency guarantees across the video processing pipeline.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Database | SQLite with file storage | PostgreSQL with connection pooling |
| Schema Management | Manual SQL schema files | Database migration framework (node-pg-migrate) |
| ORM/Query Builder | Raw SQL with prepared statements | Knex.js query builder with migrations |
| JSON Storage | TEXT columns with JSON.parse() | Native JSONB columns with indexing |
| UUID Generation | crypto.randomUUID() | uuid npm package with v4() |

**B. Recommended File Structure:**

```
video-streaming-platform/
  src/
    models/
      video.js                 ← Video entity and methods
      transcoding-job.js       ← TranscodingJob entity and state machine
      hls-manifest.js          ← HLS manifest and segment tracking
      base-model.js           ← Shared database utilities
    database/
      migrations/
        001_create_videos.sql
        002_create_transcoding_jobs.sql
        003_create_hls_manifests.sql
      seeds/
        dev_sample_videos.sql
    config/
      database.js             ← Database connection configuration
    constants/
      transcoding-states.js   ← State machine definitions
      quality-ladder.js       ← Video quality configurations
```

**C. Database Schema Creation (Complete):**

```javascript
// src/database/migrations/001_create_videos.sql
const createVideosTable = `
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  file_path TEXT NOT NULL UNIQUE,
  file_size INTEGER NOT NULL,
  mime_type TEXT NOT NULL,
  duration_seconds REAL NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  frame_rate REAL NOT NULL,
  bitrate INTEGER NOT NULL,
  codec TEXT NOT NULL,
  audio_codec TEXT,
  upload_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  processing_status TEXT NOT NULL DEFAULT 'UPLOADED',
  created_by TEXT,
  metadata_json TEXT,
  
  CHECK (file_size > 0),
  CHECK (duration_seconds > 0),
  CHECK (width > 0 AND height > 0),
  CHECK (processing_status IN ('UPLOADED', 'TRANSCODING', 'READY', 'FAILED'))
);

CREATE INDEX idx_videos_status ON videos(processing_status);
CREATE INDEX idx_videos_upload_date ON videos(upload_date);
`;

// src/database/migrations/002_create_transcoding_jobs.sql  
const createTranscodingJobsTable = `
CREATE TABLE IF NOT EXISTS transcoding_jobs (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  input_path TEXT NOT NULL,
  output_directory TEXT NOT NULL,
  quality_config TEXT NOT NULL, -- JSON string
  status TEXT NOT NULL DEFAULT 'PENDING',
  progress_percent INTEGER NOT NULL DEFAULT 0,
  started_at DATETIME,
  completed_at DATETIME,
  error_message TEXT,
  worker_id TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  estimated_duration INTEGER,
  output_manifest_path TEXT,
  segment_count INTEGER,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  
  FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
  CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')),
  CHECK (progress_percent >= 0 AND progress_percent <= 100),
  CHECK (retry_count >= 0)
);

CREATE INDEX idx_transcoding_jobs_status ON transcoding_jobs(status);
CREATE INDEX idx_transcoding_jobs_video_id ON transcoding_jobs(video_id);
CREATE INDEX idx_transcoding_jobs_worker ON transcoding_jobs(worker_id);
CREATE INDEX idx_transcoding_jobs_created_at ON transcoding_jobs(created_at);
`;

// Database connection setup
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

class DatabaseService {
  constructor(dbPath = './data/streaming.db') {
    this.db = new sqlite3.Database(dbPath);
    this.db.run('PRAGMA foreign_keys = ON'); // Enable foreign key constraints
  }
  
  // Promisify database operations for async/await usage
  async run(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(err) {
        if (err) reject(err);
        else resolve({ id: this.lastID, changes: this.changes });
      });
    });
  }
  
  async get(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.get(sql, params, (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }
  
  async all(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }
}

module.exports = { DatabaseService };
```

**D. Core Entity Models (Skeleton with TODOs):**

```javascript
// src/models/video.js
const { v4: uuidv4 } = require('uuid');

class Video {
  constructor(data = {}) {
    this.id = data.id || uuidv4();
    this.filename = data.filename;
    this.file_path = data.file_path;
    this.file_size = data.file_size;
    this.mime_type = data.mime_type;
    this.duration_seconds = data.duration_seconds;
    this.width = data.width;
    this.height = data.height;
    this.frame_rate = data.frame_rate;
    this.bitrate = data.bitrate;
    this.codec = data.codec;
    this.audio_codec = data.audio_codec;
    this.upload_date = data.upload_date;
    this.processing_status = data.processing_status || 'UPLOADED';
    this.created_by = data.created_by;
    this.metadata_json = data.metadata_json;
  }
  
  // Create new video record in database after successful upload
  static async create(db, videoData) {
    // TODO 1: Validate required fields are present (filename, file_path, file_size, etc.)
    // TODO 2: Create Video instance with provided data
    // TODO 3: Insert record into videos table using db.run()
    // TODO 4: Return created Video instance with database-generated fields
    // Hint: Use INSERT with RETURNING clause to get auto-generated timestamps
  }
  
  // Find video by ID for transcoding job creation or status checks  
  static async findById(db, videoId) {
    // TODO 1: Query videos table for record with matching id
    // TODO 2: Return null if no video found
    // TODO 3: Create Video instance from database row data
    // TODO 4: Parse metadata_json field if present
    // Hint: Handle JSON parsing errors gracefully with try/catch
  }
  
  // Update video processing status as it moves through pipeline
  async updateProcessingStatus(db, newStatus) {
    // TODO 1: Validate newStatus is valid enum value from TRANSCODING_STATES
    // TODO 2: Update processing_status field in database
    // TODO 3: Update local instance property
    // TODO 4: Log status transition for debugging
    // Hint: Consider adding timestamp field for when status last changed
  }
  
  // Extract video metadata using FFprobe during upload processing
  static async extractMetadata(filePath) {
    // TODO 1: Execute FFprobe command with JSON output format
    // TODO 2: Parse FFprobe stdout to extract video/audio metadata
    // TODO 3: Convert FFprobe field names to our schema field names  
    // TODO 4: Return metadata object ready for Video constructor
    // TODO 5: Handle FFprobe errors (file not found, unsupported format)
    // Hint: Use child_process.exec() with timeout to prevent hanging
  }
}

// src/models/transcoding-job.js
const { TRANSCODING_STATES } = require('../constants/transcoding-states');

class TranscodingJob {
  constructor(data = {}) {
    this.id = data.id || uuidv4();
    this.video_id = data.video_id;
    this.input_path = data.input_path;
    this.output_directory = data.output_directory;
    this.quality_config = data.quality_config;
    this.status = data.status || TRANSCODING_STATES.PENDING;
    this.progress_percent = data.progress_percent || 0;
    this.started_at = data.started_at;
    this.completed_at = data.completed_at;
    this.error_message = data.error_message;
    this.worker_id = data.worker_id;
    this.retry_count = data.retry_count || 0;
    this.estimated_duration = data.estimated_duration;
    this.output_manifest_path = data.output_manifest_path;
    this.segment_count = data.segment_count;
  }
  
  // Create transcoding job when video upload completes
  static async createForVideo(db, videoId, qualityConfig) {
    // TODO 1: Fetch video record to get input file path and metadata
    // TODO 2: Generate output directory path based on video ID and quality
    // TODO 3: Calculate estimated duration based on video length and target quality
    // TODO 4: Create TranscodingJob instance with job parameters
    // TODO 5: Insert job record into database and return job instance
    // Hint: Use video duration and quality settings to estimate processing time
  }
  
  // Worker claims next available job for processing
  static async claimNextJob(db, workerId) {
    // TODO 1: Start database transaction for atomic job claiming
    // TODO 2: Find oldest PENDING job using ORDER BY created_at ASC LIMIT 1
    // TODO 3: Update job status to PROCESSING and set worker_id, started_at
    // TODO 4: Commit transaction and return claimed job instance
    // TODO 5: Return null if no jobs available or claiming fails
    // Hint: Use WHERE status = 'PENDING' AND (worker_id IS NULL OR worker_id = '')
  }
  
  // Update job progress during FFmpeg transcoding
  async updateProgress(db, progressPercent, estimatedSecondsRemaining = null) {
    // TODO 1: Validate progressPercent is between 0 and 100
    // TODO 2: Update progress_percent field in database
    // TODO 3: Optionally update estimated completion time
    // TODO 4: Update local instance properties
    // TODO 5: Consider throttling progress updates to avoid excessive database writes
    // Hint: Only update database if progress changed by at least 1% since last update
  }
  
  // Mark job as completed with output file paths
  async markCompleted(db, outputManifestPath, segmentCount) {
    // TODO 1: Update status to COMPLETED in database
    // TODO 2: Set completed_at timestamp, output_manifest_path, segment_count
    // TODO 3: Clear any error_message from previous retry attempts
    // TODO 4: Update corresponding Video record processing_status to READY
    // TODO 5: Update local instance properties to reflect completion
    // Hint: Use database transaction to ensure Video and TranscodingJob updates are atomic
  }
  
  // Mark job as failed with error details and retry logic
  async markFailed(db, errorMessage, canRetry = true) {
    // TODO 1: Update status to FAILED and set error_message in database
    // TODO 2: Set completed_at timestamp and increment retry_count
    // TODO 3: If canRetry is true and retry_count < MAX_RETRIES, reset to PENDING after delay
    // TODO 4: If max retries exceeded, update Video processing_status to FAILED
    // TODO 5: Clean up any partial output files from failed transcoding attempt
    // Hint: Implement exponential backoff by storing next_retry_at timestamp
  }
}
```

**E. Constants and Configuration:**

```javascript
// src/constants/transcoding-states.js
const TRANSCODING_STATES = {
  PENDING: 'PENDING',       // Job created, waiting for worker to claim
  PROCESSING: 'PROCESSING', // Worker actively transcoding video
  COMPLETED: 'COMPLETED',   // Transcoding finished successfully
  FAILED: 'FAILED'          // Transcoding failed, may be retryable
};

// Valid state transitions for job lifecycle validation
const VALID_TRANSITIONS = {
  [TRANSCODING_STATES.PENDING]: [TRANSCODING_STATES.PROCESSING],
  [TRANSCODING_STATES.PROCESSING]: [TRANSCODING_STATES.COMPLETED, TRANSCODING_STATES.FAILED],
  [TRANSCODING_STATES.FAILED]: [TRANSCODING_STATES.PENDING], // Retry transitions
  [TRANSCODING_STATES.COMPLETED]: [] // Terminal state
};

module.exports = { TRANSCODING_STATES, VALID_TRANSITIONS };

// src/constants/quality-ladder.js
const QUALITY_LADDER = [
  {
    name: '360p',
    width: 640,
    height: 360,
    video_bitrate: 800000,    // 800 kbps
    audio_bitrate: 96000,     // 96 kbps
    framerate: 30
  },
  {
    name: '720p', 
    width: 1280,
    height: 720,
    video_bitrate: 2500000,   // 2.5 Mbps
    audio_bitrate: 128000,    // 128 kbps
    framerate: 30
  },
  {
    name: '1080p',
    width: 1920, 
    height: 1080,
    video_bitrate: 5000000,   // 5 Mbps
    audio_bitrate: 192000,    // 192 kbps
    framerate: 30
  }
];

const MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024; // 2GB
const SUPPORTED_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
const SEGMENT_DURATION = 10; // seconds per HLS segment

module.exports = { QUALITY_LADDER, MAX_FILE_SIZE, SUPPORTED_FORMATS, SEGMENT_DURATION };
```

**F. Milestone Checkpoints:**

After implementing the data model, verify functionality with these checkpoints:

- **Database Schema**: Run `node src/database/create-tables.js` and verify tables created with correct foreign key constraints
- **Video Creation**: Create a Video instance, save to database, and verify all metadata fields populated correctly
- **Job Lifecycle**: Create a TranscodingJob, simulate worker claiming it, updating progress, and marking completed
- **State Validation**: Attempt invalid state transitions (e.g., PENDING → COMPLETED) and verify they're rejected
- **Query Performance**: Insert 1000+ video records and verify status queries use indexes efficiently

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Foreign key constraint failed" | Video deleted while transcoding job exists | Check if video record exists: `SELECT * FROM videos WHERE id = ?` | Add CASCADE DELETE or cleanup orphaned jobs |
| Jobs stuck in PROCESSING | Worker crashed without updating status | Query jobs with old started_at: `SELECT * FROM transcoding_jobs WHERE status = 'PROCESSING' AND started_at < datetime('now', '-1 hour')` | Reset zombie jobs to PENDING status |
| Metadata extraction fails | FFprobe not installed or invalid file | Run FFprobe manually: `ffprobe -v quiet -print_format json -show_format video.mp4` | Install FFmpeg/FFprobe and validate file format |
| Progress updates too frequent | No throttling in updateProgress() | Check database logs for excessive UPDATE queries | Implement progress update throttling (1% minimum change) |


## Upload Service Design

> **Milestone(s):** Milestone 1 (Video Upload) - This section details the core upload service that handles chunked uploads with resumability, file validation, and metadata extraction.

Building an upload service for large video files is fundamentally different from handling typical form uploads. Think of it like constructing a building: you don't pour the entire foundation at once, but rather work in manageable sections that can be inspected and corrected before moving to the next. Similarly, **chunked uploads** break large video files into digestible pieces that can be uploaded, validated, and reassembled incrementally, providing resilience against network failures and enabling progress tracking.

The upload service serves as the entry point to our video streaming platform, transforming raw user-uploaded videos into managed assets ready for transcoding. It must handle the **impedance mismatch** between unreliable network conditions and our need for complete, valid video files. This service coordinates three critical responsibilities: accepting chunked file data from clients, validating both individual chunks and complete files against platform constraints, and extracting comprehensive metadata that drives downstream transcoding decisions.

![Chunked Upload Flow](./diagrams/upload-flow.svg)

The architectural challenge lies in managing **partial state** during uploads while maintaining data consistency. Unlike simple file uploads where success or failure is binary, chunked uploads exist in intermediate states where portions of a file are committed but the complete upload remains pending. This requires careful orchestration between temporary storage, progress tracking, and cleanup mechanisms to prevent orphaned data and ensure reliable completion detection.

### Chunked Upload Strategy

The chunked upload pattern treats large files like a jigsaw puzzle where pieces can arrive out of order, some pieces might need to be resent, and the complete picture only emerges when all pieces are validated and assembled. This approach transforms the traditionally fragile process of large file uploads into a robust, resumable operation that gracefully handles network interruptions and provides meaningful progress feedback to users.

**Chunked uploads** work by dividing files into fixed-size segments (typically 1-5MB each) that are uploaded independently. Each chunk carries metadata identifying its position within the complete file, enabling the server to reconstruct the original file regardless of arrival order. This strategy provides several critical advantages: network failures only require retransmitting the affected chunk rather than the entire file, upload progress can be tracked granularly, and bandwidth can be managed by controlling chunk upload concurrency.

> **Decision: Chunk Size Selection**
> - **Context**: Need to balance upload resilience, memory usage, and HTTP overhead for video files ranging from 100MB to 5GB
> - **Options Considered**: 
>   - 256KB chunks (high granularity, high HTTP overhead)
>   - 2MB chunks (balanced approach)
>   - 10MB chunks (low overhead, poor failure recovery)
> - **Decision**: 2MB chunk size stored in `CHUNK_SIZE` constant
> - **Rationale**: 2MB provides optimal balance between upload resilience (failed chunks represent manageable data loss) and server efficiency (reasonable HTTP overhead, fits comfortably in memory buffers)
> - **Consequences**: Upload progress updates every 2MB, network failures waste at most 2MB of transfer, server memory usage peaks at ~4MB per concurrent upload (2MB buffer + processing overhead)

| Chunk Size Option | Network Resilience | Memory Usage | HTTP Overhead | Progress Granularity |
|-------------------|-------------------|--------------|---------------|---------------------|
| 256KB | Excellent | Low | High | Excellent |
| 2MB | Good | Moderate | Low | Good |
| 10MB | Poor | High | Very Low | Poor |

The chunk reassembly process follows a **state machine** pattern where each upload progresses through distinct phases with specific validation requirements at each boundary:

| Current State | Chunk Event | Next State | Actions Taken |
|--------------|-------------|------------|---------------|
| `INITIATING` | First chunk received | `UPLOADING` | Create upload session, allocate temporary storage |
| `UPLOADING` | Intermediate chunk | `UPLOADING` | Validate chunk sequence, write to temp file |
| `UPLOADING` | Final chunk received | `ASSEMBLING` | Close temp file, begin integrity verification |
| `ASSEMBLING` | Verification complete | `COMPLETED` | Move to permanent storage, trigger metadata extraction |
| `UPLOADING` | Timeout exceeded | `EXPIRED` | Clean up temporary files, notify client |
| Any state | Duplicate chunk | Same state | Ignore duplicate (idempotency) |

The upload session management requires persistent tracking of chunk arrival state to support resumability. When a client reconnects after a network failure, it can query the upload session to determine which chunks were successfully received and resume transmission from the first missing chunk. This **resumable upload** capability is essential for large video files where complete retransmission would be prohibitively expensive.

**Upload Session Tracking** maintains state for each active upload:

| Field Name | Type | Description |
|-----------|------|-------------|
| `session_id` | UUID | Unique identifier for upload session |
| `expected_size` | integer | Total file size in bytes |
| `chunk_size` | integer | Size of each chunk (except possibly last) |
| `chunks_received` | bit array | Tracks which chunk positions have been received |
| `temp_file_path` | string | Location of temporary reassembly file |
| `created_at` | timestamp | Session creation time for timeout detection |
| `last_activity` | timestamp | Most recent chunk reception time |
| `client_ip` | string | Client IP for security and debugging |
| `original_filename` | string | Client-provided filename |
| `mime_type` | string | Client-declared content type |

The chunk reception algorithm handles the complex orchestration of partial file assembly:

1. **Session Validation**: Verify the upload session exists and hasn't expired, checking that the chunk sequence number falls within expected bounds
2. **Duplicate Detection**: Check if this chunk position was already received successfully, returning success immediately for idempotency
3. **Chunk Integrity**: Validate chunk size matches expectations (chunks 0 through N-2 must be exactly `CHUNK_SIZE` bytes, final chunk can be smaller)
4. **Temporary Storage**: Write chunk data to the appropriate offset within the temporary reassembly file, using file system seek operations for efficient random access
5. **Progress Tracking**: Update the chunks received bit array and calculate completion percentage for client progress reporting
6. **Completion Detection**: After each chunk, check if all expected chunks have been received by examining the bit array for completeness
7. **Assembly Triggering**: When the final chunk arrives, initiate file integrity verification and permanent storage transfer

> **Critical Design Insight**: The temporary storage strategy uses **sparse files** where chunks can be written to arbitrary offsets without requiring sequential assembly. This enables true out-of-order chunk reception while maintaining file structure integrity.

**Progress Tracking** provides clients with meaningful upload status that goes beyond simple byte counting. The progress calculation considers not just bytes uploaded, but also the overhead of chunk validation and file assembly:

- **Chunk Progress**: Percentage of chunks successfully received and validated
- **Byte Progress**: Percentage of total file bytes committed to storage  
- **Assembly Progress**: Progress through final file verification and metadata extraction
- **Estimated Completion**: Time remaining based on recent upload velocity and remaining work

### File Validation and Metadata

File validation in a video streaming platform extends far beyond simple file extension checking. Think of validation like airport security screening: there are multiple checkpoints with increasingly sophisticated checks, from basic document verification to detailed content scanning. Our validation pipeline operates at multiple levels to ensure uploaded files meet platform requirements while extracting the rich metadata necessary for downstream transcoding decisions.

The validation process operates in **three distinct phases** that balance upload performance with content quality assurance:

**Phase 1: Rapid Pre-Upload Validation** occurs before chunk upload begins, providing immediate feedback for obviously invalid uploads:

1. **Size Bounds Checking**: Verify file size falls within platform limits (minimum 1MB for meaningful video content, maximum defined by `MAX_FILE_SIZE`)
2. **Extension Filtering**: Check file extension against `SUPPORTED_FORMATS` list to reject clearly unsupported files immediately
3. **MIME Type Verification**: Validate client-declared MIME type matches expected video formats
4. **Client Metadata**: Ensure client provides required metadata fields like original filename and declares content type

**Phase 2: Chunk-Level Validation** runs during upload to catch corruption early:

1. **Chunk Size Validation**: Verify each chunk matches expected size boundaries
2. **Sequence Validation**: Ensure chunk sequence numbers are within expected ranges
3. **Content Sniffing**: Examine chunk headers for video format signatures, particularly in early chunks containing format metadata
4. **Upload Rate Limiting**: Monitor upload velocity to detect suspicious automation or abuse

**Phase 3: Complete File Validation** executes after chunk reassembly before permanent storage commitment:

1. **File Format Verification**: Use FFprobe to verify the assembled file is a valid video container
2. **Content Integrity**: Ensure the file can be opened and basic metadata extracted successfully  
3. **Duration Validation**: Verify video duration meets platform minimums (typically 1 second minimum)
4. **Technical Constraint Checking**: Validate resolution, bitrate, and codec parameters fall within supported ranges

> **Decision: Validation Strategy Timing**
> - **Context**: Need to balance fast upload feedback with thorough content verification while managing server resources
> - **Options Considered**:
>   - Pre-upload validation only (fast but unreliable)
>   - Post-upload validation only (reliable but poor user experience)
>   - Multi-phase validation (complex but optimal user experience)
> - **Decision**: Three-phase validation pipeline with increasing sophistication
> - **Rationale**: Early validation provides immediate feedback for obvious errors, chunk validation catches corruption during upload, final validation ensures content quality without blocking the upload experience
> - **Consequences**: Higher implementation complexity but superior user experience and platform content quality

The **metadata extraction** process transforms uploaded video files from opaque binary data into rich content descriptions that drive transcoding decisions. This process uses FFprobe as the primary metadata extraction tool, but requires sophisticated error handling since real-world video files often contain unexpected format variations.

**Video Metadata Schema** captures comprehensive video characteristics:

| Field Name | Type | Description |
|-----------|------|-------------|
| `duration_seconds` | decimal | Total video duration with millisecond precision |
| `width` | integer | Video frame width in pixels |
| `height` | integer | Video frame height in pixels |
| `frame_rate` | decimal | Frames per second (may be variable frame rate average) |
| `bitrate` | integer | Average bitrate in bits per second |
| `codec` | string | Video codec identifier (h264, h265, vp9, etc.) |
| `audio_codec` | string | Audio codec identifier (aac, mp3, opus, etc.) |
| `container_format` | string | File container format (mp4, webm, avi, etc.) |
| `color_profile` | string | Color space information (bt709, bt2020, etc.) |
| `audio_channels` | integer | Number of audio channels |
| `audio_sample_rate` | integer | Audio sampling frequency in Hz |
| `metadata_json` | JSON | Raw FFprobe output for advanced processing |

The `Video.extractMetadata(filePath)` method orchestrates the complex process of extracting reliable metadata from potentially problematic video files:

1. **FFprobe Invocation**: Execute FFprobe with JSON output format and comprehensive stream analysis flags
2. **Output Parsing**: Parse JSON response and handle FFprobe errors gracefully (some files may have metadata warnings but still be playable)
3. **Data Normalization**: Convert FFprobe's various metadata formats into consistent internal representations
4. **Validation and Defaults**: Apply sanity checks to extracted values and provide reasonable defaults for missing information
5. **Derived Calculation**: Calculate additional metadata like aspect ratio, estimated storage requirements, and transcoding complexity scores
6. **Structured Storage**: Persist both normalized metadata fields and raw FFprobe output for future reference

**Metadata Validation Rules** ensure extracted information meets platform requirements:

| Validation Rule | Minimum Value | Maximum Value | Default on Missing |
|----------------|---------------|---------------|-------------------|
| Duration | 1 second | 4 hours | Reject upload |
| Width | 160 pixels | 7680 pixels | Extract from container |
| Height | 90 pixels | 4320 pixels | Extract from container |
| Frame Rate | 1 fps | 120 fps | 30 fps |
| Bitrate | 100 kbps | 100 Mbps | Calculate from file size |
| Audio Sample Rate | 8000 Hz | 192000 Hz | 44100 Hz |

> **Important Metadata Insight**: The `metadata_json` field stores the complete FFprobe output because video format variations are extensive and future platform features may require access to metadata fields not currently processed. This raw metadata serves as a comprehensive record of the original file characteristics.

### Architecture Decision Records

The upload service architecture requires several critical decisions that significantly impact both user experience and system scalability. These decisions form the foundation for how the platform handles the complex requirements of large file uploads while maintaining data integrity and providing reliable service.

> **Decision: Storage Backend Strategy**
> - **Context**: Need to store both temporary upload chunks and permanent video files, with requirements for high throughput, reliable cleanup, and eventual CDN integration
> - **Options Considered**:
>   - Local filesystem with NFS for multi-server (simple, limited scalability)
>   - Object storage (S3/compatible) for everything (scalable, higher complexity)
>   - Hybrid: local temp + object permanent (optimized for each use case)
> - **Decision**: Hybrid storage with local temporary files and object storage for permanent assets
> - **Rationale**: Temporary files benefit from local storage's low latency and high throughput during rapid chunk assembly, while permanent storage benefits from object storage's durability, scalability, and CDN integration capabilities
> - **Consequences**: Requires two storage interfaces in `StorageService`, temporary storage cleanup becomes critical, but provides optimal performance characteristics for each storage phase

| Storage Strategy | Temp Performance | Permanent Durability | CDN Integration | Operational Complexity |
|-----------------|------------------|---------------------|-----------------|----------------------|
| All Local | Excellent | Poor | Difficult | Low |
| All Object | Good | Excellent | Native | Medium |
| Hybrid | Excellent | Excellent | Native | High |

> **Decision: Chunk Ordering Strategy**
> - **Context**: Clients may upload chunks out of order due to connection multiplexing or retry logic, requiring server-side reassembly strategy
> - **Options Considered**:
>   - Require sequential upload (simple, inflexible)
>   - Support arbitrary ordering with sparse files (complex, optimal resilience)
>   - Buffer chunks until sequential (memory intensive, partial resilience)
> - **Decision**: Support arbitrary chunk ordering using sparse file writes
> - **Rationale**: True out-of-order support provides maximum upload resilience and allows clients to implement sophisticated retry strategies, while sparse files provide efficient storage without memory pressure
> - **Consequences**: Requires more sophisticated chunk tracking and file system operations, but enables optimal client upload strategies and handles network issues gracefully

> **Decision: Progress Tracking Granularity**
> - **Context**: Clients need meaningful progress feedback, but excessive updates can overwhelm both client and server resources
> - **Options Considered**:
>   - Per-chunk progress updates (high frequency, resource intensive)
>   - Percentage-based thresholds (coarse, efficient)
>   - Adaptive frequency based on file size (optimal, complex)
> - **Decision**: Percentage-based progress with 5% update thresholds
> - **Rationale**: 5% thresholds provide meaningful progress feedback (20 updates per upload) while limiting server-side update processing and client notification overhead
> - **Consequences**: Progress updates every ~100MB for 2GB files, which provides useful feedback without overwhelming resources, though very large files may have infrequent updates

The **resumability architecture** requires persistent session state that survives server restarts and network disconnections. This decision significantly impacts both user experience and system reliability:

> **Decision: Upload Session Persistence**
> - **Context**: Upload sessions must survive server restarts, network failures, and client disconnections to provide true resumability
> - **Options Considered**:
>   - In-memory session storage (fast, not persistent)
>   - Database session storage (persistent, higher latency)
>   - File-based session storage (persistent, filesystem dependent)
> - **Decision**: Database session storage with in-memory caching layer
> - **Rationale**: Database storage ensures session persistence across server restarts while caching provides low-latency access for active uploads, balancing reliability with performance
> - **Consequences**: Requires session cleanup procedures to prevent database bloat, adds database dependency for upload operations, but provides robust resumability guarantees

**Upload Session Management Schema**:

| Field Name | Type | Purpose | Cleanup Strategy |
|-----------|------|---------|------------------|
| `session_id` | UUID | Unique session identifier | Primary key for efficient lookup |
| `chunks_received` | JSONB/TEXT | Bit array of received chunks | Deleted when session expires |
| `temp_file_path` | string | Path to assembly file | Cleaned up with session expiry |
| `created_at` | timestamp | Session start time | Used for expiry calculation |
| `expires_at` | timestamp | Session expiration | Indexed for cleanup queries |
| `last_activity` | timestamp | Most recent chunk | Used for idle detection |

### Common Upload Pitfalls

Video file upload systems present unique challenges that frequently trip up developers, particularly around memory management, partial state handling, and error recovery. These pitfalls emerge from the **impedance mismatch** between HTTP's request-response model and the long-running, stateful nature of large file uploads.

⚠️ **Pitfall: Loading Entire Files into Memory**
The most dangerous mistake is attempting to buffer complete uploaded files in server memory before processing. With video files ranging from hundreds of megabytes to several gigabytes, this approach quickly leads to memory exhaustion and server crashes. Developers often fall into this trap when using simple multipart form handling libraries that automatically buffer uploaded content.

**Why it's wrong**: A single 2GB video upload would consume 2GB of server RAM, and with multiple concurrent uploads, memory usage becomes unpredictable and unsustainable. The server process may be killed by the operating system's out-of-memory killer, interrupting all active uploads.

**How to fix**: Use streaming upload processing where chunks are immediately written to temporary storage without full buffering. The `handleChunkedUpload(req, res)` method should process data in small buffers (8KB-64KB) and write directly to the target file descriptor, keeping memory usage constant regardless of upload size.

⚠️ **Pitfall: Incomplete Upload Cleanup**
Failed or abandoned uploads leave temporary files and database session records that accumulate over time, eventually consuming all available storage space. This happens because developers often focus on successful upload paths but neglect comprehensive cleanup for failure scenarios.

**Why it's wrong**: Temporary files from abandoned uploads can grow to consume entire disk partitions, and stale database sessions can impact query performance. Without proper cleanup, the system becomes unreliable over time as resources are exhausted by orphaned data.

**How to fix**: Implement a comprehensive cleanup strategy with multiple safety nets:
- Session expiry timers that automatically clean up uploads inactive for more than 24 hours
- Periodic cleanup jobs that scan for orphaned temporary files and remove them
- Graceful shutdown handlers that clean up active uploads when the server stops
- Database constraints that automatically cascade deletion of related records

⚠️ **Pitfall: Synchronous Metadata Extraction**
Running FFprobe metadata extraction synchronously during the upload request blocks the HTTP response and creates poor user experience. Large video files can require 10-30 seconds for complete metadata analysis, during which the client connection remains open and the upload appears to hang.

**Why it's wrong**: Synchronous metadata extraction creates unpredictable response times that can trigger client timeouts, and blocks server threads from handling other requests. Users see uploads complete successfully but then experience long delays before receiving confirmation.

**How to fix**: Move metadata extraction to an asynchronous background process that runs after upload completion. The `Video.create()` method should initially set `processing_status` to `UPLOADED` and return success immediately, then trigger a background job for metadata extraction that updates the status to `READY` when complete.

⚠️ **Pitfall: Missing Chunk Deduplication**
When clients retry failed chunk uploads, servers may receive the same chunk multiple times. Without proper deduplication, this can result in corrupted file assembly where chunks are written multiple times or appended incorrectly.

**Why it's wrong**: Duplicate chunks can cause file corruption if they're appended rather than overwritten, or can waste storage space and processing time if not handled efficiently. The assembled file may be larger than expected or contain repeated data segments.

**How to fix**: Implement idempotent chunk handling where receiving the same chunk multiple times has no additional effect. Check the `chunks_received` bit array before processing each chunk, and if the chunk was already received successfully, return a success response without rewriting the data.

⚠️ **Pitfall: Inadequate Error Propagation**
Upload errors often occur deep in the processing pipeline (during chunk validation, file assembly, or metadata extraction) but are not properly communicated back to clients. Users see generic "upload failed" messages without understanding what went wrong or how to fix it.

**Why it's wrong**: Poor error messages lead to user frustration and repeated failed attempts. Users may repeatedly upload files that will never succeed (due to format issues) or abandon uploads that could succeed with minor corrections.

**How to fix**: Implement structured error reporting that captures specific failure reasons and provides actionable feedback:

| Error Category | Example Cause | User Message | Suggested Action |
|---------------|---------------|--------------|------------------|
| Format Error | Unsupported codec | "Video uses unsupported codec: HEVC" | "Please convert to H.264 format" |
| Size Error | File too large | "Video exceeds 2GB size limit" | "Please compress video or trim length" |
| Network Error | Chunk timeout | "Upload interrupted by network issue" | "Resume upload from 45% complete" |
| Server Error | Storage failure | "Temporary server issue during upload" | "Please retry upload in a few minutes" |

⚠️ **Pitfall: Race Conditions in Concurrent Uploads**
When the same client initiates multiple uploads simultaneously, or when chunk uploads arrive faster than they can be processed, race conditions can occur in session management and file assembly. This can lead to chunks being written to wrong files or session state becoming inconsistent.

**Why it's wrong**: Race conditions cause unpredictable upload failures that are difficult to reproduce and debug. Files may be corrupted in subtle ways that aren't detected until transcoding fails, creating a poor user experience with delayed error feedback.

**How to fix**: Implement proper concurrency control using file locking or database transactions to ensure atomic operations on upload sessions. Use row-level locking when updating session state and ensure chunk writes are synchronized to prevent concurrent modifications to the same file regions.

### Implementation Guidance

The upload service bridges the gap between unreliable HTTP connections and the need for robust large file handling. This section provides concrete technical recommendations and starter code to implement chunked uploads with proper error handling and metadata extraction.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | 
|-----------|--------------|----------------|
| Upload Handling | `multer` with custom storage | `busboy` for stream processing |
| File Storage | Local filesystem with `fs.promises` | AWS S3 SDK with multipart uploads |
| Progress Tracking | In-memory session store | Redis with TTL for persistence |
| Metadata Extraction | `node-ffmpeg` wrapper | Direct FFprobe subprocess with `child_process` |
| Chunk Validation | Basic size/type checking | Content-based validation with `file-type` |
| Database Layer | `sqlite3` for development | PostgreSQL with `pg` for production |

#### Recommended File Structure

```
project-root/
├── src/
│   ├── services/
│   │   ├── upload-service.js      ← core upload logic (implement this)
│   │   ├── storage-service.js     ← file storage abstraction (provided)
│   │   ├── validation-service.js  ← file validation logic (implement this)
│   │   └── metadata-service.js    ← FFprobe integration (implement this)
│   ├── models/
│   │   ├── video.js              ← Video data model (implement this)
│   │   └── upload-session.js     ← Upload session tracking (provided)
│   ├── routes/
│   │   └── upload-routes.js      ← HTTP endpoints (implement this)
│   ├── utils/
│   │   ├── chunk-utils.js        ← chunk processing helpers (provided)
│   │   └── progress-tracker.js   ← progress calculation (provided)
│   └── config/
│       └── upload-config.js      ← upload constants and limits
├── uploads/
│   ├── temp/                     ← temporary chunk assembly
│   └── videos/                   ← permanent video storage
└── tests/
    ├── upload-service.test.js    ← unit tests for upload logic
    └── integration/
        └── upload-flow.test.js   ← end-to-end upload testing
```

#### Infrastructure Starter Code

**Storage Service Abstraction** (`src/services/storage-service.js`):
```javascript
const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');

class StorageService {
  constructor(config) {
    this.basePath = config.basePath || './uploads';
    this.tempPath = path.join(this.basePath, 'temp');
    this.videoPath = path.join(this.basePath, 'videos');
  }

  async ensureDirectories() {
    await fs.mkdir(this.tempPath, { recursive: true });
    await fs.mkdir(this.videoPath, { recursive: true });
  }

  async saveUploadedFile(buffer, name, category = 'videos') {
    const filename = `${uuidv4()}-${name}`;
    const filepath = path.join(this.basePath, category, filename);
    await fs.writeFile(filepath, buffer);
    return {
      filename,
      filepath,
      size: buffer.length
    };
  }

  async createTempFile(sessionId) {
    const tempPath = path.join(this.tempPath, `${sessionId}.tmp`);
    const fd = await fs.open(tempPath, 'w');
    return { path: tempPath, descriptor: fd };
  }

  async writeChunkToFile(filepath, chunkData, offset) {
    const fd = await fs.open(filepath, 'r+');
    try {
      await fd.write(chunkData, 0, chunkData.length, offset);
    } finally {
      await fd.close();
    }
  }

  async moveToPermStorage(tempPath, finalName) {
    const finalPath = path.join(this.videoPath, finalName);
    await fs.rename(tempPath, finalPath);
    return finalPath;
  }

  async cleanup(filepath) {
    try {
      await fs.unlink(filepath);
    } catch (err) {
      if (err.code !== 'ENOENT') throw err;
    }
  }
}

module.exports = { StorageService };
```

**Upload Session Tracking** (`src/models/upload-session.js`):
```javascript
const { v4: uuidv4 } = require('uuid');

class UploadSession {
  constructor(db) {
    this.db = db;
  }

  async create(sessionData) {
    const sessionId = uuidv4();
    const session = {
      session_id: sessionId,
      expected_size: sessionData.fileSize,
      chunk_size: sessionData.chunkSize || 2097152, // 2MB default
      chunks_received: Buffer.alloc(Math.ceil(sessionData.fileSize / sessionData.chunkSize / 8)),
      temp_file_path: sessionData.tempPath,
      created_at: new Date(),
      last_activity: new Date(),
      client_ip: sessionData.clientIp,
      original_filename: sessionData.filename,
      mime_type: sessionData.mimeType,
      expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
    };

    await this.db.query(
      `INSERT INTO upload_sessions (
        session_id, expected_size, chunk_size, chunks_received,
        temp_file_path, created_at, last_activity, client_ip,
        original_filename, mime_type, expires_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      Object.values(session)
    );

    return session;
  }

  async findById(sessionId) {
    const rows = await this.db.query(
      'SELECT * FROM upload_sessions WHERE session_id = ?',
      [sessionId]
    );
    return rows[0] || null;
  }

  async markChunkReceived(sessionId, chunkIndex) {
    const session = await this.findById(sessionId);
    if (!session) throw new Error('Session not found');

    const chunksBitArray = session.chunks_received;
    const byteIndex = Math.floor(chunkIndex / 8);
    const bitIndex = chunkIndex % 8;
    chunksBitArray[byteIndex] |= (1 << bitIndex);

    await this.db.query(
      'UPDATE upload_sessions SET chunks_received = ?, last_activity = ? WHERE session_id = ?',
      [chunksBitArray, new Date(), sessionId]
    );

    return this.calculateProgress(session, chunksBitArray);
  }

  calculateProgress(session, chunksBitArray) {
    const totalChunks = Math.ceil(session.expected_size / session.chunk_size);
    let receivedChunks = 0;
    
    for (let i = 0; i < totalChunks; i++) {
      const byteIndex = Math.floor(i / 8);
      const bitIndex = i % 8;
      if (chunksBitArray[byteIndex] & (1 << bitIndex)) {
        receivedChunks++;
      }
    }

    return {
      chunksReceived: receivedChunks,
      totalChunks,
      percentComplete: Math.round((receivedChunks / totalChunks) * 100),
      isComplete: receivedChunks === totalChunks
    };
  }

  async cleanupExpiredSessions() {
    const expiredSessions = await this.db.query(
      'SELECT temp_file_path FROM upload_sessions WHERE expires_at < ?',
      [new Date()]
    );

    for (const session of expiredSessions) {
      try {
        await fs.unlink(session.temp_file_path);
      } catch (err) {
        console.warn('Failed to cleanup temp file:', session.temp_file_path, err);
      }
    }

    await this.db.query('DELETE FROM upload_sessions WHERE expires_at < ?', [new Date()]);
  }
}

module.exports = { UploadSession };
```

**Chunk Processing Utilities** (`src/utils/chunk-utils.js`):
```javascript
const crypto = require('crypto');

const CHUNK_SIZE = 2097152; // 2MB
const MAX_FILE_SIZE = 5368709120; // 5GB

function validateChunkSize(chunkData, expectedSize, isLastChunk = false) {
  if (isLastChunk) {
    return chunkData.length <= expectedSize && chunkData.length > 0;
  }
  return chunkData.length === expectedSize;
}

function calculateChunkChecksum(chunkData) {
  return crypto.createHash('md5').update(chunkData).digest('hex');
}

function parseChunkHeaders(headers) {
  const contentRange = headers['content-range'];
  if (!contentRange) {
    throw new Error('Missing Content-Range header for chunk upload');
  }

  // Parse "bytes 0-2097151/4194304" format
  const match = contentRange.match(/bytes (\d+)-(\d+)\/(\d+)/);
  if (!match) {
    throw new Error('Invalid Content-Range header format');
  }

  const [, start, end, total] = match.map(Number);
  const chunkIndex = Math.floor(start / CHUNK_SIZE);
  
  return {
    start,
    end,
    total,
    chunkIndex,
    chunkSize: end - start + 1,
    isLastChunk: end === total - 1
  };
}

module.exports = {
  CHUNK_SIZE,
  MAX_FILE_SIZE,
  validateChunkSize,
  calculateChunkChecksum,
  parseChunkHeaders
};
```

#### Core Logic Skeleton Code

**Upload Service Implementation** (`src/services/upload-service.js`):
```javascript
const { StorageService } = require('./storage-service');
const { UploadSession } = require('../models/upload-session');
const { ValidationService } = require('./validation-service');
const { parseChunkHeaders, validateChunkSize, CHUNK_SIZE } = require('../utils/chunk-utils');

class UploadService {
  constructor(db, storageConfig) {
    this.storage = new StorageService(storageConfig);
    this.sessions = new UploadSession(db);
    this.validator = new ValidationService();
    this.db = db;
  }

  // Initiates a new chunked upload session for a video file
  // Returns session metadata including upload URL and session ID
  async initiateUpload(req, res) {
    // TODO 1: Extract file metadata from request (filename, size, type)
    // TODO 2: Validate file size against MAX_FILE_SIZE limit
    // TODO 3: Perform pre-upload validation (format, extension check)
    // TODO 4: Create temporary file for chunk assembly
    // TODO 5: Create upload session in database
    // TODO 6: Return session info to client with upload endpoints
    // Hint: Use this.validator.validateUploadFile() for initial checks
  }

  // Handles individual chunk uploads during chunked upload process
  // Processes chunk data and tracks assembly progress
  async handleChunkedUpload(req, res) {
    const sessionId = req.params.sessionId;
    
    // TODO 1: Parse Content-Range header to extract chunk position and size
    // TODO 2: Retrieve upload session from database and validate it exists
    // TODO 3: Check if session has expired or been completed
    // TODO 4: Validate chunk size and sequence number within expected bounds
    // TODO 5: Check for duplicate chunk using chunks_received bit array
    // TODO 6: Write chunk data to temporary file at correct offset
    // TODO 7: Update chunks_received tracking and calculate progress
    // TODO 8: Check if upload is complete (all chunks received)
    // TODO 9: If complete, trigger file assembly and validation
    // TODO 10: Return progress information to client
    // Hint: Use parseChunkHeaders() to extract chunk position info
    // Hint: Use this.storage.writeChunkToFile() for efficient chunk storage
  }

  // Completes upload process after all chunks received
  // Validates assembled file and moves to permanent storage
  async completeUpload(sessionId, clientIp) {
    // TODO 1: Retrieve completed upload session from database
    // TODO 2: Verify all chunks were received successfully
    // TODO 3: Validate assembled file integrity using this.validator
    // TODO 4: Extract video metadata using FFprobe integration
    // TODO 5: Move file from temporary to permanent storage
    // TODO 6: Create Video record in database with extracted metadata
    // TODO 7: Clean up temporary files and upload session
    // TODO 8: Return Video object with metadata and file information
    // Hint: Use this.storage.moveToPermStorage() for atomic file moves
  }

  // Retrieves current upload progress for resumable upload support
  // Returns chunk completion status and progress percentage
  async getUploadProgress(req, res) {
    const sessionId = req.params.sessionId;
    
    // TODO 1: Retrieve upload session from database
    // TODO 2: Calculate progress from chunks_received bit array
    // TODO 3: Determine missing chunks for client resume logic
    // TODO 4: Return progress info with missing chunk ranges
    // Hint: Use this.sessions.calculateProgress() helper method
  }

  // Cancels an active upload and cleans up associated resources
  // Removes temporary files and database records
  async cancelUpload(req, res) {
    const sessionId = req.params.sessionId;
    
    // TODO 1: Retrieve upload session to get temporary file path
    // TODO 2: Delete temporary file from storage
    // TODO 3: Remove upload session record from database
    // TODO 4: Return cancellation confirmation to client
    // Hint: Handle case where files may already be cleaned up
  }
}

module.exports = { UploadService };
```

**File Validation Service** (`src/services/validation-service.js`):
```javascript
const { spawn } = require('child_process');
const path = require('path');

const SUPPORTED_FORMATS = ['.mp4', '.webm', '.avi', '.mov', '.mkv'];
const MAX_FILE_SIZE = 5368709120; // 5GB
const MIN_DURATION = 1; // 1 second minimum
const MAX_DURATION = 14400; // 4 hours maximum

class ValidationService {
  // Validates uploaded file against platform constraints
  // Checks format, size, and basic content requirements
  validateUploadFile(file, metadata) {
    // TODO 1: Check file extension against SUPPORTED_FORMATS list
    // TODO 2: Validate file size within MIN/MAX bounds
    // TODO 3: Verify MIME type matches declared content type
    // TODO 4: Check filename for security issues (path traversal, etc.)
    // TODO 5: Return validation result with specific error messages
    // Hint: Use path.extname() to extract file extension
  }

  // Validates assembled video file using FFprobe for content verification
  // Ensures file is playable and meets technical requirements  
  async validateVideoFile(filepath) {
    // TODO 1: Use FFprobe to analyze video file structure
    // TODO 2: Verify file contains valid video and audio streams
    // TODO 3: Check duration meets platform minimum/maximum requirements
    // TODO 4: Validate resolution and bitrate within acceptable ranges
    // TODO 5: Ensure codecs are supported for transcoding pipeline
    // TODO 6: Return validation status with detailed error information
    // Hint: Use this.runFFprobe() helper to execute FFprobe safely
  }

  // Executes FFprobe subprocess with proper error handling
  // Returns parsed JSON metadata or throws validation errors
  async runFFprobe(filepath) {
    return new Promise((resolve, reject) => {
      const ffprobe = spawn('ffprobe', [
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        filepath
      ]);

      let stdout = '';
      let stderr = '';

      ffprobe.stdout.on('data', (data) => stdout += data);
      ffprobe.stderr.on('data', (data) => stderr += data);
      
      ffprobe.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`FFprobe failed: ${stderr}`));
        } else {
          try {
            resolve(JSON.parse(stdout));
          } catch (err) {
            reject(new Error(`Failed to parse FFprobe output: ${err.message}`));
          }
        }
      });
    });
  }
}

module.exports = { ValidationService };
```

#### Language-Specific Hints

**Node.js Specific Recommendations:**
- Use `fs.promises` for async file operations to avoid callback nesting
- Implement proper stream handling with `fs.createWriteStream()` for large file writes
- Use `Buffer.alloc()` for binary data manipulation in chunk tracking
- Handle `SIGTERM` and `SIGINT` signals for graceful shutdown and cleanup
- Use `worker_threads` for CPU-intensive operations like metadata extraction
- Implement proper error boundaries with `process.on('unhandledRejection')`

**Memory Management:**
- Never load entire files into memory - always use streams or chunked processing
- Use `stream.pipeline()` for backpressure handling during file operations
- Implement chunk size limits to prevent memory exhaustion attacks
- Use `WeakMap` for caching upload sessions to enable garbage collection

**Database Integration:**
- Use connection pooling with `pg.Pool` for PostgreSQL connections
- Implement proper transaction boundaries for multi-step operations
- Use prepared statements to prevent SQL injection in dynamic queries
- Index `session_id` and `expires_at` columns for efficient session lookups

#### Milestone Checkpoint

After implementing the upload service, verify the following functionality:

**Test Commands:**
```bash
# Start the upload service
npm start

# Run upload service unit tests
npm test -- --testPathPattern=upload-service

# Run integration tests for complete upload flow  
npm test -- --testPathPattern=upload-flow

# Test chunked upload with curl
curl -X POST http://localhost:3000/api/uploads/initiate \
  -H "Content-Type: application/json" \
  -d '{"filename":"test-video.mp4","fileSize":10485760,"mimeType":"video/mp4"}'
```

**Expected Behavior:**
- Upload initiation returns session ID and chunk upload URL
- Chunk uploads process individual 2MB segments with progress tracking
- Progress endpoint shows accurate completion percentage
- Completed uploads trigger metadata extraction and permanent storage
- Failed uploads clean up temporary files and session state

**Success Indicators:**
- Upload sessions persist across server restarts
- Chunks can be uploaded out of order without corruption
- Network interruptions allow resumable upload from last successful chunk
- File validation catches unsupported formats and corrupted uploads
- Metadata extraction completes within 30 seconds for typical video files

**Troubleshooting Signs:**
- High memory usage indicates file buffering instead of streaming
- Slow upload responses suggest synchronous operations blocking event loop
- Missing progress updates indicate chunk tracking bit array issues
- File corruption suggests race conditions in concurrent chunk writes
- Storage space growth indicates cleanup procedures not running properly


## Transcoding Pipeline Design

> **Milestone(s):** Milestone 2 (Video Transcoding) - This section details the transcoding pipeline that converts uploaded videos to HLS format using FFmpeg with background job processing and multiple quality levels.

Converting uploaded videos into streaming-friendly formats represents one of the most computationally intensive and complex aspects of a video platform. Think of transcoding like a massive translation project - you have a source document (the uploaded video) that needs to be translated into multiple languages (different quality renditions) while preserving the meaning (visual content) but adapting the presentation for different audiences (devices with varying capabilities and network conditions). Just as a translation service needs sophisticated workflow management, quality control, and progress tracking, video transcoding requires careful orchestration of FFmpeg processes, job queuing, and state management.

![Transcoding Pipeline Flow](./diagrams/transcoding-pipeline.svg)

The transcoding pipeline serves as the critical bridge between raw uploaded content and streamable video segments. When a video upload completes successfully, the transcoding system must analyze the source material, determine appropriate output configurations, spawn multiple parallel encoding processes, monitor their progress, and coordinate the generation of HLS manifests and segments. This process can take anywhere from a few minutes for short clips to several hours for feature-length content, making robust background processing and failure recovery essential.

The pipeline operates on the principle of **database-mediated communication**, where all components coordinate through shared database state rather than direct inter-process communication. This approach provides natural persistence, failure recovery, and the ability to distribute transcoding work across multiple worker processes or even separate machines. However, it introduces the challenge of managing job state transitions, preventing duplicate work, and detecting zombie jobs from crashed workers.

### FFmpeg Integration Pattern

FFmpeg serves as the workhorse of video transcoding, but integrating it effectively requires careful process management, progress monitoring, and error handling. The key insight is that FFmpeg operates as an external subprocess that communicates through standard streams - we must parse its output to extract progress information and detect completion or failure conditions.

The `TranscodingService` manages FFmpeg integration through a spawn-and-monitor pattern. Rather than trying to embed FFmpeg as a library (which can cause memory and stability issues), the service spawns FFmpeg as a child process and establishes communication channels through stdin, stdout, and stderr pipes. This isolation protects the main application from FFmpeg crashes while enabling fine-grained control over the transcoding operation.

| Component | Responsibility | Key Methods |
|-----------|---------------|-------------|
| `TranscodingService` | Process lifecycle management, job coordination | `startTranscoding`, `monitorProgress`, `handleCompletion` |
| `FFmpegWrapper` | Direct FFmpeg interaction, command generation | `spawnFFmpegProcess`, `parseFFmpegProgress`, `buildCommand` |
| `ProgressTracker` | Progress calculation and reporting | `calculatePercent`, `estimateTimeRemaining`, `reportProgress` |
| `OutputValidator` | HLS output verification | `validateManifest`, `validateSegments`, `checkIntegrity` |

The FFmpeg command construction follows a template-based approach where base parameters are established for reliability and compatibility, then quality-specific parameters are injected based on the target rendition. The service constructs commands like:

```
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -hls_time 6 -hls_playlist_type vod -hls_segment_filename output_%03d.ts output.m3u8
```

Progress monitoring relies on parsing FFmpeg's stderr output, which contains periodic progress reports including frame numbers, timestamps, and encoding statistics. The `parseFFmpegProgress` function uses regular expressions to extract the current timestamp and calculate completion percentage based on the known video duration extracted during the upload phase.

> **Design Insight**: FFmpeg progress parsing is inherently fragile because the output format can vary between versions and configurations. The implementation includes multiple parsing strategies (timestamp-based, frame-based, and byte-based) with fallbacks to ensure progress reporting works across different FFmpeg installations.

Error detection occurs through multiple channels: process exit codes indicate overall success or failure, stderr patterns reveal specific encoding problems, and output file validation confirms that generated HLS content meets quality standards. The service maintains a timeout mechanism to detect hanging processes and implements exponential backoff for transient failures.

| Progress Indicator | Extraction Method | Calculation | Reliability |
|-------------------|------------------|-------------|-------------|
| Timestamp Progress | Parse `time=00:02:30.45` | `(currentTime / totalDuration) * 100` | High |
| Frame Progress | Parse `frame=1500` | `(currentFrame / estimatedFrames) * 100` | Medium |
| Byte Progress | Parse output file size | `(outputBytes / estimatedBytes) * 100` | Low |
| Process Status | Monitor exit codes | Binary success/failure | High |

The service implements graceful process termination using SIGTERM followed by SIGKILL if necessary, ensuring that interrupted transcoding jobs don't leave orphaned processes consuming system resources. Process cleanup includes removing partial output files and updating job status to enable retry logic.

### Quality Ladder Strategy

Determining the appropriate set of output qualities for each uploaded video requires balancing user experience across different devices and network conditions with storage and transcoding costs. The quality ladder represents a strategic decision about which resolution and bitrate combinations provide the best adaptive streaming experience.

The quality ladder strategy operates on the principle that not every video needs every possible quality level. A 480p source video doesn't benefit from 1080p transcoding, and generating excessive quality variants increases storage costs and transcoding time without improving user experience. The system analyzes source video characteristics and generates a tailored quality ladder for each upload.

| Source Resolution | Generated Qualities | Rationale |
|------------------|-------------------|-----------|
| ≤ 480p | 360p, 480p | Don't upscale, provide lower quality for poor connections |
| 720p | 360p, 480p, 720p | Full range with native quality as maximum |
| 1080p | 360p, 480p, 720p, 1080p | Complete quality ladder for maximum adaptability |
| ≥ 1440p | 480p, 720p, 1080p, 1440p | Skip lowest quality, focus on mid-to-high range |

The bitrate selection follows industry-standard guidelines adapted for modern codecs and viewing conditions. The service uses a base bitrate table that accounts for the complexity of typical video content, then applies source-specific adjustments based on the original file's characteristics.

```
Quality Configuration Table:
Resolution | Target Bitrate | Max Bitrate | Audio Bitrate | Profile
360p       | 800 Kbps      | 1.2 Mbps    | 96 Kbps      | baseline
480p       | 1.4 Mbps      | 2.1 Mbps    | 128 Kbps     | main
720p       | 2.8 Mbps      | 4.2 Mbps    | 128 Kbps     | high
1080p      | 5.0 Mbps      | 7.5 Mbps    | 192 Kbps     | high
```

The `TranscodingService` implements dynamic quality ladder generation through the `generateQualityLadder` method, which analyzes source video metadata and returns a customized configuration for each transcoding job. This analysis considers source resolution, bitrate, frame rate, and content complexity indicators derived from FFprobe output.

> **Architecture Decision: Adaptive Quality Selection**
> - **Context**: Fixed quality ladders waste resources on simple content and provide insufficient quality for complex content
> - **Options Considered**: 
>   1. Fixed ladder (same qualities for all videos)
>   2. Resolution-based ladder (qualities based only on source resolution)  
>   3. Content-aware ladder (qualities based on resolution, bitrate, and complexity)
> - **Decision**: Content-aware ladder with resolution-based fallback
> - **Rationale**: Provides optimal quality/cost balance while maintaining predictable behavior when content analysis fails
> - **Consequences**: Requires FFprobe integration and complexity analysis, but reduces storage costs by 15-30% while improving quality consistency

The complexity analysis examines factors like motion vectors, scene changes, and spatial detail to determine if content would benefit from higher bitrates. High-motion sports content receives bitrate bonuses, while static presentation content uses lower bitrates without quality degradation.

Quality validation ensures that generated renditions meet platform standards through automated checks of resolution accuracy, bitrate compliance, and visual quality metrics. The system rejects transcoding outputs that fall outside acceptable parameters and retries with adjusted settings.

### Background Job Queue

Managing concurrent transcoding operations requires a robust job queue system that coordinates work distribution, prevents resource exhaustion, and provides failure recovery. The background processing model treats each transcoding operation as an independent job that can be executed by any available worker process.

Think of the job queue like an emergency dispatch system - multiple operators (worker processes) monitor a shared board (database table) for new calls (transcoding jobs), claim responsibility for specific incidents, and report back when work is completed. Just as emergency dispatchers must handle operator unavailability and call prioritization, the transcoding queue must manage worker failures and job priority.

![Transcoding Job State Machine](./diagrams/transcoding-states.svg)

The `TranscodingJob` entity serves as the central coordination mechanism, storing all information necessary for a worker to claim and execute a transcoding operation independently. This design enables horizontal scaling by adding more worker processes without complex inter-process communication.

| TranscodingJob State | Description | Valid Transitions |
|---------------------|-------------|-------------------|
| `PENDING` | Job created, waiting for worker | → `PROCESSING`, `FAILED` |
| `PROCESSING` | Claimed by worker, transcoding active | → `COMPLETED`, `FAILED`, `PENDING` |
| `COMPLETED` | Successfully finished, output ready | None (terminal state) |
| `FAILED` | Error occurred, may be retryable | → `PENDING` (if retryable) |

The job claiming mechanism uses database-level locking to ensure that multiple workers cannot claim the same job simultaneously. The `TranscodingJob.claimNextJob` method performs an atomic update operation that both claims a pending job and marks it as processing, returning the claimed job to the worker or null if no work is available.

```
Job Claiming Algorithm:
1. Worker calls TranscodingJob.claimNextJob(workerId) with its unique identifier
2. Database query finds oldest PENDING job and attempts atomic update to PROCESSING
3. If update succeeds, return claimed job with input/output paths and quality config
4. If no pending jobs, return null and worker enters polling wait
5. Worker begins FFmpeg process using claimed job configuration
6. Worker calls updateProgress periodically during transcoding
7. Worker calls markCompleted or markFailed based on FFmpeg result
```

Worker process management follows a polling model with exponential backoff to balance responsiveness with database load. Workers continuously poll for new jobs but increase polling intervals when no work is available, reducing unnecessary database queries during quiet periods.

| Worker State | Polling Interval | Action |
|-------------|------------------|---------|
| Job Available | 1 second | Immediate claim attempt |
| No Jobs | 5 seconds | Standard polling |
| Extended Idle | 30 seconds | Low-frequency polling |
| Processing | N/A | Monitor current job |

Concurrency limits prevent system resource exhaustion by capping the number of simultaneous transcoding operations. The service maintains a configurable maximum worker count and monitors system resources (CPU, memory, disk I/O) to dynamically adjust the actual worker pool size based on current load.

The zombie job detection mechanism identifies jobs that remain in `PROCESSING` state beyond expected completion times, indicating worker crashes or hangs. A separate cleanup process periodically scans for stale jobs and returns them to `PENDING` status for retry by healthy workers.

> **Critical Design Pattern**: The job queue implements **database-mediated communication** rather than message queues or direct RPC. This choice provides automatic persistence, simple failure recovery, and eliminates the need for additional infrastructure, but requires careful attention to database performance and lock contention.

Job prioritization supports business requirements through priority scoring based on factors like user tier, content type, and upload recency. High-priority jobs move to the front of the processing queue, ensuring that premium content or time-sensitive uploads receive faster transcoding.

Error handling distinguishes between transient failures (network issues, temporary resource exhaustion) and permanent failures (corrupted input, unsupported codecs). Transient failures trigger job requeue with exponential backoff, while permanent failures mark jobs as definitively failed to prevent endless retry loops.

### Architecture Decision Records

The transcoding pipeline involves several critical architectural decisions that significantly impact performance, reliability, and maintainability. Each decision represents a trade-off between competing concerns and requires explicit justification based on project requirements and constraints.

> **Decision: H.264 + AAC Codec Standardization**
> - **Context**: Multiple codec options exist with different compatibility, quality, and performance characteristics
> - **Options Considered**: 
>   1. Modern codecs (H.265, AV1) for best compression
>   2. Legacy codecs (H.264, AAC) for maximum compatibility
>   3. Dynamic codec selection based on device capabilities
> - **Decision**: Standardize on H.264 video with AAC audio for all output renditions
> - **Rationale**: H.264 provides universal device compatibility with acceptable file sizes, while modern codecs still have limited mobile support and significantly higher encoding complexity
> - **Consequences**: Larger file sizes than possible with H.265/AV1, but guaranteed playback across all target devices and much faster transcoding

| Codec Option | Compatibility | File Size | Encode Speed | Chosen? |
|-------------|---------------|-----------|--------------|---------|
| H.264 + AAC | Universal | Baseline | Fast | ✅ Yes |
| H.265 + AAC | Limited mobile | -30% size | 3x slower | No |
| AV1 + Opus | Chrome/Firefox only | -40% size | 5x slower | No |

> **Decision: 6-Second HLS Segments**
> - **Context**: HLS segment duration affects seeking accuracy, startup time, and adaptive switching responsiveness
> - **Options Considered**:
>   1. 2-4 second segments for low latency
>   2. 6-10 second segments for efficiency balance  
>   3. 15+ second segments for maximum efficiency
> - **Decision**: Use 6-second segments across all quality levels
> - **Rationale**: Provides good seeking granularity (6-second jumps acceptable for VOD) while limiting manifest size and reducing CDN request overhead compared to shorter segments
> - **Consequences**: Seeking accuracy limited to 6-second boundaries, but significant reduction in HTTP requests and manifest complexity

> **Decision: Database-Based Job Queue vs Message Queue**
> - **Context**: Background transcoding requires work distribution across multiple worker processes
> - **Options Considered**:
>   1. Database table as job queue with polling workers
>   2. Redis/RabbitMQ message queue with job persistence
>   3. Cloud job services (AWS SQS, Google Cloud Tasks)
> - **Decision**: Use database table with optimistic locking for job queue
> - **Rationale**: Eliminates additional infrastructure dependencies, provides natural job persistence, and simplifies deployment while handling expected transcoding volumes efficiently
> - **Consequences**: Database becomes bottleneck at high job volumes, but acceptable for target scale and much simpler operations

| Queue Approach | Infrastructure | Persistence | Failure Recovery | Complexity | Chosen? |
|----------------|---------------|-------------|-----------------|------------|---------|
| Database Table | Existing DB | Automatic | Simple restart | Low | ✅ Yes |
| Redis Queue | Add Redis | Manual setup | Complex state | Medium | No |
| Cloud Service | External dependency | Managed | Service-specific | High | No |

> **Decision: Single-Process vs Multi-Process Transcoding**
> - **Context**: FFmpeg can utilize multiple CPU cores, but coordination complexity varies significantly
> - **Options Considered**:
>   1. Single FFmpeg process with multi-threading
>   2. Multiple parallel FFmpeg processes for different qualities
>   3. Pipeline parallelism (segmented transcoding)
> - **Decision**: Single FFmpeg process per job with multiple worker processes for job parallelism
> - **Rationale**: FFmpeg's built-in multi-threading efficiently uses available cores without complex coordination, while multiple workers enable processing different videos simultaneously
> - **Consequences**: Individual jobs don't fully utilize multi-core systems, but overall throughput scales with concurrent uploads

The parallel processing decision reflects a trade-off between individual job performance and system throughput. While spawning separate FFmpeg processes for each quality level could theoretically speed up individual transcoding jobs, the coordination complexity and resource contention typically result in worse overall performance compared to letting FFmpeg manage internal parallelism.

> **Decision: Proactive vs Reactive Quality Selection**
> - **Context**: Quality ladder generation can happen during upload analysis or defer until transcoding begins
> - **Options Considered**:
>   1. Generate quality ladder during upload metadata extraction
>   2. Determine qualities at transcoding job start
>   3. Progressive quality generation based on demand
> - **Decision**: Generate quality ladder during transcoding job creation with upload metadata
> - **Rationale**: Front-loads analysis cost but enables accurate time estimation, resource planning, and job prioritization before expensive transcoding begins
> - **Consequences**: Upload completion takes slightly longer due to analysis overhead, but transcoding jobs have predictable resource requirements

These architectural decisions create a transcoding pipeline that prioritizes operational simplicity and reliability over maximum theoretical performance. The choices reflect the reality that video transcoding is inherently resource-intensive and time-consuming - attempting to optimize every aspect often introduces complexity that outweighs performance benefits.

### Common Transcoding Pitfalls

Video transcoding presents numerous opportunities for subtle bugs and performance issues that can be difficult to diagnose and resolve. Understanding these common pitfalls helps developers anticipate problems and implement appropriate safeguards from the beginning.

⚠️ **Pitfall: FFmpeg Memory Accumulation**
FFmpeg processes can gradually consume increasing amounts of memory during long transcoding operations, eventually causing system memory exhaustion or process termination. This occurs particularly with complex filter chains or when processing videos with variable characteristics.

The memory accumulation happens because FFmpeg internally buffers frame data and metadata, and certain codec/filter combinations don't release buffers aggressively. For a 2-hour 4K video, memory usage can grow from 200MB to over 2GB during processing, causing crashes on memory-constrained systems.

**Prevention**: Implement memory monitoring within transcoding jobs and restart FFmpeg processes when memory usage exceeds thresholds. Configure system-level memory limits using cgroups or containers to prevent individual jobs from exhausting system resources. Monitor `/proc/[pid]/status` or equivalent to track memory growth during transcoding.

⚠️ **Pitfall: Codec Compatibility Assumptions**
Assuming that videos with standard extensions (.mp4, .mov, .avi) contain compatible codecs leads to transcoding failures when exotic or proprietary codecs are encountered. Input videos may use codecs that FFmpeg cannot decode or require specific decoder configurations.

This manifests as FFmpeg errors like "Unsupported codec" or "Invalid data found when processing input" that cause jobs to fail permanently. The error occurs even when the video plays correctly in certain media players that have different codec libraries.

**Prevention**: Implement comprehensive input validation using `ffprobe` to detect codec compatibility before starting transcoding jobs. Maintain a whitelist of supported input codecs and provide clear error messages for unsupported formats. Consider transcoding through intermediate formats when direct conversion fails.

⚠️ **Pitfall: Interrupted Job Cleanup**
When transcoding jobs are interrupted due to worker crashes, system shutdowns, or manual termination, partially completed output files may remain in storage. These partial files can cause confusion about job status and waste storage space, especially problematic with large video files.

Interrupted jobs leave behind incomplete HLS segments, partial manifests, and temporary working files. Without proper cleanup, storage usage grows continuously, and subsequent job restarts may encounter filename conflicts or assume work is already completed.

**Prevention**: Implement atomic output patterns where transcoding writes to temporary directories and only moves completed results to final locations upon success. Use job status cleanup routines that remove partial outputs when marking jobs as failed. Implement startup cleanup that scans for and removes orphaned temporary files from previous worker sessions.

⚠️ **Pitfall: Progress Reporting Inaccuracy**
FFmpeg progress parsing can provide misleading completion percentages when video characteristics change throughout the file or when processing involves multiple passes. Duration estimates from upload metadata may not match actual transcoding time, causing progress reports to exceed 100% or remain stuck.

This occurs because initial duration extraction may miss variable frame rates, codec changes within the file, or complex filter processing that takes non-linear time. Users see progress bars that jump erratically or appear to hang, reducing confidence in the platform.

**Prevention**: Use multiple progress indicators (time-based, frame-based, file-size-based) and weight them according to reliability for each job type. Implement progress smoothing algorithms that prevent large backward jumps and cap reported progress at 99% until job completion is confirmed. Validate progress against multiple metrics to detect anomalies.

⚠️ **Pitfall: Output Quality Verification Gaps**
Transcoding jobs may complete successfully according to FFmpeg exit codes but produce output with quality issues like corrupted segments, incorrect resolutions, or audio synchronization problems. These quality issues only become apparent during playback testing.

Quality problems can arise from encoder bugs, filter chain errors, or edge cases in source material that FFmpeg handles poorly. The transcoding job appears successful in the database, but users experience playback failures or degraded quality that reflects poorly on the platform.

**Prevention**: Implement comprehensive output validation that goes beyond checking file existence. Verify that generated segments have expected durations, resolutions match configuration, audio tracks are present, and manifest files contain correct segment references. Consider automated spot-checking with brief playback tests for critical content.

| Pitfall Category | Detection Method | Prevention Strategy | Recovery Action |
|-----------------|------------------|-------------------|-----------------|
| Memory Issues | Process monitoring | Memory limits, periodic restarts | Kill and restart with cleanup |
| Codec Problems | FFprobe validation | Input compatibility checks | Clear error messaging |
| Cleanup Issues | Storage monitoring | Atomic operations | Scheduled cleanup jobs |
| Progress Errors | Multiple metrics | Smoothing algorithms | Progress recalculation |
| Quality Problems | Output validation | Comprehensive testing | Job retry with different settings |

⚠️ **Pitfall: Resource Contention with Concurrent Jobs**
Running multiple transcoding jobs simultaneously can lead to resource contention that degrades performance for all jobs rather than providing expected throughput improvements. CPU, memory, and I/O resources become bottlenecks that cause all jobs to run slower than they would individually.

This is particularly problematic when transcoding multiple high-resolution videos simultaneously on systems without sufficient resources. Each job assumes it can use available CPU cores and memory, but competition leads to context switching overhead, memory swapping, and reduced efficiency.

**Prevention**: Implement dynamic worker scaling based on system resource utilization rather than fixed worker counts. Monitor CPU load, memory usage, and I/O wait times to determine optimal concurrency levels. Use resource reservations that account for expected job requirements when starting new work.

Understanding these pitfalls enables proactive design decisions that prevent problems rather than requiring reactive debugging. The key insight is that video transcoding operates at the intersection of system resources, external processes, and unpredictable input data - robust error handling and resource management are essential from the initial implementation.

### Implementation Guidance

Video transcoding implementation requires careful integration of FFmpeg subprocess management, database coordination, and progress monitoring. The following guidance provides complete infrastructure and skeleton code for building a production-ready transcoding pipeline.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| FFmpeg Integration | Direct subprocess spawn with stdio pipes | FFmpeg libraries (fluent-ffmpeg) with enhanced control |
| Job Queue | Database polling with SELECT FOR UPDATE | Redis-based queue with pub/sub notifications |
| Progress Tracking | Regex parsing of FFmpeg stderr output | Structured logging with JSON progress reports |
| Process Management | Node.js child_process module | PM2 or container orchestration for worker scaling |
| File Validation | FFprobe JSON output parsing | MediaInfo libraries for comprehensive metadata |

**Recommended File Structure:**
```
src/
  services/
    transcoding/
      TranscodingService.js        ← Main service coordination
      FFmpegWrapper.js            ← FFmpeg process management
      QualityLadder.js            ← Quality configuration logic
      JobQueue.js                 ← Database job management
      ProgressTracker.js          ← Progress parsing and reporting
      OutputValidator.js          ← HLS output verification
    storage/
      StorageService.js           ← File operations (from Upload Service)
    database/
      DatabaseService.js          ← Database connection (from Upload Service)
  models/
    TranscodingJob.js             ← Job entity and database methods
    Video.js                      ← Video entity (from Data Model)
  workers/
    transcoding-worker.js         ← Background worker process
  config/
    transcoding-config.js         ← Quality ladders and FFmpeg settings
```

**Complete Infrastructure Code:**

```javascript
// src/config/transcoding-config.js
const QUALITY_LADDER = [
  {
    name: '360p',
    width: 640,
    height: 360,
    videoBitrate: 800,
    audioBitrate: 96,
    profile: 'baseline'
  },
  {
    name: '480p', 
    width: 854,
    height: 480,
    videoBitrate: 1400,
    audioBitrate: 128,
    profile: 'main'
  },
  {
    name: '720p',
    width: 1280, 
    height: 720,
    videoBitrate: 2800,
    audioBitrate: 128,
    profile: 'high'
  },
  {
    name: '1080p',
    width: 1920,
    height: 1080, 
    videoBitrate: 5000,
    audioBitrate: 192,
    profile: 'high'
  }
];

const TRANSCODING_STATES = {
  PENDING: 'pending',
  PROCESSING: 'processing', 
  COMPLETED: 'completed',
  FAILED: 'failed'
};

const SEGMENT_DURATION = 6; // seconds
const MAX_CONCURRENT_JOBS = 3;
const JOB_TIMEOUT_MINUTES = 120;

module.exports = {
  QUALITY_LADDER,
  TRANSCODING_STATES,
  SEGMENT_DURATION,
  MAX_CONCURRENT_JOBS,
  JOB_TIMEOUT_MINUTES
};
```

```javascript
// src/services/transcoding/FFmpegWrapper.js
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

class FFmpegWrapper {
  constructor() {
    this.activeProcesses = new Map();
  }

  // Complete FFmpeg command builder with HLS output
  buildTranscodingCommand(inputPath, outputDir, quality) {
    const outputPath = path.join(outputDir, `${quality.name}.m3u8`);
    const segmentPath = path.join(outputDir, `${quality.name}_%03d.ts`);
    
    return [
      '-i', inputPath,
      '-c:v', 'libx264',
      '-c:a', 'aac',
      '-profile:v', quality.profile,
      '-level:v', '3.1',
      '-s', `${quality.width}x${quality.height}`,
      '-b:v', `${quality.videoBitrate}k`,
      '-b:a', `${quality.audioBitrate}k`,
      '-hls_time', SEGMENT_DURATION.toString(),
      '-hls_playlist_type', 'vod',
      '-hls_segment_filename', segmentPath,
      '-y', // Overwrite output files
      outputPath
    ];
  }

  // Spawn FFmpeg process with monitoring
  spawnFFmpegProcess(inputPath, outputDir, quality) {
    return new Promise((resolve, reject) => {
      const args = this.buildTranscodingCommand(inputPath, outputDir, quality);
      const process = spawn('ffmpeg', args);
      const processId = `${Date.now()}_${Math.random()}`;
      
      this.activeProcesses.set(processId, process);
      
      let stderr = '';
      let stdout = '';
      
      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      process.stderr.on('data', (data) => {
        stderr += data.toString();
        // Progress parsing happens here
        const progress = this.parseFFmpegProgress(data.toString());
        if (progress) {
          this.onProgress && this.onProgress(progress);
        }
      });
      
      process.on('close', (code) => {
        this.activeProcesses.delete(processId);
        
        if (code === 0) {
          resolve({
            success: true,
            outputPath: path.join(outputDir, `${quality.name}.m3u8`),
            stdout,
            stderr
          });
        } else {
          reject(new Error(`FFmpeg failed with code ${code}: ${stderr}`));
        }
      });
      
      process.on('error', (error) => {
        this.activeProcesses.delete(processId);
        reject(error);
      });
    });
  }

  // Parse progress from FFmpeg stderr output
  parseFFmpegProgress(line, totalDurationSeconds = null) {
    // Multiple parsing strategies for robustness
    const timeMatch = line.match(/time=(\d{2}):(\d{2}):(\d{2}\.\d{2})/);
    if (timeMatch && totalDurationSeconds) {
      const [, hours, minutes, seconds] = timeMatch;
      const currentSeconds = parseInt(hours) * 3600 + parseInt(minutes) * 60 + parseFloat(seconds);
      const progressPercent = Math.min(95, (currentSeconds / totalDurationSeconds) * 100);
      
      return {
        currentTime: currentSeconds,
        totalTime: totalDurationSeconds,
        progressPercent: Math.round(progressPercent)
      };
    }
    
    return null;
  }

  // Cleanup method for graceful shutdown
  async cleanup() {
    const processes = Array.from(this.activeProcesses.values());
    await Promise.all(processes.map(proc => {
      proc.kill('SIGTERM');
      return new Promise(resolve => {
        proc.on('close', resolve);
        setTimeout(() => {
          proc.kill('SIGKILL');
          resolve();
        }, 5000);
      });
    }));
  }
}

module.exports = FFmpegWrapper;
```

**Core Transcoding Service Skeleton:**

```javascript
// src/services/transcoding/TranscodingService.js
const FFmpegWrapper = require('./FFmpegWrapper');
const QualityLadder = require('./QualityLadder');
const { TranscodingJob } = require('../../models/TranscodingJob');
const { Video } = require('../../models/Video');

class TranscodingService {
  constructor(databaseService, storageService) {
    this.db = databaseService;
    this.storage = storageService;
    this.ffmpeg = new FFmpegWrapper();
    this.qualityLadder = new QualityLadder();
  }

  // Start transcoding for uploaded video - creates jobs for each quality
  async startTranscoding(videoId, inputPath, outputDir) {
    // TODO 1: Load video metadata from database to get duration, resolution
    // TODO 2: Generate quality ladder based on source video characteristics  
    // TODO 3: Create transcoding jobs in database for each target quality
    // TODO 4: Update video processing status to indicate transcoding started
    // TODO 5: Return job IDs for monitoring purposes
    // Hint: Use Video.findById() and TranscodingJob.createForVideo()
  }

  // Process single transcoding job - called by worker processes
  async processTranscodingJob(jobId) {
    // TODO 1: Claim the transcoding job using TranscodingJob.claimNextJob()
    // TODO 2: Validate input file exists and output directory is writable
    // TODO 3: Set up progress callback to update job progress in database
    // TODO 4: Call FFmpeg wrapper with job configuration
    // TODO 5: Validate output files meet quality requirements
    // TODO 6: Mark job completed or failed based on results
    // TODO 7: Update video status when all jobs for video complete
    // Hint: Use this.ffmpeg.spawnFFmpegProcess() with progress tracking
  }

  // Monitor progress of active transcoding job
  async updateJobProgress(jobId, progressPercent, estimatedTimeRemaining) {
    // TODO 1: Validate job is in PROCESSING state
    // TODO 2: Update progress_percent and estimated completion time
    // TODO 3: Handle progress anomalies (backwards progress, >100%)
    // TODO 4: Log progress milestones for monitoring
    // Hint: Use TranscodingJob.updateProgress() with validation
  }

  // Generate HLS manifest linking all quality variants
  async createMasterManifest(videoId, completedJobs) {
    // TODO 1: Collect all completed transcoding jobs for the video
    // TODO 2: Extract resolution and bitrate from each quality variant
    // TODO 3: Generate M3U8 master playlist with EXT-X-STREAM-INF tags
    // TODO 4: Save master manifest to storage and update database
    // TODO 5: Update video status to indicate streaming ready
    // Hint: Master manifest references individual quality playlists
  }

  // Validate transcoded output meets quality standards
  async validateTranscodingOutput(outputPath, expectedQuality) {
    // TODO 1: Check that M3U8 manifest file exists and is valid
    // TODO 2: Verify all referenced TS segments exist and have content
    // TODO 3: Validate segment durations match expected values
    // TODO 4: Check video resolution matches target quality
    // TODO 5: Ensure audio track is present and properly encoded
    // Hint: Use FFprobe to validate output characteristics
  }

  // Cleanup partial outputs from failed jobs
  async cleanupFailedJob(jobId) {
    // TODO 1: Load job details to find output directory
    // TODO 2: Remove any partial M3U8 and TS files
    // TODO 3: Clean up temporary working directories
    // TODO 4: Update job status to failed in database
    // Hint: Use storage service to remove files atomically
  }
}

module.exports = TranscodingService;
```

**Background Worker Implementation:**

```javascript
// src/workers/transcoding-worker.js  
const TranscodingService = require('../services/transcoding/TranscodingService');
const DatabaseService = require('../services/database/DatabaseService');
const StorageService = require('../services/storage/StorageService');
const { TranscodingJob } = require('../models/TranscodingJob');
const { MAX_CONCURRENT_JOBS } = require('../config/transcoding-config');

class TranscodingWorker {
  constructor() {
    this.workerId = `worker_${process.pid}_${Date.now()}`;
    this.isRunning = false;
    this.activeJobs = new Map();
    
    // Initialize services
    this.db = new DatabaseService();
    this.storage = new StorageService();
    this.transcoding = new TranscodingService(this.db, this.storage);
  }

  // Main worker loop - polls for jobs and processes them
  async start() {
    // TODO 1: Set isRunning flag and log worker startup
    // TODO 2: Enter main polling loop while isRunning is true
    // TODO 3: Check if worker capacity allows more jobs (MAX_CONCURRENT_JOBS)
    // TODO 4: Attempt to claim next available job from database
    // TODO 5: If job claimed, start processing in background
    // TODO 6: If no jobs available, wait with exponential backoff
    // TODO 7: Handle worker shutdown gracefully
    // Hint: Use TranscodingJob.claimNextJob(workerId) for atomic claiming
  }

  // Process individual transcoding job
  async processJob(job) {
    // TODO 1: Add job to activeJobs map for tracking
    // TODO 2: Set up error handling for job processing
    // TODO 3: Call TranscodingService.processTranscodingJob()
    // TODO 4: Handle job completion (success or failure)
    // TODO 5: Remove job from activeJobs map
    // TODO 6: Log job completion with timing metrics
    // Hint: Wrap in try/catch to handle FFmpeg failures gracefully
  }

  // Graceful shutdown handling
  async shutdown() {
    // TODO 1: Set isRunning to false to stop polling loop
    // TODO 2: Wait for active jobs to complete or timeout
    // TODO 3: Mark any incomplete jobs as failed for retry
    // TODO 4: Cleanup FFmpeg processes and temporary files
    // TODO 5: Close database connections
    // Hint: Use Promise.all with timeout to wait for job completion
  }
}

// Worker startup and signal handling
const worker = new TranscodingWorker();

process.on('SIGTERM', () => worker.shutdown());
process.on('SIGINT', () => worker.shutdown());

worker.start().catch(console.error);
```

**Milestone Checkpoint:**
After implementing the transcoding pipeline, verify functionality with these tests:

1. **Upload Test Video**: Use a short MP4 file (30 seconds) and confirm transcoding jobs are created
2. **Worker Processing**: Start a transcoding worker and verify it claims and processes jobs
3. **Quality Output**: Check that multiple quality variants are generated (360p, 720p, etc.)
4. **HLS Validation**: Confirm M3U8 manifests are valid and reference correct segments
5. **Progress Tracking**: Monitor job progress updates in the database during processing

Expected command: `node src/workers/transcoding-worker.js` should start processing jobs and log progress updates.

**Debugging Tips:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|-------------|-----------|-----|
| Jobs stuck in PENDING | No workers running | Check worker processes | Start transcoding worker |
| FFmpeg "command not found" | Missing FFmpeg installation | Test `ffmpeg -version` | Install FFmpeg package |
| High memory usage | Large video processing | Monitor process memory | Add memory limits, restart workers |
| Progress stays at 0% | Duration parsing failed | Check video metadata | Implement fallback progress methods |
| Segments not found | Path configuration error | Check output directory | Verify storage paths match job config |


## Streaming Service Design

> **Milestone(s):** Milestone 3 (Adaptive Streaming) - This section details the streaming service that serves HLS manifests and video segments with proper HTTP headers for adaptive streaming and CDN compatibility.

Think of the streaming service as a sophisticated video library system. Just as a library maintains a card catalog that tells you which books exist, where to find them, and what condition they're in, the streaming service maintains manifest files that tell video players which video segments exist, where to download them, and what quality they represent. The key difference is that instead of walking to a physical shelf, the player makes HTTP requests to fetch video segments, and instead of reading one book from start to finish, it can dynamically switch between different quality versions of the same content based on network conditions.

The streaming service operates on a fundamental principle: **adaptive streaming breaks the traditional download-then-play model**. Instead of forcing users to download an entire video file before playback begins, it serves bite-sized video segments that can be downloaded quickly and played immediately. This creates a smooth viewing experience where playback starts within seconds, and quality automatically adjusts to match the viewer's network bandwidth.

![HLS Streaming Protocol Flow](./diagrams/hls-streaming.svg)

### HLS Protocol Implementation

The HTTP Live Streaming (HLS) protocol solves the core challenge of adaptive video delivery through a two-tier playlist system. Think of it like a restaurant menu structure: there's a master menu that lists all the different meal categories (quality levels), and then detailed menus for each category that list the specific dishes (video segments) available. The player first requests the master playlist to understand what quality options exist, then requests specific quality playlists to get the actual segment URLs.

**Master Playlist Structure**

The master playlist serves as the entry point for all HLS streams. It contains references to variant playlists, each representing a different quality rendition of the same video content. The streaming service generates this M3U8 file by examining all available transcoded outputs for a video and creating entries that describe their characteristics.

| Field | Type | Description |
|-------|------|-------------|
| `#EXTM3U` | Header | Required HLS playlist identifier |
| `#EXT-X-VERSION` | Integer | HLS protocol version (typically 3) |
| `#EXT-X-STREAM-INF` | Attributes | Quality metadata for variant playlist |
| `BANDWIDTH` | Integer | Target bitrate in bits per second |
| `RESOLUTION` | String | Video dimensions as WIDTHxHEIGHT |
| `CODECS` | String | Video and audio codec specifications |
| Playlist URL | String | Relative path to quality variant playlist |

The streaming service constructs master playlists by querying the database for all completed `TranscodingJob` records associated with a video, then building the appropriate M3U8 content that references each quality variant.

**Variant Playlist Structure**

Each variant playlist contains the actual video segments for a specific quality level. Think of this as a detailed timeline that tells the player exactly which file to download for each few seconds of video content. The playlist includes timing information, segment durations, and URLs for each transportable stream (TS) file.

| Field | Type | Description |
|-------|------|-------------|
| `#EXTM3U` | Header | Required HLS playlist identifier |
| `#EXT-X-VERSION` | Integer | HLS protocol version |
| `#EXT-X-TARGETDURATION` | Integer | Maximum segment duration in seconds |
| `#EXT-X-MEDIA-SEQUENCE` | Integer | Sequence number of first segment |
| `#EXTINF` | Float | Duration of following segment |
| Segment URL | String | Relative path to TS video segment file |
| `#EXT-X-ENDLIST` | Marker | Indicates playlist is complete (VOD) |

The streaming service generates variant playlists by reading the segment files produced during transcoding and calculating their exact durations using FFprobe analysis.

> **Architecture Decision: M3U8 Generation Strategy**
> - **Context**: The streaming service needs to generate playlist files either dynamically on request or pre-computed during transcoding completion
> - **Options Considered**: 
>   1. Dynamic generation from database queries
>   2. Pre-computed files stored alongside segments
>   3. Hybrid approach with cached dynamic generation
> - **Decision**: Pre-computed files with database fallback
> - **Rationale**: Pre-computed playlists eliminate database queries during high-traffic streaming, reducing latency and database load. Database fallback provides flexibility for debugging and custom playlist variations
> - **Consequences**: Increases storage requirements slightly but dramatically improves streaming performance and reduces database dependency during playback

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Dynamic Generation | Always current, flexible customization | Database load, query latency | No |
| Pre-computed Files | Fast serving, no DB dependency | Storage overhead, update complexity | Yes |
| Hybrid Caching | Balance of performance and flexibility | Implementation complexity | No |

**Segment Serving Implementation**

Video segment serving requires careful attention to HTTP headers and content delivery optimization. Each TS file represents a few seconds of video content encoded at a specific quality level, and the streaming service must deliver these files with appropriate metadata for efficient playback and caching.

The segment serving process follows these steps:

1. The player requests a specific TS file based on a URL from a variant playlist
2. The streaming service validates the segment exists and the client has access permissions
3. It reads the segment file from storage (either local disk or cloud storage)
4. It sets appropriate HTTP response headers for content type, caching, and range support
5. It streams the segment content to the client with proper error handling for network interruptions

**Content Type and Header Configuration**

Proper HTTP headers are crucial for HLS streaming compatibility across different players and CDN systems. The streaming service must configure headers that optimize both immediate playback and downstream caching behavior.

| Header | Value | Purpose |
|--------|-------|---------|
| `Content-Type` | `application/vnd.apple.mpegurl` | M3U8 playlist identification |
| `Content-Type` | `video/mp2t` | TS segment identification |
| `Cache-Control` | `public, max-age=31536000` | Long-term caching for segments |
| `Cache-Control` | `public, max-age=30` | Short-term caching for playlists |
| `Accept-Ranges` | `bytes` | Enable HTTP range requests |
| `Access-Control-Allow-Origin` | `*` | CORS support for web players |
| `Access-Control-Allow-Methods` | `GET, HEAD, OPTIONS` | Allowed HTTP methods |
| `Content-Length` | File size bytes | Enables progress tracking |

### Adaptive Bitrate Logic

Adaptive bitrate streaming represents one of the most sophisticated aspects of modern video delivery. Think of it like an intelligent traffic routing system that constantly monitors road conditions (network bandwidth) and automatically directs vehicles (video data) along the best available route (quality level) to ensure smooth arrival at the destination (the viewer's screen).

The adaptive bitrate system operates through a continuous feedback loop between the player and streaming service. The player monitors its buffer levels, download speeds, and playback quality, then uses this information to request segments from higher or lower quality playlists as conditions change.

**Quality Ladder Design**

The quality ladder defines the set of available video renditions that players can choose between during adaptive streaming. Think of it as a staircase where each step represents a different balance between video quality and bandwidth requirements. The streaming service must provide enough steps to accommodate various network conditions while avoiding excessive transcoding overhead.

| Quality Level | Resolution | Target Bitrate | Use Case |
|---------------|------------|----------------|----------|
| Low | 360p (640x360) | 800 Kbps | Mobile networks, slow connections |
| Medium | 720p (1280x720) | 2500 Kbps | Standard broadband, WiFi |
| High | 1080p (1920x1080) | 5000 Kbps | High-speed connections, premium experience |
| Ultra | 1440p (2560x1440) | 8000 Kbps | High-bandwidth connections (optional) |

The streaming service generates this quality ladder during the transcoding phase, but the adaptive bitrate logic determines how players navigate between these options during playback.

**Bandwidth Estimation Algorithms**

Players estimate available bandwidth by measuring segment download performance over time. The streaming service supports this process by providing consistent segment sizes and timing information that enables accurate bandwidth calculations.

The bandwidth estimation process follows these steps:

1. Player requests a video segment and records the start time
2. Player measures actual download time and segment size when download completes
3. Player calculates instantaneous bandwidth as bytes downloaded divided by download duration
4. Player maintains a moving average of recent bandwidth measurements to smooth out temporary fluctuations
5. Player compares estimated bandwidth to quality level requirements to determine optimal quality for next segment

**Quality Switching Decision Logic**

The streaming service enables smooth quality switching by organizing segments with aligned boundaries across all quality levels. This means segment N at 360p covers the exact same time range as segment N at 720p, allowing players to switch quality levels at any segment boundary without creating playback discontinuities.

Quality switching decisions follow conservative algorithms designed to avoid excessive oscillation between quality levels:

1. **Upward Switching**: Player switches to higher quality only when estimated bandwidth exceeds the target bitrate by a significant margin (typically 25-50%) and buffer levels are healthy
2. **Downward Switching**: Player switches to lower quality immediately when estimated bandwidth drops below current quality requirements or buffer levels become critically low
3. **Startup Behavior**: Player begins with lowest quality level to minimize startup time, then quickly ramps up to appropriate quality based on initial bandwidth measurements

> The key insight for adaptive streaming is that smooth playback without buffering interruptions is more important than maximizing video quality. A player that maintains consistent playback at medium quality provides a better user experience than one that oscillates between high and low quality or frequently pauses to buffer.

### Caching and CDN Strategy

Content Delivery Network (CDN) optimization transforms the streaming service from a single-point-of-failure system into a globally distributed video delivery network. Think of CDNs like a chain of video rental stores: instead of everyone having to drive to the one central store (origin server), there are local branches (edge servers) in every neighborhood that stock popular movies (cached video segments).

**CDN-Friendly Content Organization**

The streaming service organizes content with CDN caching characteristics in mind. Video segments are immutable once created - they never change after transcoding completes - making them ideal for aggressive edge caching. Playlist files, however, might need updates for live content or debugging, requiring more nuanced cache policies.

| Content Type | Mutability | Cache Duration | CDN Behavior |
|--------------|------------|----------------|--------------|
| Video Segments (.ts) | Immutable | 1 year | Aggressive edge caching |
| Variant Playlists (.m3u8) | Semi-mutable | 30 seconds | Edge caching with revalidation |
| Master Playlists | Semi-mutable | 30 seconds | Edge caching with revalidation |
| Thumbnail Images | Immutable | 1 year | Aggressive edge caching |

**Cache Invalidation Strategy**

Cache invalidation represents one of the classic hard problems in computer science, and video streaming adds additional complexity because content is distributed across multiple quality levels and segment files. The streaming service implements a hierarchical invalidation strategy that minimizes cache churn while ensuring content consistency.

When video content needs updates (rare for video-on-demand but possible for corrections or re-transcoding), the invalidation process follows this sequence:

1. Mark the video as temporarily unavailable in the database to prevent new requests
2. Invalidate master playlist at CDN edge servers using versioned URLs or cache tags
3. Allow variant playlists to expire naturally (30-second TTL limits exposure window)
4. Update content with new segment files using different filenames to avoid cache conflicts
5. Update database records and re-enable video availability
6. New requests automatically receive updated playlists pointing to new segment files

**Origin Server Protection**

CDNs protect the streaming service origin servers from traffic spikes, but the streaming service must implement additional protections for scenarios where CDN caches are cold or invalidated simultaneously. This prevents origin server overload during popular content launches or cache invalidation events.

| Protection Mechanism | Implementation | Benefit |
|---------------------|----------------|----------|
| Rate Limiting | Token bucket per IP address | Prevents abuse and DoS attacks |
| Request Coalescing | Deduplicate identical segment requests | Reduces duplicate origin fetches |
| Stale-While-Revalidate | Serve cached content during updates | Maintains availability during refreshes |
| Circuit Breaker | Fail fast when origin is overloaded | Prevents cascade failures |

> **Architecture Decision: Segment URL Strategy**
> - **Context**: Video segments and playlists need URLs that optimize CDN caching while supporting content updates and debugging
> - **Options Considered**: 
>   1. Simple paths like `/video/123/segment001.ts`
>   2. Versioned paths like `/video/123/v2/segment001.ts`
>   3. Content-addressed paths like `/video/123/abc123def456/segment001.ts`
> - **Decision**: Content-addressed paths with transcoding job ID
> - **Rationale**: Content-addressed URLs ensure perfect cache isolation between different transcoding runs, eliminate cache invalidation complexity, and support easy debugging by encoding the transcoding job ID in the path
> - **Consequences**: URLs become longer and less human-readable, but cache behavior becomes predictable and debugging becomes significantly easier

### Architecture Decision Records

> **Decision: HLS Segment Duration**
> - **Context**: HLS segments can range from 2-10 seconds, with trade-offs between startup latency, seek accuracy, and HTTP request overhead
> - **Options Considered**:
>   1. 2-second segments for low latency
>   2. 6-second segments (Apple recommendation)
>   3. 10-second segments for efficiency
> - **Decision**: 6-second segments with configurable override
> - **Rationale**: 6-second segments balance startup time (player needs 2-3 segments buffered) with HTTP overhead. Shorter segments create excessive HTTP requests; longer segments increase startup delay and reduce seek accuracy
> - **Consequences**: Moderate startup latency (12-18 seconds of content needed before playback) but good seeking granularity and reasonable request volume

| Option | Startup Latency | HTTP Requests/Hour | Seek Accuracy | Chosen? |
|--------|-----------------|-------------------|---------------|---------|
| 2-second segments | ~6 seconds | ~1800 requests | Excellent | No |
| 6-second segments | ~12 seconds | ~600 requests | Good | Yes |
| 10-second segments | ~20 seconds | ~360 requests | Poor | No |

> **Decision: Manifest Caching Strategy**
> - **Context**: M3U8 playlists need to be served quickly but may require updates for debugging or content corrections
> - **Options Considered**:
>   1. No caching (always dynamic)
>   2. Short-term caching (30 seconds)
>   3. Long-term caching with versioned URLs
> - **Decision**: Short-term caching with ETag validation
> - **Rationale**: 30-second cache provides significant load reduction while allowing quick content updates. ETag validation enables efficient cache revalidation without full content transfer
> - **Consequences**: Slight complexity in ETag generation but major reduction in database load and faster manifest serving

> **Decision: CORS Policy Configuration**
> - **Context**: Web players need cross-origin access to video content, but overly permissive CORS policies create security risks
> - **Options Considered**:
>   1. Wildcard CORS allowing all origins
>   2. Whitelist-based CORS for known domains
>   3. Dynamic CORS based on referrer headers
> - **Decision**: Configurable whitelist with wildcard fallback for development
> - **Rationale**: Production deployments should restrict CORS to known player domains for security, but development environments benefit from permissive policies. Configuration allows environment-appropriate security
> - **Consequences**: Requires domain management but provides appropriate security posture for different deployment environments

### Common Streaming Pitfalls

Understanding the failure modes of HLS streaming helps developers build robust streaming services that provide consistent playback experiences. These pitfalls represent the most common issues encountered when implementing video streaming systems.

⚠️ **Pitfall: Inconsistent Segment Boundaries Across Quality Levels**

Many developers assume that transcoding automatically creates aligned segment boundaries across different quality levels, but FFmpeg's default behavior can create slight timing variations that break seamless quality switching.

**Why it's wrong**: When segment N at 720p covers 0:30-0:36 but segment N at 1080p covers 0:30-0:36.2, players experience brief audio/video discontinuities during quality switches. This manifests as momentary freezes, audio pops, or synchronization issues.

**How to fix it**: Use FFmpeg's `-force_key_frames` option with identical keyframe intervals across all quality levels. Generate a master keyframe timeline during the first transcoding pass, then force all subsequent quality levels to use the same keyframe positions.

⚠️ **Pitfall: CORS Headers Missing for Preflight Requests**

Web browsers send OPTIONS preflight requests before actual video segment requests, but many streaming services only configure CORS headers for GET requests, causing preflight failures that prevent video playback.

**Why it's wrong**: Modern browsers require CORS preflight approval for requests with custom headers or non-simple methods. When the streaming service returns 405 Method Not Allowed for OPTIONS requests, the browser blocks subsequent segment requests even though they would succeed individually.

**How to fix it**: Configure the streaming service to handle OPTIONS requests and return appropriate CORS headers including `Access-Control-Allow-Methods`, `Access-Control-Allow-Headers`, and `Access-Control-Max-Age`. Test with a web browser console to verify preflight handling.

⚠️ **Pitfall: Playlist Caching Prevents Content Updates**

Aggressive caching of M3U8 playlist files improves performance but can prevent content updates from reaching players for extended periods, making debugging and content corrections nearly impossible.

**Why it's wrong**: When playlists are cached for hours or days, updates to segment URLs, quality additions, or bug fixes don't propagate to players. This creates situations where the streaming service shows updated content but players continue requesting old, potentially non-existent segments.

**How to fix it**: Implement playlist versioning using query parameters (`playlist.m3u8?v=abc123`) or ETag-based cache validation. Use short cache TTLs (30 seconds) for playlists while maintaining long TTLs for segments. Provide cache invalidation mechanisms for emergency updates.

⚠️ **Pitfall: Missing Content-Length Headers Break Progress Tracking**

Streaming services that omit `Content-Length` headers prevent video players from accurately estimating download progress and buffer health, leading to poor adaptive bitrate decisions.

**Why it's wrong**: Without `Content-Length`, players cannot calculate download progress or estimate bandwidth accurately. This causes adaptive bitrate algorithms to make poor quality decisions based on incomplete information, often resulting in excessive quality oscillation or conservative quality selection.

**How to fix it**: Always include `Content-Length` headers for segment responses. For cloud storage backends, ensure the storage service provides content length metadata. Implement content-length validation during transcoding to detect corrupt segments that might have incorrect sizes.

⚠️ **Pitfall: Seek Position Inaccuracy Due to Keyframe Alignment**

Players can only seek to keyframe positions, but streaming services that don't account for keyframe placement can create seek experiences where the video jumps to unexpected positions.

**Why it's wrong**: When a user seeks to 1:30 but the nearest keyframe is at 1:26, the video jumps to 1:26 without clear indication. This creates a confusing user experience where seek operations appear inaccurate or broken.

**How to fix it**: Use consistent keyframe intervals (every 2 seconds is typical) during transcoding. Provide keyframe position metadata in playlists or as separate API responses. Implement seek preview functionality that shows the actual target position rather than the requested position.

⚠️ **Pitfall: HTTP Range Request Support Incomplete**

Many streaming services implement basic segment serving but fail to properly handle HTTP range requests, breaking features like seek preview, bandwidth testing, and progressive downloading optimizations.

**Why it's wrong**: Players often request partial segment content for bandwidth estimation or seek preview functionality. When range requests fail or return incorrect content, players fall back to downloading entire segments, wasting bandwidth and degrading performance.

**How to fix it**: Implement full HTTP range request support including `Accept-Ranges: bytes` headers, proper `Content-Range` responses, and 206 Partial Content status codes. Test range requests with curl to verify byte-range accuracy and proper header handling.

### Implementation Guidance

This implementation guidance provides the foundation for building a production-ready streaming service that handles HLS manifest generation, segment serving, and CDN optimization. The code focuses on Node.js with Express for HTTP handling and integrates with the existing transcoding pipeline data models.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| HTTP Server | Express.js with static file middleware | Fastify with custom streaming handlers |
| Manifest Generation | Template-based string building | Specialized HLS library (node-m3u8) |
| Content Storage | Local filesystem with Express static | AWS S3 with CloudFront CDN |
| Caching Layer | In-memory LRU cache | Redis with cache invalidation |
| CORS Handling | Express cors middleware | Custom CORS with domain validation |

**Recommended File Structure**

```
streaming-service/
├── src/
│   ├── services/
│   │   ├── StreamingService.js     ← Main streaming logic
│   │   ├── ManifestGenerator.js    ← M3U8 playlist generation
│   │   ├── SegmentServer.js        ← TS file serving with HTTP headers
│   │   └── CacheManager.js         ← Playlist caching and invalidation
│   ├── middleware/
│   │   ├── corsHandler.js          ← CORS configuration for web players
│   │   ├── rangeRequests.js        ← HTTP range request support
│   │   └── cacheHeaders.js         ← CDN-friendly cache headers
│   ├── routes/
│   │   ├── hls.js                  ← HLS manifest and segment routes
│   │   └── health.js               ← Service health monitoring
│   └── utils/
│       ├── urlGenerator.js         ← Content-addressed URL generation
│       └── manifestValidator.js    ← M3U8 validation utilities
├── config/
│   ├── streaming.js                ← Segment duration, quality ladder
│   └── cdn.js                      ← Cache policies and CORS settings
└── tests/
    ├── integration/
    │   └── hls-playback.test.js    ← End-to-end streaming tests
    └── unit/
        ├── manifest-generation.test.js
        └── segment-serving.test.js
```

**Infrastructure Starter Code: CORS and Cache Headers Middleware**

```javascript
// middleware/corsHandler.js
const cors = require('cors');

const createCorsHandler = (config) => {
  const corsOptions = {
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps or curl)
      if (!origin) return callback(null, true);
      
      // Check against whitelist in production
      if (config.environment === 'production') {
        if (config.allowedOrigins.includes(origin)) {
          return callback(null, true);
        }
        return callback(new Error('Not allowed by CORS'));
      }
      
      // Allow all origins in development
      callback(null, true);
    },
    methods: ['GET', 'HEAD', 'OPTIONS'],
    allowedHeaders: ['Range', 'Accept', 'Accept-Encoding'],
    exposedHeaders: ['Content-Length', 'Content-Range', 'Accept-Ranges'],
    maxAge: 86400, // 24 hours for preflight caching
    credentials: false
  };
  
  return cors(corsOptions);
};

module.exports = { createCorsHandler };
```

```javascript
// middleware/cacheHeaders.js
const setCacheHeaders = (req, res, next) => {
  const path = req.path;
  
  if (path.endsWith('.ts')) {
    // Video segments are immutable - cache aggressively
    res.set({
      'Cache-Control': 'public, max-age=31536000, immutable',
      'Accept-Ranges': 'bytes'
    });
  } else if (path.endsWith('.m3u8')) {
    // Playlists may update - short cache with revalidation
    res.set({
      'Cache-Control': 'public, max-age=30, must-revalidate',
      'Content-Type': 'application/vnd.apple.mpegurl'
    });
  }
  
  next();
};

module.exports = { setCacheHeaders };
```

**Infrastructure Starter Code: HTTP Range Request Handler**

```javascript
// middleware/rangeRequests.js
const fs = require('fs').promises;
const path = require('path');

const handleRangeRequest = async (req, res, next) => {
  // Only handle range requests for video segments
  if (!req.path.endsWith('.ts') || !req.headers.range) {
    return next();
  }
  
  try {
    const filePath = path.join(req.app.get('segmentDirectory'), req.path);
    const stat = await fs.stat(filePath);
    const fileSize = stat.size;
    
    // Parse range header (format: "bytes=start-end")
    const range = req.headers.range;
    const parts = range.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
    
    // Validate range
    if (start >= fileSize || end >= fileSize) {
      res.status(416).set({
        'Content-Range': `bytes */${fileSize}`
      });
      return res.end();
    }
    
    const chunksize = (end - start) + 1;
    const file = await fs.open(filePath, 'r');
    const buffer = Buffer.alloc(chunksize);
    
    await file.read(buffer, 0, chunksize, start);
    await file.close();
    
    res.status(206).set({
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunksize,
      'Content-Type': 'video/mp2t'
    });
    
    res.end(buffer);
  } catch (error) {
    next(error);
  }
};

module.exports = { handleRangeRequest };
```

**Core Logic Skeleton: Manifest Generator**

```javascript
// services/ManifestGenerator.js
const path = require('path');

class ManifestGenerator {
  constructor(databaseService, storageService) {
    this.db = databaseService;
    this.storage = storageService;
  }
  
  // generateMasterPlaylist creates the top-level M3U8 file that lists all quality variants
  async generateMasterPlaylist(videoId) {
    // TODO 1: Query database for all completed TranscodingJob records for this video
    // TODO 2: For each job, extract quality metadata (resolution, bitrate, codec)
    // TODO 3: Sort quality variants by bitrate (lowest to highest)
    // TODO 4: Generate #EXT-X-STREAM-INF lines with BANDWIDTH and RESOLUTION attributes
    // TODO 5: Create relative URLs pointing to variant playlists
    // TODO 6: Combine into complete M3U8 master playlist format
    // TODO 7: Return playlist content as string
    // Hint: Use QUALITY_LADDER constant to map job configs to HLS attributes
  }
  
  // generateVariantPlaylist creates quality-specific M3U8 files listing video segments
  async generateVariantPlaylist(transcodingJobId) {
    // TODO 1: Query database for TranscodingJob details and segment information
    // TODO 2: Read segment directory to find all .ts files for this job
    // TODO 3: Calculate exact duration for each segment using stored metadata
    // TODO 4: Generate #EXTINF lines with precise duration values
    // TODO 5: Create relative URLs for each segment file
    // TODO 6: Add #EXT-X-TARGETDURATION based on maximum segment duration
    // TODO 7: Add #EXT-X-ENDLIST marker for video-on-demand content
    // TODO 8: Return complete variant playlist as string
    // Hint: Use SEGMENT_DURATION constant but verify actual segment durations
  }
  
  // validatePlaylistContent ensures generated playlists conform to HLS specification
  validatePlaylistContent(playlistContent, playlistType) {
    // TODO 1: Check for required #EXTM3U header
    // TODO 2: Validate #EXT-X-VERSION is present and supported
    // TODO 3: For master playlists, verify #EXT-X-STREAM-INF format
    // TODO 4: For variant playlists, verify #EXT-X-TARGETDURATION and segment format
    // TODO 5: Check that all referenced files exist in storage
    // TODO 6: Validate duration calculations match actual segment lengths
    // TODO 7: Return validation results with specific error messages
  }
}

module.exports = { ManifestGenerator };
```

**Core Logic Skeleton: Streaming Service Routes**

```javascript
// routes/hls.js
const express = require('express');
const router = express.Router();

// Master playlist endpoint - returns top-level M3U8 with quality variants
router.get('/video/:videoId/playlist.m3u8', async (req, res) => {
  // TODO 1: Validate videoId parameter and check video exists in database
  // TODO 2: Check video processing status is COMPLETED
  // TODO 3: Generate or retrieve cached master playlist using ManifestGenerator
  // TODO 4: Set appropriate Content-Type header for M3U8
  // TODO 5: Set cache headers for playlist caching strategy
  // TODO 6: Send playlist content with proper HTTP status
  // TODO 7: Handle errors with appropriate HTTP status codes
  // Hint: Use Video.findById() and check processing_status field
});

// Quality variant playlist endpoint - returns segment list for specific quality
router.get('/video/:videoId/:quality/playlist.m3u8', async (req, res) => {
  // TODO 1: Validate parameters and find corresponding TranscodingJob
  // TODO 2: Verify transcoding job completed successfully
  // TODO 3: Generate or retrieve cached variant playlist
  // TODO 4: Set M3U8 content type and cache headers
  // TODO 5: Return playlist content to client
  // TODO 6: Handle missing quality levels with 404 responses
  // Hint: Query TranscodingJob by video_id and quality_config matching
});

// Video segment endpoint - serves individual TS files
router.get('/video/:videoId/:quality/segment:segmentNum.ts', async (req, res) => {
  // TODO 1: Validate parameters and construct segment file path
  // TODO 2: Check segment file exists in storage system
  // TODO 3: Set appropriate Content-Type header for TS files
  // TODO 4: Set cache headers for long-term segment caching
  // TODO 5: Handle range requests for seeking and bandwidth testing
  // TODO 6: Stream file content to client with error handling
  // TODO 7: Log access for analytics and debugging
  // Hint: Use StorageService.getFileStream() for efficient streaming
});

module.exports = router;
```

**Language-Specific Hints for Node.js**

- Use `express.static()` middleware for serving video segments efficiently, but override with custom middleware for range request handling
- Implement playlist caching with `node-cache` or Redis to avoid database queries on every playlist request
- Use `fs.createReadStream()` with range options for efficient segment serving without loading entire files into memory
- Set up proper error handling with Express error middleware to catch streaming failures and return appropriate HTTP status codes
- Use `helmet` middleware to set security headers, but configure it to allow the necessary CORS headers for video streaming

**Milestone Checkpoint: HLS Streaming Verification**

After implementing the streaming service, verify functionality with these tests:

1. **Master Playlist Generation**: Request `GET /video/{videoId}/playlist.m3u8` and verify response contains valid M3U8 format with multiple #EXT-X-STREAM-INF entries
2. **Variant Playlist Access**: Request each quality variant playlist and confirm segment URLs are correctly formatted and reference existing files
3. **Segment Serving**: Download individual TS segments and verify they play in VLC or other media players
4. **Range Request Support**: Use curl with range headers (`curl -H "Range: bytes=0-1023" {segment-url}`) and verify 206 Partial Content responses
5. **CORS Functionality**: Load a test HTML page from a different origin and verify video player can access manifest and segments without CORS errors

Expected behavior: A complete video should start playing within 10-15 seconds of requesting the master playlist, with quality switching working automatically based on bandwidth conditions.

Signs of problems:
- **Long startup delays**: Check segment generation and manifest accuracy
- **CORS errors in browser console**: Verify preflight OPTIONS request handling
- **Seeking problems**: Validate keyframe alignment and segment boundary consistency
- **Quality switching failures**: Check master playlist bandwidth values and variant playlist availability


## Video Player Integration

> **Milestone(s):** Milestone 4 (Video Player Integration) - This section details the frontend player implementation using HLS.js for adaptive streaming with quality controls and analytics tracking.

Building a video player for adaptive streaming is like constructing a sophisticated radio that not only receives multiple stations but automatically tunes to the clearest frequency based on signal strength. The player must understand the HLS protocol to request manifests, fetch video segments, and adapt quality based on network conditions while providing familiar playback controls. Unlike simple video players that load a single file, an adaptive streaming player orchestrates dozens of network requests, monitors bandwidth, and seamlessly switches between quality renditions without interrupting playback.

The video player represents the final component in our streaming pipeline, consuming the HLS manifests and segments generated by the transcoding pipeline and served by the streaming service. This player must handle the complexities of adaptive bitrate streaming while presenting a clean, intuitive interface to users. The implementation centers around HLS.js, a JavaScript library that brings HLS support to browsers that lack native HLS capabilities, combined with custom controls for quality switching and comprehensive analytics tracking.

### HLS.js Integration Pattern: Player initialization, event handling, and quality switching

The foundation of our video player rests on HLS.js, which acts as a translation layer between the HLS protocol and the browser's native media capabilities. Think of HLS.js as a skilled interpreter at a multilingual conference—it understands the HLS "language" of manifests and segments, then translates everything into the standardized HTML5 video API that browsers comprehend. This abstraction allows us to deliver adaptive streaming to virtually any modern browser, regardless of native HLS support.

The integration pattern follows a structured initialization sequence that establishes the player, configures event handlers, and begins the manifest loading process. The player initialization creates an HLS.js instance, attaches it to an HTML5 video element, and configures the library's behavior for our specific streaming requirements. This setup phase determines crucial performance characteristics like segment buffer sizes, quality switching aggressiveness, and error recovery strategies.

| Component | Responsibility | Key Configuration |
|-----------|---------------|-------------------|
| HLS Instance | Manages manifest parsing and segment loading | `maxBufferLength`, `maxMaxBufferLength` |
| Video Element | Provides native playback and browser integration | `preload`, `controls`, `crossorigin` |
| Event Manager | Coordinates HLS events with UI updates | Error handling, quality change notifications |
| Quality Manager | Tracks available qualities and user preferences | Manual vs automatic switching logic |

The player initialization sequence begins by creating the HTML5 video element and HLS.js instance, then establishing the critical connection between them. The HLS instance becomes responsible for all network operations—fetching manifests, parsing segment lists, downloading video chunks, and feeding them to the video element's buffer. Meanwhile, the video element handles the actual playback, seeking, and user interaction, completely unaware that it's consuming adaptive streams rather than a single video file.

Event handling forms the nervous system of the HLS integration, carrying information about manifest loading, quality switches, errors, and playback state changes throughout the player architecture. The HLS.js library emits dozens of different events during normal operation, from successful manifest parsing to buffer starvation warnings. Our player must selectively listen to relevant events and translate them into UI updates, analytics data, or error recovery actions.

| HLS.js Event | Trigger Condition | Player Response |
|--------------|------------------|-----------------|
| `MANIFEST_LOADING` | Initial manifest request begins | Show loading indicator |
| `MANIFEST_PARSED` | Master playlist successfully parsed | Populate quality selector |
| `LEVEL_SWITCHED` | Quality rendition changed | Update quality display |
| `FRAG_LOADED` | Video segment downloaded | Update buffer indicator |
| `ERROR` | Network, media, or other failure | Trigger recovery or user notification |

Quality switching represents the most sophisticated aspect of HLS.js integration, requiring coordination between bandwidth monitoring, user preferences, and playback stability. The library continuously monitors network conditions and automatically selects appropriate quality levels, but our player must also support manual overrides and intelligent switching logic. This creates a three-way interaction between automatic adaptation, user intent, and content availability.

The automatic quality switching algorithm in HLS.js operates like a feedback control system, measuring network throughput during segment downloads and adjusting quality selections to maintain smooth playback. When bandwidth increases, the algorithm gradually steps up to higher qualities to improve visual experience. When network conditions degrade, it quickly steps down to lower qualities to prevent buffer underruns and playback interruptions.

> **Key Design Insight**: The HLS.js integration must balance automation with user control. While the library excels at automatic adaptation, users occasionally need manual override capabilities—perhaps they're on a limited data plan and want to lock to low quality, or they have abundant bandwidth and prefer maximum quality despite network fluctuations.

Manual quality switching introduces additional complexity because it must override the automatic algorithm without breaking the underlying adaptation logic. When a user manually selects a quality level, our player disables automatic switching and forces HLS.js to use only the specified rendition. However, the player must monitor for severe network degradation and potentially re-enable automatic switching to prevent playback failures.

### Player Controls Design: Play/pause, seeking, volume, and quality selector implementation

The player controls serve as the user's interface to the underlying HLS streaming technology, translating familiar media player interactions into the appropriate HLS.js operations and video element manipulations. Like a car's dashboard that abstracts the complexity of the engine into simple gauges and controls, our player interface hides the intricacies of adaptive streaming behind intuitive play, pause, seek, and quality adjustment controls.

The control design follows progressive enhancement principles, starting with basic HTML5 video functionality and layering adaptive streaming features on top. This approach ensures graceful degradation when JavaScript fails or HLS.js encounters unsupported scenarios. The base video element provides fundamental playback controls, while our custom interface adds adaptive streaming-specific features like quality selection and buffer visualization.

| Control Component | Base Functionality | Adaptive Enhancement |
|------------------|-------------------|---------------------|
| Play/Pause Button | Standard video element control | Buffer state awareness |
| Progress Bar | Basic seek and position display | Segment-aware seeking |
| Volume Control | Audio level adjustment | Persistent preference storage |
| Quality Selector | Not applicable | Rendition switching and bandwidth display |
| Buffer Indicator | Not applicable | Download progress and health visualization |

The play and pause controls require careful integration with HLS.js to handle the complexities of adaptive streaming. Unlike traditional video players where play/pause directly controls a single file, our implementation must coordinate with manifest loading, segment downloading, and quality switching. When a user clicks play before the initial manifest loads, the player must show appropriate loading states and handle potential initialization failures.

Progress bar implementation for adaptive streaming presents unique challenges because seeking operations must account for segment boundaries and quality switching. Traditional video seeking involves jumping to a specific byte offset in a file, but HLS seeking requires calculating which segment contains the target time, potentially switching to a different quality rendition, and requesting new segments from that position. Our progress bar must translate user seek gestures into HLS-compatible operations while providing smooth visual feedback.

The progress bar calculation involves mapping user clicks on the visual timeline to timestamps in the video content, then determining which HLS segment contains that timestamp. This process requires knowledge of the current quality rendition's segment structure, as different quality levels may have slightly different segment durations or timing offsets. The seeking operation then instructs HLS.js to begin loading from the appropriate segment, while the UI provides immediate feedback by updating the progress position.

| Seeking Operation | User Action | HLS.js Operation | UI Feedback |
|------------------|-------------|-----------------|-------------|
| Forward Seek | Click ahead on timeline | Request future segments | Immediate position update |
| Backward Seek | Click behind on timeline | Flush buffer, request past segments | Position update with loading indicator |
| Segment Boundary | Seek to exact segment start | Direct segment request | Instant response |
| Cross-Quality Seek | Seek during quality switch | Coordinate with adaptation logic | Loading state until resolution |

Volume control implementation remains relatively straightforward since it operates directly on the HTML5 video element, independent of the streaming protocol. However, our implementation adds persistence capabilities, storing user volume preferences in local storage and restoring them across playback sessions. This enhancement improves user experience by remembering individual preferences without requiring account systems or server-side storage.

The quality selector represents the most sophisticated control component, providing users with visibility into available quality renditions and the ability to override automatic adaptation. The selector must dynamically populate with available qualities parsed from the HLS manifest, display current quality selection and automatic switching status, and provide clear indication of network-driven vs user-driven quality changes.

Quality selector population occurs after HLS.js successfully parses the master manifest and identifies available renditions. Each quality option displays both technical specifications (resolution, bitrate) and user-friendly labels (480p, 720p, 1080p). The selector also includes an "Auto" option that re-enables HLS.js automatic adaptation after manual override.

| Quality Option | Display Label | Technical Info | Selection Behavior |
|---------------|---------------|---------------|-------------------|
| Auto | "Auto (Currently 720p)" | Current rendition indication | Enable automatic adaptation |
| 1080p | "1080p (5000 kbps)" | Resolution and bitrate | Force high quality |
| 720p | "720p (2500 kbps)" | Resolution and bitrate | Force medium quality |
| 480p | "480p (1000 kbps)" | Resolution and bitrate | Force low quality |

### Playback Analytics: Tracking view duration, quality switches, and buffering events

Playback analytics transform the video player from a simple media renderer into a sophisticated data collection system that provides insights into user behavior, content performance, and technical health. Think of analytics as the black box recorder on an airplane—continuously monitoring flight conditions, recording critical events, and providing data for post-flight analysis. Our analytics system captures every significant player event, from initial play button clicks to quality switches to abandonment points.

The analytics architecture operates on an event-driven model where player interactions, HLS.js events, and technical metrics generate timestamped records that flow to analytics storage. This design separates data collection from data analysis, allowing the player to focus on smooth playback while background processes handle analytics aggregation and reporting. The analytics system must balance comprehensive data collection with performance impact, ensuring that measurement doesn't interfere with the user experience it's meant to improve.

| Analytics Category | Tracked Events | Data Points | Business Value |
|-------------------|---------------|-------------|----------------|
| Engagement | Play, pause, seek, abandon | Duration, completion rate, interaction patterns | Content optimization |
| Technical Performance | Buffer events, quality switches, errors | Switching frequency, error rates, loading times | Infrastructure optimization |
| Quality Experience | Rendition usage, adaptation triggers | Time spent per quality, user overrides | Encoding optimization |
| Device Context | Browser, screen size, connection | Capability matching, performance correlation | Platform prioritization |

View duration tracking requires sophisticated logic to distinguish between active viewing and background tab scenarios. Simple play time measurement fails to capture user attention accurately—a video might play for 30 minutes in a background tab while the user works elsewhere, or a user might pause frequently during active engagement. Our analytics system combines multiple signals including page visibility, user interactions, and playback state to estimate genuine viewing time.

The view duration calculation operates through a combination of play/pause event tracking and periodic heartbeat signals. When playback begins, the analytics system starts a viewing session and records the timestamp. Pause events, seeking operations, and quality changes all generate additional data points that contribute to engagement analysis. The system also monitors page visibility changes to detect when users switch tabs or minimize windows, adjusting view duration calculations accordingly.

| Viewing State | Detection Method | Duration Calculation | Analytics Impact |
|---------------|-----------------|---------------------|------------------|
| Active Viewing | Play state + visible tab | Continuous time accumulation | Full engagement credit |
| Paused Engagement | Pause state + recent interaction | No time accumulation | Engagement signal without duration |
| Background Playback | Play state + hidden tab | Reduced time credit | Partial engagement attribution |
| Abandoned Session | No interaction + time threshold | Stop accumulation | Abandonment event recording |

Quality switch tracking provides crucial insights into both user behavior and technical performance. Automatic quality switches indicate network conditions and adaptation effectiveness, while manual switches reveal user preferences and satisfaction with automatic selection. The analytics system must distinguish between these different switch types and correlate them with other metrics like viewing duration and user engagement.

Each quality switch event records comprehensive context including the source and destination qualities, the trigger mechanism (automatic vs manual), the network conditions at switch time, and the user's subsequent behavior. This rich dataset enables analysis of adaptation effectiveness, user satisfaction with different quality levels, and the relationship between technical quality and viewing engagement.

| Switch Type | Trigger | Context Captured | Analysis Use |
|-------------|---------|------------------|-------------|
| Automatic Up | Bandwidth increase | Network speed, buffer level | Adaptation effectiveness |
| Automatic Down | Network degradation | Error conditions, buffer status | Performance optimization |
| Manual Override | User selection | Previous quality, session duration | User preference analysis |
| Recovery Switch | Error handling | Error type, retry attempts | Technical health monitoring |

Buffering event analytics capture the technical health of the streaming experience, identifying network bottlenecks, content delivery issues, and player configuration problems. Buffer events include starvation (when playback stops due to insufficient downloaded content), recovery (when buffering completes and playback resumes), and preemptive loading (when the player downloads content ahead of current position). These events directly impact user experience and provide actionable insights for infrastructure optimization.

The buffering analytics system correlates buffer events with network conditions, content characteristics, and user context to identify patterns and root causes. For example, consistent buffering at specific points in a video might indicate transcoding issues, while widespread buffering across many users suggests CDN or server capacity problems.

> **Critical Analytics Insight**: Raw event data means nothing without context and aggregation. Individual buffer events are noise, but patterns of buffering correlated with content types, user demographics, or network conditions become actionable intelligence for improving the streaming platform.

### Architecture Decision Records: Decisions around player library, browser compatibility, and analytics

The architecture decisions for video player integration shape every aspect of implementation from browser compatibility to performance characteristics. These decisions establish the foundation for all subsequent development and significantly impact both user experience and maintenance complexity.

> **Decision: HLS.js vs Native HLS Support**
> - **Context**: Different browsers have varying levels of HLS support, with Safari providing native HLS while Chrome and Firefox require JavaScript libraries for HLS playback.
> - **Options Considered**: Rely on native HLS only, use HLS.js everywhere, or hybrid approach detecting native support
> - **Decision**: Use HLS.js universally across all browsers
> - **Rationale**: HLS.js provides consistent behavior, features, and debugging across all browsers, eliminating cross-browser compatibility issues and providing unified event handling
> - **Consequences**: Adds JavaScript bundle size but ensures predictable behavior and simplifies maintenance

| Option | Browser Coverage | Feature Consistency | Bundle Size | Maintenance |
|--------|-----------------|-------------------|------------|-------------|
| Native Only | Limited (Safari only) | Inconsistent APIs | Minimal | Complex cross-browser code |
| HLS.js Universal | Full modern browser support | Identical across browsers | ~200KB | Single codebase |
| Hybrid Detection | Full coverage | Mixed APIs | Medium | Dual code paths |

> **Decision: Custom Controls vs HTML5 Native Controls**
> - **Context**: HTML5 video elements provide built-in controls, but adaptive streaming requires additional functionality like quality selection
> - **Options Considered**: Extend native controls, completely custom controls, or hybrid approach
> - **Decision**: Implement completely custom controls with native controls as fallback
> - **Rationale**: Adaptive streaming features like quality selection, buffer visualization, and analytics integration require custom UI that can't be achieved with native controls
> - **Consequences**: Increased development complexity but provides full control over user experience and adaptive streaming features

The custom controls decision enables sophisticated adaptive streaming features but requires comprehensive accessibility implementation and extensive cross-browser testing. Native controls provide automatic accessibility compliance and familiar user interfaces, but they can't accommodate quality selection, buffer status, or other adaptive streaming requirements.

> **Decision: Client-Side vs Server-Side Analytics**
> - **Context**: Analytics data can be processed in the browser for immediate feedback or sent to servers for centralized analysis
> - **Options Considered**: Pure client-side processing, real-time server streaming, or batched server transmission
> - **Decision**: Hybrid approach with immediate client-side processing and batched server transmission
> - **Rationale**: Client-side processing enables immediate UI responses to analytics data while server-side storage provides comprehensive analysis and cross-session insights
> - **Consequences**: Requires robust error handling for network failures and careful privacy consideration for data transmission

| Analytics Architecture | Real-time Capability | Data Persistence | Privacy Control | Implementation Complexity |
|------------------------|-------------------|------------------|-----------------|-------------------------|
| Client-only | High | Session-limited | Full user control | Low |
| Server-only | Medium (network dependent) | Comprehensive | Server-controlled | Medium |
| Hybrid Batch | High (client) + Medium (server) | Best of both | Configurable | High |

> **Decision: Quality Selection Algorithm**
> - **Context**: Users need both automatic adaptation and manual override capabilities for quality selection
> - **Options Considered**: Automatic only, manual only, or user-controllable automatic with manual override
> - **Decision**: Default to automatic adaptation with persistent manual override option
> - **Rationale**: Most users benefit from automatic adaptation, but power users and specific network scenarios require manual control
> - **Consequences**: Requires complex UI state management but provides optimal experience for different user types

### Common Player Pitfalls: Browser compatibility, memory leaks, and bandwidth estimation

Video player implementation presents numerous subtle pitfalls that can severely impact user experience or application stability. These issues often manifest inconsistently across browsers, network conditions, or content types, making them particularly challenging to diagnose and fix. Understanding these common pitfalls and their solutions prevents costly debugging sessions and user experience degradation.

⚠️ **Pitfall: Memory Leaks from Event Listeners**
Event listeners attached to HLS.js instances, video elements, or window objects frequently cause memory leaks when players are destroyed and recreated. JavaScript's garbage collector cannot reclaim objects that still have active event listeners, leading to accumulating memory usage during navigation or playlist playback. The leak occurs because event listeners maintain references to player instances, preventing cleanup even after the video element is removed from the DOM.

The solution requires systematic cleanup of all event listeners during player destruction. Create a cleanup method that explicitly removes every event listener attached during initialization, and ensure this cleanup runs when users navigate away from video content or when the player component unmounts in single-page applications.

| Leak Source | Symptoms | Detection Method | Prevention |
|-------------|----------|------------------|------------|
| HLS.js Events | Memory usage grows with video switches | Browser dev tools memory tab | Explicit `hls.off()` calls |
| Video Element Events | Player controls stop responding | DOM listener count inspection | `removeEventListener()` matching |
| Window Events | Global state corruption | Memory profiler snapshots | Cleanup on unmount/navigation |
| Timer References | Background processing continues | CPU usage after player removal | `clearInterval()`/`clearTimeout()` |

⚠️ **Pitfall: Cross-Origin Resource Sharing (CORS) Issues**
HLS streaming requires loading manifests and segments from potentially different domains than the main web application, triggering CORS restrictions that prevent playback. Modern browsers block cross-origin requests unless the server explicitly allows them through appropriate headers. This issue particularly affects development environments where video content is served from different ports or domains than the player application.

The CORS problem manifests as network errors when HLS.js attempts to fetch manifest files or video segments, with browser developer consoles showing "blocked by CORS policy" messages. The solution requires coordination between the streaming service configuration and the player initialization, ensuring proper headers are set and the video element is configured for cross-origin content.

| CORS Scenario | Error Symptoms | Server Solution | Client Solution |
|---------------|----------------|-----------------|-----------------|
| Manifest Loading | "Failed to fetch" errors | `Access-Control-Allow-Origin` header | `crossorigin="anonymous"` attribute |
| Segment Requests | Playback fails after manifest loads | CORS headers on segment endpoints | No client changes needed |
| Development Setup | Works in production, fails locally | Development proxy configuration | Local server CORS middleware |

⚠️ **Pitfall: Bandwidth Estimation Accuracy**
HLS.js bandwidth estimation can be inaccurate during initial playback or network condition changes, leading to inappropriate quality selection that hurts user experience. The library estimates bandwidth by measuring segment download times, but initial segments may not be representative of sustained throughput, and network conditions can change rapidly during playback sessions.

Bandwidth estimation problems appear as excessive quality switching (thrashing between levels), persistent selection of inappropriate quality levels, or slow adaptation to network changes. These issues are particularly problematic on mobile networks where conditions fluctuate rapidly or during video startup when limited measurement data is available.

The solution involves configuring HLS.js bandwidth estimation parameters to match expected network conditions and implementing custom logic to smooth bandwidth estimates over longer time windows. Additionally, providing manual quality override options gives users escape mechanisms when automatic adaptation fails.

| Estimation Problem | User Impact | HLS.js Configuration | Custom Logic |
|-------------------|-------------|---------------------|-------------|
| Initial Overestimation | Immediate buffering after play | Lower `abrEwmaDefaultEstimate` | Gradual quality increase |
| Network Change Lag | Poor quality during good conditions | Faster `abrEwmaFastLive` update | Monitor multiple metrics |
| Mobile Fluctuation | Constant quality switching | Higher switching thresholds | Hysteresis in quality decisions |

⚠️ **Pitfall: Safari HLS Behavior Differences**
Safari's native HLS implementation behaves differently from HLS.js, creating inconsistent experiences when using hybrid approaches or when HLS.js detection logic incorrectly defers to native support. Safari may handle quality switching, error recovery, or seeking differently than the HLS.js implementation, leading to feature differences or functionality gaps.

These differences become apparent when testing across browsers reveals different quality switching behaviors, seeking accuracy, or error handling. The solution requires either forcing HLS.js usage across all browsers for consistency or implementing browser-specific code paths that account for native HLS behavior differences.

⚠️ **Pitfall: Buffer Management Configuration**
Incorrect buffer configuration in HLS.js can cause either excessive memory usage from over-buffering or playback interruptions from under-buffering. The default buffer settings may not match application requirements or user network conditions, leading to suboptimal performance. Over-buffering consumes unnecessary bandwidth and memory, while under-buffering increases the risk of playback starvation during network hiccups.

Buffer management requires tuning multiple HLS.js parameters including `maxBufferLength`, `maxBufferSize`, and `maxMaxBufferLength` based on expected content duration, user network conditions, and device capabilities. The optimal settings balance smooth playback with resource efficiency.

### Implementation Guidance

The video player integration requires careful orchestration of HLS.js, custom controls, and analytics systems. The implementation balances browser compatibility with advanced streaming features while maintaining clean separation between media handling and user interface concerns.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| HLS Library | HLS.js with default configuration | HLS.js with custom buffer management |
| UI Framework | Vanilla JavaScript with DOM manipulation | React/Vue component with state management |
| Analytics | Simple event logging to console | Comprehensive analytics with batched server transmission |
| Styling | CSS with manual responsive design | CSS framework with component system |

**Recommended File Structure:**

```
frontend/
  src/
    components/
      VideoPlayer/
        VideoPlayer.js           ← main player component
        VideoPlayer.css          ← player styling
        controls/
          PlayButton.js          ← play/pause control
          ProgressBar.js         ← seek and progress
          VolumeControl.js       ← audio control
          QualitySelector.js     ← quality switching
        analytics/
          AnalyticsManager.js    ← event tracking
          EventBuffer.js         ← batched transmission
    utils/
      HlsWrapper.js             ← HLS.js integration
      StorageUtils.js           ← preference persistence
  public/
    index.html                  ← player demo page
```

**HLS.js Integration Wrapper (COMPLETE):**

```javascript
// HlsWrapper.js - Complete HLS.js integration utility
import Hls from 'hls.js';

export class HlsWrapper {
  constructor(videoElement, options = {}) {
    this.videoElement = videoElement;
    this.hls = null;
    this.eventHandlers = new Map();
    this.options = {
      maxBufferLength: 30,
      maxMaxBufferLength: 600,
      enableWorker: true,
      ...options
    };
  }

  isSupported() {
    return Hls.isSupported();
  }

  initialize() {
    if (!this.isSupported()) {
      throw new Error('HLS is not supported in this browser');
    }

    this.hls = new Hls(this.options);
    this.hls.attachMedia(this.videoElement);

    // Set up default error recovery
    this.hls.on(Hls.Events.ERROR, (event, data) => {
      if (data.fatal) {
        this.handleFatalError(data);
      }
    });

    return this.hls;
  }

  loadSource(manifestUrl) {
    if (!this.hls) {
      this.initialize();
    }
    this.hls.loadSource(manifestUrl);
  }

  on(eventType, handler) {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, []);
    }
    this.eventHandlers.get(eventType).push(handler);
    
    if (this.hls) {
      this.hls.on(eventType, handler);
    }
  }

  off(eventType, handler) {
    if (this.hls) {
      this.hls.off(eventType, handler);
    }
    
    const handlers = this.eventHandlers.get(eventType);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  getCurrentQuality() {
    return this.hls ? this.hls.currentLevel : -1;
  }

  getQualities() {
    return this.hls ? this.hls.levels : [];
  }

  setQuality(levelIndex) {
    if (this.hls) {
      this.hls.currentLevel = levelIndex;
    }
  }

  enableAutoQuality() {
    if (this.hls) {
      this.hls.currentLevel = -1; // -1 enables automatic quality selection
    }
  }

  handleFatalError(data) {
    switch (data.type) {
      case Hls.ErrorTypes.NETWORK_ERROR:
        console.log('Fatal network error, trying recovery...');
        this.hls.startLoad();
        break;
      case Hls.ErrorTypes.MEDIA_ERROR:
        console.log('Fatal media error, trying recovery...');
        this.hls.recoverMediaError();
        break;
      default:
        console.error('Fatal error, cannot recover', data);
        break;
    }
  }

  destroy() {
    if (this.hls) {
      // Remove all event listeners
      this.eventHandlers.forEach((handlers, eventType) => {
        handlers.forEach(handler => {
          this.hls.off(eventType, handler);
        });
      });
      
      this.hls.destroy();
      this.hls = null;
    }
    this.eventHandlers.clear();
  }
}
```

**Analytics Manager (COMPLETE):**

```javascript
// AnalyticsManager.js - Complete analytics tracking system
export class AnalyticsManager {
  constructor(options = {}) {
    this.events = [];
    this.sessionId = this.generateSessionId();
    this.startTime = Date.now();
    this.lastHeartbeat = Date.now();
    this.isVisible = !document.hidden;
    this.batchSize = options.batchSize || 10;
    this.flushInterval = options.flushInterval || 30000; // 30 seconds
    this.endpoint = options.endpoint || '/api/analytics';
    
    this.setupVisibilityTracking();
    this.startPeriodicFlush();
  }

  generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  }

  setupVisibilityTracking() {
    document.addEventListener('visibilitychange', () => {
      this.isVisible = !document.hidden;
      this.trackEvent('visibility_change', {
        visible: this.isVisible,
        timestamp: Date.now()
      });
    });
  }

  trackEvent(eventType, data = {}) {
    const event = {
      sessionId: this.sessionId,
      eventType,
      timestamp: Date.now(),
      sessionDuration: Date.now() - this.startTime,
      isVisible: this.isVisible,
      ...data
    };

    this.events.push(event);
    console.log('Analytics event:', event);

    if (this.events.length >= this.batchSize) {
      this.flush();
    }
  }

  trackPlaybackStart(videoId, manifestUrl) {
    this.trackEvent('playback_start', {
      videoId,
      manifestUrl,
      userAgent: navigator.userAgent,
      screenSize: `${screen.width}x${screen.height}`
    });
  }

  trackQualitySwitch(fromQuality, toQuality, trigger) {
    this.trackEvent('quality_switch', {
      fromQuality,
      toQuality,
      trigger, // 'automatic' or 'manual'
      networkSpeed: this.estimateNetworkSpeed()
    });
  }

  trackBufferEvent(eventType, bufferLength, position) {
    this.trackEvent('buffer_event', {
      bufferEventType: eventType, // 'starvation', 'recovery', 'healthy'
      bufferLength,
      playbackPosition: position
    });
  }

  trackViewingHeartbeat(position, duration, currentQuality) {
    // Only track if user is actively viewing
    if (this.isVisible && Date.now() - this.lastHeartbeat > 10000) {
      this.trackEvent('viewing_heartbeat', {
        position,
        duration,
        currentQuality,
        playbackRate: this.calculatePlaybackRate()
      });
      this.lastHeartbeat = Date.now();
    }
  }

  estimateNetworkSpeed() {
    // Simple network speed estimation based on navigator connection API
    if (navigator.connection) {
      return {
        downlink: navigator.connection.downlink,
        effectiveType: navigator.connection.effectiveType,
        rtt: navigator.connection.rtt
      };
    }
    return null;
  }

  calculatePlaybackRate() {
    // Calculate actual playback rate vs real time
    const sessionDuration = Date.now() - this.startTime;
    return sessionDuration > 0 ? (this.lastHeartbeat - this.startTime) / sessionDuration : 1;
  }

  async flush() {
    if (this.events.length === 0) return;

    const eventsToSend = [...this.events];
    this.events = [];

    try {
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          events: eventsToSend,
          sessionId: this.sessionId
        })
      });

      if (!response.ok) {
        // Re-add events to queue if sending failed
        this.events.unshift(...eventsToSend);
        console.error('Failed to send analytics events');
      }
    } catch (error) {
      // Re-add events to queue if network failed
      this.events.unshift(...eventsToSend);
      console.error('Analytics transmission error:', error);
    }
  }

  startPeriodicFlush() {
    setInterval(() => {
      this.flush();
    }, this.flushInterval);
  }

  destroy() {
    this.flush(); // Final flush
    clearInterval(this.flushTimer);
  }
}
```

**Core Video Player Component (SKELETON):**

```javascript
// VideoPlayer.js - Main player component (implement the TODOs)
import { HlsWrapper } from '../utils/HlsWrapper.js';
import { AnalyticsManager } from './analytics/AnalyticsManager.js';

export class VideoPlayer {
  constructor(containerElement, options = {}) {
    this.container = containerElement;
    this.options = options;
    this.hlsWrapper = null;
    this.analytics = null;
    this.currentVideoId = null;
    this.qualities = [];
    this.isManualQuality = false;
    
    this.initializePlayer();
  }

  initializePlayer() {
    // TODO 1: Create HTML5 video element with proper attributes
    // Set crossorigin="anonymous", preload="metadata"
    // Add to container element
    
    // TODO 2: Initialize HLS wrapper with video element
    // Pass buffer configuration options
    // Store reference for later use
    
    // TODO 3: Initialize analytics manager
    // Configure batch size and flush interval
    // Set up analytics endpoint if provided
    
    // TODO 4: Create custom control elements
    // Build play/pause, progress bar, volume, quality selector
    // Add to container with proper CSS classes
    
    // TODO 5: Set up HLS event handlers
    // Listen for MANIFEST_PARSED, LEVEL_SWITCHED, ERROR events
    // Connect HLS events to UI updates and analytics
  }

  loadVideo(videoId, manifestUrl) {
    // TODO 1: Store video ID and reset player state
    // Clear previous qualities, reset manual selection flag
    
    // TODO 2: Load HLS source through wrapper
    // Call hlsWrapper.loadSource with manifest URL
    
    // TODO 3: Track playback start in analytics
    // Record video ID, manifest URL, user context
    
    // TODO 4: Update UI for loading state
    // Show loading indicator, disable controls until ready
  }

  setupHlsEventHandlers() {
    // TODO 1: Handle manifest parsed event
    // Extract available qualities from HLS levels
    // Populate quality selector dropdown
    // Enable player controls
    
    // TODO 2: Handle quality switch events
    // Update quality selector display
    // Track switch in analytics (automatic vs manual)
    // Update quality indicator in UI
    
    // TODO 3: Handle buffer events
    // Monitor FRAG_LOADED for buffer health
    // Track buffer starvation and recovery
    // Update buffer indicator in UI
    
    // TODO 4: Handle error events
    // Distinguish between recoverable and fatal errors
    // Show appropriate user messages
    // Track error events in analytics
  }

  setupCustomControls() {
    // TODO 1: Create play/pause button
    // Add click handler that calls video.play() or video.pause()
    // Update button icon based on video play state
    
    // TODO 2: Create progress bar
    // Add click handler for seeking (calculate position from mouse)
    // Update progress during playback with requestAnimationFrame
    // Handle dragging for smooth seeking
    
    // TODO 3: Create volume control
    // Add slider input that controls video.volume
    // Add mute button that toggles video.muted
    // Persist volume setting in localStorage
    
    // TODO 4: Create quality selector
    // Populate dropdown with available qualities from HLS
    // Add "Auto" option for automatic quality selection
    // Handle quality selection changes
  }

  handleQualitySelection(qualityIndex) {
    // TODO 1: Check if "Auto" selected (index -1)
    // If auto, call hlsWrapper.enableAutoQuality()
    // Set isManualQuality flag appropriately
    
    // TODO 2: Handle manual quality selection
    // Call hlsWrapper.setQuality(qualityIndex)
    // Set isManualQuality = true
    
    // TODO 3: Track quality switch in analytics
    // Record previous and new quality levels
    // Mark as manual trigger
    
    // TODO 4: Update quality selector UI
    // Update displayed current quality
    // Update auto/manual indicator
  }

  startAnalyticsHeartbeat() {
    // TODO 1: Set up interval timer (every 10 seconds)
    // Check if video is currently playing
    // Only send heartbeat if user is actively viewing
    
    // TODO 2: Gather current playback state
    // Get current position, duration, quality level
    // Calculate playback progress percentage
    
    // TODO 3: Send heartbeat to analytics
    // Call analytics.trackViewingHeartbeat with current state
    // Include buffer health and network conditions if available
  }

  destroy() {
    // TODO 1: Clean up HLS wrapper
    // Call hlsWrapper.destroy() to remove event listeners
    // Clear HLS instance reference
    
    // TODO 2: Clean up analytics
    // Flush any pending analytics events
    // Call analytics.destroy()
    
    // TODO 3: Remove DOM elements
    // Remove video element and custom controls
    // Clear container innerHTML
    
    // TODO 4: Clear timers and references
    // Clear analytics heartbeat interval
    // Set all object references to null
  }
}
```

**Language-Specific Hints:**

- Use `fetch()` API for analytics transmission with proper error handling and retry logic
- Leverage `localStorage` for persisting user preferences like volume and quality selection
- Use `requestAnimationFrame()` for smooth progress bar updates during playback
- Implement proper cleanup in `destroy()` methods to prevent memory leaks in single-page applications
- Use CSS custom properties (CSS variables) for theming and responsive design
- Consider using `IntersectionObserver` API for detecting when video enters/exits viewport
- Use `MediaQueryList` API for responsive control layout based on screen size

**Milestone Checkpoint:**

After implementing the video player integration:

1. **Test HLS Playback**: Load a test HLS manifest URL and verify video plays with quality adaptation
2. **Verify Custom Controls**: Test all custom controls (play/pause, seeking, volume, quality selection)
3. **Check Analytics**: Open browser console and verify analytics events are being logged correctly
4. **Test Browser Compatibility**: Verify player works in Chrome, Firefox, Safari, and Edge
5. **Validate Error Handling**: Test with invalid manifest URLs and network interruptions

Expected behavior: The player should load HLS content smoothly, provide responsive custom controls, automatically adapt quality based on network conditions, and track comprehensive analytics data.

Signs of issues:
- CORS errors in browser console indicate streaming service headers need configuration
- Memory usage climbing during quality switches suggests missing event listener cleanup
- Inconsistent behavior across browsers indicates need for better HLS.js configuration


## Interactions and Data Flow

> **Milestone(s):** All milestones (1-4) - Understanding the complete data flow from upload through transcoding to playback is essential for integrating all components and handling status propagation across the entire video processing pipeline.

Think of the video streaming platform as a manufacturing assembly line where raw materials (uploaded videos) flow through multiple processing stations (upload, transcoding, streaming) before becoming finished products (playable streams). Each station has specific responsibilities, quality checks, and handoff procedures. The critical challenge is coordinating these stations so they work together smoothly, with clear communication about status, progress, and any problems that arise. Unlike a physical assembly line where you can see the product moving between stations, our digital pipeline requires explicit status tracking and event propagation to coordinate the distributed components.

The video streaming platform operates as a **database-mediated workflow system** where components coordinate through shared database state rather than direct API calls. This design choice provides durability, observability, and fault tolerance at the cost of some latency. Each major workflow stage (upload completion, transcoding initiation, streaming activation) involves multiple database transactions that update entity states and trigger downstream processing.

![End-to-End Video Processing Flow](./diagrams/end-to-end-flow.svg)

The complete data flow follows this high-level pattern: client uploads video chunks → upload service assembles file and extracts metadata → transcoding service discovers new video and creates jobs → FFmpeg workers process jobs and generate HLS segments → streaming service serves manifests and segments → player client requests and plays adaptive streams. Each transition involves database updates, file system operations, and potentially external notifications.

### Upload to Transcoding Flow

The transition from completed upload to transcoding initiation represents the first major handoff in our processing pipeline. This flow must handle the **impedance mismatch** between the synchronous upload completion event and the asynchronous transcoding pipeline startup. The upload service operates in request-response cycles, while the transcoding pipeline operates on a background job queue pattern.

When the upload service receives the final chunk for a video file, it performs several critical operations atomically. First, it assembles all chunks into the complete video file by concatenating them in the correct sequence. The chunk assembly process validates that all expected chunks were received by checking the `chunks_received` bit array in the `UploadSession` record. If any chunks are missing, the upload service returns an error to the client and preserves the partial upload state for resumability.

Upon successful file assembly, the upload service initiates **metadata extraction** using FFprobe to analyze the video file properties. This extraction process determines the video duration, resolution, frame rate, codec, and bitrate - information essential for the transcoding quality ladder decisions. The metadata extraction happens synchronously during upload completion to ensure we have complete video information before triggering transcoding.

The upload completion workflow follows these steps:

1. Validate that all expected chunks have been received by checking the `chunks_received` bit array
2. Concatenate chunk files in sequence to assemble the complete video file
3. Calculate and verify the final file checksum against the expected hash
4. Extract video metadata using `Video.extractMetadata(filePath)` with FFprobe
5. Create the `Video` record in the database with status `UPLOADED` and extracted metadata
6. Delete temporary chunk files and the `UploadSession` record
7. Trigger transcoding pipeline discovery through database polling or event notification

> **Decision: Database Polling vs Event-Driven Transcoding Trigger**
> - **Context**: The transcoding service needs to know when new videos are ready for processing, but the upload service and transcoding service run as separate processes
> - **Options Considered**: Direct API calls, message queues, database polling, file system watchers
> - **Decision**: Database polling with exponential backoff
> - **Rationale**: Database polling provides fault tolerance (crashed transcoding workers automatically resume), simplicity (no additional infrastructure), and exactly-once processing (database transactions prevent duplicate jobs). Direct API calls create tight coupling and single points of failure. Message queues add operational complexity for this use case.
> - **Consequences**: Introduces 1-5 second latency between upload completion and transcoding start, but provides excellent reliability and operational simplicity

| Transcoding Trigger Option | Pros | Cons | Chosen? |
|---|---|---|---|
| Database Polling | Fault tolerant, simple, exactly-once processing | 1-5 second latency, periodic database load | ✅ Yes |
| Direct API Calls | Low latency, immediate processing | Tight coupling, failure propagation, retry complexity | ❌ No |
| Message Queue | Decoupled, scalable, persistent | Additional infrastructure, operational complexity | ❌ No |

The transcoding service runs a background discovery loop that periodically queries for `Video` records with `processing_status = 'UPLOADED'`. When it finds new videos, it evaluates each one against the **quality ladder** to determine which transcoding jobs to create. The quality ladder considers the source video resolution and bitrate to avoid upscaling (generating higher quality outputs than the source supports).

For a 1080p source video, the transcoding service typically creates three `TranscodingJob` records with different `quality_config` settings: 360p for mobile compatibility, 720p for standard viewing, and 1080p for high-quality displays. Each job specifies the target resolution, bitrate, codec parameters, and HLS segment duration. The jobs are created with status `PENDING` and added to the processing queue.

The database schema supports this workflow through careful transaction isolation. The upload service's final transaction atomically creates the `Video` record and deletes the `UploadSession`, ensuring that partially uploaded videos never trigger transcoding. The transcoding service's job creation transaction atomically updates the video status to `TRANSCODING` and creates all required `TranscodingJob` records, preventing duplicate job creation if multiple workers discover the same video simultaneously.

**State Management During Upload-to-Transcoding Transition:**

| Video Processing Status | Transcoding Jobs | Upload Session | Description |
|---|---|---|---|
| `UPLOADING` | None | Exists | Upload in progress, chunks being received |
| `UPLOADED` | None | Deleted | Upload complete, awaiting transcoding job creation |
| `TRANSCODING` | Created (PENDING) | Deleted | Transcoding jobs queued, workers can claim them |
| `TRANSCODING` | In Progress | Deleted | At least one job actively processing |

> The critical insight here is that the `Video` entity serves as the central coordination point between upload completion and transcoding initiation. Its `processing_status` field acts as a state machine that prevents race conditions and ensures exactly-once processing semantics.

### Common Upload-to-Transcoding Pitfalls

⚠️ **Pitfall: Creating Transcoding Jobs Before Upload Completion**
The upload service might attempt to optimize by creating transcoding jobs as soon as it has enough metadata, before all chunks are received. This creates a race condition where transcoding could start on an incomplete file. Always wait for complete file assembly and checksum verification before updating the video status to `UPLOADED`.

⚠️ **Pitfall: Duplicate Transcoding Job Creation**
Multiple transcoding workers discovering the same newly uploaded video simultaneously can create duplicate jobs. Use database transactions with proper isolation levels and update the video status atomically when creating jobs. Check that `processing_status = 'UPLOADED'` and update it to `TRANSCODING` in the same transaction as job creation.

⚠️ **Pitfall: Metadata Extraction Timeout Handling**
FFprobe can hang indefinitely on corrupted video files, blocking the upload completion. Always run metadata extraction with a timeout (30-60 seconds) and handle failures gracefully by marking the video as `FAILED` with appropriate error messages rather than leaving it in `UPLOADING` state permanently.

### Transcoding to Streaming Flow

The transition from completed transcoding jobs to available streaming content represents the second major workflow handoff. This flow must coordinate multiple parallel transcoding jobs (one per quality level) and make their outputs available through the streaming service only when sufficient quality levels are ready for adaptive streaming.

Think of this flow as a restaurant kitchen where multiple chefs prepare different components of a meal simultaneously. The meal isn't ready to serve until all essential components are completed, but some components (like appetizers) can be served independently. Similarly, our streaming service can begin serving lower quality renditions while higher quality versions are still processing, but it needs at least one complete rendition before streaming can begin.

Each `TranscodingJob` produces two types of output: HLS video segments (TS files) and a quality-specific manifest (M3U8 playlist). The segments contain the actual video data split into fixed-duration chunks (typically 6 seconds each), while the manifest lists all segments in playback order with timing information. The transcoding worker calls `TranscodingJob.markCompleted(db, outputPath, segmentCount)` when FFmpeg finishes processing and all output files are validated.

The streaming service activation process monitors transcoding job completion through database polling, similar to how the transcoding service monitors upload completion. When the first transcoding job for a video completes successfully, the streaming service generates the **master playlist** - a top-level M3U8 file that references all available quality variants. As additional transcoding jobs complete, the master playlist is updated to include the new quality options.

**Streaming Activation Workflow:**

1. Monitor for `TranscodingJob` records with status `COMPLETED` that don't have corresponding `HLSManifest` entries
2. For each completed job, validate that all expected output files exist and are readable
3. Generate the quality-specific **variant playlist** using `generateVariantPlaylist(transcodingJobId)`
4. Create or update the master playlist using `generateMasterPlaylist(videoId)` to include this quality level
5. Create `HLSManifest` record linking the video to its streaming endpoints
6. Update `Video.processing_status` to `AVAILABLE` when the first quality level is ready for streaming
7. Configure CDN-friendly caching headers for manifest and segment files

The master playlist generation process requires careful coordination of multiple quality variants. The `generateMasterPlaylist(videoId)` function queries all completed transcoding jobs for the video and constructs an M3U8 file with `#EXT-X-STREAM-INF` entries for each quality level. Each entry specifies the bandwidth requirements, resolution, and URL for the quality-specific variant playlist.

```
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
360p/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION=1280x720
720p/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
1080p/playlist.m3u8
```

The streaming service must handle **partial completion scenarios** where some transcoding jobs succeed while others fail. A video with completed 360p and 720p transcoding but failed 1080p transcoding should still be available for streaming with the working quality levels. The streaming service implements a configurable policy for minimum quality requirements before activation.

> **Decision: Progressive vs All-or-Nothing Streaming Activation**
> - **Context**: Transcoding jobs for different quality levels complete at different times, and some may fail while others succeed
> - **Options Considered**: Wait for all jobs to complete, activate as soon as any job completes, activate when majority of jobs complete
> - **Decision**: Progressive activation starting with first completed job
> - **Rationale**: Users can begin watching immediately with available quality, improving perceived performance. Failed high-quality jobs don't block streaming of working lower qualities. Additional qualities can be added to the master playlist as they complete.
> - **Consequences**: Master playlist URLs change as new qualities are added, requiring cache invalidation. Player clients must handle dynamic quality availability.

| Activation Strategy | User Experience | Implementation Complexity | Resource Efficiency | Chosen? |
|---|---|---|---|---|
| Progressive (First Job) | Best - immediate playback | Medium - dynamic playlists | Best - no waiting for failures | ✅ Yes |
| All-or-Nothing | Poor - long wait times | Low - static playlists | Poor - blocked by slowest job | ❌ No |
| Majority Complete | Medium - some waiting | High - complex thresholds | Medium - blocked by minority failures | ❌ No |

The streaming service coordinates with the CDN or caching layer to ensure proper cache invalidation when master playlists are updated. Each playlist update increments a version number in the URL path to create **content-addressed URLs** that prevent stale cached responses from serving incomplete quality lists to players.

**Transcoding Job to Streaming Readiness State Transitions:**

| Job States | Video Status | Streaming State | Available Actions |
|---|---|---|---|
| All PENDING | TRANSCODING | Not Available | Wait for job completion |
| First COMPLETED | AVAILABLE | Limited Streaming | Serve available qualities |
| Majority COMPLETED | AVAILABLE | Full Streaming | Serve all completed qualities |
| All COMPLETED | READY | Complete Streaming | Serve full quality ladder |
| Any FAILED | PARTIAL or READY | Degraded Streaming | Serve working qualities only |

### Client Streaming Flow

The client streaming flow represents the final stage where all our backend processing delivers value to end users. This flow involves the video player requesting manifests and segments through the HLS protocol, with adaptive bitrate switching based on network conditions and device capabilities.

Think of the client streaming flow as a conversation between an attentive waiter (the player) and a well-organized kitchen (the streaming service). The waiter first asks for the menu (master playlist) to see what dishes are available, then orders specific items (variant playlists) based on the customer's preferences, and finally receives individual courses (video segments) in sequence. If the customer's appetite changes or the kitchen gets busy, the waiter can switch to different portion sizes (quality levels) seamlessly.

The streaming flow begins when the player calls `loadVideo(videoId, manifestUrl)` with the master playlist URL. This URL follows the pattern `/videos/{videoId}/master.m3u8` and includes cache-busting parameters to ensure fresh content. The streaming service responds with the master playlist containing all available quality variants, along with CDN-friendly cache headers.

**HLS Player Request Sequence:**

1. **Master Playlist Request**: Player requests `/videos/{videoId}/master.m3u8` to discover available quality levels
2. **Initial Quality Selection**: Player chooses starting quality based on bandwidth estimation or default configuration
3. **Variant Playlist Request**: Player requests quality-specific playlist (e.g., `/videos/{videoId}/720p/playlist.m3u8`)
4. **Segment Request Initiation**: Player begins downloading video segments starting from the first TS file
5. **Playback Buffer Management**: Player maintains 10-30 seconds of buffered content ahead of playback position
6. **Adaptive Quality Switching**: Player monitors bandwidth and buffer health to switch quality levels
7. **Segment Prefetching**: Player requests upcoming segments based on current playback position and buffer status

The variant playlist provides the roadmap for segment downloading, listing each TS file with its duration and optional byte range information. Players typically start downloading segments 2-3 segments ahead of the current playback position to maintain smooth playback despite network variability.

```
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:6.0,
segment_000.ts
#EXTINF:6.0,
segment_001.ts
#EXTINF:6.0,
segment_002.ts
```

The streaming service handles segment requests through optimized static file serving with support for **HTTP range requests**. Range requests allow players to download partial segments for seeking or to resume interrupted downloads. The `handleRangeRequest(req, res, next)` middleware processes `Range: bytes=start-end` headers and responds with the appropriate byte range and `206 Partial Content` status code.

**Adaptive Bitrate Decision Making Process:**

The player's adaptive bitrate algorithm continuously evaluates network conditions and playback buffer status to select the optimal quality level. The decision process runs every few seconds and considers multiple factors:

| Factor | Weight | Measurement Method | Decision Impact |
|---|---|---|---|
| Available Bandwidth | High | Download time of recent segments | Primary quality selector |
| Buffer Health | High | Seconds of video buffered ahead | Prevents quality increases when low |
| Device Capabilities | Medium | Screen resolution, CPU usage | Caps maximum useful quality |
| User Preference | Medium | Manual quality selection override | Overrides automatic decisions |
| Historical Performance | Low | Average quality over session | Influences starting quality |

The player implements **conservative quality switching** to avoid thrashing between quality levels. Quality increases require sustained good bandwidth measurements, while quality decreases happen more aggressively when buffer health degrades. This asymmetric behavior provides stable playback experience.

> **Decision: Conservative vs Aggressive Quality Switching**
> - **Context**: Players must balance quick adaptation to network changes with stable quality experience
> - **Options Considered**: Immediate switching on every measurement, hysteresis with different up/down thresholds, machine learning-based prediction
> - **Decision**: Conservative switching with hysteresis bands
> - **Rationale**: Prevents quality thrashing that degrades user experience. Users prefer stable medium quality over constantly changing high/low quality. Simple threshold-based approach is predictable and debuggable.
> - **Consequences**: Slower adaptation to improved network conditions, but better overall stability and user satisfaction

When the player decides to switch quality levels, it implements **seamless switching** by completing the current segment download at the old quality, then requesting subsequent segments from the new quality variant playlist. The switch happens at segment boundaries to maintain synchronization, typically resulting in 6-second maximum switching delay.

**Client-Side Buffer Management Strategy:**

The player maintains multiple buffers for optimal streaming performance:

| Buffer Type | Target Size | Purpose | Refill Trigger |
|---|---|---|---|
| Playback Buffer | 10-30 seconds | Smooth continuous playback | Below 5 seconds |
| Quality Buffer | 2-3 segments | Enable quality switches | After each segment |
| Seek Buffer | 1-2 segments | Fast seeking response | On seek operations |
| Prefetch Buffer | 5-10 segments | Optimize bandwidth usage | During idle periods |

The streaming service supports these client requirements through carefully designed HTTP headers and caching policies. The `setCacheHeaders(req, res, next)` middleware applies different cache strategies for manifests (short TTL, must-revalidate) versus segments (long TTL, immutable) to balance freshness with CDN efficiency.

**Error Handling in Client Streaming Flow:**

Players must handle various failure modes gracefully:

| Failure Mode | Detection Method | Recovery Strategy | Fallback Behavior |
|---|---|---|---|
| Manifest 404 | HTTP status code | Retry with exponential backoff | Show "content unavailable" message |
| Segment 404 | HTTP status code | Skip segment, request next | Continue playback with gap |
| Network Timeout | Request timeout | Retry same segment up to 3 times | Switch to lower quality |
| Bandwidth Drop | Slow download speed | Switch to lower quality immediately | Pause if no quality works |
| Buffer Underrun | Playback catches up to buffer | Pause playback, refill buffer | Show buffering indicator |

### Common Streaming Flow Pitfalls

⚠️ **Pitfall: Master Playlist Caching Issues**
Overly aggressive caching of master playlists prevents players from discovering new quality levels as transcoding jobs complete. Use short TTL (30-60 seconds) and ETags for master playlists, while using long TTL (hours) for immutable video segments. Include version numbers in playlist URLs to enable cache busting when content changes.

⚠️ **Pitfall: CORS Configuration for Cross-Origin Players**
Players hosted on different domains than the streaming service will fail to load manifests and segments without proper CORS headers. Configure the streaming service to return appropriate `Access-Control-Allow-Origin`, `Access-Control-Allow-Methods`, and `Access-Control-Allow-Headers` for all HLS endpoints.

⚠️ **Pitfall: Segment URL Generation Inconsistencies**
Variant playlists must use consistent URL patterns for segments to enable proper CDN caching and client prefetching. Use absolute URLs or consistent relative URL patterns. Avoid including session-specific parameters that prevent cache sharing between clients.

### Implementation Guidance

This implementation guidance provides the infrastructure and patterns needed to orchestrate the complete video processing workflow from upload through streaming.

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|---|---|---|
| Workflow Coordination | Database polling with cron jobs | Event-driven with Redis Streams |
| State Management | PostgreSQL with row-level locking | Distributed state machine with etcd |
| Progress Tracking | Periodic database updates | WebSocket real-time notifications |
| Error Handling | Database error logs with retry counts | Dead letter queues with exponential backoff |
| File System Operations | Node.js fs/promises with atomic moves | S3 with multipart upload completion |

**B. Recommended File Structure**

```
video-streaming-platform/
├── src/
│   ├── services/
│   │   ├── upload/
│   │   │   ├── UploadService.js
│   │   │   ├── ChunkedUploadHandler.js
│   │   │   └── MetadataExtractor.js
│   │   ├── transcoding/
│   │   │   ├── TranscodingService.js
│   │   │   ├── JobQueue.js
│   │   │   ├── FFmpegWrapper.js
│   │   │   └── QualityLadder.js
│   │   ├── streaming/
│   │   │   ├── StreamingService.js
│   │   │   ├── ManifestGenerator.js
│   │   │   └── SegmentServer.js
│   │   └── shared/
│   │       ├── DatabaseService.js
│   │       ├── StorageService.js
│   │       └── ValidationService.js
│   ├── models/
│   │   ├── Video.js
│   │   ├── TranscodingJob.js
│   │   ├── HLSManifest.js
│   │   └── UploadSession.js
│   ├── workflows/
│   │   ├── UploadCompletionHandler.js     ← this section's focus
│   │   ├── TranscodingCoordinator.js      ← this section's focus
│   │   └── StreamingActivator.js          ← this section's focus
│   └── client/
│       ├── VideoPlayer.js
│       ├── HlsWrapper.js
│       └── AnalyticsManager.js
├── tests/
│   ├── integration/
│   │   ├── upload-to-transcoding.test.js
│   │   ├── transcoding-to-streaming.test.js
│   │   └── end-to-end-flow.test.js
│   └── unit/
├── config/
│   └── workflow-config.js
└── scripts/
    ├── start-workers.js
    └── cleanup-failed-jobs.js
```

**C. Infrastructure Starter Code**

**Workflow Coordinator Base Class:**
```javascript
const { DatabaseService } = require('../services/shared/DatabaseService');
const { EventEmitter } = require('events');

/**
 * Base class for workflow coordination between video processing stages.
 * Provides polling, state transitions, and error handling patterns.
 */
class WorkflowCoordinator extends EventEmitter {
    constructor(databaseService, pollingIntervalMs = 5000) {
        super();
        this.databaseService = databaseService;
        this.pollingIntervalMs = pollingIntervalMs;
        this.isRunning = false;
        this.activeWorkers = new Map();
    }

    /**
     * Start the workflow coordinator with exponential backoff on errors.
     */
    async start() {
        this.isRunning = true;
        let backoffMs = 1000;
        const maxBackoffMs = 60000;

        while (this.isRunning) {
            try {
                await this.processPendingItems();
                backoffMs = 1000; // Reset backoff on success
                await this.sleep(this.pollingIntervalMs);
            } catch (error) {
                console.error(`Workflow coordinator error:`, error);
                this.emit('error', error);
                
                await this.sleep(backoffMs);
                backoffMs = Math.min(backoffMs * 2, maxBackoffMs);
            }
        }
    }

    /**
     * Stop the coordinator and wait for active workers to complete.
     */
    async stop() {
        this.isRunning = false;
        
        // Wait for active workers with timeout
        const timeout = 30000; // 30 seconds
        const start = Date.now();
        
        while (this.activeWorkers.size > 0 && Date.now() - start < timeout) {
            await this.sleep(1000);
        }
        
        if (this.activeWorkers.size > 0) {
            console.warn(`Coordinator stopped with ${this.activeWorkers.size} active workers`);
        }
    }

    /**
     * Process pending items - implemented by subclasses.
     * Should be idempotent and handle partial failures gracefully.
     */
    async processPendingItems() {
        throw new Error('Subclasses must implement processPendingItems');
    }

    /**
     * Track active work to enable graceful shutdown.
     */
    trackWorker(workerId, promise) {
        this.activeWorkers.set(workerId, promise);
        
        promise.finally(() => {
            this.activeWorkers.delete(workerId);
        });
        
        return promise;
    }

    async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = { WorkflowCoordinator };
```

**Database Transaction Helper:**
```javascript
/**
 * Database transaction utilities for workflow coordination.
 * Provides atomic state transitions with proper error handling.
 */
class TransactionHelper {
    constructor(databaseService) {
        this.db = databaseService;
    }

    /**
     * Execute a function within a database transaction with retry logic.
     * Automatically rolls back on errors and provides isolation.
     */
    async withTransaction(operation, maxRetries = 3) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            const transaction = await this.db.beginTransaction();
            
            try {
                const result = await operation(transaction);
                await transaction.commit();
                return result;
            } catch (error) {
                await transaction.rollback();
                lastError = error;
                
                // Don't retry application logic errors, only infrastructure errors
                if (this.isRetriableError(error) && attempt < maxRetries) {
                    await this.sleep(Math.pow(2, attempt - 1) * 1000); // Exponential backoff
                    continue;
                }
                
                throw error;
            }
        }
        
        throw lastError;
    }

    /**
     * Check if an error is retriable (network, deadlock) vs application error.
     */
    isRetriableError(error) {
        const retriableCodes = ['ECONNRESET', 'ETIMEDOUT', 'DEADLOCK_DETECTED'];
        return retriableCodes.some(code => 
            error.code === code || error.message.includes(code)
        );
    }

    async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = { TransactionHelper };
```

**D. Core Logic Skeleton Code**

**Upload Completion Handler:**
```javascript
const { WorkflowCoordinator } = require('./WorkflowCoordinator');
const { TransactionHelper } = require('./TransactionHelper');

/**
 * Handles the transition from completed uploads to transcoding initiation.
 * Monitors for videos with UPLOADED status and creates transcoding jobs.
 */
class UploadCompletionHandler extends WorkflowCoordinator {
    constructor(databaseService, transcodingService, storageService) {
        super(databaseService);
        this.transcodingService = transcodingService;
        this.storageService = storageService;
        this.transactionHelper = new TransactionHelper(databaseService);
    }

    async processPendingItems() {
        // TODO 1: Query for Video records with processing_status = 'UPLOADED'
        // Use SELECT FOR UPDATE to prevent multiple workers claiming same video
        
        // TODO 2: For each uploaded video, validate the file exists and is readable
        // Call storageService.validateFile(video.file_path) to ensure integrity
        
        // TODO 3: Generate quality ladder based on source video metadata
        // Use QualityLadder.generateConfigs(video.metadata) to get target qualities
        
        // TODO 4: Create transcoding jobs within a database transaction
        // Call this.createTranscodingJobsForVideo(video, qualityConfigs) atomically
        
        // TODO 5: Update video status to TRANSCODING after successful job creation
        // This prevents duplicate job creation by other workers
        
        // TODO 6: Emit events for monitoring and debugging
        // this.emit('upload-to-transcoding', { videoId, jobCount, duration })
    }

    async createTranscodingJobsForVideo(video, qualityConfigs) {
        return this.transactionHelper.withTransaction(async (tx) => {
            // TODO 1: Verify video is still in UPLOADED status (prevent race conditions)
            // SELECT processing_status FROM videos WHERE id = ? FOR UPDATE
            
            // TODO 2: Create TranscodingJob record for each quality configuration
            // Include video_id, quality_config JSON, status='PENDING', created_at
            
            // TODO 3: Update Video.processing_status to 'TRANSCODING' atomically
            // This marks the video as claimed by the transcoding pipeline
            
            // TODO 4: Return array of created job IDs for tracking
            // Format: { videoId, jobIds: [uuid1, uuid2, uuid3], qualityCount: 3 }
        });
    }

    async validateVideoFile(video) {
        // TODO 1: Check that video file exists at the expected storage path
        // TODO 2: Verify file size matches the recorded metadata
        // TODO 3: Validate that metadata extraction completed successfully
        // TODO 4: Return validation result with any error details
    }
}

module.exports = { UploadCompletionHandler };
```

**Transcoding to Streaming Coordinator:**
```javascript
/**
 * Coordinates the transition from completed transcoding jobs to streaming availability.
 * Generates HLS manifests and activates streaming when sufficient qualities are ready.
 */
class TranscodingToStreamingCoordinator extends WorkflowCoordinator {
    constructor(databaseService, manifestGenerator, streamingService) {
        super(databaseService);
        this.manifestGenerator = manifestGenerator;
        this.streamingService = streamingService;
        this.transactionHelper = new TransactionHelper(databaseService);
    }

    async processPendingItems() {
        // TODO 1: Find TranscodingJob records with status='COMPLETED' that don't have HLSManifest entries
        // JOIN transcoding_jobs tj LEFT JOIN hls_manifests hm ON tj.id = hm.transcoding_job_id WHERE hm.id IS NULL
        
        // TODO 2: Group completed jobs by video_id to handle multiple qualities together
        // Use Map<videoId, TranscodingJob[]> to organize jobs for batch processing
        
        // TODO 3: For each video group, validate all transcoded outputs exist and are valid
        // Call this.validateTranscodingOutputs(jobs) to verify segments and playlists
        
        // TODO 4: Generate or update master playlist for the video
        // Call manifestGenerator.generateMasterPlaylist(videoId) with all completed qualities
        
        // TODO 5: Create HLSManifest records linking jobs to streaming endpoints
        // Include transcoding_job_id, video_id, quality, manifest_path, segment_count
        
        // TODO 6: Update Video.processing_status based on streaming readiness policy
        // First completed job -> 'AVAILABLE', all jobs done -> 'READY'
    }

    async validateTranscodingOutputs(jobs) {
        // TODO 1: For each job, verify the output directory contains expected files
        // TODO 2: Validate segment count matches the expected duration/segment_length ratio
        // TODO 3: Check that variant playlist is valid M3U8 format
        // TODO 4: Verify segment files are readable and have expected duration
        // TODO 5: Return validation results with details for failed jobs
    }

    async updateStreamingAvailability(videoId, completedJobs) {
        return this.transactionHelper.withTransaction(async (tx) => {
            // TODO 1: Count total transcoding jobs for this video
            // TODO 2: Count completed jobs that now have HLS manifests
            // TODO 3: Apply streaming activation policy (first job vs all jobs)
            // TODO 4: Update Video.processing_status accordingly
            // TODO 5: Generate master playlist with available qualities
            // TODO 6: Return activation status and available quality count
        });
    }
}

module.exports = { TranscodingToStreamingCoordinator };
```

**Client Streaming Flow Handler:**
```javascript
/**
 * Handles client requests for HLS manifests and video segments.
 * Implements adaptive streaming protocol with proper caching and error handling.
 */
class ClientStreamingHandler {
    constructor(databaseService, storageService, manifestGenerator) {
        this.databaseService = databaseService;
        this.storageService = storageService;
        this.manifestGenerator = manifestGenerator;
    }

    async handleMasterPlaylistRequest(req, res) {
        // TODO 1: Extract videoId from request parameters
        // TODO 2: Verify video exists and is available for streaming (status != 'FAILED')
        // TODO 3: Generate master playlist with all available quality variants
        // TODO 4: Set CDN-friendly cache headers (short TTL, ETags)
        // TODO 5: Return M3U8 content with proper MIME type
        // TODO 6: Log request for analytics and debugging
    }

    async handleVariantPlaylistRequest(req, res) {
        // TODO 1: Extract videoId and quality from request path
        // TODO 2: Find HLSManifest record for this video/quality combination
        // TODO 3: Read variant playlist file from storage
        // TODO 4: Set appropriate cache headers (longer TTL for variant playlists)
        // TODO 5: Return playlist content with segment URLs
        // TODO 6: Handle 404 if quality not available yet
    }

    async handleSegmentRequest(req, res) {
        // TODO 1: Parse segment path to extract video, quality, and segment filename
        // TODO 2: Validate segment exists and is readable
        // TODO 3: Handle HTTP Range requests for partial content delivery
        // TODO 4: Set long-term cache headers (segments are immutable)
        // TODO 5: Stream segment file to response with proper content-type
        // TODO 6: Handle errors gracefully (404 for missing segments)
    }

    setCacheHeaders(req, res, next) {
        // TODO 1: Determine content type (master playlist, variant playlist, segment)
        // TODO 2: Set Cache-Control headers based on content mutability
        // TODO 3: Add ETags for master playlists to enable conditional requests
        // TODO 4: Set CORS headers for cross-origin player support
        // TODO 5: Add Content-Length and Accept-Ranges headers for segments
    }

    handleRangeRequest(req, res, next) {
        // TODO 1: Parse Range header to extract byte range
        // TODO 2: Validate range against file size
        // TODO 3: Set 206 Partial Content status and Content-Range header
        // TODO 4: Stream the requested byte range
        // TODO 5: Handle malformed range requests with 416 Range Not Satisfiable
    }
}

module.exports = { ClientStreamingHandler };
```

**E. Language-Specific Hints**

**Node.js Workflow Coordination:**
- Use `setInterval` with `clearInterval` for polling loops, not recursive `setTimeout` to avoid stack buildup
- Implement graceful shutdown with `process.on('SIGTERM')` to complete active workflows before exit
- Use `Promise.allSettled` instead of `Promise.all` when processing multiple items to handle partial failures
- Leverage `fs.promises.access()` with `fs.constants.R_OK` to check file readability before processing

**Database Transaction Management:**
- Use connection pooling with proper timeout configuration to handle high workflow concurrency
- Implement row-level locking with `SELECT ... FOR UPDATE` to prevent race conditions between workers
- Use `SERIALIZABLE` isolation level for critical state transitions, `READ_COMMITTED` for reads
- Always use parameterized queries to prevent SQL injection in dynamic workflow queries

**Error Handling and Retry Logic:**
- Distinguish between retriable errors (network timeouts) and permanent errors (file corruption)
- Implement exponential backoff with jitter to prevent thundering herd when services recover
- Use circuit breaker pattern for external service calls (FFmpeg, storage APIs)
- Log structured workflow events with correlation IDs for debugging distributed processing

**F. Milestone Checkpoints**

**Upload-to-Transcoding Flow (Milestone 1 → 2):**
```bash
# Start upload completion handler
npm run start:upload-handler

# Simulate completed upload
curl -X POST http://localhost:3000/api/upload/complete \
  -H "Content-Type: application/json" \
  -d '{"sessionId": "test-session-123"}'

# Verify transcoding jobs created
npm run check:transcoding-jobs -- --video-id test-video-123

# Expected: 3 transcoding jobs (360p, 720p, 1080p) in PENDING status
```

**Transcoding-to-Streaming Flow (Milestone 2 → 3):**
```bash
# Mark transcoding job as completed (simulate)
npm run simulate:transcoding-complete -- --job-id test-job-123

# Start streaming coordinator
npm run start:streaming-coordinator

# Check master playlist generation
curl http://localhost:3000/videos/test-video-123/master.m3u8

# Expected: M3U8 file with available quality variants
```

**Client Streaming Flow (Milestone 3 → 4):**
```bash
# Start streaming service
npm run start:streaming-service

# Test master playlist request
curl -H "Accept: application/vnd.apple.mpegurl" \
  http://localhost:3000/videos/test-video-123/master.m3u8

# Test variant playlist request  
curl http://localhost:3000/videos/test-video-123/720p/playlist.m3u8

# Test segment request with range
curl -H "Range: bytes=0-1023" \
  http://localhost:3000/videos/test-video-123/720p/segment_000.ts

# Expected: Proper HTTP responses with correct content-types and headers
```

**G. Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---|---|---|---|
| Upload completes but transcoding never starts | Upload completion handler not running or failing | Check handler logs, query videos with status='UPLOADED' | Restart handler, check database connectivity |
| Transcoding jobs complete but streaming not available | Missing HLSManifest creation or master playlist generation | Query completed jobs without manifests | Run streaming coordinator manually |
| Player requests fail with CORS errors | Missing Access-Control headers in streaming service | Check browser network tab for CORS preflight failures | Add CORS middleware to all HLS endpoints |
| Master playlist shows no qualities | Transcoding jobs not marked as completed properly | Check job status and output file validation | Verify FFmpeg output validation logic |
| Segments return 404 during playback | Inconsistent URL generation between playlist and serving | Compare playlist URLs with actual file paths | Standardize URL generation patterns |
| Quality switching not working | Master playlist bandwidth values incorrect | Check #EXT-X-STREAM-INF bandwidth values | Recalculate based on actual transcoded bitrates |


## Error Handling and Edge Cases

> **Milestone(s):** All milestones (1-4) - Error handling is critical across video upload (M1), transcoding failures (M2), streaming delivery issues (M3), and player error recovery (M4).

Building a robust video streaming platform requires anticipating and gracefully handling failures at every stage of the pipeline. Think of error handling like building a suspension bridge - you need redundant safety systems at every critical joint because when one component fails under stress, the entire structure must remain stable. Unlike simple web applications where a failed request might just show an error page, video streaming involves long-running processes (uploads, transcoding), expensive computations (FFmpeg), and real-time delivery constraints (streaming) where failures can cascade across multiple components and affect user experience for minutes or hours.

The video streaming pipeline presents unique error handling challenges because it combines **stateful long-running operations** (chunked uploads spanning minutes, transcoding jobs running for hours) with **real-time delivery requirements** (streaming must start immediately when requested) and **expensive resource consumption** (FFmpeg processes consuming CPU/memory, storage growing rapidly). When a transcoding job fails after processing 80% of a 2-hour video, simply restarting from the beginning wastes computational resources and delays content availability. When a streaming client loses network connectivity during playback, the player must seamlessly reconnect and resume without the user noticing a disruption.

This section analyzes the three major failure categories that can occur in our video streaming platform: upload failures that prevent content from entering the system, transcoding failures that block content processing, and streaming failures that prevent content delivery. Each category requires different detection mechanisms, recovery strategies, and user communication approaches.

![Error Handling and Recovery Flows](./diagrams/error-handling-flows.svg)

### Upload Failure Scenarios

Upload failures represent the first potential failure point in our video pipeline. Think of chunked uploads like shipping a large piece of furniture that must be broken down, shipped in separate boxes, and reassembled at the destination. If any box gets lost, damaged, or arrives corrupted, the entire piece becomes unusable until the missing parts are replaced. Unlike furniture shipping, however, video uploads must handle dynamic network conditions, concurrent access patterns, and storage constraints while providing real-time progress feedback to users.

The `UploadService` must detect and recover from several distinct failure modes that can occur during the chunked upload process. Network interruptions can occur at any point during chunk transmission, leaving upload sessions in partially completed states. File validation can fail after significant data has already been uploaded, requiring cleanup of temporary storage. Storage systems can experience capacity limits, permission errors, or corruption that affects upload completion.

**Network Interruption Recovery**

Network interruptions during chunked uploads create the most common failure scenario users encounter. When a client loses connectivity while uploading chunk 15 of 50, the server must preserve the session state and allow resumption from the exact chunk boundary rather than forcing a complete restart. The `UploadSession` entity tracks which chunks have been successfully received using a bit array, enabling precise resumption.

| Network Failure Mode | Detection Mechanism | Recovery Strategy | User Experience Impact |
|----------------------|---------------------|------------------|------------------------|
| Client disconnect during chunk upload | TCP connection timeout or reset | Preserve session state, allow chunk retry | Upload progress bar shows "reconnecting" |
| Intermittent packet loss | HTTP request timeout or incomplete chunk | Retry individual chunk with exponential backoff | Progress temporarily pauses, then continues |
| Complete network outage | Multiple consecutive chunk timeouts | Pause upload session, resume when connectivity returns | Upload pauses with "offline" indicator |
| Server restart during upload | Active sessions lost from memory | Reload sessions from database on startup | Temporary "service unavailable" then automatic resume |

The challenge with network interruption recovery lies in **session state persistence**. Upload sessions exist in server memory for performance, but must be persisted to database storage for durability across server restarts. The `UploadService` implements a hybrid approach where active session metadata lives in memory while chunk completion status is immediately written to persistent storage.

> **Design Insight:** The key to robust upload resumption is treating chunk completion as write-ahead logging. Each successful chunk write creates a durable record before acknowledging success to the client, ensuring no chunk completion is ever lost even if the server crashes immediately afterward.

**File Validation Failures**

File validation failures occur after the client has invested significant time and bandwidth uploading content, making the user experience particularly sensitive. When validation detects an unsupported codec, excessive file size, or corrupted data, the system must provide clear feedback about what went wrong and whether the user can fix the issue.

| Validation Failure Type | Detection Point | Recovery Options | Cleanup Required |
|-------------------------|----------------|------------------|------------------|
| Unsupported file format | After first chunk upload | User must convert file format | Delete temporary chunks |
| File size exceeds limits | During chunk assembly | User must reduce file size or upgrade account | Delete all uploaded chunks |
| Corrupted video data | After complete upload during metadata extraction | User must re-upload from different source | Delete corrupted file and chunks |
| Missing or invalid metadata | After FFprobe analysis | Depends on metadata type - some recoverable | Partial cleanup based on error severity |

The `validateUploadFile` function implements a **staged validation approach** where basic checks (file extension, size estimates) occur early to provide rapid feedback, while expensive checks (codec validation, integrity verification) occur after upload completion to avoid wasting upload bandwidth on files that will ultimately be rejected.

```
Staged Validation Process:
1. Pre-upload validation: Check file extension against SUPPORTED_FORMATS
2. First chunk validation: Verify file signature matches declared MIME type
3. Progressive size validation: Check accumulated size against MAX_FILE_SIZE during upload
4. Post-upload validation: Run FFprobe to extract metadata and verify codec compatibility
5. Integrity validation: Verify file can be opened and basic properties extracted
```

**Storage System Failures**

Storage failures during upload can result in partially written files, inconsistent metadata, or complete data loss. The `StorageService` must detect these conditions and provide recovery mechanisms that preserve user data while maintaining system consistency.

| Storage Failure Mode | Symptoms | Detection Method | Recovery Strategy |
|----------------------|----------|------------------|------------------|
| Disk space exhausted | Write operations fail with ENOSPC | Monitor storage metrics, catch write errors | Reject new uploads, complete in-progress uploads |
| Permission errors | Write operations fail with EACCES | File system error codes | Check and repair file permissions |
| Storage corruption | Successful writes but unreadable files | Checksum validation on read | Re-upload affected chunks |
| Temporary file cleanup failure | Disk space gradually consumed | Periodic cleanup job monitoring | Background cleanup with age-based deletion |
| Network storage disconnection | I/O operations hang or timeout | Operation timeouts | Failover to backup storage or queue for retry |

The most critical storage failure scenario involves **partial file corruption** where chunks write successfully but become unreadable due to hardware issues or file system corruption. The `StorageService` implements content checksums for each chunk, enabling detection of corruption during subsequent read operations.

> **Architecture Decision: Checksum Strategy**
> - **Context**: Storage corruption can make uploaded content unreadable without obvious failure indicators
> - **Options Considered**: 
>   - No checksums (rely on file system integrity)
>   - SHA-256 checksums for complete files only
>   - CRC32 checksums for individual chunks
> - **Decision**: CRC32 checksums for individual chunks
> - **Rationale**: Individual chunk checksums enable granular corruption detection and selective re-upload of affected chunks rather than complete file re-upload. CRC32 provides sufficient corruption detection with minimal computational overhead.
> - **Consequences**: Slightly increased storage overhead (8 bytes per chunk) and computation during upload, but significantly improved recovery granularity and user experience.

### Transcoding Failure Scenarios

Transcoding failures represent the most complex error handling challenge in the video streaming platform because they involve long-running external processes (FFmpeg), substantial computational resources, and multiple output artifacts that must remain consistent. Think of video transcoding like an assembly line manufacturing different product variants from raw materials - when the line breaks down, you need to determine whether the problem is with the raw materials (source video), the machinery (FFmpeg configuration), the assembly instructions (transcoding parameters), or the factory environment (system resources).

![Transcoding Job State Machine](./diagrams/transcoding-states.svg)

The `TranscodingService` manages FFmpeg processes that can fail in numerous ways: crashes due to unsupported codecs, resource exhaustion from processing high-resolution content, configuration errors that produce invalid output, and system-level failures that interrupt processing. Unlike simple compute tasks, transcoding jobs represent significant invested computation time (potentially hours) and must support intelligent retry strategies that avoid wasting resources on repeatedly failing operations.

**FFmpeg Process Crashes**

FFmpeg crashes represent the most common transcoding failure mode, typically resulting from unsupported input formats, corrupted source data, or resource constraints. When an FFmpeg process terminates unexpectedly, the `TranscodingService` must determine whether the failure is recoverable through retry, requires different transcoding parameters, or indicates a fundamental problem with the source video.

| FFmpeg Crash Type | Exit Code Pattern | Diagnostic Approach | Recovery Strategy |
|-------------------|-------------------|---------------------|------------------|
| Unsupported codec | Exit code 1 with codec error message | Parse stderr for codec-specific errors | Mark job as failed, suggest format conversion |
| Corrupted input | Exit code 1 with stream error | Attempt partial transcode to identify corruption point | Retry with error tolerance flags or mark failed |
| Memory exhaustion | Exit code 137 (SIGKILL) or OOM messages | Check system memory usage and FFmpeg memory options | Retry with reduced memory settings or queue for later |
| Invalid parameters | Exit code 1 with parameter error | Validate transcoding configuration against input metadata | Regenerate configuration and retry |
| Interrupt/timeout | Exit code 130 (SIGINT) or 124 (timeout) | Check job duration against expected completion time | Retry with extended timeout or different quality settings |

The `spawnFFmpegProcess` function implements **comprehensive error detection** by monitoring both process exit codes and stderr output patterns. FFmpeg provides detailed error messages that can be parsed to determine the specific failure cause, enabling targeted recovery strategies rather than generic retry logic.

```
FFmpeg Error Detection Process:
1. Start FFmpeg process with stderr capture enabled
2. Monitor process output for progress updates and error messages
3. Set timeout based on estimated transcoding duration (3x expected time)
4. On process exit, check exit code against known error patterns
5. Parse stderr output for specific error indicators (codec, format, resource)
6. Determine retry eligibility based on error classification
7. Update TranscodingJob status with detailed failure information
```

**Resource Exhaustion Handling**

Video transcoding consumes substantial CPU, memory, and disk I/O resources, making resource exhaustion a common failure mode when processing high-resolution content or running multiple concurrent jobs. The `TranscodingService` must detect resource constraints and implement back-pressure mechanisms that prevent system overload while maintaining reasonable throughput.

| Resource Constraint | Detection Indicators | Mitigation Strategy | Performance Impact |
|---------------------|---------------------|---------------------|-------------------|
| CPU saturation | Load average > core count, FFmpeg progress slows | Reduce concurrent job count, lower process priority | Increased transcoding latency |
| Memory exhaustion | High memory usage, swap activity, OOM killer | Reduce FFmpeg memory buffers, queue large jobs | Some jobs deferred to low-usage periods |
| Disk I/O bottleneck | High I/O wait time, slow file writes | Batch segment writes, use faster storage tier | Slightly longer segment generation |
| Network bandwidth limits | Slow source file reads, upload failures | Cache source files locally, compress intermediate files | Delayed job start for remote sources |

The transcoding pipeline implements **adaptive resource management** where the `TranscodingService` monitors system resources and adjusts job scheduling dynamically. When resource utilization exceeds safe thresholds, new jobs are queued rather than started immediately, and running jobs may have their priority reduced to maintain system responsiveness.

> **Design Insight:** Resource exhaustion often manifests as gradual performance degradation rather than immediate failures. Monitoring trends in transcoding duration and memory usage provides early warning signals that prevent system overload before failures occur.

**Output Validation Failures**

Transcoding can complete successfully from FFmpeg's perspective while producing output that fails validation for streaming delivery. Invalid HLS manifests, corrupted video segments, or incorrect encoding parameters can make transcoded content unusable despite successful job completion. The `validateHLSOutput` function detects these issues before marking transcoding jobs as complete.

| Output Validation Failure | Detection Method | Root Cause Analysis | Recovery Action |
|---------------------------|------------------|---------------------|----------------|
| Invalid M3U8 manifest syntax | Parse manifest against HLS specification | Check FFmpeg HLS output options | Regenerate manifest with corrected parameters |
| Missing video segments | Check file existence for all manifest references | Verify FFmpeg segmentation completed | Re-run segmentation phase of transcoding |
| Segment duration mismatch | Parse segment durations from manifest | Review source video timing and FFmpeg settings | Regenerate segments with fixed timing |
| Corrupted segments | Attempt to read segment headers | Check disk space during transcoding | Re-transcode affected segments |
| Incorrect quality parameters | Validate bitrate, resolution against configuration | Review quality ladder generation | Regenerate with corrected quality configuration |

Output validation implements **multi-stage verification** where basic file system checks (file existence, size) occur first, followed by format validation (manifest syntax, segment headers), and finally content validation (playback testing, quality verification).

**Zombie Job Detection and Recovery**

Long-running transcoding jobs can become "zombie jobs" when worker processes crash, servers restart, or network connectivity is lost, leaving jobs in PROCESSING state indefinitely. The `TranscodingService` implements **zombie job detection** through periodic scanning and heartbeat mechanisms that identify stuck jobs and enable recovery.

| Zombie Job Scenario | Detection Criteria | Recovery Approach | Prevention Mechanism |
|---------------------|-------------------|-------------------|---------------------|
| Worker process crash | Job in PROCESSING state > maximum duration | Reset job to PENDING for retry | Worker heartbeat with job ownership |
| Server restart | Active jobs lost from worker memory | Scan database for orphaned jobs on startup | Persist worker state to database |
| Network partition | Worker cannot update job progress | Timeout-based job release | Distributed worker coordination |
| Resource deadlock | Job consuming resources but making no progress | Progress monitoring with stall detection | Resource limits and timeout enforcement |

⚠️ **Pitfall: Infinite Retry Loops**  
A common mistake is implementing transcoding retry logic without retry limits or backoff delays, causing jobs with fundamental issues (unsupported codecs, corrupted source files) to consume resources indefinitely. Always implement maximum retry counts and exponential backoff, and distinguish between transient failures (resource exhaustion) and permanent failures (unsupported content).

### Streaming Failure Scenarios

Streaming failures occur during content delivery when videos have been successfully uploaded and transcoded but clients cannot access the content for playback. Think of streaming like operating a television broadcast station - the content exists and is ready for transmission, but various technical issues can prevent viewers from receiving a clear signal. Unlike traditional broadcasting, however, HTTP-based streaming involves individual client connections, dynamic quality adaptation, and stateless request/response cycles that create unique failure modes.

The `StreamingService` must handle failures that occur during manifest delivery, segment serving, and client-server communication. These failures often manifest as playback interruptions, quality degradation, or complete inability to start video playback, directly impacting user experience in ways that are immediately visible.

**Missing or Corrupted Segment Files**

Video segments can become missing or corrupted between transcoding completion and client requests due to storage failures, accidental deletion, or file system corruption. When clients request missing segments, the streaming service must detect the issue and provide recovery mechanisms that maintain playback continuity.

| Segment Availability Issue | Client Symptoms | Detection Method | Recovery Strategy |
|----------------------------|-----------------|------------------|------------------|
| Segment file deleted | HTTP 404 errors, playback stalls | File system checks on segment requests | Regenerate missing segments from source |
| Segment corruption | HTTP 200 but unplayable content | Content-length mismatches, checksum validation | Re-transcode affected segments |
| Manifest-segment mismatch | Player requests non-existent segments | Validate manifest references during serving | Update manifest or regenerate missing segments |
| Storage system failure | Multiple segments unavailable | Storage health monitoring | Failover to backup storage or CDN |
| Partial segment writes | Truncated files with correct names | File size validation against expected durations | Complete segment regeneration |

The `handleSegmentRequest` function implements **defensive serving** where segment availability is verified before responding to client requests. When segments are missing, the streaming service can either regenerate them on-demand (for transient failures) or serve alternative quality levels to maintain playback continuity.

> **Architecture Decision: Segment Recovery Strategy**
> - **Context**: Missing video segments cause immediate playback failures that users notice within seconds
> - **Options Considered**:
>   - On-demand regeneration from source video
>   - Failover to lower quality segments
>   - Graceful degradation with gap indication
> - **Decision**: Failover to lower quality segments with background regeneration
> - **Rationale**: Maintains continuous playback experience while recovering missing content. Users prefer quality reduction over playback interruption.
> - **Consequences**: Requires cross-quality segment mapping and background job scheduling, but provides better user experience than alternatives.

**Manifest File Issues**

HLS manifest files (M3U8) coordinate client playback by providing segment references, timing information, and quality metadata. Corrupted, outdated, or incorrectly formatted manifests can prevent playback initialization or cause clients to request non-existent content.

| Manifest Problem Type | Impact on Playback | Diagnostic Approach | Resolution Method |
|-----------------------|-------------------|---------------------|------------------|
| Invalid M3U8 syntax | Player initialization fails | Parse manifest against HLS specification | Regenerate manifest from database metadata |
| Stale segment references | 404 errors during playback | Cross-reference manifest with actual files | Update manifest with current segment list |
| Incorrect segment durations | Seeking accuracy problems | Validate durations against actual segments | Recalculate and update manifest timing |
| Missing quality variants | No quality switching available | Check transcoding job completion status | Generate missing quality levels |
| Caching issues | Clients receive outdated manifests | Monitor cache headers and CDN behavior | Update cache invalidation strategy |

The `ManifestGenerator` implements **consistency validation** where generated manifests are verified against actual file system state before serving to clients. This prevents race conditions where manifests reference segments that haven't been fully written to storage.

**Cross-Origin and CORS Configuration Errors**

Video streaming often involves serving content from different domains than the player application, requiring proper Cross-Origin Resource Sharing (CORS) configuration. Incorrect CORS settings prevent players from accessing manifests or segments, causing cryptic JavaScript errors that can be difficult to diagnose.

| CORS Configuration Issue | Browser Error Symptoms | Technical Cause | Configuration Fix |
|--------------------------|------------------------|-----------------|------------------|
| Missing CORS headers | "Access to fetch blocked by CORS policy" | Server doesn't include Access-Control-Allow-Origin | Add CORS middleware to streaming endpoints |
| Restrictive origin policy | CORS errors from specific domains | Whitelist doesn't include player domain | Update allowed origins configuration |
| Missing preflight support | OPTIONS requests fail | Server rejects preflight requests | Handle OPTIONS method for all streaming endpoints |
| Credential policy mismatch | Authentication errors with CORS | Credentials policy conflicts with origin policy | Align credential and origin policies |
| Cache header conflicts | Intermittent CORS failures | CDN caching CORS headers incorrectly | Configure CDN to vary on Origin header |

⚠️ **Pitfall: CDN CORS Configuration**  
When using a Content Delivery Network (CDN) for video delivery, CORS headers must be configured at both the origin server and CDN level. A common mistake is configuring CORS correctly on the origin but forgetting to enable CORS header forwarding in the CDN, causing intermittent failures depending on whether requests hit cached or origin content.

**HTTP Range Request Handling**

Video players use HTTP range requests to seek within video content and implement adaptive streaming. Incorrect range request handling causes seeking failures, prevents quality switching, and can break progressive download functionality.

| Range Request Problem | Player Behavior | Server-Side Issue | Implementation Fix |
|-----------------------|-----------------|------------------|-------------------|
| Range requests ignored | Seeking doesn't work, downloads entire file | Server doesn't parse Range header | Implement partial content support |
| Incorrect Content-Range headers | Player shows wrong duration | Byte range calculations wrong | Fix range calculation logic |
| Missing Accept-Ranges header | Player doesn't attempt seeking | Server doesn't advertise range support | Add Accept-Ranges: bytes header |
| Range boundary errors | Partial downloads fail | Off-by-one errors in range handling | Validate range boundaries against file size |
| Concurrent range request issues | Multiple quality downloads conflict | Shared state between range handlers | Isolate range request processing |

The `handleSegmentRequest` function implements **RFC 7233 compliant range handling** with careful attention to byte boundary calculations and proper HTTP status codes (206 for partial content, 416 for invalid ranges).

```
Range Request Processing Algorithm:
1. Parse Range header to extract byte range specifications
2. Validate range boundaries against actual file size
3. For single ranges: stream requested bytes with 206 status
4. For multiple ranges: create multipart response with boundaries
5. Include Content-Range header with total file size
6. Set Accept-Ranges header to indicate range support
7. Handle edge cases: empty ranges, ranges beyond file end
```

### Implementation Guidance

This section provides practical implementation patterns for robust error handling across the video streaming platform components. The error handling infrastructure must balance comprehensive failure detection with maintainable code that doesn't obscure the primary business logic.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Tracking | Console logging with structured JSON | Sentry or DataDog APM with custom dashboards |
| Retry Logic | Exponential backoff with jitter | Bull queue with Redis for distributed retry |
| Health Monitoring | HTTP health endpoints | Prometheus metrics with Grafana alerts |
| Circuit Breaking | Simple counter-based logic | Hystrix-style circuit breaker library |
| Database Transactions | Basic try/catch with rollback | Saga pattern for distributed transactions |

**Recommended File Structure:**
```
src/
  services/
    upload/
      upload-service.js          ← Main upload logic
      upload-error-handler.js    ← Upload-specific error handling
      upload-validators.js       ← File validation logic
    transcoding/
      transcoding-service.js     ← Job management
      ffmpeg-wrapper.js         ← FFmpeg process handling
      transcoding-recovery.js    ← Failure recovery logic
    streaming/
      streaming-service.js       ← Content delivery
      manifest-generator.js      ← Playlist generation
      cors-middleware.js        ← CORS configuration
  utils/
    error-types.js              ← Custom error classes
    retry-helper.js             ← Reusable retry logic
    circuit-breaker.js          ← Circuit breaker implementation
    health-monitor.js           ← System health checking
```

**Core Error Handling Infrastructure:**

```javascript
// utils/error-types.js - Custom error classes for different failure modes
class VideoStreamingError extends Error {
  constructor(message, code, retryable = false, context = {}) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.retryable = retryable;
    this.context = context;
    this.timestamp = new Date().toISOString();
  }
}

class UploadError extends VideoStreamingError {
  constructor(message, code, retryable = true, context = {}) {
    super(message, code, retryable, context);
  }
}

class TranscodingError extends VideoStreamingError {
  constructor(message, code, retryable = false, context = {}) {
    super(message, code, retryable, context);
  }
}

class StreamingError extends VideoStreamingError {
  constructor(message, code, retryable = true, context = {}) {
    super(message, code, retryable, context);
  }
}

// utils/retry-helper.js - Exponential backoff retry implementation
class RetryHelper {
  static async withRetry(operation, options = {}) {
    const {
      maxRetries = 3,
      initialDelayMs = 1000,
      maxDelayMs = 30000,
      backoffMultiplier = 2,
      jitter = true
    } = options;

    let lastError;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await operation(attempt);
      } catch (error) {
        lastError = error;
        
        // Don't retry if error is not retryable or we've exhausted attempts
        if (!error.retryable || attempt === maxRetries) {
          throw error;
        }
        
        // Calculate delay with exponential backoff and jitter
        let delay = Math.min(initialDelayMs * Math.pow(backoffMultiplier, attempt), maxDelayMs);
        if (jitter) {
          delay += Math.random() * 0.1 * delay; // Add 10% jitter
        }
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError;
  }
}
```

**Upload Error Handling Skeleton:**

```javascript
// services/upload/upload-error-handler.js
class UploadErrorHandler {
  constructor(storageService, databaseService) {
    this.storageService = storageService;
    this.databaseService = databaseService;
  }

  async handleChunkUploadError(error, sessionId, chunkIndex) {
    // TODO 1: Log error with session and chunk context
    // TODO 2: Determine if error is retryable (network vs validation vs storage)
    // TODO 3: For retryable errors, preserve session state and return retry instructions
    // TODO 4: For non-retryable errors, cleanup partial uploads and return failure
    // TODO 5: Update session last_activity to prevent timeout during error handling
    // Hint: Use error.code to distinguish between ENOSPC, EACCES, and network errors
  }

  async handleValidationFailure(validationError, uploadedFile, sessionId) {
    // TODO 1: Classify validation failure type (format, size, corruption, metadata)
    // TODO 2: Provide user-friendly error message with specific guidance
    // TODO 3: Cleanup uploaded chunks and temporary files
    // TODO 4: Record validation failure metrics for monitoring
    // TODO 5: Return structured error response with retry guidance if applicable
    // Hint: Some validation failures (corrupted data) may indicate user should re-upload
  }

  async recoverOrphanedSessions() {
    // TODO 1: Query database for upload sessions with last_activity > session timeout
    // TODO 2: For each orphaned session, check if temporary files still exist
    // TODO 3: Remove orphaned temporary files to free disk space
    // TODO 4: Update session status to EXPIRED or remove from database
    // TODO 5: Log recovery actions for monitoring and debugging
    // Hint: Run this as a periodic background job to prevent storage leaks
  }
}
```

**Transcoding Error Handling Skeleton:**

```javascript
// services/transcoding/transcoding-recovery.js
class TranscodingRecoveryService {
  constructor(transcodingService, databaseService, storageService) {
    this.transcodingService = transcodingService;
    this.databaseService = databaseService;
    this.storageService = storageService;
  }

  async handleFFmpegFailure(job, exitCode, stderr) {
    // TODO 1: Parse stderr output to identify specific failure type
    // TODO 2: Classify failure as codec error, resource exhaustion, or corruption
    // TODO 3: For resource exhaustion, retry with reduced quality or delayed scheduling
    // TODO 4: For codec errors, mark job as permanently failed with specific error message
    // TODO 5: For corruption, attempt partial recovery or suggest re-upload
    // TODO 6: Update job status with detailed failure information
    // Hint: Use regex patterns to extract meaningful error details from FFmpeg stderr
  }

  async detectZombieJobs() {
    // TODO 1: Query for jobs in PROCESSING status longer than maximum expected duration
    // TODO 2: For each potential zombie, check if worker process is still running
    // TODO 3: If worker is dead, reset job to PENDING status for retry
    // TODO 4: If worker is alive but job is stalled, send termination signal
    // TODO 5: Clean up any partial output files from failed jobs
    // TODO 6: Log zombie detection events for monitoring
    // Hint: Maximum expected duration = source_duration * transcode_speed_factor * safety_margin
  }

  async validateTranscodingOutput(jobId, outputPath) {
    // TODO 1: Check that all expected segments and manifest files exist
    // TODO 2: Validate M3U8 manifest syntax against HLS specification
    // TODO 3: Verify segment durations match manifest declarations
    // TODO 4: Test that segments can be opened and basic headers read
    // TODO 5: Compare output quality parameters against job configuration
    // TODO 6: If validation fails, determine whether to retry or mark job failed
    // Hint: Use ffprobe to validate individual segments without full decode
  }
}
```

**Streaming Error Handling Skeleton:**

```javascript
// services/streaming/streaming-error-handler.js
class StreamingErrorHandler {
  constructor(storageService, manifestGenerator) {
    this.storageService = storageService;
    this.manifestGenerator = manifestGenerator;
  }

  async handleMissingSegment(videoId, segmentPath, qualityLevel) {
    // TODO 1: Verify that segment is actually missing vs temporarily unavailable
    // TODO 2: Check if segment exists in other quality levels for fallback
    // TODO 3: Attempt to regenerate missing segment from source if available
    // TODO 4: If regeneration fails, update manifest to skip missing segment
    // TODO 5: Log missing segment for monitoring and alerting
    // TODO 6: Return appropriate HTTP status and recovery instructions
    // Hint: 404 for permanently missing, 503 for temporarily unavailable
  }

  async validateManifestConsistency(videoId, manifestType) {
    // TODO 1: Load manifest from storage and parse M3U8 format
    // TODO 2: Verify all segment references point to existing files
    // TODO 3: Check segment durations and timing consistency
    // TODO 4: Validate quality variant references in master playlist
    // TODO 5: If inconsistencies found, regenerate manifest from database state
    // TODO 6: Update manifest with corrected references and timing
    // Hint: Use HLS specification validation rules for manifest format checking
  }

  setupCorsHeaders(req, res, next) {
    // TODO 1: Extract origin from request headers
    // TODO 2: Check origin against allowed domains list
    // TODO 3: Set appropriate Access-Control-Allow-Origin header
    // TODO 4: Add Access-Control-Allow-Methods for OPTIONS requests
    // TODO 5: Set Access-Control-Allow-Headers for custom headers
    // TODO 6: Handle preflight OPTIONS requests
    // Hint: Be careful with wildcard origins when credentials are involved
  }
}
```

**Milestone Checkpoints:**

**After Milestone 1 (Upload Error Handling):**
- Test chunked upload interruption: start large upload, kill client mid-transfer, verify session state preserved
- Test storage exhaustion: fill disk during upload, verify graceful failure and cleanup
- Test invalid file upload: upload unsupported format, verify validation error and cleanup
- Expected behavior: uploads fail gracefully with clear error messages, no storage leaks

**After Milestone 2 (Transcoding Error Handling):**
- Test FFmpeg crash: upload video with unsupported codec, verify job marked as failed with specific error
- Test resource exhaustion: run multiple concurrent transcoding jobs, verify graceful degradation
- Test zombie job detection: kill transcoding worker process, verify job is reset to pending
- Expected behavior: transcoding failures provide actionable error messages, system remains stable

**After Milestone 3 (Streaming Error Handling):**
- Test missing segment: delete random segment file, verify player fallback or regeneration
- Test CORS failure: serve video from different domain, verify CORS headers allow access
- Test manifest corruption: modify M3U8 file syntax, verify regeneration from database
- Expected behavior: streaming errors don't interrupt playback, manifests self-heal

**After Milestone 4 (Player Error Handling):**
- Test network interruption: disable network during playback, verify reconnection and resume
- Test quality switching failure: corrupt one quality level, verify fallback to other qualities
- Test manifest reload failure: serve stale manifest, verify player adaptation
- Expected behavior: player gracefully handles errors with minimal user impact


## Testing Strategy

> **Milestone(s):** All milestones (1-4) - Testing strategy applies to video upload (M1), transcoding pipeline (M2), adaptive streaming (M3), and player integration (M4) with comprehensive validation at each stage.

Testing a video streaming platform requires a multi-layered approach that validates both individual components and their complex interactions. Think of testing like quality control in a manufacturing pipeline — we need checks at every stage (unit tests), validation of how stages work together (integration tests), and verification that the final product works for customers (end-to-end tests). The video streaming pipeline introduces unique testing challenges: large file handling, long-running FFmpeg processes, timing-sensitive HLS delivery, and browser compatibility across different devices.

The testing strategy must account for the **impedance mismatch** between components — upload operations complete in seconds, transcoding jobs run for minutes, and streaming playback can last hours. Each component operates at different time scales with different failure modes, requiring specialized test approaches that can validate behavior across these temporal boundaries.

### Testing Architecture Overview

The video streaming platform testing architecture mirrors the production system but with controllable inputs and observable outputs. We establish three primary test environments: isolated component testing with mocked dependencies, integration testing with real FFmpeg processing using small test videos, and end-to-end testing with actual browser players consuming HLS streams.

| Test Layer | Scope | Duration | Purpose |
|------------|--------|----------|---------|
| Unit Tests | Individual functions/classes | < 100ms | Validate component logic, error handling |
| Integration Tests | Component interactions | 1-10 seconds | Validate workflows, data persistence |
| End-to-End Tests | Complete pipeline | 30-60 seconds | Validate user experience, browser compatibility |
| Performance Tests | System under load | 5-10 minutes | Validate scalability, resource usage |

The testing framework must handle **asynchronous workflows** where upload completion triggers transcoding, which eventually enables streaming. Traditional synchronous test patterns break down when dealing with background job processing and FFmpeg subprocess execution.

> **Decision: Test Video Asset Strategy**
> - **Context**: Testing requires video files, but large videos make tests slow and storage-intensive
> - **Options Considered**: Generate synthetic videos, use tiny real videos, mock all video operations
> - **Decision**: Use a set of small (< 1MB) real video files covering common formats and edge cases
> - **Rationale**: Real videos catch format-specific issues that synthetic data misses, while small files keep tests fast
> - **Consequences**: Requires maintaining test asset repository, but provides confidence in real-world compatibility

### Component-Level Testing Strategy

Each major component requires specialized testing approaches that account for their unique operational characteristics and failure modes.

**Upload Service Testing**

The `UploadService` testing focuses on chunked upload mechanics, resumability, and file validation. The primary challenge is simulating network interruptions and partial upload scenarios without requiring actual network failures.

| Test Category | Test Cases | Validation Focus |
|---------------|------------|------------------|
| Chunked Upload | Sequential chunks, out-of-order chunks, duplicate chunks | Upload session state management |
| Resumability | Interrupted uploads, expired sessions, corrupted temp files | Recovery and cleanup behavior |
| File Validation | Invalid formats, oversized files, corrupted headers | Security and constraint enforcement |
| Progress Tracking | Chunk acknowledgment, completion percentage, ETA calculation | Client progress reporting accuracy |

The upload testing strategy uses **controlled chunk simulation** where test cases can inject failures at specific chunk boundaries. This allows validation of edge cases like receiving chunk 5 before chunk 4, or handling a session that expires during upload.

**Transcoding Service Testing**

The `TranscodingService` presents unique testing challenges because it spawns external FFmpeg processes. Testing must validate both the job coordination logic and the actual video processing results.

| Test Category | Approach | Key Validations |
|---------------|----------|-----------------|
| Job Queue | Mock FFmpeg, focus on scheduling | Concurrent job limits, priority handling |
| FFmpeg Integration | Small test videos, real transcoding | Progress parsing, error detection |
| Quality Ladder | Various input resolutions | Appropriate quality selection |
| Output Validation | HLS manifest parsing | Segment count, duration accuracy |

The transcoding tests employ a **tiered mocking strategy**: pure unit tests mock FFmpeg entirely for speed, while integration tests use real FFmpeg with tiny video files to validate actual transcoding behavior.

```markdown
Test Video Assets:
- test-720p.mp4 (500KB, 5 seconds, H.264/AAC)
- test-480p-vp9.webm (300KB, 3 seconds, VP9/Opus)
- test-corrupted.mp4 (invalid headers)
- test-audio-only.mp3 (no video track)
- test-huge-metadata.mov (excessive metadata size)
```

**Streaming Service Testing**

The `StreamingService` testing validates HLS protocol compliance, HTTP header correctness, and CDN compatibility. The challenge is ensuring manifest files and segments work with real HLS players.

| Test Focus | Validation Method | Expected Behavior |
|------------|-------------------|-------------------|
| Manifest Generation | HLS specification compliance check | Valid M3U8 syntax, correct URIs |
| Segment Delivery | HTTP range request simulation | Proper content-type, cache headers |
| Adaptive Logic | Quality variant selection | Correct bitrate ordering |
| CORS Headers | Cross-origin request simulation | Appropriate access-control headers |

**Player Integration Testing**

The `VideoPlayer` testing must validate browser compatibility, HLS.js integration, and user interface behavior across different devices and network conditions.

| Test Category | Testing Approach | Key Metrics |
|---------------|------------------|-------------|
| HLS.js Integration | Browser automation with real streams | Playback initiation, quality switching |
| Analytics Tracking | Event capture and validation | Accurate metrics, no data loss |
| Error Handling | Network interruption simulation | Graceful degradation, recovery |
| Performance | Memory usage monitoring | No memory leaks on video change |

### Database Testing Patterns

The video streaming platform relies heavily on database state to coordinate between components. Database testing must validate both individual record operations and complex multi-table transactions.

| Database Operation | Test Pattern | Validation Focus |
|--------------------|--------------|------------------|
| Video Creation | Single transaction test | Metadata persistence, constraint enforcement |
| Transcoding Jobs | Multi-job creation test | Atomic batch operations, rollback behavior |
| Status Updates | Concurrent update test | Race condition handling, consistency |
| Job Claims | Worker coordination test | Exclusive claim semantics |

The database tests use **transaction isolation testing** to ensure that concurrent operations behave correctly under load. This is critical because multiple transcoding workers may attempt to claim jobs simultaneously.

### Error Injection Testing

Video streaming workflows involve many external dependencies — file systems, FFmpeg processes, network requests — that can fail in various ways. **Error injection testing** systematically introduces failures to validate error handling and recovery logic.

| Failure Type | Injection Method | Expected Recovery |
|--------------|------------------|-------------------|
| Disk Full | Mock storage service failure | Upload rejection, cleanup |
| FFmpeg Crash | Process termination signal | Job marked as failed, retry logic |
| Network Timeout | HTTP client timeout simulation | Graceful degradation |
| Database Deadlock | Concurrent transaction test | Automatic retry with backoff |

The error injection framework allows tests to specify failure points: "fail on the 3rd FFmpeg progress update" or "return disk full error after 50% upload completion." This enables testing of complex failure scenarios that are difficult to reproduce naturally.

### Performance and Load Testing

Video streaming platforms must handle concurrent uploads, multiple transcoding jobs, and high-bandwidth streaming delivery. **Performance testing** validates that the system maintains acceptable response times and resource usage under realistic load.

| Load Test Type | Scenario | Success Criteria |
|----------------|----------|------------------|
| Concurrent Uploads | 10 simultaneous 100MB uploads | < 5% failure rate, memory stable |
| Transcoding Queue | 20 videos queued for transcoding | Jobs complete within 2x expected time |
| Streaming Load | 100 concurrent HLS players | < 100ms manifest response time |
| Mixed Workload | Upload + transcode + stream simultaneously | No component starvation |

Performance tests use **resource monitoring** to track CPU usage, memory consumption, disk I/O, and network bandwidth throughout test execution. This helps identify resource bottlenecks and memory leaks before they impact production.

> The critical insight for video streaming performance testing is that different components have vastly different resource profiles — uploads are network-bound, transcoding is CPU-bound, and streaming is bandwidth-bound. Tests must validate that these different workload patterns don't interfere with each other.

### Milestone Checkpoints

Each development milestone requires specific verification steps to ensure the implementation meets functional and performance requirements before proceeding to the next stage.

**Milestone 1: Video Upload Checkpoint**

After implementing the upload service, learners should validate chunked upload functionality, resumability, and proper error handling.

| Verification Step | Test Method | Expected Result |
|-------------------|-------------|-----------------|
| Basic Upload | POST 10MB video file | 200 response, file persisted |
| Chunked Upload | Upload 50MB file in 2MB chunks | Progress tracking, successful completion |
| Resume Upload | Interrupt and resume upload | Continues from last chunk |
| Invalid File | Upload .txt file as video | 400 error, no file persistence |
| Oversized File | Upload 6GB file | 413 error, connection closed |

**Manual Verification Commands:**
```bash
# Test basic upload
curl -X POST -F "video=@test-video.mp4" http://localhost:3000/api/upload

# Test chunked upload with progress
./upload-client --chunk-size=2MB --file=large-video.mp4

# Verify file storage
ls -la uploads/ | grep video-uuid
```

**Milestone 2: Video Transcoding Checkpoint**

After implementing the transcoding pipeline, validation focuses on FFmpeg integration, job queue management, and HLS output quality.

| Verification Step | Test Method | Expected Result |
|-------------------|-------------|-----------------|
| Single Video Transcode | Submit video for transcoding | Multiple quality outputs generated |
| Progress Tracking | Monitor transcoding job status | Accurate progress percentage |
| Queue Management | Submit 5 videos simultaneously | Jobs processed with concurrency limits |
| Output Validation | Check generated HLS files | Valid M3U8 and TS segments |
| Error Handling | Submit corrupted video file | Job marked as failed with error details |

**HLS Output Validation:**
```bash
# Check master playlist
curl http://localhost:3000/hls/video-id/master.m3u8

# Validate playlist format
ffprobe -v error -show_format video-output/playlist.m3u8

# Test segment accessibility
curl -I http://localhost:3000/hls/video-id/720p/segment000.ts
```

**Milestone 3: Adaptive Streaming Checkpoint**

After implementing the streaming service, validation ensures proper HLS delivery, HTTP caching headers, and CDN compatibility.

| Verification Step | Test Method | Expected Result |
|-------------------|-------------|-----------------|
| Manifest Serving | Request master playlist | Valid M3U8 with quality variants |
| Segment Delivery | Request video segment | TS file with correct MIME type |
| Range Requests | HTTP range header request | Partial content support |
| Cache Headers | Inspect response headers | CDN-friendly cache-control headers |
| CORS Support | Cross-origin manifest request | Appropriate CORS headers |

**Browser Compatibility Testing:**
```bash
# Test in multiple browsers
npm run test:browsers -- --browsers=chrome,firefox,safari

# Validate HLS.js compatibility
node validate-hls-output.js --manifest=master.m3u8
```

**Milestone 4: Video Player Integration Checkpoint**

After implementing the player integration, validation covers HLS.js functionality, quality switching, and analytics tracking.

| Verification Step | Test Method | Expected Result |
|-------------------|-------------|-----------------|
| Video Playback | Load video in player | Smooth playback initiation |
| Quality Switching | Change quality during playback | Seamless transition without rebuffering |
| Seeking Support | Click on progress bar | Accurate seek to target position |
| Analytics Tracking | Monitor analytics events | Accurate view duration and quality metrics |
| Error Recovery | Simulate network interruption | Player recovers and continues playback |

**Player Integration Testing:**
```javascript
// Automated player testing
const playerTest = new PlayerTestSuite();
await playerTest.loadVideo('test-video-id');
await playerTest.verifyPlaybackStart();
await playerTest.testQualitySwitching(['720p', '480p']);
await playerTest.validateAnalyticsEvents();
```

### Integration Testing Approach

Integration testing validates the **complete upload-transcode-stream pipeline** by orchestrating workflows across multiple components. Unlike unit tests that isolate individual functions, integration tests verify that components coordinate correctly through database state changes, file system operations, and HTTP API calls.

**End-to-End Workflow Testing**

The primary integration test follows a complete video from upload through first playback, validating every major workflow transition and data transformation.

| Workflow Stage | Integration Test Focus | Success Criteria |
|----------------|------------------------|-------------------|
| Upload Completion | Video record creation, file persistence | Database record exists, file accessible |
| Transcoding Trigger | Job creation, worker pickup | Jobs created for each quality level |
| Transcoding Progress | Status updates, progress tracking | Accurate progress reporting |
| Output Generation | HLS file creation, manifest generation | Valid playlists and segments |
| Streaming Activation | Video status update, availability | Player can access HLS streams |

**Cross-Component State Validation**

Integration tests must validate that state changes in one component correctly trigger behavior in downstream components. This requires **database-mediated communication testing** that verifies the coordination mechanisms between services.

```markdown
Integration Test Workflow:

1. Upload Service creates Video record with UPLOADED status
2. WorkflowCoordinator detects new video, creates TranscodingJob records
3. TranscodingService claims jobs, updates status to PROCESSING
4. FFmpeg completes transcoding, jobs marked as COMPLETED
5. ManifestGenerator creates HLSManifest records
6. StreamingService serves manifests and segments
7. Player successfully initiates playback
```

The integration test framework uses **workflow coordination testing** that monitors database state changes and validates that each transition occurs within expected time bounds. This catches issues where components fail to detect state changes or process them incorrectly.

**Database Transaction Testing**

Video streaming workflows require **atomic operations** across multiple database tables. Integration tests validate that complex operations either complete entirely or roll back cleanly without leaving inconsistent state.

| Transaction Type | Test Scenario | Rollback Trigger |
|------------------|---------------|------------------|
| Video Upload Completion | Create video + metadata + storage record | Storage service failure |
| Transcoding Job Creation | Create jobs for all quality levels | Quality configuration error |
| Streaming Activation | Update video status + manifest records | Manifest validation failure |

The transaction testing uses **controlled failure injection** at different points in multi-step operations to ensure that partial failures don't corrupt the database state.

**File System Integration Testing**

The video streaming platform performs extensive file operations — storing uploads, creating transcoding outputs, serving HLS segments. Integration tests must validate that file operations coordinate correctly with database state.

| File Operation | Integration Challenge | Test Validation |
|----------------|----------------------|-----------------|
| Upload Storage | Atomic file + database update | File exists if database record exists |
| Transcoding Output | Multiple output files per job | All expected files created or none |
| Segment Serving | File accessibility through HTTP | URL routing matches file system structure |

**Performance Integration Testing**

Integration tests must validate performance characteristics across component boundaries, particularly how components handle load and resource contention.

| Performance Test | Load Pattern | Success Criteria |
|------------------|--------------|-------------------|
| Upload + Transcode | 5 concurrent uploads with immediate transcoding | No resource starvation |
| Queue Management | 20 videos submitted for transcoding | Fair job distribution |
| Streaming Under Load | 50 concurrent players | Consistent manifest response time |

**Error Propagation Testing**

Integration tests validate that errors in downstream components correctly propagate to upstream components and user interfaces. This ensures that system failures are properly communicated rather than causing silent failures or timeout scenarios.

| Error Source | Propagation Path | Expected Behavior |
|--------------|------------------|-------------------|
| FFmpeg Crash | TranscodingService → Database → UI | Job status shows failure with error details |
| Storage Full | StorageService → UploadService → Client | Upload rejected with clear error message |
| Manifest Corruption | StreamingService → Player | Player shows error, attempts recovery |

**Temporal Coordination Testing**

Video streaming workflows span different time scales — uploads complete in seconds, transcoding takes minutes, streaming sessions last hours. Integration tests must validate behavior across these **temporal boundaries**.

| Time-Based Test | Duration | Validation Focus |
|-----------------|----------|------------------|
| Upload Timeout | 30 seconds | Incomplete uploads properly cleaned up |
| Transcoding Timeout | 10 minutes | Long-running jobs don't block queue |
| Streaming Session | 1 hour | Player maintains connection, handles quality switches |

The integration testing framework includes **time acceleration capabilities** that can simulate longer time periods for testing timeout and cleanup behavior without requiring tests to run for actual hours.

⚠️ **Pitfall: Integration Test Data Isolation**
Many integration tests fail intermittently because they share database state or file system resources. Each integration test should create isolated test data (unique video IDs, separate upload directories) and clean up completely after execution. Consider using database transactions that roll back after each test or separate test database schemas.

⚠️ **Pitfall: Async Workflow Testing**
Integration tests often fail because they don't properly wait for asynchronous workflows to complete. When testing upload → transcoding → streaming workflows, use polling with timeouts rather than fixed delays. For example, poll the database every 100ms for up to 30 seconds waiting for transcoding jobs to reach COMPLETED status, rather than assuming all jobs complete in exactly 10 seconds.

### Implementation Guidance

The testing implementation provides comprehensive validation infrastructure that supports both development-time testing and production monitoring. The testing framework must handle the unique challenges of video processing: large files, long-running operations, and browser compatibility.

**Technology Recommendations**

| Testing Type | Simple Option | Advanced Option |
|--------------|---------------|-----------------|
| Unit Testing | Jest with file mocks | Jest + Docker containers |
| Integration Testing | Supertest for HTTP APIs | Testcontainers with real databases |
| Browser Testing | Puppeteer automation | Sauce Labs cross-browser grid |
| Load Testing | Artillery.io HTTP load | Custom WebSocket load simulator |
| Video Validation | FFprobe output parsing | Custom video analysis tools |

**Recommended Test File Structure**

```
project-root/
├── tests/
│   ├── unit/
│   │   ├── upload-service.test.js     ← Unit tests for upload logic
│   │   ├── transcoding-service.test.js ← Unit tests for job management
│   │   ├── streaming-service.test.js   ← Unit tests for HLS serving
│   │   └── video-player.test.js       ← Unit tests for player logic
│   ├── integration/
│   │   ├── upload-workflow.test.js     ← Upload → storage integration
│   │   ├── transcoding-pipeline.test.js ← Upload → transcode → HLS
│   │   ├── streaming-delivery.test.js  ← HLS manifest and segment serving
│   │   └── end-to-end.test.js         ← Complete video pipeline
│   ├── fixtures/
│   │   ├── test-videos/               ← Small test video files
│   │   ├── expected-outputs/          ← Expected transcoding results
│   │   └── mock-responses/            ← Canned API responses
│   └── helpers/
│       ├── test-database.js           ← Database setup/teardown
│       ├── mock-ffmpeg.js             ← FFmpeg process mocking
│       ├── video-validator.js         ← HLS output validation
│       └── browser-automation.js      ← Player testing utilities
```

**Test Database Infrastructure**

```javascript
// tests/helpers/test-database.js
const { DatabaseService } = require('../../src/services/database');
const { v4: uuidv4 } = require('uuid');

class TestDatabaseService extends DatabaseService {
    constructor() {
        super({
            host: process.env.TEST_DB_HOST || 'localhost',
            database: `test_video_streaming_${uuidv4()}`,
            port: process.env.TEST_DB_PORT || 5432
        });
        this.testSchema = `test_schema_${Date.now()}`;
    }

    async setupTestSchema() {
        // TODO: Create isolated test schema for this test run
        // TODO: Run all migration scripts to create tables
        // TODO: Insert test data fixtures if needed
        // TODO: Return cleanup function for test teardown
    }

    async cleanupTestData() {
        // TODO: Drop test schema and all data
        // TODO: Close database connections
        // TODO: Clean up any test files or external resources
    }

    async createTestVideo(overrides = {}) {
        const videoData = {
            id: uuidv4(),
            filename: 'test-video.mp4',
            original_size: 1024 * 1024,
            processing_status: 'UPLOADED',
            created_at: new Date(),
            ...overrides
        };
        return await Video.create(this.db, videoData);
    }
}

module.exports = { TestDatabaseService };
```

**FFmpeg Integration Testing**

```javascript
// tests/helpers/mock-ffmpeg.js
const EventEmitter = require('events');
const path = require('path');

class MockFFmpegProcess extends EventEmitter {
    constructor(inputPath, outputPath, quality) {
        super();
        this.inputPath = inputPath;
        this.outputPath = outputPath;
        this.quality = quality;
        this.pid = Math.floor(Math.random() * 10000);
        this.killed = false;
    }

    simulateTranscoding() {
        // TODO: Emit progress events at realistic intervals
        // TODO: Create expected output files (empty TS segments, M3U8 playlists)
        // TODO: Emit 'close' event with exit code 0 for success
        // Hint: Use setTimeout to simulate processing time based on input size
    }

    simulateFailure(errorMessage) {
        // TODO: Emit 'error' event with realistic FFmpeg error
        // TODO: Emit 'close' event with non-zero exit code
        // TODO: Clean up any partial output files
    }

    kill() {
        this.killed = true;
        this.emit('close', 1);
    }
}

// Mock the spawn function to return MockFFmpegProcess
function mockSpawnFFmpeg(enabled = true) {
    if (!enabled) return;
    
    const originalSpawn = require('child_process').spawn;
    require('child_process').spawn = function(command, args, options) {
        if (command === 'ffmpeg') {
            // TODO: Parse FFmpeg arguments to extract input/output paths
            // TODO: Return MockFFmpegProcess instead of real process
            // TODO: Support both success and failure scenarios based on test context
        }
        return originalSpawn(command, args, options);
    };
}

module.exports = { MockFFmpegProcess, mockSpawnFFmpeg };
```

**Video Output Validation**

```javascript
// tests/helpers/video-validator.js
const fs = require('fs').promises;
const path = require('path');

class VideoOutputValidator {
    constructor(outputDirectory) {
        this.outputDirectory = outputDirectory;
    }

    async validateHLSOutput(videoId, expectedQualities) {
        // TODO: Check that master.m3u8 exists and has valid syntax
        // TODO: Verify each quality variant has its own playlist
        // TODO: Validate that all referenced segments exist as files
        // TODO: Check segment duration consistency (should be ~6 seconds)
        // TODO: Ensure total playlist duration matches source video
        // Return validation results with specific failures
    }

    async validateTranscodingJob(jobId, expectedOutputs) {
        // TODO: Check that all expected output files were created
        // TODO: Validate file sizes are reasonable (not zero, not too large)
        // TODO: Use FFprobe to verify video/audio codec correctness
        // TODO: Check that resolution matches expected quality settings
        // Return detailed validation report
    }

    async parseM3U8Manifest(manifestPath) {
        // TODO: Read and parse M3U8 file format
        // TODO: Extract segment references, duration info, quality metadata
        // TODO: Validate HLS specification compliance
        // TODO: Return structured manifest data for test assertions
    }
}

module.exports = { VideoOutputValidator };
```

**Integration Test Framework**

```javascript
// tests/integration/upload-workflow.test.js
const request = require('supertest');
const { TestDatabaseService } = require('../helpers/test-database');
const { VideoOutputValidator } = require('../helpers/video-validator');
const { mockSpawnFFmpeg } = require('../helpers/mock-ffmpeg');

describe('Upload Workflow Integration', () => {
    let testDb;
    let app;
    let validator;

    beforeEach(async () => {
        // TODO: Initialize test database with clean schema
        // TODO: Start test server with test configuration
        // TODO: Set up mock FFmpeg if needed for this test
        // TODO: Prepare test video files and expected outputs
    });

    afterEach(async () => {
        // TODO: Clean up test database and files
        // TODO: Stop test server
        // TODO: Reset any global mocks
    });

    test('complete upload to transcoding workflow', async () => {
        const testVideoPath = path.join(__dirname, '../fixtures/test-videos/test-720p.mp4');
        
        // TODO 1: Upload video file via chunked upload API
        // TODO 2: Verify video record created in database with UPLOADED status
        // TODO 3: Wait for transcoding jobs to be created automatically
        // TODO 4: Simulate transcoding completion by updating job status
        // TODO 5: Verify HLS manifest and segments are generated
        // TODO 6: Test that streaming endpoints return valid content
        
        // Use validator.validateHLSOutput() to check transcoding results
        // Use polling with timeout to wait for asynchronous operations
    });

    test('upload failure cleanup', async () => {
        // TODO: Simulate disk full error during upload
        // TODO: Verify no partial files remain in storage
        // TODO: Verify no video records remain in database
        // TODO: Verify client receives appropriate error response
    });
});
```

**Browser Player Testing**

```javascript
// tests/integration/player-integration.test.js
const puppeteer = require('puppeteer');
const { TestDatabaseService } = require('../helpers/test-database');

describe('Video Player Integration', () => {
    let browser;
    let page;
    let testDb;

    beforeAll(async () => {
        browser = await puppeteer.launch({ 
            headless: process.env.CI === 'true',
            args: ['--no-sandbox', '--disable-dev-shm-usage']
        });
    });

    beforeEach(async () => {
        page = await browser.newPage();
        // TODO: Set up test database with sample video
        // TODO: Navigate to test player page
        // TODO: Enable console logging for debugging
    });

    test('HLS.js player loads and plays video', async () => {
        const videoId = await testDb.createTestVideo({ 
            processing_status: 'COMPLETED' 
        });

        // TODO: Load player page with test video ID
        // TODO: Wait for HLS.js to initialize
        // TODO: Verify video element shows correct duration
        // TODO: Start playback and verify progress updates
        // TODO: Check that no JavaScript errors occurred
        
        // Example assertion pattern:
        // await page.waitForSelector('.video-player[data-status="playing"]');
        // const duration = await page.$eval('video', v => v.duration);
        // expect(duration).toBeGreaterThan(0);
    });

    test('quality switching during playback', async () => {
        // TODO: Start video playback
        // TODO: Wait for multiple quality levels to be detected
        // TODO: Programmatically switch to different quality
        // TODO: Verify smooth transition without rebuffering
        // TODO: Check analytics events are tracked correctly
    });
});
```

**Performance Test Implementation**

```javascript
// tests/performance/streaming-load.test.js
const { Worker } = require('worker_threads');
const { performance } = require('perf_hooks');

describe('Streaming Performance Tests', () => {
    test('concurrent HLS manifest requests', async () => {
        const concurrency = 50;
        const duration = 30000; // 30 seconds
        
        // TODO: Create worker threads to simulate concurrent players
        // TODO: Each worker requests manifest and segments repeatedly
        // TODO: Measure response times and error rates
        // TODO: Verify system maintains performance under load
        
        const results = await runConcurrentLoad({
            concurrency,
            duration,
            endpoint: '/hls/test-video-id/master.m3u8'
        });
        
        expect(results.averageResponseTime).toBeLessThan(100); // ms
        expect(results.errorRate).toBeLessThan(0.01); // < 1%
    });

    async function runConcurrentLoad({ concurrency, duration, endpoint }) {
        // TODO: Spawn worker threads for concurrent requests
        // TODO: Collect timing and error metrics from all workers
        // TODO: Calculate aggregate statistics
        // TODO: Return performance summary
    }
});
```

**Milestone Checkpoint Automation**

```javascript
// tests/milestones/milestone-checkpoints.test.js
const { execSync } = require('child_process');
const path = require('path');

describe('Milestone Checkpoints', () => {
    test('Milestone 1: Upload Service Checkpoint', async () => {
        // TODO: Start upload service
        // TODO: Test basic file upload via HTTP
        // TODO: Test chunked upload with progress tracking
        // TODO: Test resumable upload after interruption
        // TODO: Test file validation and error handling
        // TODO: Verify all files are stored correctly
        
        const checkpointResults = await validateUploadMilestone();
        expect(checkpointResults.basicUpload).toBe(true);
        expect(checkpointResults.chunkedUpload).toBe(true);
        expect(checkpointResults.resumableUpload).toBe(true);
        expect(checkpointResults.fileValidation).toBe(true);
    });

    test('Milestone 2: Transcoding Pipeline Checkpoint', async () => {
        // TODO: Upload test video and trigger transcoding
        // TODO: Monitor transcoding job progress
        // TODO: Validate HLS output files are created
        // TODO: Check FFmpeg integration works correctly
        // TODO: Test transcoding queue management
        
        const results = await validateTranscodingMilestone();
        expect(results.jobCreation).toBe(true);
        expect(results.ffmpegIntegration).toBe(true);
        expect(results.hlsOutput).toBe(true);
        expect(results.queueManagement).toBe(true);
    });

    async function validateUploadMilestone() {
        // TODO: Implement comprehensive upload validation
        // TODO: Return structured results for each capability
    }

    async function validateTranscodingMilestone() {
        // TODO: Implement comprehensive transcoding validation
        // TODO: Return structured results for each capability
    }
});
```

**Language-Specific Testing Hints**

For Node.js video streaming testing:
- Use `supertest` for HTTP API testing with file uploads
- Use `jest.setTimeout(30000)` for tests involving FFmpeg processing
- Use `fs.promises` for async file operations in test setup/teardown
- Use `child_process.spawn` mocking for FFmpeg integration tests
- Use `puppeteer` for browser-based player testing
- Use `worker_threads` for concurrent load testing

**Debugging Integration Test Failures**

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| Upload tests timeout | Large test files or slow disk I/O | Check test file sizes, monitor disk usage | Use smaller test files (< 1MB) |
| Transcoding tests fail intermittently | FFmpeg process conflicts or resource limits | Check process list, monitor CPU/memory | Reduce concurrent job limits in tests |
| Browser tests hang | Page load errors or JavaScript exceptions | Enable browser console logging | Fix CORS headers, check for JS errors |
| Integration tests corrupt each other | Shared database or file system state | Check test isolation, verify cleanup | Use unique test schemas and directories |
| Performance tests show high variance | Background system activity or resource contention | Run tests on dedicated systems, monitor resource usage | Use consistent test environment |


## Debugging Guide

> **Milestone(s):** All milestones (1-4) - Debugging techniques are essential across video upload (M1), transcoding failures (M2), streaming delivery issues (M3), and player integration problems (M4) with systematic diagnostic approaches for each component.

Building a video streaming platform involves complex interactions between upload handling, FFmpeg transcoding, HLS manifest generation, and browser-based playback. When issues arise, they often manifest as mysterious symptoms that span multiple components, making diagnosis challenging for developers new to video streaming systems.

Think of debugging a video streaming platform like diagnosing a multi-stage assembly line where each station depends on the previous one's output. A problem at the upload stage might not become visible until transcoding fails mysteriously, or a transcoding configuration error might only surface when the player attempts adaptive streaming. The key to effective debugging is understanding the expected data flow and having systematic techniques to isolate failures at each boundary.

This debugging guide provides concrete diagnostic techniques, common symptom patterns, and step-by-step resolution procedures for issues learners encounter when building the streaming platform. Rather than generic troubleshooting advice, we focus on the specific failure modes of chunked uploads, FFmpeg integration, HLS delivery, and browser player compatibility.

![Error Handling and Recovery Flows](./diagrams/error-handling-flows.svg)

### Upload Issues Debugging

The upload service handles large video files through chunked transfer, resumable uploads, and file validation. Upload failures often present as timeout errors, incomplete transfers, or storage inconsistencies that can be difficult to distinguish from network issues.

**Mental Model: Upload as Multi-Stage Pipeline**

Think of chunked upload debugging like investigating a conveyor belt system where items (chunks) arrive sequentially and must be assembled into a final product (complete video file). Each chunk carries metadata about its position and the total expected assembly. When the final product is malformed, you need to trace back through the assembly log to find which chunk was damaged, missing, or arrived out of order.

The `UploadService` maintains state for each active upload session through the `UploadSession` entity, which tracks chunk arrival patterns and validates file integrity at completion. Understanding this state model is crucial for diagnosing upload failures systematically.

#### Chunked Upload Diagnostic Techniques

Upload failures typically fall into three categories: **network interruption** (chunks stop arriving), **chunk corruption** (data integrity problems), and **session management** (state tracking errors). Each category requires different diagnostic approaches.

**Network Interruption Symptoms and Diagnosis**

| Symptom | Likely Cause | Diagnostic Steps | Resolution |
|---------|-------------|------------------|------------|
| Upload stalls at specific percentage | Network timeout during chunk transfer | Check `UploadSession.chunks_received` bit array for gaps | Resume from last successful chunk using session state |
| Client reports "connection reset" | Server process crash during chunk handling | Examine server logs for memory errors or uncaught exceptions | Implement chunk size reduction and memory cleanup |
| Upload appears successful but file is incomplete | Final assembly step failed silently | Validate `expected_size` matches actual assembled file size | Add explicit size validation before marking session complete |
| Resume attempts fail with "session not found" | Session expired between upload attempts | Check `UploadSession.expires_at` timestamp against current time | Extend session timeout or implement session refresh mechanism |

The most common network interruption pattern occurs when the client successfully uploads 80-90% of chunks but the final chunks timeout due to server resource exhaustion. This happens because the `handleChunkedUpload` method accumulates memory pressure as it processes more chunks, eventually triggering garbage collection pauses or out-of-memory conditions.

To diagnose this pattern, examine the `UploadSession.last_activity` timestamp progression:

1. Query the database for the stalled session using `UploadSession.session_id`
2. Check if `chunks_received` bit array shows a contiguous pattern that stops abruptly
3. Calculate the time gap between `last_activity` and the current timestamp
4. If the gap exceeds network timeout thresholds, this indicates server-side resource issues

**Chunk Corruption Detection and Recovery**

Chunk corruption manifests as validation failures during final file assembly, even when all expected chunks arrived successfully. The `validateUploadFile` method detects these issues through file format validation and metadata extraction.

> **Critical Insight**: Chunk corruption often occurs at chunk boundaries when the client miscalculates byte ranges or the server incorrectly assembles received chunks. The symptom appears as FFprobe metadata extraction failures during the final validation step.

Common chunk corruption patterns include:

- **Byte boundary misalignment**: Client calculates chunk boundaries that split video frame headers
- **Content-Length mismatch**: HTTP headers report different sizes than actual chunk content
- **Encoding issues**: Binary video data corrupted during multipart form processing

To diagnose chunk corruption:

1. Extract a small sample from the assembled file at suspected chunk boundaries
2. Use hexdump or similar tools to examine byte patterns around chunk transitions
3. Compare the assembled file's magic bytes against expected video format signatures
4. Validate that `Video.extractMetadata` can successfully parse the assembled file

**Session Management State Debugging**

The `UploadSession` entity maintains complex state that coordinates chunk arrival, tracks progress, and manages cleanup. Session state bugs often manifest as race conditions when multiple chunks arrive concurrently or when cleanup processes interfere with active uploads.

| State Issue | Detection Method | Common Cause | Fix Strategy |
|-------------|------------------|-------------|-------------|
| Duplicate chunk acceptance | Multiple chunks with same sequence number succeed | Race condition in chunk validation | Add atomic check-and-set for chunk marking |
| Session cleanup during active upload | Upload fails with "session expired" despite recent activity | Background cleanup process removes active sessions | Update `last_activity` timestamp atomically with chunk processing |
| Bit array corruption | `chunks_received` shows impossible patterns (gaps after completion) | Concurrent modification without proper locking | Implement session-level locking for state updates |
| Temporary file orphaning | Storage fills with abandoned partial uploads | Session cleanup fails to remove temp files | Add cleanup verification and retry logic |

#### Storage Integration Debugging

The upload service integrates with the `StorageService` to persist uploaded files and manage temporary storage during chunk assembly. Storage-related upload failures often present as disk space errors, permission issues, or file system corruption.

**Storage Path and Permission Validation**

Storage failures frequently occur due to incorrect file system permissions or storage path misconfiguration. The `StorageService.basePath` must be writable by the upload service process, and temporary file locations need sufficient space for the largest expected uploads.

To diagnose storage permission issues:

1. Verify that the upload service process has write permissions to `StorageService.basePath`
2. Check available disk space against `MAX_FILE_SIZE` limits
3. Test file creation and deletion in the configured storage directory
4. Validate that temporary file paths don't conflict with permanent storage locations

**Temporary File Management**

During chunked uploads, the `UploadService` creates temporary files to accumulate chunks before final assembly. Improper temporary file handling leads to storage exhaustion, file descriptor leaks, and assembly corruption.

> **Design Principle**: Temporary files should be created with unique names based on `UploadSession.session_id` to prevent conflicts between concurrent uploads. The final assembly process must atomically move temporary files to permanent storage to avoid partial write visibility.

Common temporary file issues:

- **File descriptor leaks**: Opening chunks for append without proper cleanup
- **Atomic move failures**: Cross-filesystem moves that fail partway through
- **Cleanup race conditions**: Session expiration cleanup interfering with active assembly
- **Storage exhaustion**: Temporary files consuming all available disk space

### Transcoding Issues Debugging

The transcoding pipeline integrates FFmpeg for video format conversion, manages background job queues, and generates HLS segments with manifest files. Transcoding failures often involve FFmpeg configuration errors, resource exhaustion, or job queue coordination problems that require systematic diagnosis across multiple system layers.

**Mental Model: Transcoding as Factory Production Line**

Think of transcoding debugging like diagnosing a factory production line where raw materials (uploaded videos) flow through multiple processing stations (FFmpeg workers) to produce finished goods (HLS segments). Each station has specific input requirements, processing capabilities, and quality control checks. When the production line fails, you must trace the failure back through the processing stages to identify whether the problem lies in raw materials (source video), processing configuration (FFmpeg parameters), worker capacity (system resources), or quality control (output validation).

The transcoding pipeline uses a **job queue pattern** where `TranscodingJob` entities represent work items that `FFmpegWrapper` processes asynchronously. Understanding this distributed processing model is essential for diagnosing coordination failures and resource contention issues.

#### FFmpeg Integration Debugging

FFmpeg integration failures manifest in several distinct patterns: **process startup failures** (FFmpeg cannot launch), **runtime crashes** (FFmpeg terminates unexpectedly), **configuration errors** (incorrect encoding parameters), and **output validation failures** (FFmpeg completes but produces unusable results).

**Process Startup and Configuration Diagnosis**

| Symptom | Root Cause | Diagnostic Command | Resolution |
|---------|-----------|-------------------|------------|
| "FFmpeg command not found" | FFmpeg not installed or not in PATH | `which ffmpeg` and `ffmpeg -version` | Install FFmpeg or update PATH environment |
| "Unsupported codec" error | FFmpeg built without required codec support | `ffmpeg -codecs \| grep h264` | Rebuild FFmpeg with codec support or use different codec |
| "Invalid argument" during startup | Incorrect FFmpeg parameter formatting | Run `spawnFFmpegProcess` command manually in shell | Fix parameter escaping and format validation |
| Process starts but immediately exits | Input file corruption or unsupported format | `ffmpeg -i input.mp4 -f null -` to test decoding | Validate input file with `Video.extractMetadata` |

The `spawnFFmpegProcess` method constructs FFmpeg command lines with complex parameter sets for quality ladders and HLS segmentation. Command construction errors often involve parameter escaping, codec selection, or filter chain syntax that only becomes apparent when FFmpeg attempts to parse the arguments.

To debug FFmpeg command construction:

1. Log the complete command line generated by `spawnFFmpegProcess` before execution
2. Execute the logged command manually in a terminal to isolate syntax errors
3. Validate that input file paths are accessible and output directories exist
4. Test simplified versions of the command to isolate problematic parameters

**Runtime Progress Monitoring and Crash Detection**

FFmpeg provides progress information through stderr output that the `parseFFmpegProgress` method parses to update `TranscodingJob.updateProgress`. Progress parsing failures can cause jobs to appear stalled even when FFmpeg is working correctly.

> **Common Pitfall**: FFmpeg progress output format varies between versions and can include warning messages that break parsing regular expressions. Robust progress parsing must handle unexpected output and provide fallback progress estimation.

Runtime crash patterns include:

- **Memory exhaustion**: FFmpeg process killed by OS due to excessive memory usage
- **Codec crashes**: Specific input content triggers crashes in codec libraries  
- **Filter chain errors**: Complex filter graphs fail during processing
- **I/O errors**: Output filesystem issues cause FFmpeg to abort

To diagnose FFmpeg runtime crashes:

1. Examine FFmpeg stderr logs for error messages before termination
2. Check system resources (memory, disk space) during transcoding execution
3. Test the same input with simplified FFmpeg commands to isolate failure points
4. Validate that output directories remain writable throughout processing

**Quality Ladder Configuration Validation**

The `QualityLadder` generates encoding parameters for multiple output renditions based on source video characteristics. Configuration errors lead to failed transcoding jobs or outputs that don't meet streaming requirements.

Quality ladder issues include:

- **Upscaling prevention**: Ensuring output resolution doesn't exceed source resolution
- **Bitrate allocation**: Matching bitrates to resolution and complexity appropriately  
- **Codec compatibility**: Using encoding settings supported by target playback devices
- **Segment duration consistency**: Maintaining uniform segment timing across quality levels

#### Job Queue Coordination Debugging

The transcoding pipeline uses background job processing where multiple worker processes claim jobs from a shared queue. Job queue coordination failures involve **job claiming conflicts** (multiple workers processing same job), **zombie job detection** (jobs stuck in processing state), and **job retry logic** (handling failed jobs appropriately).

**Job State Consistency Validation**

| State Issue | Symptoms | Diagnostic Query | Resolution Strategy |
|-------------|----------|------------------|-------------------|
| Jobs stuck in PROCESSING state | No progress updates for extended period | `SELECT * FROM transcoding_jobs WHERE status='PROCESSING' AND updated_at < NOW() - INTERVAL '1 hour'` | Implement zombie job detection and cleanup |
| Multiple workers claim same job | Duplicate output files or conflicting progress updates | Check for concurrent `TranscodingJob.claimNextJob` calls with same result | Add proper locking or atomic job claiming |
| Job retry loops | Same job repeatedly fails and retries without resolution | Count retry attempts in job history logs | Implement exponential backoff and max retry limits |
| Completed jobs not marked as done | FFmpeg succeeds but job remains in PROCESSING state | Validate `TranscodingJob.markCompleted` execution after FFmpeg exit | Add job state verification and recovery logic |

The `TranscodingJob.claimNextJob` method must atomically transition jobs from PENDING to PROCESSING state while recording the worker ID. Race conditions in job claiming lead to duplicate processing or lost jobs that appear to hang indefinitely.

**Worker Resource Management and Scaling**

Transcoding workers consume significant CPU and memory resources during FFmpeg processing. Resource contention between concurrent jobs can cause system-wide performance degradation or individual job failures.

> **Resource Management Principle**: The `MAX_CONCURRENT_JOBS` constant should account for both system hardware limitations and FFmpeg memory requirements. A conservative approach limits concurrent jobs to prevent resource exhaustion at the cost of reduced throughput.

Resource management debugging focuses on:

- **Memory pressure monitoring**: Tracking system memory usage during peak transcoding load
- **CPU utilization patterns**: Understanding FFmpeg CPU usage characteristics for different content types
- **Disk I/O bottlenecks**: Identifying storage bandwidth limitations during concurrent processing
- **Worker coordination**: Ensuring proper job distribution across available worker processes

#### HLS Output Validation Debugging

Transcoding jobs produce HLS manifest files and video segments that must conform to streaming protocol specifications. Output validation failures often don't become apparent until playback attempts fail, making diagnosis challenging.

**Manifest File Structure Validation**

The `ManifestGenerator` creates M3U8 playlist files that reference video segments with timing and quality metadata. Manifest validation ensures HLS specification compliance and player compatibility.

Common manifest issues:

- **Segment duration inconsistency**: Segments with timing that doesn't match manifest declarations
- **Missing segment files**: Manifest references segments that weren't created successfully  
- **Incorrect MIME types**: Segment files served with wrong Content-Type headers
- **Playlist caching problems**: Manifest updates not reflected due to aggressive caching

To validate HLS output systematically:

1. Parse manifest files using `parseM3U8Manifest` to verify structure and syntax
2. Check that all referenced segment files exist and are readable
3. Validate segment durations match the configured `SEGMENT_DURATION` constant
4. Test manifest and segment serving with appropriate HTTP headers for streaming

**Segment File Integrity Verification**

HLS segments are short video files that players download and concatenate for seamless playback. Segment integrity issues cause playback stuttering, quality switching failures, or complete playback failure.

Segment validation includes:

- **File format verification**: Ensuring segments use compatible video and audio codecs
- **Timing consistency**: Verifying that segment boundaries align properly for smooth concatenation
- **Quality rendition matching**: Confirming that segment quality matches manifest declarations
- **Byte range compatibility**: Testing that segments support HTTP range requests for seeking

### Streaming Issues Debugging

The streaming service delivers HLS manifests and video segments through HTTP with headers optimized for adaptive streaming and CDN compatibility. Streaming failures often involve **CORS configuration** (cross-origin access restrictions), **manifest serving problems** (playlist delivery issues), and **player compatibility** (browser-specific streaming limitations).

**Mental Model: Streaming as Content Distribution Network**

Think of streaming debugging like diagnosing a content distribution system where multiple types of resources (manifests, segments) must be delivered with precise timing and caching characteristics. The browser player acts as an intelligent client that makes adaptive decisions based on manifest content and network conditions. When streaming fails, the problem usually lies in the coordination between server-side resource delivery and client-side player expectations.

The streaming service implements the **HLS protocol** through careful HTTP header management, URL structure, and caching policies that influence both player behavior and CDN effectiveness. Understanding these protocol requirements is essential for diagnosing streaming delivery issues.

#### HLS Protocol Compliance Debugging

HLS streaming relies on strict protocol compliance for manifest structure, segment timing, and HTTP response characteristics. Protocol violations often cause subtle playback issues that vary between different player implementations and browser versions.

**Master Playlist Serving Validation**

The master playlist serves as the entry point for adaptive streaming, listing available quality variants and their characteristics. Master playlist issues prevent players from discovering available quality levels or making appropriate adaptation decisions.

| Issue Pattern | Player Symptom | Server-Side Diagnosis | Resolution |
|---------------|----------------|----------------------|------------|
| Invalid M3U8 syntax | Player fails to load video entirely | Validate playlist syntax with `validatePlaylistContent` | Fix manifest generation template syntax |
| Missing quality variants | Player only shows single quality option | Check that `TranscodingJob.findByVideoId` returns all expected qualities | Ensure all transcoding jobs complete before playlist generation |
| Incorrect bandwidth declarations | Player makes poor quality switching decisions | Compare manifest bitrate values with actual segment file sizes | Update `QualityLadder` bitrate calculations |
| Broken variant URLs | Specific qualities fail to load | Test individual variant playlist URLs manually | Fix URL generation in `generateMasterPlaylist` |

The `handleMasterPlaylistRequest` method must generate syntactically correct M3U8 content that accurately reflects available transcoding outputs. Playlist generation errors often involve template formatting, URL construction, or missing quality metadata.

**Variant Playlist and Segment Delivery**

Individual quality variant playlists list video segments for specific renditions. Variant playlist issues cause quality-specific playback failures or prevent adaptive streaming from switching between qualities smoothly.

Common variant playlist problems:

- **Segment URL path errors**: Playlist references incorrect segment file locations
- **Duration mismatch**: Playlist segment durations don't match actual file timing
- **Sequence number gaps**: Missing or incorrectly numbered segments break playback continuity
- **Content-Type headers**: Segments served with incorrect MIME types cause player rejection

To debug variant playlist delivery:

1. Request variant playlists directly and validate M3U8 syntax manually
2. Check that all referenced segment URLs return successful HTTP responses
3. Verify segment files have correct `Content-Type: video/MP2T` headers
4. Test segment timing by playing individual files in a media player

#### CORS and Cross-Origin Access Debugging

Modern web browsers enforce **Cross-Origin Resource Sharing (CORS)** policies that can block video streaming requests when the player page and streaming service operate on different domains. CORS failures manifest as cryptic browser console errors that don't clearly indicate streaming-specific issues.

**CORS Configuration Validation**

| Browser Error | Underlying Issue | Server Configuration Fix | Validation Method |
|---------------|------------------|-------------------------|-------------------|
| "Access to fetch blocked by CORS policy" | Missing `Access-Control-Allow-Origin` header | Add CORS headers to manifest and segment responses | Use browser dev tools Network tab to inspect response headers |
| "Preflight request failed" | Missing support for OPTIONS requests | Implement OPTIONS handler for streaming endpoints | Send OPTIONS request manually and verify response |
| "Credentials include flag mismatch" | Cookie/credential handling conflicts with CORS | Configure `Access-Control-Allow-Credentials` appropriately | Test streaming with and without authentication cookies |
| Video loads but seeking fails | Range request CORS headers missing | Add CORS headers to partial content responses | Test HTTP range requests with different Origin headers |

The streaming service must include appropriate CORS headers on all HLS-related responses, including master playlists, variant playlists, and video segments. Missing CORS headers on any response type can cause intermittent playback failures that are difficult to reproduce consistently.

**CDN and Caching Integration Issues**

Content Delivery Networks (CDNs) introduce additional complexity to CORS configuration because they must proxy and cache streaming responses while preserving CORS headers. CDN caching policies can also interfere with manifest updates and segment delivery timing.

> **CDN Debugging Principle**: When diagnosing CDN-related streaming issues, always test direct server access first to isolate whether the problem lies in origin server configuration or CDN behavior. CDN-specific problems often involve cached responses with incorrect headers or stale manifest content.

CDN-related debugging steps:

1. Compare direct server responses with CDN-proxied responses for header differences
2. Test manifest and segment caching behavior with different `Cache-Control` header values  
3. Verify that CDN respects `Vary` headers for adaptive streaming content negotiation
4. Monitor CDN cache hit rates and purge behavior for frequently updated manifests

#### Player Compatibility and Browser-Specific Issues

Different browsers implement HLS support with varying levels of native compatibility, requiring JavaScript libraries like HLS.js for consistent behavior. Player compatibility issues often involve **codec support variations**, **media API differences**, and **performance characteristics** that vary significantly between browser engines.

**Native HLS vs HLS.js Compatibility**

Safari browsers provide native HLS support through HTML5 video elements, while Chrome, Firefox, and Edge require JavaScript-based HLS implementation. This compatibility split creates different failure modes and debugging requirements.

Browser-specific debugging approaches:

- **Safari native HLS**: Monitor browser console for media element error events and HTTP request failures
- **HLS.js in Chrome/Firefox**: Examine HLS.js event logs and buffer management statistics  
- **Edge compatibility**: Test both native and JavaScript HLS modes depending on version
- **Mobile browser variations**: Account for different performance characteristics and codec support

**Memory Management and Performance Debugging**

Video streaming places significant memory pressure on browsers through buffer management, decoded video frames, and JavaScript processing overhead. Memory-related streaming issues cause browser tab crashes, playback stuttering, or gradual performance degradation.

Memory debugging focuses on:

- **Buffer size monitoring**: Tracking HLS.js buffer levels and garbage collection impact
- **Video element cleanup**: Ensuring proper cleanup when switching videos or unmounting components
- **Event listener management**: Preventing memory leaks from unremoved event handlers
- **Garbage collection patterns**: Understanding how video buffer management triggers GC pauses

> **Performance Insight**: Video streaming memory usage patterns differ significantly from typical web application memory management. Video buffers accumulate decoded frame data that can consume hundreds of megabytes, making aggressive buffer management and cleanup essential for stable long-term playback.

### Implementation Guidance

The debugging workflow for a video streaming platform requires systematic diagnostic tools, structured logging, and automated error detection mechanisms. This implementation guidance provides concrete debugging infrastructure and diagnostic procedures.

#### Technology Recommendations for Debugging Infrastructure

| Debugging Component | Simple Option | Advanced Option |
|-------------------|---------------|-----------------|
| Log Aggregation | Console output with structured JSON logging | ELK Stack (Elasticsearch, Logstash, Kibana) or similar |
| Error Tracking | File-based error logs with rotation | Sentry or Rollbar for error aggregation and alerting |
| Performance Monitoring | Basic metrics collection with periodic dumps | Prometheus + Grafana for real-time dashboards |
| FFmpeg Debugging | Stderr capture and parsing | FFmpeg with detailed logging levels and progress callbacks |
| HLS Validation | Manual playlist testing | Automated HLS validation tools and compliance testing |

#### Debugging Infrastructure File Structure

```
project-root/
├── src/
│   ├── debugging/
│   │   ├── logger.js              ← Structured logging configuration
│   │   ├── errorCollector.js      ← Error aggregation and reporting
│   │   ├── metricsCollector.js    ← Performance and health metrics
│   │   └── diagnostics.js         ← Automated diagnostic routines
│   ├── upload/
│   │   ├── uploadDiagnostics.js   ← Upload-specific debugging tools
│   │   └── chunkValidator.js      ← Chunk integrity validation
│   ├── transcoding/
│   │   ├── ffmpegDebugger.js      ← FFmpeg process monitoring and debugging
│   │   ├── jobQueueMonitor.js     ← Job queue health monitoring
│   │   └── outputValidator.js     ← HLS output validation tools
│   └── streaming/
│       ├── hlsValidator.js        ← HLS protocol compliance testing
│       ├── corsDebugger.js        ← CORS configuration validation
│       └── playerDiagnostics.js   ← Client-side debugging utilities
├── tools/
│   ├── healthcheck.js             ← Overall system health validation
│   ├── diagnosticRunner.js       ← Automated diagnostic test suite
│   └── debugDashboard.js          ← Real-time debugging interface
└── logs/
    ├── upload/                    ← Upload service logs by date
    ├── transcoding/               ← Transcoding logs organized by job
    ├── streaming/                 ← Streaming access logs and errors
    └── system/                    ← System-wide health and performance logs
```

#### Core Debugging Infrastructure Implementation

**Structured Logging Service**

```javascript
// src/debugging/logger.js
const winston = require('winston');
const path = require('path');

// Centralized logging configuration with component-specific formatting
class DebugLogger {
    constructor() {
        this.loggers = new Map();
        this.initializeLoggers();
    }

    initializeLoggers() {
        const components = ['upload', 'transcoding', 'streaming', 'player'];
        
        components.forEach(component => {
            const logger = winston.createLogger({
                level: 'debug',
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.errors({ stack: true }),
                    winston.format.json()
                ),
                defaultMeta: { component },
                transports: [
                    new winston.transports.File({
                        filename: path.join('logs', component, 'debug.log'),
                        level: 'debug'
                    }),
                    new winston.transports.File({
                        filename: path.join('logs', component, 'error.log'),
                        level: 'error'
                    }),
                    new winston.transports.Console({
                        format: winston.format.simple()
                    })
                ]
            });
            
            this.loggers.set(component, logger);
        });
    }

    // TODO: Implement component-specific logging with contextual metadata
    // TODO: Add log correlation IDs for tracing requests across components  
    // TODO: Implement log level filtering based on environment configuration
    // TODO: Add structured error reporting with stack trace preservation
    // TODO: Implement log rotation and cleanup policies
}

// Export singleton logger instance for consistent usage across components
module.exports = new DebugLogger();
```

**Upload Diagnostics Implementation**

```javascript
// src/upload/uploadDiagnostics.js
const fs = require('fs').promises;
const path = require('path');
const { UploadSession } = require('../models/uploadSession');

class UploadDiagnostics {
    constructor(databaseService, storageService) {
        this.db = databaseService;
        this.storage = storageService;
    }

    // Comprehensive upload session analysis for stuck or failed uploads
    async diagnoseUploadSession(sessionId) {
        // TODO: Query UploadSession record and validate state consistency
        // TODO: Check chunks_received bit array for gaps or inconsistencies
        // TODO: Validate temporary file exists and matches expected size
        // TODO: Test storage permissions and available disk space
        // TODO: Generate detailed diagnostic report with remediation suggestions
        
        const session = await UploadSession.findById(this.db, sessionId);
        if (!session) {
            return { error: 'Session not found', sessionId };
        }

        const diagnostics = {
            sessionId,
            status: 'analyzing',
            checks: {}
        };

        // Chunk consistency analysis
        diagnostics.checks.chunkConsistency = await this.validateChunkConsistency(session);
        
        // Storage validation
        diagnostics.checks.storageStatus = await this.validateStorageAccess(session);
        
        // Session timing analysis
        diagnostics.checks.timingAnalysis = this.analyzeSessionTiming(session);

        return diagnostics;
    }

    async validateChunkConsistency(session) {
        // TODO: Parse chunks_received bit array and identify missing chunks
        // TODO: Calculate expected vs actual chunk count based on file size
        // TODO: Check for duplicate or out-of-order chunk markers
        // TODO: Validate chunk size consistency across the upload
    }

    async validateStorageAccess(session) {
        // TODO: Test read/write permissions for session temp file location
        // TODO: Check available disk space against session requirements
        // TODO: Validate temp file integrity and size
        // TODO: Test atomic move operation to final storage location
    }

    analyzeSessionTiming(session) {
        // TODO: Calculate upload rate based on received chunks and timing
        // TODO: Identify potential timeout or stall periods
        // TODO: Compare session duration against expected completion time
        // TODO: Flag sessions that exceed reasonable completion timeframes
    }
}

module.exports = UploadDiagnostics;
```

**FFmpeg Process Debugging Tools**

```javascript
// src/transcoding/ffmpegDebugger.js
const { spawn } = require('child_process');
const { TranscodingJob } = require('../models/transcodingJob');

class FFmpegDebugger {
    constructor() {
        this.activeProcesses = new Map();
        this.processLogs = new Map();
    }

    // Enhanced FFmpeg process spawning with comprehensive logging and monitoring
    spawnFFmpegWithDebugging(inputPath, outputPath, qualityConfig, jobId) {
        // TODO: Construct FFmpeg command with debug-friendly parameters
        // TODO: Add detailed logging flags to FFmpeg command line
        // TODO: Set up stderr/stdout capture for progress and error monitoring
        // TODO: Implement process timeout and resource limit monitoring
        // TODO: Create process monitoring entry for health tracking

        const ffmpegArgs = this.buildDebugCommand(inputPath, outputPath, qualityConfig);
        
        const process = spawn('ffmpeg', ffmpegArgs, {
            stdio: ['ignore', 'pipe', 'pipe']
        });

        // Process monitoring setup
        this.activeProcesses.set(jobId, {
            process,
            startTime: Date.now(),
            inputPath,
            outputPath,
            qualityConfig,
            progressData: []
        });

        this.setupProcessMonitoring(jobId, process);
        
        return process;
    }

    buildDebugCommand(inputPath, outputPath, qualityConfig) {
        // TODO: Add verbose logging flags for detailed FFmpeg debugging
        // TODO: Include progress reporting with machine-parseable format
        // TODO: Add error detection and recovery options
        // TODO: Configure output validation and quality verification
        
        const baseArgs = [
            '-i', inputPath,
            '-loglevel', 'verbose',
            '-progress', 'pipe:2',
            '-nostdin'
        ];

        // Quality-specific encoding parameters
        const qualityArgs = [
            '-c:v', 'libx264',
            '-b:v', qualityConfig.bitrate,
            '-s', qualityConfig.resolution,
            '-c:a', 'aac',
            '-b:a', '128k'
        ];

        // HLS-specific parameters
        const hlsArgs = [
            '-f', 'hls',
            '-hls_time', '6',
            '-hls_list_size', '0',
            '-hls_segment_filename', `${outputPath}/segment_%03d.ts`,
            `${outputPath}/playlist.m3u8`
        ];

        return [...baseArgs, ...qualityArgs, ...hlsArgs];
    }

    setupProcessMonitoring(jobId, process) {
        // TODO: Parse stderr output for progress updates and errors
        // TODO: Monitor process memory and CPU usage
        // TODO: Implement timeout detection and cleanup
        // TODO: Log detailed error information for failed processes
        // TODO: Update TranscodingJob progress in database
    }

    // Diagnostic analysis for failed or stalled transcoding jobs
    async diagnoseTranscodingJob(jobId) {
        // TODO: Retrieve TranscodingJob record and analyze current state
        // TODO: Check FFmpeg process status and resource usage
        // TODO: Validate input file accessibility and format compatibility
        // TODO: Test output directory permissions and disk space
        // TODO: Analyze FFmpeg logs for specific error patterns
        // TODO: Generate remediation recommendations based on failure type
    }
}

module.exports = FFmpegDebugger;
```

**HLS Protocol Validation Tools**

```javascript
// src/streaming/hlsValidator.js
const fs = require('fs').promises;
const path = require('path');

class HLSValidator {
    constructor(storageService) {
        this.storage = storageService;
    }

    // Comprehensive HLS output validation for transcoding completion
    async validateHLSOutput(videoId, expectedQualities) {
        // TODO: Locate master playlist file for video
        // TODO: Parse master playlist and extract quality variant references
        // TODO: Validate each quality variant playlist exists and is well-formed
        // TODO: Check that all referenced segments exist and are accessible
        // TODO: Verify segment timing consistency and duration accuracy
        // TODO: Test HTTP serving with appropriate headers and CORS configuration

        const validationResults = {
            videoId,
            masterPlaylist: null,
            variants: [],
            segments: [],
            errors: [],
            warnings: []
        };

        try {
            // Master playlist validation
            validationResults.masterPlaylist = await this.validateMasterPlaylist(videoId);
            
            // Individual variant validation  
            for (const quality of expectedQualities) {
                const variantResult = await this.validateVariantPlaylist(videoId, quality);
                validationResults.variants.push(variantResult);
            }

        } catch (error) {
            validationResults.errors.push({
                type: 'validation_failure',
                message: error.message,
                stack: error.stack
            });
        }

        return validationResults;
    }

    async validateMasterPlaylist(videoId) {
        // TODO: Read master playlist file and parse M3U8 syntax
        // TODO: Validate playlist structure against HLS specification
        // TODO: Check that variant streams are properly declared
        // TODO: Verify bandwidth and resolution metadata accuracy
        // TODO: Test playlist serving with proper Content-Type headers
    }

    async validateVariantPlaylist(videoId, quality) {
        // TODO: Read quality-specific playlist file
        // TODO: Parse segment list and validate segment references
        // TODO: Check segment duration consistency with SEGMENT_DURATION
        // TODO: Verify all referenced segment files exist and are readable
        // TODO: Test segment serving with HTTP range request support
    }

    // Automated CORS configuration testing for streaming endpoints
    async validateCORSConfiguration(baseUrl) {
        // TODO: Test master playlist request with different Origin headers
        // TODO: Validate variant playlist CORS headers
        // TODO: Test segment requests with range headers and CORS
        // TODO: Check OPTIONS preflight request handling
        // TODO: Verify credentials and caching header compatibility
    }
}

module.exports = HLSValidator;
```

#### Milestone Debugging Checkpoints

**Milestone 1: Upload Service Debugging Verification**

After implementing upload functionality, verify debugging capabilities:

1. **Upload Session Diagnostics**: Create a test upload session and intentionally interrupt it midway. Run `UploadDiagnostics.diagnoseUploadSession()` to verify it correctly identifies missing chunks and provides recovery guidance.

2. **Chunk Corruption Detection**: Upload a video file with intentionally corrupted chunks. Verify that validation catches corruption and provides specific error messages about which chunks are problematic.

3. **Storage Permission Testing**: Configure incorrect storage permissions and verify that diagnostic tools detect and report permission issues with clear remediation steps.

Expected diagnostic output should include specific chunk numbers, byte ranges, and actionable error messages rather than generic failure notifications.

**Milestone 2: Transcoding Debugging Verification**

Verify transcoding debugging tools handle FFmpeg integration issues:

1. **FFmpeg Process Monitoring**: Start a transcoding job and monitor the FFmpeg process through `FFmpegDebugger`. Verify that progress updates, resource usage, and error detection work correctly.

2. **Job Queue Diagnostics**: Create multiple transcoding jobs and simulate worker crashes. Verify that zombie job detection identifies stuck jobs and provides recovery mechanisms.

3. **Output Validation Testing**: Complete transcoding jobs and run `validateHLSOutput()` to ensure all expected segments and manifests are generated correctly.

**Milestone 3: Streaming Debugging Verification**

Test streaming service diagnostics:

1. **HLS Protocol Validation**: Use `HLSValidator` to verify that generated manifests and segments conform to HLS specifications and serve correctly through HTTP.

2. **CORS Configuration Testing**: Test streaming from different origins and verify that CORS diagnostics correctly identify configuration issues and provide specific header recommendations.

3. **CDN Compatibility**: If using a CDN, verify that diagnostic tools can differentiate between origin server issues and CDN-specific problems.

#### Common Debugging Pitfalls

⚠️ **Pitfall: Incomplete Error Context**
Many debugging issues arise from insufficient error context that makes root cause analysis difficult. Always include relevant state information (session IDs, job IDs, file paths) in error messages and logs.

⚠️ **Pitfall: Ignoring Timing Issues**
Video streaming involves precise timing coordination between components. Debugging tools must account for temporal aspects like session timeouts, job processing duration, and player buffering behavior.

⚠️ **Pitfall: Browser-Specific Debugging**
Player compatibility issues often manifest differently across browsers. Maintain separate debugging strategies for native HLS support (Safari) versus JavaScript implementation (other browsers).

⚠️ **Pitfall: Resource Exhaustion Masquerading as Logic Errors**
Many apparent logic errors in video streaming are actually resource exhaustion problems (memory, disk space, file descriptors). Always check system resources as part of diagnostic procedures.


## Future Extensions

> **Milestone(s):** All milestones (1-4) - Future extensions build upon the complete video streaming platform, extending upload (M1), transcoding (M2), streaming (M3), and player (M4) capabilities with advanced features.

The video streaming platform provides a solid foundation for basic video-on-demand streaming, but modern video platforms require additional capabilities to remain competitive and meet evolving user expectations. Think of the current platform as a sturdy house with good bones - the foundation, framing, and basic systems are in place, but there's tremendous opportunity to add rooms, upgrade systems, and enhance the user experience. This section explores natural extensions that build upon the existing architecture while introducing new technical challenges and opportunities for learning advanced streaming concepts.

Each extension represents a significant engineering undertaking that touches multiple components of the existing system. Rather than being simple add-ons, these enhancements often require architectural evolution, new infrastructure components, and sophisticated coordination between existing services. The extensions are designed to be implemented incrementally, allowing developers to choose which capabilities align with their learning goals and business requirements.

### Live Streaming Extension

Live streaming transforms the platform from a video-on-demand service into a real-time broadcasting platform, introducing entirely new technical challenges around latency, real-time transcoding, and concurrent viewer management. Unlike the current upload-then-transcode workflow, live streaming requires processing video streams in real-time while they're being generated.

Think of live streaming as the difference between shipping a finished product (video-on-demand) versus manufacturing on an assembly line while customers watch through the factory windows (live broadcasting). Every decision about buffering, quality switching, and transcoding must be made with incomplete information about what's coming next, while maintaining the lowest possible latency to keep viewers engaged.

The live streaming extension introduces several new components and significantly modifies existing ones. A **Live Ingest Service** handles real-time video streams from broadcasters using protocols like RTMP or WebRTC. This service must buffer incoming video segments while simultaneously feeding them to a **Real-time Transcoding Pipeline** that generates multiple quality levels with minimal delay. The **Streaming Service** requires enhancement to serve live manifests that continuously update with new segments, and the **Video Player** needs adaptive logic for live content where seeking is limited and latency optimization becomes critical.

| Component | Live Streaming Role | New Responsibilities |
|-----------|-------------------|---------------------|
| Live Ingest Service | Accept real-time streams | RTMP/WebRTC endpoint, stream authentication, connection management |
| Real-time Transcoding | Process live video | Low-latency FFmpeg pipeline, segment generation, quality ladder adaptation |
| Enhanced Streaming Service | Serve live manifests | Dynamic playlist updates, segment availability windows, live edge management |
| Live-aware Player | Optimize live playback | Latency reduction, live seek restrictions, connection recovery |
| Broadcast Management | Coordinate live sessions | Stream lifecycle, viewer analytics, broadcaster controls |

**Architecture Decisions for Live Streaming:**

> **Decision: Low-Latency HLS vs WebRTC for Live Delivery**
> - **Context**: Live streaming requires minimizing latency between broadcaster and viewers while maintaining quality adaptation and scalability
> - **Options Considered**: Low-Latency HLS (LL-HLS), WebRTC with simulcast, DASH with low-latency extensions
> - **Decision**: Implement Low-Latency HLS with 2-second segment duration and partial segment delivery
> - **Rationale**: LL-HLS provides 3-6 second latency while reusing existing HLS infrastructure, WebRTC requires complete streaming service rewrite and lacks quality adaptation
> - **Consequences**: Enables sub-10-second latency streaming while leveraging existing manifest generation and player code, but requires enhanced segment generation pipeline

The live streaming data model introduces new entities to track active broadcasts and their real-time state:

| Entity | Fields | Purpose |
|--------|--------|---------|
| LiveStream | id UUID, broadcaster_id UUID, title string, status enum, start_time timestamp, viewer_count integer, ingest_url string, playback_url string | Track active live broadcasts |
| LiveSegment | id UUID, stream_id UUID, sequence_number integer, duration_ms integer, file_path string, created_at timestamp, quality string | Individual live video segments |
| BroadcastSession | id UUID, stream_id UUID, connection_id string, bitrate integer, resolution string, started_at timestamp, last_heartbeat timestamp | Active broadcaster connections |
| ViewerSession | id UUID, stream_id UUID, client_ip string, user_agent string, joined_at timestamp, last_ping timestamp, quality string | Track live viewers for analytics |

Real-time transcoding presents unique challenges compared to the existing batch transcoding pipeline. Instead of processing complete files, the system must transcode streaming video with strict latency requirements. The **FFmpegWrapper** requires enhancement to handle live input streams, managing continuous transcoding processes that run indefinitely until the broadcast ends.

Live streaming workflow coordination differs fundamentally from video-on-demand processing. The **WorkflowCoordinator** must manage active broadcast sessions, monitoring broadcaster connections for health and automatically cleaning up resources when streams end. Real-time analytics become critical for detecting stream quality issues and viewer experience problems as they occur.

**Common Live Streaming Pitfalls:**

⚠️ **Pitfall: Accumulating Transcoding Delay**
Live transcoding can gradually accumulate latency as the FFmpeg processes fall behind real-time input. Without careful monitoring, a stream that starts with 3-second latency can drift to 30+ seconds, making interactive features unusable. Implement transcoding process monitoring that detects when processes fall behind real-time and can restart them with frame skipping to catch up.

⚠️ **Pitfall: Manifest Caching with Live Content**
Standard HLS manifests are heavily cached for performance, but live manifests must update frequently to include new segments. Incorrectly cached live manifests cause players to miss new content or request segments that don't exist yet. Configure manifest caching with very short TTL values (1-2 seconds) and implement proper cache invalidation when new segments are available.

⚠️ **Pitfall: Connection Recovery Without Stream Position Tracking**
Live streams require careful coordination between ingest disconnections and transcoding pipeline cleanup. When a broadcaster's connection drops temporarily, the transcoding pipeline may continue processing buffered content, creating gaps when they reconnect. Implement stream position tracking that correlates ingest timestamps with transcoded output to detect and handle discontinuities.

### Thumbnail Generation Extension

Automatic thumbnail generation enhances user experience by providing visual previews of video content, but introduces new challenges around image processing, storage optimization, and timeline-based content extraction. Think of thumbnails as the storefront windows for video content - they need to be visually appealing, representative of the content, and generated efficiently at scale.

Thumbnail generation extends the existing transcoding pipeline with image extraction capabilities, requiring coordination between video analysis, image processing, and storage management. The system must determine optimal thumbnail extraction points, generate multiple thumbnail sizes for different display contexts, and integrate thumbnail URLs into the video metadata for player consumption.

The **Thumbnail Generation Service** integrates with the existing **TranscodingService**, adding image extraction jobs to the transcoding workflow. This service uses FFmpeg's image extraction capabilities to generate thumbnails at specific timestamps, applying intelligent algorithms to select visually interesting frames while avoiding scene transitions or low-quality frames.

| Thumbnail Type | Generation Strategy | Use Cases |
|----------------|-------------------|-----------|
| Poster Frame | Single representative image | Video catalog display, social media sharing |
| Timeline Thumbnails | Multiple images at regular intervals | Seek bar previews, chapter navigation |
| Adaptive Thumbnails | Quality-based frame selection | Avoid black frames, scene transitions, low-quality content |
| Custom Thumbnails | User-uploaded alternatives | Editorial control, branded content |

**Architecture Decisions for Thumbnail Generation:**

> **Decision: Pipeline Integration vs Separate Processing Service**
> - **Context**: Thumbnails can be generated during transcoding or as a separate post-processing step, each with different resource utilization patterns
> - **Options Considered**: Integrate with existing transcoding pipeline, separate thumbnail generation service, client-side thumbnail extraction
> - **Decision**: Integrate thumbnail generation into the transcoding pipeline with separate image optimization service
> - **Rationale**: Leverages existing video file access and FFmpeg processes while allowing specialized image processing and storage optimization
> - **Consequences**: Reduces file I/O by processing during transcoding, but increases transcoding job complexity and may slow video processing if image generation fails

The thumbnail extension introduces new data model entities to track generated images and their relationships to source videos:

| Entity | Fields | Purpose |
|--------|--------|---------|
| VideoThumbnail | id UUID, video_id UUID, thumbnail_type enum, timestamp_ms integer, file_path string, width integer, height integer, file_size integer, created_at timestamp | Track generated thumbnail images |
| ThumbnailSet | id UUID, video_id UUID, poster_thumbnail_id UUID, timeline_interval_ms integer, total_count integer, storage_path string | Group related thumbnails for efficient retrieval |
| CustomThumbnail | id UUID, video_id UUID, uploaded_by UUID, file_path string, is_active boolean, upload_timestamp timestamp | User-uploaded custom thumbnails |

Timeline thumbnail generation requires careful coordination with video content analysis to ensure thumbnail quality and relevance. The system must avoid extracting thumbnails from scene transitions, black frames, or other low-quality content that doesn't represent the video effectively.

**Enhanced TranscodingJob Configuration:**

| Field | Type | Purpose |
|-------|------|---------|
| generate_thumbnails | boolean | Enable thumbnail extraction for this job |
| thumbnail_interval_ms | integer | Timeline thumbnail spacing (e.g., every 10 seconds) |
| thumbnail_sizes | JSON array | Required thumbnail dimensions [(width, height)] |
| poster_frame_timestamp | integer | Specific timestamp for poster frame extraction |
| avoid_scene_transitions | boolean | Enable intelligent frame selection |

The **ImageOptimizationService** processes raw thumbnail extractions, applying compression, resizing, and format optimization to minimize storage costs while maintaining visual quality. This service operates asynchronously after thumbnail extraction, allowing transcoding jobs to complete quickly while optimizing images in the background.

**Common Thumbnail Generation Pitfalls:**

⚠️ **Pitfall: Extracting Thumbnails from Scene Transitions**
Naive timestamp-based thumbnail extraction often captures frames during scene transitions, resulting in blurry or black thumbnails that don't represent the video content. Implement frame quality analysis that evaluates brightness, contrast, and motion blur before selecting thumbnail frames, with fallback logic to try nearby timestamps if the initial frame is unsuitable.

⚠️ **Pitfall: Thumbnail Storage Without Cleanup Strategy**
Thumbnail generation can create thousands of small image files per video, leading to storage bloat and high I/O overhead if not managed properly. Implement thumbnail lifecycle management that removes unused thumbnails, archives old timeline previews, and optimizes storage layout to group related thumbnails for efficient retrieval.

### Analytics Dashboard Extension

The analytics dashboard transforms raw playback data into actionable insights about viewer behavior, content performance, and system health, requiring sophisticated data aggregation, real-time processing, and visualization capabilities. Think of analytics as the nervous system of the streaming platform - it must collect signals from every component, process them intelligently, and present information that enables data-driven decision making.

Building an effective analytics system requires extending the existing **AnalyticsManager** with comprehensive data collection, implementing a time-series data storage solution, and creating aggregation pipelines that can process millions of events efficiently. The dashboard must present both real-time metrics for operational monitoring and historical trends for content strategy decisions.

The analytics extension introduces several new architectural components that work together to collect, process, and visualize streaming data:

| Component | Responsibility | Key Capabilities |
|-----------|---------------|------------------|
| Event Collection Service | Gather analytics events | High-throughput ingestion, event validation, batching |
| Data Processing Pipeline | Transform raw events | Aggregation, sessionization, anomaly detection |
| Time-series Database | Store metrics efficiently | High write throughput, time-based queries, data retention |
| Analytics API Service | Query processed data | Dashboard endpoints, report generation, real-time metrics |
| Dashboard Frontend | Visualize insights | Interactive charts, real-time updates, custom reports |

**Architecture Decisions for Analytics Infrastructure:**

> **Decision: Time-series Database vs Traditional Relational Database**
> - **Context**: Analytics requires storing and querying millions of time-stamped events with high write throughput and efficient time-based aggregations
> - **Options Considered**: PostgreSQL with time partitioning, InfluxDB, TimescaleDB, Clickhouse
> - **Decision**: Implement TimescaleDB for time-series storage with PostgreSQL for metadata
> - **Rationale**: TimescaleDB provides time-series optimization while maintaining SQL compatibility, enabling complex joins with existing video metadata tables
> - **Consequences**: Enables efficient time-series queries and automatic data retention policies, but requires learning time-series query patterns and managing dual database setup

The analytics data model captures detailed information about viewer behavior, content performance, and system operations:

| Entity | Fields | Purpose |
|--------|--------|---------|
| PlaybackEvent | timestamp timestamptz, session_id UUID, video_id UUID, event_type string, quality string, position_ms integer, client_ip inet, user_agent text | Individual viewer interactions |
| ViewingSession | session_id UUID, video_id UUID, viewer_id UUID, started_at timestamptz, ended_at timestamptz, total_watch_time_ms integer, qualities_used text[], device_type string | Aggregated session data |
| ContentMetrics | video_id UUID, date date, view_count integer, unique_viewers integer, total_watch_time_ms bigint, completion_rate decimal, avg_quality string | Daily video performance |
| SystemMetrics | timestamp timestamptz, component string, metric_name string, metric_value double, labels jsonb | Infrastructure monitoring data |

Real-time analytics processing requires implementing event streaming pipelines that can handle high-throughput data ingestion while performing complex aggregations. The **DataProcessingPipeline** consumes events from the collection service, applies sessionization logic to group related events, and generates both real-time metrics and batch-processed reports.

Event sessionization presents particular challenges in video analytics, as viewer sessions can span multiple quality switches, pause/resume cycles, and even disconnection/reconnection scenarios. The processing pipeline must implement sophisticated logic to correlate events across time and reconstruct meaningful viewer journeys.

| Metric Category | Key Metrics | Update Frequency |
|-----------------|-------------|------------------|
| Real-time Metrics | Active viewers, current bandwidth usage, error rates | Every 10 seconds |
| Engagement Metrics | View duration, completion rates, interaction patterns | Every 5 minutes |
| Content Performance | Popular videos, trending content, viewer retention curves | Every hour |
| System Health | Transcoding queue depth, storage utilization, CDN performance | Every minute |

The **AnalyticsDashboard** frontend provides interactive visualization of collected metrics, supporting both operational monitoring dashboards for system administrators and content performance analytics for content creators and business stakeholders. The dashboard must handle real-time data updates while providing responsive interaction for historical data exploration.

**Common Analytics Implementation Pitfalls:**

⚠️ **Pitfall: Collecting Too Much Data Without Retention Policies**
Video analytics can generate enormous amounts of raw event data - potentially gigabytes per day for active platforms. Without proper data retention and aggregation strategies, storage costs explode and query performance degrades. Implement hierarchical data retention that keeps raw events for short periods (7-30 days), hourly aggregates for medium periods (90 days), and daily/weekly summaries for long-term analysis.

⚠️ **Pitfall: Real-time Dashboard Updates That Overwhelm the Backend**
Live dashboard updates can create significant load on analytics databases if not implemented carefully. Naive implementations that query raw event data every few seconds can saturate database connections and slow down data ingestion. Use pre-computed metrics tables, implement dashboard-specific caching layers, and batch multiple metric updates into single API calls.

⚠️ **Pitfall: Session Tracking Without Handling Player Reconnections**
Video players frequently reconnect during playback due to network issues, quality switches, or browser behavior, creating artificial session boundaries that skew analytics. Simple timeout-based sessionization can split single viewing sessions into multiple records. Implement intelligent session stitching that can correlate events across brief disconnections and player reinitialization.

### Content Delivery Network Integration

CDN integration transforms the streaming platform from a single-server solution into a globally distributed system capable of serving millions of concurrent viewers with low latency and high reliability. Think of CDN integration as building a global network of local video stores - instead of all customers traveling to one central location, they can get the same content from a nearby location optimized for their geographic region.

CDN integration touches every component of the streaming platform, requiring changes to storage architecture, manifest generation, caching strategies, and analytics collection. The system must coordinate between origin servers that host the master content and edge servers that cache and deliver content to end users, while maintaining consistency and handling cache invalidation scenarios.

The CDN extension introduces new architectural patterns around multi-tier content distribution, cache-aware URL generation, and geo-distributed analytics collection:

| CDN Component | Responsibility | Integration Points |
|---------------|---------------|-------------------|
| Origin Server | Authoritative content source | Enhanced **StreamingService** with CDN-aware URLs |
| Edge Cache Layer | Geographically distributed content | Modified manifest generation for edge URLs |
| Cache Invalidation Service | Coordinate content updates | Integration with **TranscodingService** completion |
| CDN Analytics Aggregator | Collect distributed metrics | Enhanced **AnalyticsManager** with multi-source data |
| Geo-routing Logic | Direct users to optimal edges | DNS-based or application-level routing |

**Architecture Decisions for CDN Implementation:**

> **Decision: Push vs Pull CDN Caching Strategy**
> - **Context**: Video content can be proactively pushed to CDN edges or pulled on-demand when first requested, each with different cost and performance characteristics
> - **Options Considered**: Push all content immediately after transcoding, pull content on first request, hybrid approach with popularity-based pushing
> - **Decision**: Implement hybrid caching with immediate push for popular content and pull-through caching for long-tail content
> - **Rationale**: Optimizes cache hit rates for popular content while avoiding unnecessary storage costs for rarely-viewed videos
> - **Consequences**: Provides optimal performance for popular content with cost efficiency, but requires implementing content popularity prediction algorithms

CDN integration requires significant modifications to the manifest generation system. The **ManifestGenerator** must create CDN-aware manifests that reference edge server URLs instead of origin server paths, while maintaining the ability to generate origin-direct manifests for testing and fallback scenarios.

CDN-aware manifest generation introduces new complexity around URL construction and cache management:

| Manifest Type | URL Pattern | Caching Strategy |
|---------------|-------------|------------------|
| Master Playlist | `https://cdn.example.com/v/{video_id}/master.m3u8` | Short TTL (60 seconds) for metadata updates |
| Variant Playlist | `https://edge.cdn.com/v/{video_id}/{quality}/playlist.m3u8` | Medium TTL (300 seconds) for segment lists |
| Video Segments | `https://edge.cdn.com/v/{video_id}/{quality}/seg{N}.ts` | Long TTL (86400 seconds) for immutable content |
| Thumbnail Images | `https://cdn.example.com/t/{video_id}/thumb_{timestamp}.jpg` | Long TTL (604800 seconds) for static images |

The **CDNOrchestrationService** coordinates between the origin platform and CDN infrastructure, managing content distribution, cache warming, and invalidation workflows. This service must integrate with the existing **TranscodingService** to automatically distribute newly transcoded content and update CDN configurations when content changes.

Cache invalidation presents particular challenges in video streaming, where content updates must be coordinated across potentially hundreds of edge servers while maintaining service availability. The system must implement intelligent invalidation strategies that minimize cache churn while ensuring content consistency.

| Cache Event | Invalidation Strategy | Performance Impact |
|-------------|----------------------|-------------------|
| New Video Upload | No invalidation needed | No impact - new URLs |
| Transcoding Completion | Warm cache proactively | Positive - content pre-positioned |
| Thumbnail Update | Invalidate image caches | Low - small file sizes |
| Manifest Changes | Invalidate playlist caches | Medium - affects all viewers |

CDN analytics integration requires collecting and correlating performance data from multiple sources, including origin servers, edge servers, and client-side player metrics. The **CDNAnalyticsAggregator** must handle data from distributed sources with different formats and timing characteristics.

**Enhanced Analytics for CDN Integration:**

| Metric Category | Origin Metrics | Edge Metrics | Client Metrics |
|-----------------|----------------|--------------|----------------|
| Performance | Transcoding throughput | Cache hit rates | Startup latency |
| Geographic | Upload origins | Viewer locations | Regional quality |
| Cost Optimization | Storage utilization | Bandwidth usage | CDN costs by region |
| Reliability | Origin availability | Edge health | Playback error rates |

**Common CDN Integration Pitfalls:**

⚠️ **Pitfall: Cache Warming Race Conditions During High-Traffic Events**
When popular content is published, simultaneous requests from multiple edge servers can overwhelm the origin server during cache warming. This creates a thundering herd problem where CDN amplifies load instead of reducing it. Implement staged cache warming with rate limiting and prioritization logic that warms the most important edges first while preventing origin overload.

⚠️ **Pitfall: Inconsistent Manifest Caching Across Edge Servers**
Different CDN edge servers may cache manifests at slightly different times, leading to inconsistencies where some viewers see updated playlists while others see stale versions. This causes players to request segments that don't exist at their edge server. Implement manifest versioning with atomic updates and coordinate cache invalidation timing across all edges before making content available.

⚠️ **Pitfall: Analytics Double-Counting with CDN and Origin Metrics**
When collecting analytics from both CDN edge servers and origin servers, it's easy to double-count viewer metrics or miss events that only occur at one layer. CDN analytics may report cache hits while origin analytics miss those same views. Implement hierarchical analytics collection that designates either CDN or origin as the authoritative source for each metric type and correlates data using session identifiers.

### Advanced Quality Selection Extension

Advanced quality selection transforms the basic adaptive streaming implementation into an intelligent system that considers multiple factors beyond simple bandwidth estimation, including device capabilities, power consumption, viewer preferences, and content characteristics. Think of it as upgrading from a simple thermostat that only measures temperature to a smart climate control system that considers humidity, occupancy, time of day, and energy costs.

The current **HlsWrapper** implements basic adaptive bitrate selection based on bandwidth measurements, but advanced quality selection requires considering device-specific capabilities, battery optimization for mobile devices, accessibility requirements, and content-aware quality decisions. This extension introduces machine learning elements, device fingerprinting, and sophisticated decision algorithms.

Advanced quality selection requires new components that work together to make intelligent streaming decisions:

| Component | Responsibility | Key Capabilities |
|-----------|---------------|------------------|
| Device Capability Detector | Identify client capabilities | Screen resolution, codec support, processing power |
| Quality Decision Engine | Intelligent quality selection | Multi-factor decision making, learning algorithms |
| Content Analysis Service | Analyze video characteristics | Motion complexity, darkness levels, detail density |
| Viewer Preference Manager | Track user quality choices | Quality preferences, accessibility needs |
| Battery Optimization Service | Mobile power management | Quality reduction for battery conservation |

**Architecture Decisions for Advanced Quality Selection:**

> **Decision: Client-side vs Server-side Quality Decision Making**
> - **Context**: Quality decisions can be made entirely on the client using local information, on the server using global data, or through hybrid approaches
> - **Options Considered**: Pure client-side adaptive bitrate, server-side recommendation with client override, hybrid system with server hints
> - **Decision**: Implement hybrid system with server-provided quality recommendations and client-side fine-tuning
> - **Rationale**: Combines server-side global knowledge and content analysis with client-side real-time conditions and user preferences
> - **Consequences**: Provides optimal quality decisions while maintaining responsiveness, but requires additional API complexity and client-server coordination

The advanced quality selection system extends the existing data model with new entities to track quality decisions and their effectiveness:

| Entity | Fields | Purpose |
|--------|--------|---------|
| QualityDecision | timestamp timestamptz, session_id UUID, selected_quality string, available_qualities string[], decision_factors jsonb, bandwidth_estimate integer, device_score decimal | Track quality selection rationale |
| DeviceCapability | fingerprint_hash string, screen_width integer, screen_height integer, supported_codecs string[], processing_score decimal, is_mobile boolean, battery_conscious boolean | Cache device capability detection |
| ContentCharacteristics | video_id UUID, motion_complexity decimal, average_brightness decimal, detail_density decimal, recommended_qualities jsonb, encoding_efficiency jsonb | Store content analysis results |
| ViewerPreference | viewer_id UUID, preferred_max_quality string, accessibility_needs string[], battery_optimization boolean, data_saver boolean, updated_at timestamptz | Track user preferences and needs |

Content-aware quality selection requires analyzing video characteristics to understand how different content types respond to quality changes. High-motion sports content requires different quality strategies than static presentation slides or low-light documentary footage.

The **ContentAnalysisService** processes transcoded videos to extract characteristics that inform quality selection:

| Analysis Type | Extracted Metrics | Quality Impact |
|---------------|------------------|----------------|
| Motion Analysis | Average motion vectors, scene complexity | High motion needs higher bitrates |
| Brightness Analysis | Histogram distribution, dark scene percentage | Dark content hides compression artifacts |
| Detail Analysis | Edge density, texture complexity | High detail content needs higher resolution |
| Audio Analysis | Dynamic range, speech vs music | Affects quality/bandwidth tradeoffs |

The **QualityDecisionEngine** implements sophisticated algorithms that consider multiple factors when selecting optimal quality levels. Unlike simple bandwidth-based selection, this engine weighs device capabilities, content characteristics, user preferences, and contextual factors like battery level or data plan constraints.

Quality decision factors and their relative weights:

| Decision Factor | Weight Range | Calculation Method |
|-----------------|-------------|-------------------|
| Bandwidth Estimate | 0.3-0.5 | Exponential moving average of throughput measurements |
| Device Capability Score | 0.2-0.3 | Composite of screen size, processing power, codec support |
| Content Complexity | 0.1-0.2 | Motion analysis, detail density, encoding efficiency |
| User Preferences | 0.1-0.2 | Explicit quality settings, accessibility needs, data saver mode |
| Context Factors | 0.1-0.2 | Battery level, connection type, time of day |

**Enhanced Player Integration:**

The **VideoPlayer** requires significant enhancements to support advanced quality selection, including integration with device capability detection, preference management, and contextual awareness:

| Player Enhancement | Implementation | Purpose |
|-------------------|---------------|---------|
| Device Fingerprinting | Hardware detection, capability testing | Determine optimal quality ceiling |
| Battery API Integration | Battery status monitoring | Reduce quality when battery is low |
| Connection Type Detection | Network API usage | Adapt to WiFi vs cellular connections |
| User Preference UI | Advanced quality settings panel | Allow fine-grained quality control |
| A/B Testing Framework | Quality algorithm variants | Test and optimize decision algorithms |

**Common Advanced Quality Selection Pitfalls:**

⚠️ **Pitfall: Over-Engineering Quality Decisions with Too Many Factors**
Advanced quality selection can become counterproductively complex when trying to optimize for every possible factor. Systems that consider dozens of variables often become unpredictable and difficult to debug, leading to suboptimal quality choices that frustrate users. Focus on the 3-5 most impactful factors for your specific use case and implement them reliably before adding additional complexity.

⚠️ **Pitfall: Ignoring Quality Selection Latency in Real-time Decisions**
Sophisticated quality decision algorithms can introduce significant computational latency, causing delays in quality switching that harm the viewing experience. Complex machine learning models or extensive content analysis during playback can cause stuttering or buffering. Implement quality decision caching, pre-computation where possible, and fallback to simple algorithms when decision latency exceeds acceptable thresholds.

⚠️ **Pitfall: Quality Thrashing from Conflicting Decision Factors**
When multiple decision factors provide contradictory signals (e.g., high bandwidth but low battery), poorly designed systems can rapidly switch between quality levels, creating a jarring viewing experience. Implement quality switching hysteresis with different thresholds for switching up versus down, and add minimum time intervals between quality changes to prevent rapid oscillation.

### Implementation Guidance

The future extensions represent significant architectural evolution beyond the base streaming platform. Each extension introduces new technical challenges and learning opportunities that build upon the foundational components while exploring advanced streaming concepts.

**A. Technology Recommendations Table:**

| Extension | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Live Streaming | Node.js with `node-rtmp-server` + FFmpeg | Go with custom RTMP server + distributed transcoding |
| Thumbnails | FFmpeg direct extraction + ImageMagick | OpenCV-based intelligent frame selection |
| Analytics | PostgreSQL with time partitioning | TimescaleDB + Apache Kafka streaming |
| CDN Integration | Cloudflare API integration | Custom edge server deployment |
| Advanced Quality | Rule-based decision tree | TensorFlow.js machine learning models |

**B. Recommended File Structure for Extensions:**

```
video-platform/
  services/
    live-streaming/
      ingest-server.js        ← RTMP/WebRTC endpoint
      live-transcoder.js      ← Real-time FFmpeg pipeline  
      broadcast-manager.js    ← Session lifecycle management
    thumbnails/
      thumbnail-service.js    ← Image extraction coordinator
      frame-analyzer.js       ← Intelligent frame selection
      image-optimizer.js      ← Compression and resizing
    analytics/
      event-collector.js      ← High-throughput event ingestion
      data-processor.js       ← Event aggregation pipeline
      dashboard-api.js        ← Analytics query endpoints
    cdn/
      cdn-orchestrator.js     ← Cache management and distribution
      manifest-enhancer.js    ← CDN-aware URL generation
      invalidation-service.js ← Cache invalidation coordination
    quality/
      decision-engine.js      ← Advanced quality selection algorithms
      content-analyzer.js     ← Video characteristics extraction
      device-detector.js      ← Client capability identification
  extensions/
    shared/
      metrics-collector.js    ← Common analytics utilities
      config-manager.js       ← Extension configuration
      health-monitor.js       ← Extension health checking
  frontend/
    extensions/
      live-player.js          ← Live streaming player enhancements
      analytics-dashboard.js  ← Metrics visualization
      quality-controls.js     ← Advanced quality selection UI
```

**C. Infrastructure Starter Code:**

**Live Streaming RTMP Server Foundation:**

```javascript
// services/live-streaming/ingest-server.js
const NodeMediaServer = require('node-media-server');
const EventEmitter = require('events');

class LiveIngestServer extends EventEmitter {
    constructor(config) {
        super();
        this.config = {
            rtmp: {
                port: 1935,
                chunk_size: 60000,
                gop_cache: true,
                ping: 30,
                ping_timeout: 60
            },
            http: {
                port: 8000,
                allow_origin: '*',
                mediaroot: './live-media'
            },
            ...config
        };
        
        this.mediaServer = new NodeMediaServer(this.config);
        this.activeStreams = new Map();
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.mediaServer.on('preConnect', (id, args) => {
            // TODO: Implement stream authentication
            // TODO: Validate broadcaster credentials
            // TODO: Check concurrent stream limits
            console.log('[NodeEvent on preConnect]', `id=${id} args=${JSON.stringify(args)}`);
        });
        
        this.mediaServer.on('postPublish', (id, StreamPath, args) => {
            // TODO: Extract stream key from StreamPath
            // TODO: Initialize real-time transcoding pipeline
            // TODO: Create LiveStream database record
            // TODO: Emit 'streamStarted' event for downstream processing
            console.log('[NodeEvent on postPublish]', `id=${id} StreamPath=${StreamPath} args=${JSON.stringify(args)}`);
        });
        
        this.mediaServer.on('donePublish', (id, StreamPath, args) => {
            // TODO: Stop transcoding processes
            // TODO: Update LiveStream status to ended
            // TODO: Cleanup temporary files and resources
            // TODO: Emit 'streamEnded' event
            console.log('[NodeEvent on donePublish]', `id=${id} StreamPath=${StreamPath} args=${JSON.stringify(args)}`);
        });
    }
    
    start() {
        return new Promise((resolve, reject) => {
            try {
                this.mediaServer.run();
                console.log(`Live ingest server started on RTMP port ${this.config.rtmp.port}`);
                resolve();
            } catch (error) {
                reject(error);
            }
        });
    }
    
    stop() {
        this.mediaServer.stop();
        this.activeStreams.clear();
    }
    
    getActiveStreamCount() {
        return this.activeStreams.size;
    }
    
    getStreamStatus(streamKey) {
        return this.activeStreams.get(streamKey);
    }
}

module.exports = LiveIngestServer;
```

**Analytics Event Collection Infrastructure:**

```javascript
// services/analytics/event-collector.js
const express = require('express');
const { Pool } = require('pg');

class AnalyticsEventCollector {
    constructor(dbConfig, options = {}) {
        this.db = new Pool(dbConfig);
        this.batchSize = options.batchSize || 1000;
        this.flushInterval = options.flushInterval || 5000; // 5 seconds
        this.eventBuffer = [];
        this.flushTimer = null;
        
        this.setupPeriodicFlush();
    }
    
    setupPeriodicFlush() {
        this.flushTimer = setInterval(() => {
            if (this.eventBuffer.length > 0) {
                this.flushEvents().catch(console.error);
            }
        }, this.flushInterval);
    }
    
    async collectEvent(event) {
        // TODO: Validate event schema against expected format
        // TODO: Add server-side timestamp and IP address
        // TODO: Apply rate limiting per session/IP to prevent abuse
        // TODO: Add event to buffer and check if flush is needed
        
        const enrichedEvent = {
            ...event,
            server_timestamp: new Date(),
            // Additional server-side enrichment goes here
        };
        
        this.eventBuffer.push(enrichedEvent);
        
        if (this.eventBuffer.length >= this.batchSize) {
            await this.flushEvents();
        }
        
        return { status: 'accepted', buffered: this.eventBuffer.length };
    }
    
    async flushEvents() {
        if (this.eventBuffer.length === 0) return;
        
        const eventsToFlush = [...this.eventBuffer];
        this.eventBuffer = [];
        
        try {
            // TODO: Implement efficient bulk insert using PostgreSQL COPY
            // TODO: Handle partial failures and retry logic
            // TODO: Update metrics for successful/failed event processing
            await this.bulkInsertEvents(eventsToFlush);
            console.log(`Flushed ${eventsToFlush.length} analytics events`);
        } catch (error) {
            console.error('Failed to flush analytics events:', error);
            // TODO: Implement dead letter queue for failed events
        }
    }
    
    async bulkInsertEvents(events) {
        // TODO: Generate efficient bulk insert SQL
        // TODO: Use prepared statements for performance
        // TODO: Handle connection retries and timeouts
        const query = `
            INSERT INTO analytics_events 
            (timestamp, session_id, video_id, event_type, quality, position_ms, client_ip, user_agent, server_timestamp)
            VALUES ${events.map((_, index) => `($${index * 9 + 1}, $${index * 9 + 2}, $${index * 9 + 3}, $${index * 9 + 4}, $${index * 9 + 5}, $${index * 9 + 6}, $${index * 9 + 7}, $${index * 9 + 8}, $${index * 9 + 9})`).join(', ')}
        `;
        
        const values = events.flatMap(event => [
            event.timestamp, event.session_id, event.video_id, 
            event.event_type, event.quality, event.position_ms,
            event.client_ip, event.user_agent, event.server_timestamp
        ]);
        
        await this.db.query(query, values);
    }
    
    createExpressMiddleware() {
        const router = express.Router();
        
        router.post('/events', express.json(), async (req, res) => {
            try {
                const result = await this.collectEvent(req.body);
                res.json(result);
            } catch (error) {
                console.error('Event collection error:', error);
                res.status(500).json({ error: 'Failed to collect event' });
            }
        });
        
        router.post('/events/batch', express.json(), async (req, res) => {
            try {
                const events = req.body.events || [];
                const results = await Promise.all(
                    events.map(event => this.collectEvent(event))
                );
                res.json({ results, accepted: results.length });
            } catch (error) {
                console.error('Batch event collection error:', error);
                res.status(500).json({ error: 'Failed to collect batch events' });
            }
        });
        
        return router;
    }
    
    async stop() {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
        }
        
        // Flush remaining events before stopping
        await this.flushEvents();
        await this.db.end();
    }
}

module.exports = AnalyticsEventCollector;
```

**D. Core Logic Skeleton Code:**

**Advanced Quality Selection Decision Engine:**

```javascript
// services/quality/decision-engine.js
class QualityDecisionEngine {
    constructor(config) {
        this.decisionWeights = {
            bandwidth: 0.4,
            deviceCapability: 0.25,
            contentComplexity: 0.15,
            userPreference: 0.1,
            contextFactors: 0.1
        };
        this.config = config;
        this.decisionHistory = new Map(); // session_id -> decisions[]
    }
    
    async selectOptimalQuality(selectionContext) {
        // TODO 1: Extract and validate all decision factors from context
        // TODO 2: Calculate bandwidth score from recent throughput measurements
        // TODO 3: Determine device capability score from hardware detection
        // TODO 4: Load content complexity metrics from video analysis
        // TODO 5: Apply user preferences and accessibility requirements
        // TODO 6: Consider context factors (battery, connection type, time)
        // TODO 7: Calculate weighted decision score for each available quality
        // TODO 8: Apply quality switching hysteresis to prevent thrashing
        // TODO 9: Log decision rationale for analytics and debugging
        // TODO 10: Return selected quality with confidence score
        
        const factors = await this.extractDecisionFactors(selectionContext);
        const qualityScores = this.calculateQualityScores(factors, selectionContext.availableQualities);
        const selectedQuality = this.applySelectionHysteresis(qualityScores, selectionContext);
        
        await this.logDecision(selectionContext, factors, selectedQuality);
        
        return {
            quality: selectedQuality,
            confidence: qualityScores[selectedQuality],
            factors: factors,
            reasoning: this.generateDecisionReasoning(factors, selectedQuality)
        };
    }
    
    async extractDecisionFactors(context) {
        // TODO: Parse bandwidth measurements and calculate trend
        // TODO: Query device capabilities from fingerprint or detect real-time  
        // TODO: Load content characteristics from analysis cache
        // TODO: Apply user preference overrides and constraints
        // TODO: Check battery level and connection type from client
        return {
            bandwidthScore: 0, // 0.0-1.0 based on available throughput
            deviceScore: 0,    // 0.0-1.0 based on device capabilities  
            contentScore: 0,   // 0.0-1.0 based on content complexity
            preferenceScore: 0, // 0.0-1.0 based on user settings
            contextScore: 0    // 0.0-1.0 based on situational factors
        };
    }
    
    calculateQualityScores(factors, availableQualities) {
        // TODO: For each available quality level:
        // TODO: - Calculate how well it matches bandwidth requirements
        // TODO: - Check device capability compatibility  
        // TODO: - Consider content-specific quality needs
        // TODO: - Apply user preference modifiers
        // TODO: - Weight all factors according to configuration
        // TODO: Return map of quality -> composite score
        const scores = {};
        return scores;
    }
    
    applySelectionHysteresis(qualityScores, context) {
        // TODO: Get previous quality selection for this session
        // TODO: Check if current top choice is significantly better than previous
        // TODO: Apply different thresholds for switching up vs down
        // TODO: Prevent quality changes more frequent than minimum interval
        // TODO: Return quality choice with hysteresis applied
        const currentQuality = context.currentQuality;
        const bestQuality = Object.keys(qualityScores).reduce((a, b) => 
            qualityScores[a] > qualityScores[b] ? a : b
        );
        return bestQuality;
    }
}

module.exports = QualityDecisionEngine;
```

**CDN Cache Orchestration Service:**

```javascript
// services/cdn/cdn-orchestrator.js
class CDNOrchestrator {
    constructor(cdnConfig, databaseService, storageService) {
        this.cdnConfig = cdnConfig;
        this.db = databaseService;
        this.storage = storageService;
        this.cacheWarmingQueue = [];
        this.invalidationQueue = [];
    }
    
    async distributeVideoContent(videoId, transcodingJobs) {
        // TODO 1: Validate that all transcoding jobs are completed successfully  
        // TODO 2: Generate CDN-aware manifest files with edge server URLs
        // TODO 3: Determine cache warming strategy based on content popularity prediction
        // TODO 4: Upload manifests and segments to CDN origin servers
        // TODO 5: Trigger cache warming for high-priority edge locations
        // TODO 6: Update video metadata with CDN URLs and distribution status
        // TODO 7: Set up cache invalidation rules for future content updates
        // TODO 8: Monitor distribution progress and handle failures gracefully
        
        console.log(`Starting CDN distribution for video ${videoId}`);
        
        const video = await this.db.Video.findById(videoId);
        if (!video) {
            throw new Error(`Video ${videoId} not found`);
        }
        
        const completedJobs = transcodingJobs.filter(job => job.status === 'COMPLETED');
        if (completedJobs.length === 0) {
            throw new Error(`No completed transcoding jobs found for video ${videoId}`);
        }
        
        // Generate CDN-aware manifests and coordinate distribution
        const distributionResult = await this.executeDistribution(video, completedJobs);
        return distributionResult;
    }
    
    async executeDistribution(video, jobs) {
        // TODO: Create CDN-aware master and variant playlists
        // TODO: Upload content to CDN origin with proper cache headers  
        // TODO: Implement staged cache warming to prevent origin overload
        // TODO: Track distribution progress and update database records
        return {
            success: true,
            cdnUrls: {},
            distributionId: `dist_${Date.now()}`
        };
    }
    
    async invalidateVideoContent(videoId, invalidationType = 'full') {
        // TODO 1: Identify all CDN URLs associated with the video
        // TODO 2: Determine invalidation scope (manifests only, segments, thumbnails)
        // TODO 3: Generate CDN API requests for cache invalidation
        // TODO 4: Handle rate limiting and batch invalidation requests
        // TODO 5: Monitor invalidation progress across all edge servers
        // TODO 6: Update database with invalidation status and timestamps
        // TODO 7: Implement rollback strategy if invalidation fails partially
        
        console.log(`Invalidating CDN cache for video ${videoId}, type: ${invalidationType}`);
        
        const invalidationId = `inv_${videoId}_${Date.now()}`;
        // Implementation details for cache invalidation coordination
        
        return {
            invalidationId,
            status: 'pending',
            affectedUrls: []
        };
    }
}

module.exports = CDNOrchestrator;
```

**E. Language-Specific Hints for Extensions:**

- **Live Streaming**: Use `node-media-server` for RTMP ingestion, `child_process.spawn()` for real-time FFmpeg transcoding with continuous monitoring
- **Analytics**: Leverage PostgreSQL's `COPY` command for high-throughput event insertion, use TimescaleDB extensions for automatic time-based partitioning  
- **Thumbnails**: Use FFmpeg's `-vf thumbnail` filter for intelligent frame selection, `sharp` library for efficient image processing and resizing
- **CDN Integration**: Use `axios` for CDN API integration with retry logic, implement exponential backoff for cache warming operations
- **Quality Selection**: Use `user-agent` parsing libraries for device detection, implement decision caching with `node-cache` for performance

**F. Milestone Checkpoints for Extensions:**

**Live Streaming Checkpoint:**
- Start RTMP server: `node services/live-streaming/ingest-server.js`
- Test broadcast: Use OBS Studio to stream to `rtmp://localhost:1935/live/test-stream`
- Verify real-time transcoding: Check that HLS segments are generated continuously
- Expected: Low-latency HLS playback within 5-10 seconds of broadcast

**Analytics Dashboard Checkpoint:**
- Start event collection: `node services/analytics/event-collector.js`
- Send test events: POST analytics events to `/events` endpoint
- Verify data storage: Query analytics database for received events
- Expected: Real-time dashboard showing playback metrics and viewer statistics

**CDN Integration Checkpoint:**
- Configure CDN credentials in environment variables
- Transcode test video and trigger CDN distribution
- Verify CDN URLs: Check that manifests reference edge server locations
- Expected: Video playback through CDN with improved global performance


## Glossary

> **Milestone(s):** All milestones (1-4) - This glossary provides essential terminology for video upload (M1), transcoding pipeline (M2), adaptive streaming (M3), and player integration (M4), as well as future extensions and debugging concepts.

The video streaming domain combines web development, multimedia processing, networking protocols, and client-side media APIs. This glossary defines the specialized terminology used throughout the design document, organized by conceptual area to help developers build mental models as they progress through implementation milestones.

Understanding this terminology is crucial because video streaming systems bridge multiple technical domains that each have their own vocabulary. A term like "manifest" means something different in web development (application cache) versus streaming protocols (playlist file). This glossary establishes precise definitions within the context of our video streaming platform.

### Core Video Streaming Concepts

Think of video streaming like a restaurant with a complex kitchen operation. Traditional video download is like ordering takeout - you wait for the entire meal to be prepared before you can eat anything. Streaming is like a buffet where dishes are continuously replenished, and adaptive streaming is like a smart buffet that automatically adjusts portion sizes and dish complexity based on how hungry you are and how fast you can eat.

| Term | Definition | Context |
|------|------------|---------|
| **Progressive Download** | Downloading entire video file before playback begins | Traditional approach; simple but requires complete download wait time |
| **Adaptive Streaming** | Automatic quality switching based on bandwidth and device capabilities | Core streaming approach; enables optimal experience across network conditions |
| **Manifest File** | M3U8 playlist containing segment references and metadata | HLS protocol; acts as index for video segments |
| **Transcoding Pipeline** | Process converting videos to streaming-friendly formats | Video processing; transforms uploaded content into playable segments |
| **Quality Rendition** | Video encoded at specific resolution and bitrate | Adaptive streaming; multiple renditions enable quality switching |
| **HLS Segmentation** | Splitting video into small downloadable chunks | HTTP Live Streaming; enables efficient streaming and seeking |
| **Quality Ladder** | Set of different quality levels available for streaming | Adaptive bitrate; typically 360p/720p/1080p with corresponding bitrates |
| **Master Playlist** | Top-level M3U8 listing all quality variants | HLS protocol; entry point for adaptive streaming |
| **Variant Playlist** | Quality-specific M3U8 containing segment URLs | HLS protocol; lists actual video segments for one quality level |

### Upload and File Processing

The upload process resembles shipping a large package through a courier service. Instead of requiring one massive truck that might break down, chunked upload is like breaking the package into many small boxes that can be sent via different routes and reassembled at the destination. If one box gets lost, you only need to resend that specific piece rather than starting over.

| Term | Definition | Context |
|------|------------|---------|
| **Chunked Upload** | Uploading large files in small sequential pieces | File handling; enables resumability and progress tracking |
| **Resumable Upload** | Ability to continue interrupted uploads from last successful chunk | Upload resilience; critical for large video files over unreliable networks |
| **Metadata Extraction** | Reading video properties using tools like FFprobe | Video processing; extracts duration, resolution, codec, bitrate information |
| **Upload Session** | Stateful context tracking chunked upload progress | Session management; maintains chunk state and expiration |
| **File Validation** | Checking uploaded files against format and size constraints | Security and quality; ensures platform compatibility |
| **Storage Service** | Abstraction layer for file persistence operations | Infrastructure; handles local disk or cloud storage |
| **Chunk Reassembly** | Combining uploaded chunks into complete video file | Upload completion; validates integrity and triggers transcoding |

### Video Processing and Encoding

Video transcoding is like translating a book into multiple languages while also creating different reading levels - picture books, regular text, and academic versions. Each "translation" (quality level) serves different audiences (devices and network conditions), and the translation process must be carefully managed to ensure quality and efficiency.

| Term | Definition | Context |
|------|------------|---------|
| **Transcoding Job** | Background task converting video to specific quality level | Job processing; represents one quality rendition transcoding |
| **Quality Configuration** | Settings defining output resolution, bitrate, and codec | Transcoding; determines characteristics of output video |
| **FFmpeg Process** | External process performing video encoding operations | Video processing; industry-standard transcoding tool |
| **Progress Parsing** | Extracting completion percentage from FFmpeg output | Job monitoring; tracks transcoding advancement |
| **Job Queue Pattern** | Background processing system managing transcoding tasks | Distributed systems; handles concurrent video processing |
| **Zombie Job Detection** | Identifying stuck jobs from crashed workers | Error recovery; prevents indefinite job hang |
| **Quality Ladder Generation** | Creating appropriate quality levels based on source video | Adaptive streaming; ensures optimal quality distribution |
| **Codec Selection** | Choosing appropriate video encoder for target compatibility | Video encoding; balances quality, size, and device support |

### Streaming Protocols and Delivery

HTTP Live Streaming works like a sophisticated TV guide system. The master playlist is like a channel guide showing all available quality options, variant playlists are like program schedules for each channel, and segments are like individual TV show episodes that can be watched in sequence.

| Term | Definition | Context |
|------|------------|---------|
| **HLS Protocol** | HTTP Live Streaming specification for adaptive video delivery | Streaming standard; defines manifest format and segment serving |
| **M3U8 Format** | Playlist file format used by HLS for manifests | File format; UTF-8 encoded M3U playlist with video-specific extensions |
| **Video Segment** | Individual video file representing small time slice | HLS delivery; typically 2-10 seconds of video content |
| **Segment Duration** | Time length of individual video segments | HLS tuning; affects seeking accuracy and startup latency |
| **Content-Addressed URLs** | URLs containing content hash for cache isolation | CDN optimization; prevents cache pollution across versions |
| **HTTP Range Requests** | Partial content delivery for efficient seeking | Streaming optimization; enables jumping to arbitrary video positions |
| **CDN-Friendly Headers** | HTTP headers optimized for content delivery networks | Performance; includes cache-control and content-length |
| **Progressive Activation** | Enabling streaming as each quality level completes transcoding | User experience; allows playback before all qualities are ready |

### Client-Side Streaming

The video player acts like a smart TV that automatically adjusts picture quality based on your internet connection speed, while also learning your preferences and device capabilities. It continuously monitors conditions and makes decisions to provide the best possible viewing experience.

| Term | Definition | Context |
|------|------------|---------|
| **HLS.js Integration** | JavaScript library enabling HLS playback in browsers | Browser compatibility; adds HLS support to browsers lacking native support |
| **Quality Switching** | Changing video quality during playback | Adaptive streaming; manual selection or automatic adaptation |
| **Bandwidth Estimation** | Measuring available network throughput for quality decisions | Adaptive bitrate; determines optimal quality for current conditions |
| **Quality Thrashing** | Rapid switching between quality levels that degrades experience | Player optimization; undesirable behavior requiring hysteresis logic |
| **Conservative Quality Switching** | Stable quality changes avoiding unnecessary switches | Adaptive bitrate; prioritizes stability over immediate optimization |
| **Playback Analytics** | Recording viewing behavior and performance metrics | Data collection; tracks user engagement and technical performance |
| **Buffer Health Monitoring** | Tracking downloaded content ahead of playback position | Streaming optimization; prevents playback interruption |

### System Architecture Patterns

The video streaming system uses several distributed systems patterns that can be understood through analogies. Database-mediated communication is like using a shared bulletin board where different teams post updates and check for work. The job queue pattern is like a restaurant kitchen with order tickets that different cooks can claim and complete.

| Term | Definition | Context |
|------|------------|---------|
| **Database-Mediated Communication** | Components coordinating through shared database state | Architecture pattern; enables loose coupling between services |
| **Workflow Coordination** | Orchestrating multi-step processes across components | System integration; manages upload→transcoding→streaming pipeline |
| **Impedance Mismatch** | Different operational characteristics between components | System design; upload is real-time while transcoding is batch |
| **Event-Driven Architecture** | Components reacting to state changes and notifications | Distributed systems; enables scalable component coordination |
| **Stateful Session Management** | Maintaining client state across multiple requests | Upload handling; tracks chunked upload progress and resumption |
| **Background Job Processing** | Asynchronous task execution without blocking user requests | Performance; transcoding happens independently of user interactions |
| **Transactional Boundaries** | Ensuring data consistency across related database operations | Data integrity; atomic updates spanning multiple tables |

### Testing and Quality Assurance

Testing a video streaming system is like quality control in a complex assembly line with multiple interdependent stations. You need unit tests that verify each station works correctly in isolation, integration tests that ensure the conveyor belts between stations function properly, and end-to-end tests that validate the entire product from raw materials to finished output.

| Term | Definition | Context |
|------|------------|---------|
| **Controlled Chunk Simulation** | Test framework injecting failures at specific upload boundaries | Testing infrastructure; validates resumable upload behavior |
| **Tiered Mocking Strategy** | Different levels of mocking for unit vs integration tests | Test architecture; balances test isolation with realistic interactions |
| **Error Injection Testing** | Systematically introducing failures to validate error handling | Reliability testing; ensures robust failure recovery |
| **Workflow Coordination Testing** | Monitoring state changes across component boundaries | Integration testing; validates end-to-end pipeline behavior |
| **Temporal Coordination Testing** | Validating behavior across different time scales | System testing; ensures proper sequencing of asynchronous operations |
| **Performance Testing** | Validating system behavior under load and resource constraints | Scalability testing; identifies bottlenecks and capacity limits |
| **Milestone Checkpoints** | Verification steps after each development stage | Development process; ensures progressive functionality validation |

### Error Handling and Recovery

Error handling in video streaming systems requires understanding the different failure modes across the pipeline. Upload errors are like postal service problems - packages can get lost in transit and need redelivery. Transcoding errors are like factory machinery breakdowns - jobs need to be restarted or rerouted to different workers. Streaming errors are like broadcast interruptions - playback needs to gracefully recover without user intervention.

| Term | Definition | Context |
|------|------------|---------|
| **Upload Recovery** | Resuming interrupted uploads from last successful chunk | Error handling; handles network failures during large file uploads |
| **Transcoding Retry Logic** | Automatic retrying of failed transcoding jobs | Job processing; handles temporary FFmpeg failures |
| **Graceful Degradation** | System continuing operation with reduced functionality | Resilience; serving available qualities when some transcoding fails |
| **Error Propagation** | Communicating failures up through system layers | Architecture; ensures errors reach appropriate handlers |
| **Circuit Breaker Pattern** | Preventing cascading failures in distributed components | Fault tolerance; isolates failing components from healthy ones |
| **Exponential Backoff** | Progressively longer delays between retry attempts | Rate limiting; prevents overwhelming failed services |
| **Dead Letter Queuing** | Isolating persistently failing jobs for investigation | Job processing; prevents poison messages from blocking queue |

### Performance and Optimization

Performance optimization in video streaming involves multiple levels, like optimizing a complex transportation network. You need local optimizations (efficient video encoding), regional optimizations (CDN placement), and global optimizations (adaptive bitrate algorithms). Each level has different constraints and optimization strategies.

| Term | Definition | Context |
|------|------------|---------|
| **Cache Hit Rate** | Percentage of requests served from CDN cache versus origin | CDN performance; measures effectiveness of content distribution |
| **Startup Latency** | Time from user clicking play until video begins | User experience; affected by manifest download and initial buffering |
| **Seeking Performance** | Speed of jumping to arbitrary positions in video | User experience; depends on segment size and HTTP range support |
| **Transcoding Efficiency** | Balancing encoding speed with output quality | Resource optimization; affects both cost and time-to-availability |
| **Concurrent Job Limits** | Maximum parallel transcoding operations | Resource management; prevents server overload |
| **Memory Management** | Efficient handling of large video files in processing | Performance; prevents out-of-memory errors during transcoding |
| **Connection Pooling** | Reusing HTTP connections for multiple requests | Network optimization; reduces connection overhead |

### Advanced Features and Extensions

Advanced features build upon the core streaming platform like adding specialized equipment to a well-functioning factory. Live streaming adds real-time conveyor belts, analytics systems add quality control monitoring, and CDN integration adds global distribution networks.

| Term | Definition | Context |
|------|------------|---------|
| **Live Streaming** | Real-time video broadcasting with minimal latency | Advanced feature; requires real-time transcoding and distribution |
| **Real-Time Transcoding** | Converting live video streams with minimal delay | Live streaming; much more demanding than on-demand transcoding |
| **Thumbnail Generation** | Extracting representative images from video content | User interface; provides preview images and timeline scrubbing |
| **Timeline Thumbnails** | Preview images for video seeking interface | User experience; shows video content at different time positions |
| **Analytics Aggregation** | Processing raw events into meaningful metrics | Data analysis; transforms playback events into business insights |
| **Content Delivery Network** | Geographically distributed servers for global content delivery | Scalability; reduces latency and bandwidth costs |
| **Cache Warming** | Proactively loading content to CDN edge servers | Performance optimization; prepares content before user requests |
| **Cache Invalidation** | Coordinated removal of stale content from CDN caches | Content management; ensures users receive updated video versions |
| **Quality Decision Engine** | Intelligent system for optimal streaming quality selection | Adaptive streaming; makes sophisticated quality decisions |
| **Device Capability Detection** | Identifying client hardware and software capabilities | Optimization; tailors streaming parameters to device constraints |
| **Content-Aware Quality Selection** | Quality decisions based on video characteristics | Advanced optimization; considers video complexity in quality selection |

### Implementation Guidance

The glossary serves as a reference throughout development, but certain terms become critical at specific milestones. Understanding the progression of terminology helps developers build vocabulary incrementally.

#### Milestone-Specific Term Clusters

**Milestone 1 (Upload):** Focus on chunked upload, resumable upload, upload session, file validation, metadata extraction, and storage service concepts. These form the foundation vocabulary for implementing reliable large file uploads.

**Milestone 2 (Transcoding):** Emphasis shifts to transcoding job, FFmpeg process, quality configuration, job queue pattern, and progress parsing. Understanding these terms is essential for building the video processing pipeline.

**Milestone 3 (Streaming):** HLS protocol, manifest file, video segment, master playlist, variant playlist, and CDN-friendly headers become central. These terms define the streaming delivery architecture.

**Milestone 4 (Player):** HLS.js integration, quality switching, bandwidth estimation, playback analytics, and conservative quality switching guide the client-side implementation.

#### Technology-Specific Terminology

Different implementation technologies introduce their own vocabulary that layers on top of these concepts:

**Node.js Context:** Terms like "streaming upload" refer to Node.js streams API, while "chunked transfer" relates to HTTP chunked encoding. Understanding how Node.js handles asynchronous I/O affects implementation of upload sessions and transcoding job monitoring.

**FFmpeg Context:** "Progress parsing" involves understanding FFmpeg's stderr output format. "Process management" requires knowledge of child process spawning and signal handling. "Codec selection" maps to specific FFmpeg encoder names and parameters.

**Browser Context:** "HLS.js integration" assumes understanding of HTML5 video element API and JavaScript event handling. "Bandwidth estimation" involves browser network APIs and performance measurement techniques.

#### Common Term Confusion Points

Several terms have meanings specific to video streaming that differ from their usage in general web development:

- **Manifest:** In web development, often refers to application cache manifest or PWA manifest. In streaming, specifically refers to HLS playlist files.
- **Segment:** In networking, refers to TCP segments. In streaming, refers to video file chunks.
- **Quality:** In general software, often means code quality or testing quality. In streaming, specifically refers to video resolution and bitrate combinations.
- **Adaptive:** In web development, often refers to responsive design. In streaming, refers to automatic quality switching based on network conditions.
- **Progressive:** In web development, often refers to progressive web apps. In video, refers to downloading entire file before playback.

Understanding these distinctions helps avoid confusion when reading streaming-focused documentation and tutorials.

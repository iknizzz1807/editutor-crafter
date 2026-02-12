# Media Processing Pipeline: Design Document


## Overview

This system builds a scalable media processing service that handles image resizing, video transcoding, and thumbnail generation through an asynchronous job queue. The key architectural challenge is efficiently processing large media files while providing real-time progress tracking and reliable error recovery across distributed worker processes.

![System Architecture Overview](./diagrams/system-architecture.svg)


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** All milestones (1-3) as this foundational understanding applies throughout the system

The explosive growth of digital media content creation and consumption has created an unprecedented demand for scalable media processing services. Modern applications routinely handle thousands of user-uploaded images and videos daily, each requiring transformation into multiple formats, resolutions, and quality levels to support diverse devices and network conditions. This section establishes the real-world drivers behind media processing systems and examines the core technical challenges that make building such systems complex.

### Mental Model: Digital Photo Lab

> **Mental Model**: Think of this system as a modern digital photo lab that never closes and can process thousands of orders simultaneously.

In the era of film photography, a photo lab operated as a specialized facility where customers dropped off rolls of film and returned later to collect developed prints. The lab employed skilled technicians who understood the chemistry of different film types, the characteristics of various paper stocks, and the precise timing required for each processing step. Customers could request different print sizes, finishes, and quantities, with each order requiring specific handling procedures.

Our media processing pipeline operates on remarkably similar principles, but at digital scale. Users submit "orders" by uploading raw media files through an API gateway, much like dropping off film canisters at a lab counter. Each upload triggers the creation of a **processing job** that contains detailed instructions about desired output formats, resolutions, and quality settings—analogous to the order forms customers filled out for their prints.

The job queue functions as the lab's work order system, organizing incoming requests by priority and distributing them to available technicians. In our digital lab, these technicians are **worker processes** that specialize in different types of media transformation. Just as a photo lab might have separate stations for developing, printing, and finishing, our workers are equipped with specialized tools: Pillow for image manipulation, FFmpeg for video transcoding, and custom progress tracking systems.

The most crucial similarity lies in the concept of **batch processing with progress tracking**. Traditional photo labs provided customers with claim tickets and estimated completion times. Our system maintains detailed progress records for each job, sending webhook notifications that function like the phone calls labs made when orders were ready for pickup. Both systems must handle the reality that some processes take much longer than others—a simple wallet-sized print versus a large format enlargement in the traditional lab, or thumbnail generation versus 4K video transcoding in our digital system.

This analogy helps clarify why certain architectural decisions are essential: job queues prevent the system from being overwhelmed during peak times (like the Christmas photo rush), worker specialization ensures optimal resource utilization, and progress tracking provides transparency that builds user trust in a process they cannot directly observe.

### Existing Solutions Comparison

The media processing landscape offers three primary architectural approaches, each with distinct trade-offs that influence system design decisions. Understanding these alternatives provides context for the architectural choices made in this implementation.

**Cloud Service Integration Approach**

Cloud-based solutions like AWS MediaConvert, Google Cloud Video Intelligence, and Azure Media Services represent the "software-as-a-service" model for media processing. These services handle infrastructure scaling automatically and provide pre-optimized processing pipelines with minimal configuration requirements.

| Aspect | Cloud Services | Assessment |
|--------|---------------|------------|
| **Setup Complexity** | Minimal - API integration only | Fastest time to market |
| **Scaling** | Automatic horizontal scaling | Handles traffic spikes seamlessly |
| **Cost Model** | Pay-per-operation pricing | Can become expensive at scale |
| **Customization** | Limited to provided parameters | Insufficient for specialized workflows |
| **Latency** | Network round-trip overhead | Adds 200-500ms per operation |
| **Data Residency** | Files transmitted to cloud | Potential privacy/compliance issues |
| **Vendor Lock-in** | Tight coupling to provider APIs | Migration complexity increases over time |

Cloud services excel for applications with straightforward processing requirements and variable usage patterns. However, they become prohibitively expensive for high-volume applications and offer insufficient control for specialized processing workflows that require custom algorithms or unusual format support.

**On-Premise Processing Solutions**

Self-hosted media processing represents the traditional approach where organizations deploy and manage their own processing infrastructure. This includes both custom-built solutions using libraries like FFmpeg and Pillow, as well as commercial software packages designed for media workflow automation.

| Aspect | On-Premise | Assessment |
|--------|------------|------------|
| **Infrastructure Control** | Complete hardware/software control | Optimal for specialized requirements |
| **Processing Latency** | Local processing - minimal network overhead | Best performance for real-time needs |
| **Customization** | Unlimited - full source code access | Supports any processing algorithm |
| **Scaling Complexity** | Manual capacity planning required | Requires infrastructure expertise |
| **Operational Overhead** | Full system administration burden | Significant ongoing maintenance costs |
| **Initial Investment** | High upfront hardware/software costs | Substantial capital expenditure |
| **Reliability** | Single points of failure without redundancy | Requires sophisticated disaster recovery |

On-premise solutions provide maximum control and can achieve the lowest per-operation costs at high volumes. They are essential for organizations with strict data residency requirements or highly specialized processing needs. However, they demand significant infrastructure expertise and cannot easily handle unpredictable traffic patterns.

**Hybrid Processing Architecture**

The hybrid approach combines self-hosted processing capabilities with cloud service integration, allowing systems to optimize for both control and scalability. This architecture typically handles routine processing operations locally while leveraging cloud services for overflow capacity or specialized operations.

| Aspect | Hybrid Approach | Assessment |
|--------|----------------|------------|
| **Cost Optimization** | Local processing for base load, cloud for peaks | Balances fixed and variable costs |
| **Flexibility** | Custom algorithms locally, cloud for standard operations | Best of both worlds approach |
| **Complexity** | Requires orchestration between multiple systems | Significant architectural complexity |
| **Reliability** | Cloud failover for local system outages | Enhanced disaster recovery options |
| **Data Management** | Intelligent routing based on content sensitivity | Supports compliance requirements |
| **Scaling Strategy** | Predictable local capacity with elastic overflow | Handles both steady state and spikes |

> **Decision: Self-Hosted Processing with Cloud-Ready Architecture**
> - **Context**: This implementation targets organizations that need control over processing algorithms, data residency, and per-operation costs, while maintaining the flexibility to integrate cloud services in the future
> - **Options Considered**: Pure cloud integration, pure on-premise deployment, hybrid architecture from the start
> - **Decision**: Build a self-hosted system with modular components that can integrate with cloud services later
> - **Rationale**: Self-hosted provides learning value for understanding media processing fundamentals, offers maximum customization for specialized requirements, and delivers the lowest long-term operational costs for high-volume scenarios. The modular architecture ensures cloud integration remains possible without major refactoring.
> - **Consequences**: Higher initial development complexity and operational overhead, but provides deep understanding of media processing challenges and maximum future flexibility

### Core Technical Challenges

Building a scalable media processing pipeline presents several interconnected technical challenges that differentiate it from typical web application development. Each challenge requires specific architectural decisions and implementation strategies that influence the entire system design.

**Memory Management and Resource Utilization**

Media processing operations are inherently memory-intensive, with resource requirements that scale dramatically based on content characteristics. A single 4K video file can require several gigabytes of RAM during transcoding operations, while high-resolution image processing may need temporary buffers that exceed available system memory.

The fundamental challenge lies in the **unpredictable nature of resource requirements**. Unlike traditional web services where request processing consumes predictable amounts of memory for predictable durations, media processing operations vary enormously based on input characteristics. A 30-second 1080p video might require 500MB of working memory, while a 2-hour 4K video could need 8GB or more.

| Resource Type | Challenge | Impact | Mitigation Strategy |
|---------------|-----------|--------|-------------------|
| **Memory Usage** | Unpredictable working set sizes | Worker process crashes, system instability | Pre-flight content analysis, memory-based job routing |
| **Disk I/O** | Large temporary file requirements | Storage exhaustion, I/O bottlenecks | Streaming processing, temporary file cleanup |
| **CPU Utilization** | CPU-intensive encoding operations | Worker starvation, system responsiveness | Process isolation, CPU quota enforcement |
| **Network Bandwidth** | Large file transfers during processing | Network congestion, timeout issues | Local processing, chunked transfers |

The memory management challenge extends beyond simple allocation tracking to include **temporal memory usage patterns**. Video transcoding typically follows a sawtooth pattern where memory usage builds during frame buffering, spikes during encoding operations, then drops during I/O operations. Image processing may require sudden allocation of large contiguous buffers for pixel manipulation operations.

Effective memory management requires implementing **resource-aware job scheduling** that considers both current system utilization and estimated resource requirements for queued jobs. This involves analyzing input media characteristics before processing begins and routing jobs to workers with appropriate resource availability.

**Format Diversity and Compatibility**

The media processing ecosystem encompasses dozens of container formats, codecs, and encoding parameters, each with unique characteristics and compatibility requirements. Modern applications must support legacy formats for backward compatibility while adopting emerging formats like WebP, AVIF, and AV1 for optimal efficiency.

The complexity extends beyond simple format conversion to include **parameter optimization for different use cases**. A single input image might need transformation into JPEG for broad compatibility, WebP for size optimization, and AVIF for cutting-edge browsers, each with different quality settings optimized for the intended display context.

| Format Category | Complexity Factor | Technical Challenge | Implementation Requirement |
|----------------|------------------|-------------------|--------------------------|
| **Image Formats** | Color space variations (RGB, CMYK, YUV) | Color accuracy preservation | Comprehensive color management |
| **Video Containers** | Codec/container compatibility matrices | Format selection optimization | Intelligent codec mapping |
| **Encoding Parameters** | Quality vs. size trade-offs | Context-aware optimization | Adaptive parameter selection |
| **Metadata Handling** | Format-specific metadata standards | Information preservation/privacy | Configurable metadata policies |

The challenge intensifies when considering **cross-format metadata preservation**. EXIF data in JPEG images contains orientation, camera settings, and potentially sensitive location information. Video files include complex metadata about encoding parameters, timestamps, and technical characteristics that may need preservation or stripping based on privacy policies.

Format diversity also introduces **version compatibility issues**. The WebP format supports both lossy and lossless compression with different browser support timelines. H.265 video encoding provides superior compression but requires hardware acceleration for practical performance and has complex licensing requirements.

**Processing Time Estimation and Progress Tracking**

Unlike typical web service operations that complete in milliseconds, media processing operations can require minutes or hours depending on content characteristics and output requirements. Users expect accurate progress estimates and reliable completion notifications, but media processing progress is notoriously difficult to predict accurately.

The fundamental challenge stems from the **non-linear nature of media processing operations**. Video transcoding progress varies dramatically based on scene complexity—static scenes encode quickly while high-motion sequences require significantly more processing time per frame. Image processing operations may complete in stages with vastly different durations for each phase.

Traditional percentage-based progress indicators prove inadequate for media processing workflows. A more effective approach involves **stage-based progress tracking** that provides users with meaningful information about current processing phases rather than potentially inaccurate time estimates.

| Processing Stage | Progress Characteristics | Estimation Challenge | Tracking Strategy |
|-----------------|------------------------|-------------------|------------------|
| **Content Analysis** | Fast, predictable duration | Minimal - file I/O bound | Simple percentage based on bytes read |
| **Format Conversion** | Highly variable based on complexity | Significant - depends on content characteristics | Stage completion milestones |
| **Quality Optimization** | Iterative with unknown convergence | Extreme - optimization algorithms vary | Iteration count with quality targets |
| **Output Generation** | Predictable but format-dependent | Moderate - based on output size estimates | Bytes written vs. estimated output size |

Progress tracking complexity increases with **multi-output processing scenarios**. A single input video might generate multiple resolution variants, each requiring separate transcoding operations with different complexity characteristics. Users need visibility into overall job progress rather than individual operation progress.

The challenge extends to **failure recovery and retry scenarios**. When processing operations fail partway through completion, the system must determine whether to restart from the beginning or resume from intermediate checkpoints. This decision impacts both progress accuracy and resource utilization efficiency.

**Worker Coordination and Distributed Processing**

Media processing systems require sophisticated coordination between multiple worker processes to achieve optimal throughput while avoiding resource conflicts and ensuring reliable job completion. Unlike stateless web service workers, media processing workers maintain significant state during long-running operations and require careful coordination to prevent resource exhaustion.

The primary coordination challenge involves **load balancing with resource awareness**. Traditional round-robin or random load balancing proves inadequate when workers have different resource profiles and jobs have varying resource requirements. A worker processing a large video file may be unavailable for new jobs for hours, while a worker handling image resizing might complete dozens of operations in the same timeframe.

| Coordination Aspect | Challenge | Impact | Solution Approach |
|-------------------|-----------|--------|------------------|
| **Resource-Aware Scheduling** | Matching job requirements with worker capabilities | Suboptimal resource utilization, job failures | Dynamic worker capability advertising |
| **Failure Detection** | Long-running operations mask worker failures | Jobs stuck in processing state indefinitely | Heartbeat monitoring with timeout escalation |
| **State Management** | Workers maintain processing state across operations | Inconsistent state after failures | Checkpoint-based recovery mechanisms |
| **Priority Handling** | High-priority jobs blocked by long-running operations | Poor user experience for urgent requests | Preemptive scheduling with job priority queues |

Worker coordination must also handle **graceful degradation scenarios** where individual workers become unavailable due to resource exhaustion, process failures, or system maintenance. The coordination system needs mechanisms to redistribute work from failed workers while preserving processing progress and avoiding duplicate work.

The challenge extends to **dynamic scaling scenarios** where worker capacity needs to increase or decrease based on current load patterns. Adding new workers requires capability discovery and job redistribution, while removing workers requires graceful job migration and state preservation.

> **Architecture Insight**: The combination of these four core challenges—memory management, format diversity, progress estimation, and worker coordination—creates a system complexity profile that differs significantly from typical web applications. Success requires architectural patterns that embrace asynchronous processing, resource-aware scheduling, and comprehensive error recovery mechanisms.

These technical challenges inform every major architectural decision in the media processing pipeline, from the choice of message queue technology to the design of progress tracking APIs. Understanding their interconnected nature helps explain why media processing systems require specialized patterns and cannot simply apply traditional web application architectures.

The following sections will detail how each system component addresses these challenges through specific design decisions, implementation patterns, and architectural trade-offs that provide reliable, scalable media processing capabilities.

### Implementation Guidance

This implementation guidance provides practical technology choices and starter code to address the core challenges identified in the problem analysis. The recommendations balance learning value with production readiness, focusing on Python-based solutions that provide clear visibility into media processing concepts.

**Technology Recommendations**

| Component | Simple Option | Advanced Option | Rationale |
|-----------|---------------|----------------|-----------|
| **Message Queue** | Redis with `rq` library | RabbitMQ with Celery | Redis offers simpler setup for learning; RabbitMQ provides better production reliability |
| **Image Processing** | Pillow (PIL fork) | OpenCV with Pillow fallback | Pillow handles most use cases; OpenCV adds computer vision capabilities |
| **Video Processing** | subprocess calls to FFmpeg | python-ffmpeg wrapper | Direct subprocess gives full control; wrapper adds convenience |
| **Progress Storage** | Redis key-value store | PostgreSQL with real-time updates | Redis provides fast updates; PostgreSQL offers ACID guarantees |
| **File Storage** | Local filesystem | AWS S3 with local caching | Local files simplify development; S3 enables production scaling |
| **Worker Management** | Simple process pool | Kubernetes job scheduling | Process pool suitable for single-machine; K8s enables distributed workers |

**Recommended Project Structure**

The project structure separates core media processing logic from infrastructure concerns, enabling focused development on each component while maintaining clear boundaries for testing and maintenance.

```
media_pipeline/
├── src/
│   ├── media_pipeline/
│   │   ├── __init__.py
│   │   ├── api/                    # REST API and job submission
│   │   │   ├── __init__.py
│   │   │   ├── handlers.py         # Request handlers
│   │   │   └── schemas.py          # Request/response models
│   │   ├── core/                   # Core business logic
│   │   │   ├── __init__.py
│   │   │   ├── job_manager.py      # Job lifecycle management
│   │   │   ├── progress_tracker.py # Progress monitoring
│   │   │   └── webhook_sender.py   # Notification delivery
│   │   ├── processors/             # Media processing components
│   │   │   ├── __init__.py
│   │   │   ├── image_processor.py  # Image operations
│   │   │   ├── video_processor.py  # Video transcoding
│   │   │   └── base_processor.py   # Common processing interface
│   │   ├── queue/                  # Job queue implementation
│   │   │   ├── __init__.py
│   │   │   ├── redis_queue.py      # Redis-based queue
│   │   │   ├── worker.py           # Worker process logic
│   │   │   └── scheduler.py        # Job scheduling
│   │   ├── storage/                # File and metadata storage
│   │   │   ├── __init__.py
│   │   │   ├── file_manager.py     # File I/O operations
│   │   │   └── metadata_store.py   # Job and progress storage
│   │   └── utils/                  # Common utilities
│   │       ├── __init__.py
│   │       ├── config.py           # Configuration management
│   │       ├── logging.py          # Logging setup
│   │       └── errors.py           # Custom exception types
├── tests/                          # Test files mirror src structure
│   ├── unit/
│   ├── integration/
│   └── fixtures/                   # Sample media files
├── scripts/                        # Operational scripts
│   ├── start_workers.py            # Worker process management
│   └── health_check.py             # System monitoring
├── config/                         # Configuration files
│   ├── development.yaml
│   └── production.yaml
├── requirements/                   # Dependency specifications
│   ├── base.txt                    # Core dependencies
│   ├── dev.txt                     # Development tools
│   └── prod.txt                    # Production additions
└── docker/                        # Container definitions
    ├── Dockerfile.api
    └── Dockerfile.worker
```

**Infrastructure Starter Code**

The following components provide complete, production-ready infrastructure that supports the core learning objectives without requiring deep implementation of supporting systems.

**Configuration Management System** (`src/media_pipeline/utils/config.py`):

```python
"""
Configuration management for media processing pipeline.
Handles environment-specific settings and validation.
"""
import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
@dataclass
class StorageConfig:
    base_path: str = "./storage"
    temp_path: str = "./temp"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
@dataclass
class ProcessingConfig:
    max_workers: int = 4
    job_timeout: int = 3600  # 1 hour
    retry_attempts: int = 3
    webhook_timeout: int = 30
    
@dataclass
class AppConfig:
    redis: RedisConfig
    storage: StorageConfig
    processing: ProcessingConfig
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AppConfig':
        """Load configuration from YAML file with environment variable substitution."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Substitute environment variables
        config_data = cls._substitute_env_vars(config_data)
        
        return cls(
            redis=RedisConfig(**config_data.get('redis', {})),
            storage=StorageConfig(**config_data.get('storage', {})),
            processing=ProcessingConfig(**config_data.get('processing', {})),
            debug=config_data.get('debug', False)
        )
    
    @staticmethod
    def _substitute_env_vars(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in configuration."""
        if isinstance(data, dict):
            return {k: AppConfig._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        else:
            return data

# Global configuration instance
config: Optional[AppConfig] = None

def init_config(config_path: Optional[str] = None) -> AppConfig:
    """Initialize global configuration from file or environment."""
    global config
    
    if config_path is None:
        env = os.getenv('ENVIRONMENT', 'development')
        config_path = f"config/{env}.yaml"
    
    config = AppConfig.from_yaml(config_path)
    return config

def get_config() -> AppConfig:
    """Get global configuration, initializing if necessary."""
    global config
    if config is None:
        config = init_config()
    return config
```

**Logging Infrastructure** (`src/media_pipeline/utils/logging.py`):

```python
"""
Centralized logging configuration for media processing pipeline.
Provides structured logging with correlation IDs for job tracking.
"""
import logging
import logging.config
import json
import uuid
from contextlib import contextmanager
from typing import Optional, Dict, Any
from datetime import datetime

class JobContextFilter(logging.Filter):
    """Add job context information to log records."""
    
    def filter(self, record):
        # Add job_id from context if available
        record.job_id = getattr(self, 'job_id', 'no-job')
        record.correlation_id = getattr(self, 'correlation_id', str(uuid.uuid4()))
        return True

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'job_id': getattr(record, 'job_id', None),
            'correlation_id': getattr(record, 'correlation_id', None),
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s (job=%(job_id)s)'
        },
        'json': {
            '()': JSONFormatter,
        },
    },
    'filters': {
        'job_context': {
            '()': JobContextFilter,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'filters': ['job_context'],
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filters': ['job_context'],
            'filename': 'logs/media_pipeline.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

def setup_logging(debug: bool = False):
    """Initialize logging configuration."""
    if debug:
        LOGGING_CONFIG['root']['level'] = 'DEBUG'
        LOGGING_CONFIG['handlers']['console']['level'] = 'DEBUG'
    
    logging.config.dictConfig(LOGGING_CONFIG)

@contextmanager
def job_logging_context(job_id: str, correlation_id: Optional[str] = None):
    """Context manager that adds job information to all log records."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Get the job context filter and set context
    logger = logging.getLogger()
    job_filter = None
    
    for handler in logger.handlers:
        for filter_obj in handler.filters:
            if isinstance(filter_obj, JobContextFilter):
                job_filter = filter_obj
                break
    
    if job_filter:
        old_job_id = getattr(job_filter, 'job_id', None)
        old_correlation_id = getattr(job_filter, 'correlation_id', None)
        
        job_filter.job_id = job_id
        job_filter.correlation_id = correlation_id
        
        try:
            yield
        finally:
            job_filter.job_id = old_job_id
            job_filter.correlation_id = old_correlation_id
    else:
        yield
```

**Core Logic Skeleton Code**

The following skeletons provide the structure for core components that learners should implement themselves, with detailed TODO comments mapping to the architectural concepts discussed above.

**Job Management Core** (`src/media_pipeline/core/job_manager.py`):

```python
"""
Core job management functionality for media processing pipeline.
Handles job lifecycle, state transitions, and resource coordination.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20

@dataclass
class ProcessingJob:
    """Represents a media processing job with all required metadata."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_file_path: str = ""
    output_specifications: List[Dict[str, Any]] = field(default_factory=list)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    webhook_url: Optional[str] = None
    progress_percentage: float = 0.0
    estimated_duration: Optional[int] = None

class JobManager:
    """
    Manages the complete lifecycle of media processing jobs.
    Coordinates between job queue, progress tracking, and worker processes.
    """
    
    def __init__(self, queue_backend, progress_store, webhook_sender):
        self.queue = queue_backend
        self.progress_store = progress_store
        self.webhook_sender = webhook_sender
        self.active_jobs: Dict[str, ProcessingJob] = {}
    
    def submit_job(self, input_file: str, output_specs: List[Dict[str, Any]], 
                   priority: JobPriority = JobPriority.NORMAL, 
                   webhook_url: Optional[str] = None) -> ProcessingJob:
        """
        Submit a new processing job to the queue.
        
        Args:
            input_file: Path to input media file
            output_specs: List of output format specifications
            priority: Job processing priority
            webhook_url: Optional webhook for status notifications
            
        Returns:
            ProcessingJob instance with assigned job_id
        """
        # TODO 1: Create ProcessingJob instance with unique job_id
        # TODO 2: Validate input file exists and is readable
        # TODO 3: Validate output specifications contain required fields
        # TODO 4: Estimate processing duration based on input file characteristics
        # TODO 5: Store job metadata in progress store
        # TODO 6: Submit job to queue with priority handling
        # TODO 7: Send initial webhook notification if webhook_url provided
        # TODO 8: Return created job instance
        pass
    
    def update_job_progress(self, job_id: str, progress_percentage: float, 
                           stage: str, details: Optional[Dict[str, Any]] = None):
        """
        Update job progress and notify interested parties.
        
        Args:
            job_id: Unique job identifier
            progress_percentage: Completion percentage (0.0 - 100.0)
            stage: Current processing stage description
            details: Optional additional progress details
        """
        # TODO 1: Validate job_id exists in active jobs
        # TODO 2: Update job progress in memory and persistent storage
        # TODO 3: Calculate estimated time remaining based on progress rate
        # TODO 4: Check for progress milestone thresholds (25%, 50%, 75%)
        # TODO 5: Send webhook notifications for milestone progress
        # TODO 6: Update job status if progress indicates completion
        # Hint: Progress updates can arrive out of order - handle gracefully
        pass
    
    def mark_job_completed(self, job_id: str, output_files: List[str]):
        """
        Mark job as completed and perform cleanup operations.
        
        Args:
            job_id: Unique job identifier
            output_files: List of generated output file paths
        """
        # TODO 1: Validate job exists and is in PROCESSING state
        # TODO 2: Update job status to COMPLETED with completion timestamp
        # TODO 3: Store output file paths in job metadata
        # TODO 4: Remove job from active jobs tracking
        # TODO 5: Send completion webhook notification with output file details
        # TODO 6: Schedule cleanup of temporary files after retention period
        pass
    
    def mark_job_failed(self, job_id: str, error_message: str, 
                       retry_eligible: bool = True):
        """
        Handle job failure with retry logic and error reporting.
        
        Args:
            job_id: Unique job identifier  
            error_message: Detailed error description
            retry_eligible: Whether job can be retried
        """
        # TODO 1: Retrieve job and increment retry_count
        # TODO 2: Check if retry_count exceeds maximum retry attempts
        # TODO 3: If retries available and retry_eligible, requeue job with exponential backoff
        # TODO 4: If no retries remaining, mark job as permanently FAILED
        # TODO 5: Send failure webhook notification with error details
        # TODO 6: Clean up any partial output files
        # TODO 7: Log failure details for debugging and monitoring
        # Hint: Exponential backoff delays should be: 1min, 2min, 4min, 8min...
        pass
```

**Language-Specific Implementation Hints**

**Python Media Processing Essentials:**

- Use `Pillow.Image.MAX_IMAGE_PIXELS = None` to handle large images, but implement custom size limits to prevent memory exhaustion
- Install FFmpeg system package: `apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)  
- Use `subprocess.run()` with `capture_output=True` and `text=True` for FFmpeg integration
- Implement timeout handling with `subprocess.run(timeout=seconds)` to prevent hung processes
- Use `pathlib.Path` for cross-platform file path handling instead of string concatenation
- Redis connection pooling: `redis.ConnectionPool(max_connections=20)` prevents connection exhaustion

**Memory Management Patterns:**

- Process images in chunks for large files: read → process → write → release memory
- Use `gc.collect()` after processing large files to force garbage collection
- Monitor memory usage with `psutil.Process().memory_info().rss` and fail fast when limits exceeded  
- Implement file streaming: `with open(file, 'rb') as f: for chunk in iter(lambda: f.read(8192), b''):`

**Error Handling Strategies:**

- Catch `PIL.Image.DecompressionBombError` for oversized images
- Handle `subprocess.CalledProcessError` for FFmpeg failures and parse stderr for specific error types
- Use `try/except/finally` blocks to ensure temporary file cleanup even when processing fails
- Implement circuit breaker pattern for external services (webhook delivery, storage APIs)

**Milestone Checkpoint**

After implementing the core infrastructure, verify the following behaviors:

**Configuration and Logging Verification:**
```bash
# Test configuration loading
python -c "from media_pipeline.utils.config import init_config; print(init_config('config/development.yaml'))"

# Verify logging output includes job context
python -c "
from media_pipeline.utils.logging import setup_logging, job_logging_context
import logging
setup_logging(debug=True)
with job_logging_context('test-job-123'):
    logging.info('Test message with job context')
"
```

**Expected Output Indicators:**
- Configuration loads without errors and displays all nested settings
- Log messages include job_id and correlation_id in structured format
- No import errors when loading core modules

**Common Setup Issues:**
- **ModuleNotFoundError**: Ensure `PYTHONPATH` includes `src/` directory or install package in development mode
- **FFmpeg not found**: Install system FFmpeg package, verify with `ffmpeg -version`
- **Redis connection errors**: Start Redis server with `redis-server` or use Docker container
- **Permission errors**: Ensure write access to configured storage and log directories

The infrastructure code provides a solid foundation that handles configuration management, logging, and error handling patterns. This allows learners to focus on implementing the core media processing algorithms and job coordination logic without getting distracted by supporting infrastructure concerns.


## Goals and Non-Goals

> **Milestone(s):** All milestones (1-3) as these requirements drive the entire system design and implementation scope

### Mental Model: Restaurant Menu Planning

Think of defining system goals like planning a restaurant menu. A successful restaurant doesn't try to serve every possible dish—instead, it carefully selects a focused set of offerings that it can execute exceptionally well with its available kitchen staff, equipment, and ingredients. The chef explicitly decides what goes on the menu (functional requirements), sets quality standards for each dish (non-functional requirements), and importantly, decides what NOT to serve (non-goals) to maintain focus and quality.

Our media processing pipeline follows this same principle. We must clearly define what we will build, how well it must perform, and what we intentionally exclude to ensure we can deliver a high-quality, maintainable system within our resource constraints.

### Functional Requirements

The **functional requirements** define the core capabilities our media processing pipeline must deliver. These represent the essential features that users directly interact with and depend upon for their media processing workflows.

> **Decision: Core Media Processing Operations**
> - **Context**: Users need to process various media types with different output requirements, from simple image resizing to complex video transcoding workflows
> - **Options Considered**: 
>   1. Image-only processing system with simple resize operations
>   2. Video-only transcoding service focusing on streaming formats
>   3. Comprehensive media pipeline supporting both images and videos with advanced features
> - **Decision**: Comprehensive media pipeline with full image and video processing capabilities
> - **Rationale**: Modern applications require both image and video processing, and building separate systems would create operational complexity and code duplication. A unified pipeline provides better resource utilization and simpler client integration.
> - **Consequences**: Increased system complexity requiring careful resource management, but provides complete media processing solution that scales with user needs

| Capability Category | Specific Requirements | Input Formats | Output Formats | Configuration Options |
|---------------------|----------------------|---------------|----------------|---------------------|
| **Image Processing** | Resize with aspect ratio preservation | JPEG, PNG, WebP, GIF | JPEG, PNG, WebP, AVIF | Target dimensions, interpolation algorithm, quality settings |
| | Format conversion with quality control | | | Compression level, progressive encoding, color space |
| | Thumbnail generation with smart cropping | | | Standard sizes, center-crop vs smart-crop, preview quality |
| | EXIF metadata handling | | | Preserve, strip, or selectively copy metadata fields |
| **Video Processing** | Multi-format transcoding | MP4, AVI, MOV, WebM, MKV | MP4, WebM, HLS, DASH | Codec selection, bitrate, resolution, frame rate |
| | Adaptive bitrate variant generation | | | Multiple quality levels, resolution ladder, segment duration |
| | Video thumbnail extraction | | JPEG, PNG, WebP | Time offset, frame selection, thumbnail size |
| | Audio track handling | | | Audio codec, bitrate, channel configuration, synchronization |
| **Job Management** | Asynchronous processing with priority queuing | All supported formats | All output formats | Priority levels, processing timeout, resource allocation |
| | Progress tracking and status reporting | | | Real-time updates, stage-based progress, time estimation |
| | Webhook notifications for status changes | | | Completion, failure, progress milestones, retry events |
| | Error handling with automatic retry logic | | | Exponential backoff, retry limits, dead letter handling |

The system must support **batch processing operations** where multiple output variants are generated from a single input file. For example, a single uploaded image might produce thumbnails at three different sizes, convert to WebP format for web display, and generate a high-quality JPEG for download—all as part of one `ProcessingJob` with multiple output specifications.

**Resource-aware scheduling** ensures that memory-intensive video transcoding jobs don't overwhelm the system while lightweight image resizing operations continue processing. The job queue must intelligently distribute work based on worker process capabilities and current system resource utilization.

> **Critical Design Insight**: The functional requirements prioritize **flexibility over simplicity**. Rather than building separate specialized tools, we create a unified pipeline that handles diverse media processing scenarios through configurable job specifications. This approach serves both simple use cases (single image resize) and complex workflows (multi-variant video transcoding with thumbnails).

### Non-Functional Requirements

The **non-functional requirements** establish the quality standards and operational characteristics that make our media processing pipeline suitable for production environments. These requirements directly impact architecture decisions and implementation strategies.

> **Decision: Performance and Scalability Targets**
> - **Context**: Media processing is computationally intensive with highly variable workloads, from small profile images to large video files requiring hours of processing time
> - **Options Considered**:
>   1. Single-node processing with simple scaling through instance size increases
>   2. Horizontally scalable worker pool with shared job queue
>   3. Serverless processing using cloud functions with automatic scaling
> - **Decision**: Horizontally scalable worker pool architecture with intelligent job distribution
> - **Rationale**: Provides predictable costs, allows optimization for different media types, and enables resource-aware scheduling that serverless cannot provide. Single-node scaling hits memory and I/O limits with large video files.
> - **Consequences**: Requires distributed system complexity but provides linear scaling and cost control

| Requirement Category | Specification | Measurement Method | Business Justification |
|---------------------|--------------|-------------------|----------------------|
| **Processing Performance** | Image resize operations complete within 5 seconds for files up to 50MB | End-to-end job completion time | User experience for real-time workflows |
| | Video transcoding processes at 2x real-time speed for 1080p content | FFmpeg processing rate measurement | Acceptable wait times for uploaded content |
| | System processes 1000 concurrent image jobs | Active job queue depth monitoring | Peak load handling for content publishing |
| | 95th percentile job completion time under 30 seconds for standard operations | Job duration histogram analysis | Predictable processing times for SLA commitments |
| **Reliability and Availability** | 99.5% job success rate excluding invalid input files | Success/failure ratio tracking | Minimize user frustration from processing failures |
| | Failed jobs automatically retry with exponential backoff | Retry attempt logging and outcome tracking | Handles transient failures without manual intervention |
| | System recovers from worker crashes within 30 seconds | Worker health monitoring and replacement time | Maintains processing capacity during failures |
| | Zero data loss for accepted processing jobs | Job persistence and completion verification | Protects user content and maintains service trust |
| **Scalability Characteristics** | Linear scaling to 50 worker processes per node | Throughput measurement across worker counts | Handles traffic growth through horizontal scaling |
| | Supports 10,000 queued jobs without performance degradation | Queue depth vs processing latency correlation | Manages traffic spikes and batch processing loads |
| | Worker processes scale up/down based on queue depth | Auto-scaling trigger timing and effectiveness | Cost optimization during variable load periods |
| | Storage scales to 100TB of processing temp files | Disk usage monitoring and cleanup verification | Supports large-scale video processing operations |
| **Resource Management** | Individual jobs limited to 8GB memory usage | Process memory monitoring and enforcement | Prevents single large jobs from destabilizing system |
| | Temp file cleanup within 1 hour of job completion | File system monitoring and cleanup verification | Maintains disk space availability for ongoing operations |
| | CPU utilization averaged across workers stays below 80% | System resource monitoring and alert thresholds | Maintains responsive performance during peak loads |
| | Network bandwidth utilization for job coordination under 100Mbps | Inter-component traffic measurement | Ensures job queue doesn't become network bottleneck |

**Progress tracking accuracy** represents a particularly challenging non-functional requirement. Video transcoding progress estimation requires stage-based reporting rather than simple percentage completion, since FFmpeg progress varies significantly based on content complexity and encoding parameters. The system must provide meaningful progress updates that help users understand processing status without making overly precise time commitments.

> **Decision: Consistency vs Availability Trade-offs**
> - **Context**: Media processing jobs represent significant computational work that shouldn't be lost, but the system must remain responsive during partial failures
> - **Options Considered**:
>   1. Strict consistency requiring all components to acknowledge job state changes
>   2. Eventual consistency with job state synchronization through message queue
>   3. Hybrid approach with strong consistency for job submission, eventual consistency for progress updates
> - **Decision**: Hybrid consistency model with durable job submission and eventually consistent progress reporting
> - **Rationale**: Job submission must be durable to prevent lost work, but progress updates can be eventually consistent since they're informational rather than critical for system correctness
> - **Consequences**: Simplified client integration with reliable job submission, but progress updates may occasionally show stale information

**Security and compliance requirements** focus on protecting user content and metadata privacy. The system must provide configurable EXIF metadata stripping to remove potentially sensitive location and device information from processed images. All temporary files must be securely deleted after processing completion, and webhook notifications must use signature verification to prevent spoofing attacks.

### Explicit Non-Goals

The **explicit non-goals** define functionality that we intentionally exclude from this media processing pipeline implementation. These boundaries prevent scope creep and ensure we can deliver high-quality core functionality rather than attempting to solve every possible media processing challenge.

> **Decision: Content Storage and Management Exclusion**
> - **Context**: Many media processing systems also provide content storage, CDN integration, and digital asset management features
> - **Options Considered**:
>   1. Full-featured media management platform with storage, metadata, and delivery
>   2. Processing-only service that integrates with external storage systems
>   3. Hybrid approach with basic storage and advanced processing capabilities
> - **Decision**: Processing-only service with clear storage integration points
> - **Rationale**: Content storage, CDN management, and digital asset management represent separate domain expertise areas with different scaling characteristics and operational requirements
> - **Consequences**: Simpler system design focused on processing excellence, but requires clients to manage their own storage and delivery infrastructure

| Non-Goal Category | Specific Exclusions | Rationale | Alternative Solutions |
|-------------------|-------------------|-----------|---------------------|
| **Content Management** | Long-term storage of original or processed media files | Storage requirements vary dramatically by use case and compliance needs | Integrate with S3, Google Cloud Storage, or local file systems |
| | Digital asset management with metadata search and organization | Requires different expertise in search indexing and content categorization | Use dedicated DAM systems like Adobe Experience Manager or custom solutions |
| | Content delivery network integration and optimization | CDN management requires separate operational expertise and provider relationships | Integrate processed outputs with CloudFlare, AWS CloudFront, or similar |
| | User authentication and access control for media files | Authentication adds complexity unrelated to media processing core competency | Handle auth at API gateway or application layer before job submission |
| **Advanced Processing Features** | AI-powered content analysis and automatic tagging | Machine learning model management requires different infrastructure and expertise | Integrate with cloud AI services or specialized content analysis tools |
| | Real-time live streaming and broadcast processing | Live streaming has different latency, resource, and reliability requirements than batch processing | Use dedicated streaming platforms like Wowza, OBS, or cloud streaming services |
| | Advanced video editing operations beyond transcoding | Complex editing requires timeline management, effects processing, and interactive workflows | Integrate with video editing APIs or desktop applications for advanced editing |
| | Content moderation and inappropriate content detection | Moderation requires specialized models, human review workflows, and policy management | Use dedicated moderation services like AWS Rekognition or manual review processes |
| **Specialized Format Support** | Legacy or proprietary media formats with complex licensing requirements | Specialized formats require additional dependencies and potentially expensive licensing | Handle conversion to standard formats before submitting to processing pipeline |
| | Professional broadcast formats and workflows | Broadcast requirements include specialized metadata, color spaces, and workflow integration | Use professional broadcast tools and integrate outputs with standard pipeline |
| | 3D media processing, VR/AR content optimization | Immersive media has different processing requirements and specialized libraries | Develop separate processing pipeline optimized for 3D workflows |
| | Medical or scientific imaging format support | Specialized domains require DICOM, microscopy format expertise | Use domain-specific processing tools and convert outputs to standard formats |
| **Infrastructure and Operations** | Built-in monitoring, alerting, and observability dashboards | Monitoring requirements vary by deployment environment and operational preferences | Integrate with existing monitoring stack using structured logging and metrics |
| | Automatic cloud provider provisioning and infrastructure management | Infrastructure automation requires deep cloud-specific knowledge and operational procedures | Use existing infrastructure-as-code tools like Terraform or cloud-native auto-scaling |
| | Multi-tenant processing with resource isolation and billing | Multi-tenancy adds complexity in resource accounting, security isolation, and billing integration | Deploy separate instances per tenant or add tenancy features in future versions |
| | Geographic distribution and edge processing capabilities | Edge deployment requires different architectural patterns and content synchronization strategies | Deploy multiple regional instances or integrate with edge computing platforms |

**Real-time processing requirements** represent a significant non-goal that affects architecture decisions. While the system provides progress tracking and webhook notifications, it does not attempt to provide sub-second processing latency or real-time streaming capabilities. Batch processing optimization allows for better resource utilization and more reliable error recovery compared to real-time processing constraints.

**Custom processing algorithm development** falls outside our scope. The system integrates with proven libraries like Pillow for image processing and FFmpeg for video transcoding rather than implementing custom algorithms. This decision ensures reliability and compatibility while focusing development effort on job orchestration, progress tracking, and error recovery—the areas where custom logic provides the most value.

> **Critical Boundary Decision**: We explicitly exclude **content-aware processing features** like automatic cropping based on face detection, intelligent compression based on content analysis, or quality optimization based on viewing context. These features require machine learning integration and significantly more complex configuration management. Clients needing content-aware features should implement them as separate preprocessing steps or integrate with specialized AI services.

The non-goals serve as architectural constraints that simplify system design and clarify integration points. By excluding storage management, we avoid the complexity of data durability guarantees and focus on processing reliability. By excluding real-time requirements, we can optimize for throughput and implement simpler retry mechanisms. By excluding content analysis, we avoid the operational complexity of managing machine learning models and their dependencies.

These boundaries are not permanent limitations but represent conscious decisions about the current implementation scope. Future versions might incorporate some excluded features as optional components or extension points, but the initial system focuses on delivering excellent core media processing capabilities with clear integration patterns for external systems to provide the excluded functionality.

### Implementation Guidance

This implementation guidance provides practical direction for translating the goals and requirements into working code structure and development milestones.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|--------------|----------------|
| **Job Queue** | Redis with simple pub/sub | Celery with RabbitMQ broker |
| **Progress Tracking** | Redis hash with periodic updates | WebSocket server with real-time updates |
| **Webhook Delivery** | Python requests with basic retry | Task queue with exponential backoff |
| **Configuration Management** | JSON config files | Environment-based config with validation |
| **Logging** | Python logging to files | Structured logging with correlation IDs |
| **Metrics** | Simple counters in Redis | Prometheus metrics with custom collectors |

**B. Recommended File Structure:**

```python
media_pipeline/
├── config/
│   ├── __init__.py
│   ├── app_config.py          # AppConfig, RedisConfig, StorageConfig
│   └── settings.py            # Configuration loading and validation
├── core/
│   ├── __init__.py
│   ├── job_manager.py         # ProcessingJob operations
│   ├── progress_tracker.py    # Progress reporting and webhook delivery
│   └── types.py              # JobStatus, JobPriority enums
├── workers/
│   ├── __init__.py
│   ├── base_worker.py        # Abstract worker interface
│   ├── image_worker.py       # Image processing implementation
│   └── video_worker.py       # Video transcoding implementation
├── api/
│   ├── __init__.py
│   ├── handlers.py           # HTTP request handlers
│   └── validation.py         # Input validation and sanitization
├── utils/
│   ├── __init__.py
│   ├── logging.py            # Structured logging setup
│   └── storage.py            # File system utilities
└── tests/
    ├── integration/          # End-to-end job processing tests
    ├── unit/                 # Component unit tests
    └── fixtures/             # Sample media files for testing
```

**C. Infrastructure Starter Code:**

```python
# config/app_config.py - Complete configuration management
from dataclasses import dataclass
from typing import Optional
import json
import os

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    def connection_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class StorageConfig:
    base_path: str = "/tmp/media_processing"
    temp_path: str = "/tmp/media_processing/temp"
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    
    def __post_init__(self):
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)

@dataclass
class ProcessingConfig:
    max_workers: int = 4
    job_timeout: int = 3600  # 1 hour
    retry_attempts: int = 3
    webhook_timeout: int = 30
    
@dataclass
class AppConfig:
    redis: RedisConfig
    storage: StorageConfig
    processing: ProcessingConfig
    debug: bool = False

def init_config(config_path: Optional[str] = None) -> AppConfig:
    """Initialize application configuration from file or environment variables."""
    # TODO 1: Load config from JSON file if config_path provided
    # TODO 2: Override with environment variables where present
    # TODO 3: Validate configuration values and ranges
    # TODO 4: Initialize storage directories
    # TODO 5: Return complete AppConfig instance
    pass

# utils/logging.py - Complete logging infrastructure
import logging
import json
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add job context if present
        if hasattr(record, 'job_id'):
            log_entry['job_id'] = record.job_id
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
            
        return json.dumps(log_entry)

def setup_logging(debug: bool = False):
    """Configure structured logging system."""
    level = logging.DEBUG if debug else logging.INFO
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    # Suppress noisy third-party logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

@contextmanager
def job_logging_context(job_id: str, correlation_id: Optional[str] = None):
    """Context manager for job-aware logging."""
    # TODO 1: Get current logger
    # TODO 2: Create logging adapter with job_id and correlation_id
    # TODO 3: Yield adapter for use in job processing
    # TODO 4: Clean up context on exit
    pass
```

**D. Core Logic Skeleton Code:**

```python
# core/types.py - Core data types
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobPriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20

@dataclass
class ProcessingJob:
    job_id: str
    input_file_path: str
    output_specifications: List[Dict[str, Any]]
    priority: JobPriority
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    webhook_url: Optional[str] = None
    progress_percentage: float = 0.0
    estimated_duration: Optional[int] = None

# core/job_manager.py - Job lifecycle management
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from .types import ProcessingJob, JobStatus, JobPriority

class JobManager:
    """Manages processing job lifecycle and state transitions."""
    
    def __init__(self, redis_client, storage_config):
        self.redis = redis_client
        self.storage = storage_config
    
    def submit_job(self, input_file: str, output_specs: List[Dict[str, Any]], 
                   priority: JobPriority, webhook_url: Optional[str] = None) -> ProcessingJob:
        """Create and queue new processing job."""
        # TODO 1: Generate unique job_id using uuid4
        # TODO 2: Validate input_file exists and is readable
        # TODO 3: Validate output_specs format and required fields
        # TODO 4: Create ProcessingJob instance with provided parameters
        # TODO 5: Serialize job to Redis with job:{job_id} key
        # TODO 6: Add job to priority queue based on priority level
        # TODO 7: Return created ProcessingJob instance
        pass
    
    def update_job_progress(self, job_id: str, progress_percentage: float, 
                           stage: str, details: Dict[str, Any]):
        """Update job progress and send notifications."""
        # TODO 1: Retrieve job from Redis using job_id
        # TODO 2: Update progress_percentage and add stage information
        # TODO 3: Calculate estimated_duration based on progress rate
        # TODO 4: Save updated job state to Redis
        # TODO 5: Send webhook notification if configured
        # TODO 6: Log progress update with job context
        pass
    
    def mark_job_completed(self, job_id: str, output_files: List[str]):
        """Finalize completed job and cleanup."""
        # TODO 1: Retrieve job from Redis and validate current status
        # TODO 2: Update job status to COMPLETED with completion timestamp
        # TODO 3: Store output_files list in job record
        # TODO 4: Send completion webhook notification
        # TODO 5: Schedule cleanup of temporary files
        # TODO 6: Update job completion metrics
        pass
    
    def mark_job_failed(self, job_id: str, error_message: str, retry_eligible: bool):
        """Handle job failure with retry logic."""
        # TODO 1: Retrieve job from Redis and increment retry_count
        # TODO 2: Check if retry_eligible and retry_count < max_retry_attempts
        # TODO 3: If retryable, requeue job with exponential backoff delay
        # TODO 4: If not retryable, mark job as FAILED permanently
        # TODO 5: Store error_message and send failure webhook notification
        # TODO 6: Clean up temporary files for failed job
        pass
```

**E. Language-Specific Hints:**

- **Redis Integration**: Use `redis-py` library with connection pooling for job queue operations. Store jobs as JSON-serialized strings with TTL for automatic cleanup.
- **File Processing**: Use `pathlib.Path` for cross-platform file path handling. Always use context managers (`with open()`) for file operations to ensure proper cleanup.
- **Process Management**: Use `subprocess.run()` with timeout parameters for FFmpeg integration. Capture stdout/stderr for progress parsing and error reporting.
- **Error Handling**: Create custom exception classes for different failure types (transient vs permanent). Use try/except blocks with specific exception types rather than bare `except:` clauses.
- **Async Operations**: Consider `asyncio` for webhook delivery and I/O operations, but keep media processing in separate worker processes to avoid blocking the event loop.

**F. Milestone Checkpoints:**

**After Requirements Definition:**
- Run `python -m pytest tests/test_requirements.py -v` to verify requirement validation logic
- Expected: All functional requirement validation passes, non-functional requirement measurements return reasonable defaults
- Manual verification: Create sample `ProcessingJob` instances with different priorities and validate serialization/deserialization

**After Goal Validation Implementation:**
- Test command: `python scripts/validate_goals.py --config config/test.json`
- Expected output: "✓ All functional requirements validated", "✓ Non-functional requirements initialized", "✓ Non-goals properly excluded"
- Manual verification: Attempt to submit job with excluded feature (should raise clear error), verify job priority ordering in queue

**G. Common Implementation Pitfalls:**

| Pitfall | Symptoms | Root Cause | Solution |
|---------|----------|------------|----------|
| **Requirement Validation Missing** | Jobs fail mysteriously during processing | No upfront validation of input parameters | Add comprehensive validation in `submit_job()` before queuing |
| **Priority Queue Not Working** | High priority jobs wait behind low priority | Using simple FIFO queue instead of priority queue | Implement Redis sorted sets with priority scores for job ordering |
| **Non-Functional Requirements Not Measured** | Performance degrades without notice | No monitoring of actual vs required performance | Add metrics collection for all non-functional requirements |
| **Goal Scope Creep** | System becomes complex and unstable | Adding features that should be non-goals | Regularly review and reject features not in functional requirements |

The implementation should start with requirement validation and configuration management before building any processing capabilities. This foundation ensures all subsequent components operate within the defined goals and can be tested against concrete acceptance criteria.


## High-Level Architecture

> **Milestone(s):** All milestones (1-3) as this architectural foundation supports image processing, video transcoding, and job queue components

### Mental Model: Modern Manufacturing Assembly Line

Think of our media processing pipeline as a modern manufacturing assembly line, but instead of building cars or electronics, we're transforming raw media files into optimized, web-ready content. Just like Toyota's production system revolutionized manufacturing with just-in-time delivery and quality control at every stage, our pipeline processes media files through specialized stations with real-time monitoring and automatic error recovery.

The **API gateway** acts as the customer service desk where orders (processing jobs) are received, validated, and entered into the system with detailed specifications. The **job queue** functions as the production control system that schedules work orders and routes them to the appropriate assembly stations based on priority and worker availability. Each **worker process** represents a specialized assembly station equipped with specific tools—some stations excel at image manipulation using precision instruments (Pillow), while others handle complex video assembly using industrial-grade equipment (FFmpeg). The **storage layer** serves as both the incoming materials warehouse and the finished goods depot, organizing raw inputs and polished outputs with proper inventory tracking.

What makes this assembly line particularly sophisticated is its ability to handle rush orders (high-priority jobs), automatically retry failed operations when a station encounters problems, and provide real-time progress updates to customers waiting for their custom media products. Unlike traditional assembly lines that process identical widgets, our pipeline handles diverse media formats and transforms them according to unique specifications for each job.

### Component Overview

The media processing pipeline consists of four primary architectural layers, each with distinct responsibilities and clear interfaces for maximum modularity and testability.

#### API Gateway Layer

The **API Gateway** serves as the system's primary entry point, handling all external client interactions and providing a clean REST interface for job submission and status monitoring. This component owns the responsibility for request validation, authentication, rate limiting, and translating external API contracts into internal job specifications.

| Component | Primary Responsibility | Key Operations | External Dependencies |
|-----------|----------------------|----------------|---------------------|
| REST API Server | HTTP request handling and response formatting | Job submission, status queries, file uploads | Flask/FastAPI framework |
| Request Validator | Input sanitization and schema validation | Media file type checking, parameter validation | JSON schema libraries |
| Authentication Handler | API key and webhook signature verification | Token validation, signature generation | JWT libraries |
| Rate Limiter | Traffic control and abuse prevention | Request counting, backoff enforcement | Redis for distributed counting |

The gateway operates stateless by design, allowing horizontal scaling through simple load balancing. All persistent state lives in the job queue and storage layers, while the gateway focuses purely on protocol translation and input validation.

#### Job Queue Layer

The **Job Queue** implements the core orchestration logic, managing the lifecycle of processing jobs from submission through completion. This layer abstracts away the complexities of distributed task scheduling and provides reliable message delivery semantics with exactly-once processing guarantees.

| Component | Primary Responsibility | Key Operations | Storage Requirements |
|-----------|----------------------|----------------|-------------------|
| Job Manager | Job lifecycle coordination | Job creation, state transitions, cleanup | Job metadata persistence |
| Priority Scheduler | Work distribution based on priority and resources | Queue ordering, worker assignment | Priority queue implementation |
| Progress Tracker | Real-time status monitoring and reporting | Progress updates, time estimation | Progress state storage |
| Webhook Dispatcher | External notification delivery | HTTP callbacks, retry logic | Webhook delivery logs |

The queue layer maintains strong consistency for job state while providing eventual consistency for progress updates. This design ensures that jobs never get lost while allowing for high-throughput progress reporting without blocking core operations.

> **Decision: Redis vs RabbitMQ for Message Queuing**
> - **Context**: Need reliable message delivery with priority support and progress tracking
> - **Options Considered**: Redis with sorted sets, RabbitMQ with priority queues, Amazon SQS with message attributes
> - **Decision**: Redis with sorted sets for primary queue, RabbitMQ for complex routing scenarios
> - **Rationale**: Redis provides atomic operations for priority management and excellent performance for frequent progress updates. RabbitMQ adds complexity but offers superior durability guarantees for critical jobs.
> - **Consequences**: Enables sub-second job scheduling with atomic priority updates but requires Redis persistence configuration and monitoring

| Queue Option | Pros | Cons | Use Case Fit |
|--------------|------|------|-------------|
| Redis Sorted Sets | Atomic priority operations, excellent performance, built-in progress storage | Less durable than dedicated message brokers, memory-based | High-throughput processing with frequent priority changes |
| RabbitMQ Priority Queues | Strong durability, mature ecosystem, complex routing | Higher latency for simple operations, more operational overhead | Mission-critical jobs requiring guaranteed delivery |
| Amazon SQS | Fully managed, infinite scale, integrated with AWS | Vendor lock-in, limited priority levels, higher per-message cost | Cloud-native deployments with moderate priority requirements |

#### Worker Process Layer

**Worker Processes** execute the actual media transformation logic, providing isolated execution environments with resource monitoring and automatic recovery capabilities. Each worker specializes in specific media types while sharing common infrastructure for job acquisition, progress reporting, and error handling.

| Worker Type | Specialization | Resource Requirements | Typical Processing Time |
|-------------|----------------|-------------------|---------------------|
| Image Worker | JPEG/PNG/WebP processing, thumbnail generation | 512MB RAM, moderate CPU | 1-30 seconds per job |
| Video Worker | FFmpeg-based transcoding, segment generation | 2GB+ RAM, high CPU/disk I/O | 30 seconds to 2+ hours per job |
| Batch Worker | Multi-file operations, archive processing | Variable based on batch size | Minutes to hours |

Workers implement a polling model with intelligent backoff, claiming jobs from the priority queue based on their capabilities and current resource utilization. Each worker process runs in isolation with configurable memory and CPU limits to prevent resource exhaustion from affecting other jobs.

#### Storage Layer

The **Storage Layer** provides persistent storage for input files, intermediate processing artifacts, and final output files with proper lifecycle management and cleanup policies. This layer abstracts storage implementation details while providing consistent interfaces for file operations across different backend systems.

| Storage Component | Purpose | Consistency Requirements | Cleanup Policy |
|------------------|---------|------------------------|---------------|
| Input File Storage | Raw uploaded media files | Strong consistency for uploads | Retain until processing complete |
| Work Directory Manager | Temporary processing workspace | Local consistency per worker | Immediate cleanup after job completion |
| Output File Storage | Processed media deliverables | Strong consistency for results | Configurable retention policy |
| Metadata Database | Job state and processing logs | Strong consistency for job state | Archive after job expiration |

### Request Processing Flow

The end-to-end processing flow orchestrates multiple components to transform uploaded media files into optimized outputs while providing comprehensive monitoring and error recovery.

#### Job Submission Phase

When a client submits a new processing job, the request flows through several validation and preparation stages before entering the execution queue:

1. **Request Reception**: The API gateway receives an HTTP POST request containing the input media file, output specifications, priority level, and optional webhook URL for notifications.

2. **Input Validation**: The request validator examines the uploaded file to determine media type, validates format support, checks file size limits, and ensures output specifications are achievable given the input characteristics.

3. **Job Creation**: The job manager generates a unique job identifier, creates a `ProcessingJob` record with status `JobStatus.PENDING`, and calculates initial processing time estimates based on file size and requested operations.

4. **Queue Insertion**: The priority scheduler inserts the job into the appropriate queue based on the requested `JobPriority` level and current system load, using atomic Redis operations to maintain queue consistency.

5. **Client Response**: The API gateway returns the job identifier and initial status to the client, allowing them to track progress through subsequent status queries.

| Phase | Duration | Failure Modes | Recovery Strategy |
|-------|----------|--------------|------------------|
| File Upload | 1-60 seconds | Network interruption, file corruption | Client retry with resumable uploads |
| Validation | <1 second | Invalid format, unsupported operations | Immediate error response with specific details |
| Job Creation | <100ms | Database connectivity, resource exhaustion | Automatic retry with exponential backoff |
| Queue Insertion | <50ms | Redis connectivity, memory limits | Failover to secondary queue or delayed retry |

#### Job Execution Phase

Once queued, jobs wait for available worker processes that match their resource requirements and processing capabilities:

1. **Worker Job Acquisition**: Workers poll the priority queue using Redis ZPOPMIN operations to atomically claim the highest-priority job matching their capabilities, updating job status to `JobStatus.PROCESSING`.

2. **Resource Preparation**: The assigned worker creates an isolated working directory, downloads the input file from storage, and initializes processing tools (Pillow for images, FFmpeg for videos) with job-specific configuration parameters.

3. **Processing Execution**: The worker executes the media transformation pipeline, reporting progress updates at regular intervals using `update_job_progress()` calls that broadcast status changes to monitoring systems and trigger webhook notifications.

4. **Output Generation**: Completed processing artifacts are uploaded to the output storage layer with proper naming conventions and metadata tags, ensuring atomic replacement of any existing files.

5. **Job Completion**: The worker calls `mark_job_completed()` to update the job status, trigger final webhook notifications, and schedule cleanup of temporary resources.

The execution phase implements comprehensive error handling with automatic retries for transient failures and clear failure classification for permanent errors that require human intervention.

#### Progress Reporting and Notification Flow

Throughout processing, the system maintains real-time visibility into job status and estimated completion times:

1. **Stage-Based Progress**: Workers report progress using discrete stages rather than time-based percentages, providing more accurate estimates for complex operations like video transcoding where processing speed varies significantly across different content sections.

2. **Webhook Delivery**: The webhook dispatcher sends HTTP POST notifications to client-provided URLs for all significant job state changes, implementing retry logic with exponential backoff for failed delivery attempts.

3. **Status Queries**: Clients can query job status at any time through the API gateway, receiving current progress percentage, estimated completion time, and detailed stage information without impacting processing performance.

| Notification Type | Trigger Event | Payload Contents | Retry Policy |
|-------------------|--------------|-----------------|-------------|
| Job Started | Status change to PROCESSING | Job ID, worker assignment, estimated duration | 3 retries over 15 minutes |
| Progress Update | Configurable percentage milestones | Current progress, stage description, time remaining | 2 retries over 5 minutes |
| Job Completed | Successful processing finish | Job ID, output file URLs, final metrics | 5 retries over 1 hour |
| Job Failed | Permanent processing failure | Job ID, error details, retry eligibility | 5 retries over 1 hour |

### Recommended Project Structure

The codebase organization reflects the architectural layers while promoting clear separation of concerns and enabling independent testing of each component.

#### Top-Level Directory Organization

The project structure follows Python packaging best practices with clear separation between application logic, configuration, and external interfaces:

```
media-processing-pipeline/
├── src/
│   └── media_processor/           # Main application package
│       ├── __init__.py
│       ├── api/                   # API Gateway Layer
│       ├── queue/                 # Job Queue Layer
│       ├── workers/               # Worker Process Layer
│       ├── storage/               # Storage Layer
│       ├── models/                # Shared data models
│       └── common/                # Cross-cutting utilities
├── config/                        # Configuration files
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
├── migrations/                    # Database schema migrations
├── tests/                         # Test organization mirrors src/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                  # Sample media files
├── scripts/                       # Deployment and maintenance scripts
├── docker/                        # Container configurations
└── docs/                          # Architecture and API documentation
```

#### API Gateway Module Structure

The API layer implements a clean separation between HTTP handling, business logic, and external integrations:

```
src/media_processor/api/
├── __init__.py
├── app.py                         # FastAPI application factory
├── routes/
│   ├── __init__.py
│   ├── jobs.py                    # Job submission and status endpoints
│   ├── health.py                  # System health and monitoring
│   └── webhooks.py                # Webhook management endpoints
├── middleware/
│   ├── __init__.py
│   ├── authentication.py         # API key validation
│   ├── rate_limiting.py          # Request throttling
│   └── request_logging.py        # Structured request/response logging
├── schemas/
│   ├── __init__.py
│   ├── request_models.py         # Input validation schemas
│   └── response_models.py        # Output serialization schemas
└── dependencies.py               # FastAPI dependency injection setup
```

#### Job Queue Module Structure

The queue layer encapsulates all aspects of job lifecycle management and worker coordination:

```
src/media_processor/queue/
├── __init__.py
├── job_manager.py                # Core job lifecycle operations
├── priority_scheduler.py         # Queue ordering and worker assignment
├── progress_tracker.py          # Real-time progress monitoring
├── webhook_dispatcher.py        # External notification delivery
├── backends/                     # Queue implementation adapters
│   ├── __init__.py
│   ├── redis_queue.py           # Redis-based queue operations
│   └── rabbitmq_queue.py        # RabbitMQ integration (optional)
└── serializers.py               # Job data serialization formats
```

#### Worker Process Module Structure

The worker layer organizes media processing capabilities into specialized modules with shared infrastructure:

```
src/media_processor/workers/
├── __init__.py
├── base_worker.py               # Common worker infrastructure
├── worker_manager.py            # Worker process lifecycle management
├── image_worker.py              # Image processing specialization
├── video_worker.py              # Video transcoding specialization
├── processors/                  # Media processing implementations
│   ├── __init__.py
│   ├── image_processor.py       # Pillow-based image operations
│   ├── video_processor.py       # FFmpeg integration layer
│   └── thumbnail_generator.py   # Cross-format thumbnail creation
└── resource_monitor.py          # Memory and CPU usage tracking
```

#### Shared Models and Utilities Structure

Common code shared across all layers resides in dedicated modules to prevent circular dependencies:

```
src/media_processor/models/
├── __init__.py
├── job_models.py                # ProcessingJob and related entities
├── config_models.py            # AppConfig and component configurations
└── enums.py                     # JobStatus, JobPriority, and other constants

src/media_processor/common/
├── __init__.py
├── logging_setup.py             # Structured logging configuration
├── config_loader.py            # Configuration file parsing
├── exceptions.py               # Custom exception hierarchy
└── utils.py                    # Generic utility functions
```

> **Decision: Package Organization Strategy**
> - **Context**: Need to balance modularity with import simplicity while preventing circular dependencies
> - **Options Considered**: Flat structure with all modules at top level, strict layered packages, domain-driven module grouping
> - **Decision**: Layered packages with shared models and utilities at the same level
> - **Rationale**: Reflects architectural boundaries while making dependencies explicit. Shared models prevent import cycles between layers.
> - **Consequences**: Enables independent testing of each layer and clear dependency direction from API → Queue → Workers → Storage

#### Configuration and Deployment Structure

Environment-specific configuration and deployment artifacts maintain clear separation to support multiple deployment targets:

| Configuration File | Purpose | Contains | Environment |
|-------------------|---------|-----------|-------------|
| `config/development.yaml` | Local development settings | Debug logging, local Redis, relaxed timeouts | Developer workstations |
| `config/testing.yaml` | Automated test configuration | In-memory queues, mock external services | CI/CD pipelines |
| `config/production.yaml` | Production deployment settings | Cluster Redis, strict timeouts, monitoring | Production clusters |
| `docker/worker.Dockerfile` | Worker container image | Media processing tools, Python runtime | All containerized environments |

⚠️ **Pitfall: Circular Import Dependencies**

A common mistake in Python projects is creating circular imports between layers, especially when the API layer needs to import job models that also reference API-specific types. This manifests as `ImportError: cannot import name` exceptions during module loading.

**Why it's wrong**: Python cannot resolve module dependencies when they form cycles, leading to runtime import failures that only appear when specific code paths are executed.

**How to fix**: Place all shared data models (`ProcessingJob`, `AppConfig`, etc.) in a dedicated `models` package that doesn't import from other application layers. Use dependency injection at the application boundary to provide layer-specific implementations to shared models.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|--------------|----------------|
| Web Framework | Flask with Flask-RESTful | FastAPI with async support |
| Message Queue | Redis with manual job management | Celery with Redis/RabbitMQ backend |
| Configuration | Python configparser with INI files | Pydantic with YAML configuration |
| Database | SQLite with simple schema | PostgreSQL with SQLAlchemy ORM |
| File Storage | Local filesystem with organized directories | AWS S3 with boto3 integration |
| Logging | Python logging with file rotation | Structured logging with JSON output |
| Testing | unittest with manual mocks | pytest with pytest-asyncio and fixtures |
| Containerization | Single Docker container | Multi-stage builds with Alpine Linux |

#### Recommended File Structure

```
media-processing-pipeline/
├── src/
│   └── media_processor/
│       ├── __init__.py
│       ├── main.py                    # Application entry point
│       ├── config.py                  # Configuration management
│       ├── models/
│       │   ├── __init__.py
│       │   ├── job_models.py          # ProcessingJob, JobStatus, JobPriority
│       │   └── config_models.py       # AppConfig, RedisConfig, etc.
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py                 # FastAPI application setup
│       │   └── routes/
│       │       ├── __init__.py
│       │       └── jobs.py            # Job submission endpoints
│       ├── queue/
│       │   ├── __init__.py
│       │   ├── job_manager.py         # Core job operations
│       │   └── redis_queue.py         # Redis integration
│       ├── workers/
│       │   ├── __init__.py
│       │   ├── base_worker.py         # Worker base class
│       │   ├── image_worker.py        # Image processing worker
│       │   └── video_worker.py        # Video processing worker
│       └── storage/
│           ├── __init__.py
│           └── file_manager.py        # File operations
├── config/
│   ├── development.yaml
│   └── production.yaml
├── requirements.txt
├── requirements-dev.txt
└── tests/
    ├── __init__.py
    ├── test_job_manager.py
    └── fixtures/
        ├── sample.jpg
        └── sample.mp4
```

#### Infrastructure Starter Code

**Configuration Management (src/media_processor/config.py)**

```python
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

@dataclass
class StorageConfig:
    base_path: str = "./storage"
    temp_path: str = "./temp"
    max_file_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class ProcessingConfig:
    max_workers: int = 4
    job_timeout: int = 3600  # 1 hour
    retry_attempts: int = 3
    webhook_timeout: int = 30

@dataclass
class AppConfig:
    redis: RedisConfig
    storage: StorageConfig
    processing: ProcessingConfig
    debug: bool = False

def init_config(config_path: str) -> AppConfig:
    """Initialize application configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return AppConfig(
        redis=RedisConfig(**config_data.get('redis', {})),
        storage=StorageConfig(**config_data.get('storage', {})),
        processing=ProcessingConfig(**config_data.get('processing', {})),
        debug=config_data.get('debug', False)
    )
```

**Logging Setup (src/media_processor/common/logging_setup.py)**

```python
import logging
import sys
from contextlib import contextmanager
from typing import Optional

def setup_logging(debug: bool = False) -> None:
    """Configure structured logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(message)s [%(filename)s:%(lineno)d]"
    )
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('media_processor.log')
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('redis').setLevel(logging.WARNING)

@contextmanager
def job_logging_context(job_id: str, correlation_id: Optional[str] = None):
    """Context manager for job-aware logging."""
    logger = logging.getLogger()
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.job_id = job_id
        record.correlation_id = correlation_id or job_id
        return record
    
    logging.setLogRecordFactory(record_factory)
    try:
        yield logger
    finally:
        logging.setLogRecordFactory(old_factory)
```

#### Core Logic Skeleton Code

**Job Manager Core (src/media_processor/queue/job_manager.py)**

```python
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from ..models.job_models import ProcessingJob, JobStatus, JobPriority

logger = logging.getLogger(__name__)

class JobManager:
    """Manages job lifecycle from creation to completion."""
    
    def __init__(self, redis_client, storage_manager):
        self.redis = redis_client
        self.storage = storage_manager
    
    def submit_job(
        self, 
        input_file: str, 
        output_specs: List[dict], 
        priority: JobPriority, 
        webhook_url: Optional[str] = None
    ) -> ProcessingJob:
        """Create and queue new processing job."""
        # TODO 1: Generate unique job_id using uuid.uuid4()
        # TODO 2: Validate input_file exists in storage
        # TODO 3: Create ProcessingJob instance with PENDING status
        # TODO 4: Store job metadata in Redis hash
        # TODO 5: Add job to priority queue using Redis ZADD with priority score
        # TODO 6: Log job creation with structured metadata
        # Hint: Use job priority value as Redis sorted set score for automatic ordering
        pass
    
    def update_job_progress(
        self, 
        job_id: str, 
        progress_percentage: float, 
        stage: str, 
        details: Optional[str] = None
    ) -> None:
        """Update job progress and send notifications."""
        # TODO 1: Retrieve current job from Redis
        # TODO 2: Validate job exists and is in PROCESSING status
        # TODO 3: Update progress_percentage and stage in job record
        # TODO 4: Store updated job back to Redis
        # TODO 5: Trigger webhook notification if configured
        # TODO 6: Publish progress update to monitoring channels
        # Hint: Use Redis HSET for atomic field updates
        pass
    
    def mark_job_completed(
        self, 
        job_id: str, 
        output_files: List[str]
    ) -> None:
        """Finalize completed job and cleanup."""
        # TODO 1: Update job status to COMPLETED
        # TODO 2: Set completed_at timestamp
        # TODO 3: Store output file references
        # TODO 4: Remove job from active processing queue
        # TODO 5: Send completion webhook notification
        # TODO 6: Schedule temporary file cleanup
        # Hint: Use Redis transaction (MULTI/EXEC) for atomic state updates
        pass
    
    def mark_job_failed(
        self, 
        job_id: str, 
        error_message: str, 
        retry_eligible: bool
    ) -> None:
        """Handle job failure with retry logic."""
        # TODO 1: Retrieve current job and increment retry_count
        # TODO 2: Check if retry_count exceeds max_retry_attempts
        # TODO 3: If retry eligible and under limit, requeue with exponential backoff
        # TODO 4: Otherwise, mark as permanently FAILED
        # TODO 5: Send failure webhook notification
        # TODO 6: Log detailed error information for debugging
        # Hint: Calculate backoff delay as 2^retry_count * base_delay seconds
        pass
```

**API Application Setup (src/media_processor/api/app.py)**

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import redis
from ..config import AppConfig
from ..queue.job_manager import JobManager
from ..storage.file_manager import FileManager

def create_app(config: AppConfig) -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="Media Processing Pipeline",
        description="Scalable media processing with job queue",
        version="1.0.0"
    )
    
    # Configure CORS for web client access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Initialize core services
    redis_client = redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        password=config.redis.password,
        decode_responses=True
    )
    
    storage_manager = FileManager(config.storage)
    job_manager = JobManager(redis_client, storage_manager)
    
    # Dependency injection for route handlers
    def get_job_manager() -> JobManager:
        return job_manager
    
    # TODO: Import and include route modules
    # from .routes.jobs import router as jobs_router
    # app.include_router(jobs_router, prefix="/api/v1")
    
    return app
```

#### Language-Specific Hints

**Python-Specific Considerations:**
- Use `redis-py` client with connection pooling for high-throughput job operations
- Configure `multiprocessing.Process` for worker isolation with proper signal handling
- Use `pathlib.Path` instead of `os.path` for cross-platform file operations
- Implement proper exception handling with custom exception hierarchy in `common/exceptions.py`
- Use `dataclasses` or `pydantic` models for type safety and automatic validation
- Configure proper virtual environment with `requirements.txt` and `requirements-dev.txt`

**Redis Integration Tips:**
- Use Redis hash structures (`HSET`/`HGET`) for job metadata storage
- Implement atomic operations with Redis transactions (`MULTI`/`EXEC`) for job state changes
- Use sorted sets (`ZADD`/`ZPOPMIN`) for priority queue implementation
- Configure Redis persistence (AOF + RDB) for job durability in production
- Set appropriate Redis memory policies for handling job queue growth

**File Operation Best Practices:**
- Create temporary directories using `tempfile.mkdtemp()` for isolated processing
- Use context managers (`with` statements) for automatic file handle cleanup
- Implement atomic file operations by writing to temporary files and renaming
- Configure proper file permissions (0o600) for sensitive temporary files
- Use `shutil.move()` for atomic cross-filesystem file operations

#### Milestone Checkpoint

After implementing the high-level architecture foundation:

**What to verify manually:**
1. Start the application with `python -m media_processor.main --config config/development.yaml`
2. Submit a test job via curl: `curl -X POST http://localhost:8000/api/v1/jobs -F "file=@test.jpg"`
3. Check Redis for job data: `redis-cli HGETALL job:[job_id]`
4. Verify job appears in priority queue: `redis-cli ZRANGE jobs:pending 0 -1 WITHSCORES`

**Expected behavior:**
- Application starts without import errors
- Configuration loads successfully from YAML file
- Redis connection establishes and accepts job data
- File uploads save to configured storage directory
- Job IDs are properly formatted UUIDs
- Priority queue maintains correct score ordering

**Signs something is wrong:**
- Import errors: Check Python path and package `__init__.py` files
- Redis connection failures: Verify Redis server is running and configuration matches
- File permission errors: Check storage directory exists and is writable
- Configuration errors: Validate YAML syntax and required fields are present


## Data Model and Types

> **Milestone(s):** All milestones (1-3) as these core data structures underpin image processing, video transcoding, and job queue operations

### Mental Model: Blueprint Library

Think of our data model as a comprehensive blueprint library for a construction company. Just as architects need detailed blueprints that specify every dimension, material, and connection point before breaking ground, our media processing system needs precise data structures that define every job parameter, metadata field, and configuration option before processing begins. Each blueprint (data type) serves a specific purpose: job blueprints define what work needs to be done, media metadata blueprints capture the current state of materials, and configuration blueprints establish the standards and limits for all operations. Without these detailed specifications, workers would make inconsistent decisions, leading to structural failures (processing errors) and cost overruns (resource waste).

The power of a well-designed blueprint library lies in its ability to handle complexity through standardization. A residential blueprint looks different from a commercial one, just as image processing jobs have different data requirements than video transcoding jobs. However, both follow the same foundational patterns for dimensions, materials, and safety requirements. Similarly, our data structures establish common patterns for job identification, progress tracking, and error handling while allowing specialization for different media types and processing operations.

![Data Model Relationships](./diagrams/data-model.svg)

### Job and Task Entities

The job and task entities form the backbone of our asynchronous processing system, representing units of work from initial submission through final completion. These structures must capture not only the processing requirements but also the complete lifecycle state, error conditions, and progress information needed for reliable distributed processing.

A **processing job** represents a single media processing request from a client, containing the input file, desired output specifications, and all metadata needed for execution and tracking. Each job moves through a well-defined lifecycle managed by the job queue system, with state transitions triggered by worker processes and external events. The job entity serves as the primary coordination point between the client-facing API, the job queue infrastructure, and the worker processes that perform actual media processing.

**Processing jobs** contain both the specification of work to be performed and the runtime state of that work as it progresses through the system. This dual nature requires careful design to separate immutable job parameters (input file, output specifications) from mutable runtime state (current status, progress percentage, error conditions). The job entity must be serializable for storage in the job queue and database, while providing efficient access patterns for both job submission and progress tracking operations.

| Field Name | Type | Description |
|------------|------|-------------|
| `job_id` | `str` | Unique identifier generated at job creation, used for all tracking and reference operations |
| `input_file_path` | `str` | Absolute path to the source media file in the storage system, validated at submission time |
| `output_specifications` | `List[OutputSpec]` | Collection of desired output formats, resolutions, and quality settings for this job |
| `priority` | `JobPriority` | Processing priority level determining queue position and resource allocation preferences |
| `status` | `JobStatus` | Current lifecycle state of the job, updated atomically by worker processes and queue system |
| `created_at` | `datetime` | UTC timestamp when job was first submitted to the system |
| `started_at` | `datetime` | UTC timestamp when worker process began processing, null if not yet started |
| `completed_at` | `datetime` | UTC timestamp when processing finished (success or permanent failure) |
| `error_message` | `str` | Human-readable error description for failed jobs, null for successful or pending jobs |
| `retry_count` | `int` | Number of processing attempts made so far, incremented before each retry |
| `webhook_url` | `str` | Optional HTTP endpoint for status notifications, validated as proper URL at submission |
| `progress_percentage` | `float` | Current completion percentage (0.0-100.0) updated by worker during processing |
| `estimated_duration` | `int` | Estimated total processing time in seconds, calculated based on file size and type |

> **Design Insight**: The `ProcessingJob` structure intentionally separates immutable job specification (input file, output specs, webhook URL) from mutable runtime state (status, progress, timestamps). This separation enables safe concurrent access patterns where multiple processes can read the job specification while only the assigned worker updates the runtime state.

The job priority system uses numerical values to enable flexible scheduling while maintaining intuitive semantic meaning. Higher numerical values represent higher priority, allowing the job queue to use simple numerical comparison for ordering operations.

| Priority Level | Numeric Value | Use Case | Typical Processing Time |
|----------------|---------------|----------|------------------------|
| `JobPriority.LOW` | 1 | Batch processing, background tasks | No specific SLA |
| `JobPriority.NORMAL` | 5 | Standard user uploads, scheduled processing | Process within 1 hour |
| `JobPriority.HIGH` | 10 | Interactive user requests, preview generation | Process within 15 minutes |
| `JobPriority.URGENT` | 20 | Critical system operations, error recovery | Process immediately |

The job status enumeration defines a finite state machine that governs job lifecycle management and ensures consistent state transitions across the distributed system.

| Status Value | Description | Valid Transitions | Worker Actions |
|--------------|-------------|------------------|----------------|
| `JobStatus.PENDING` | Job queued awaiting worker assignment | → PROCESSING | Worker claims job and begins processing |
| `JobStatus.PROCESSING` | Job actively being processed by worker | → COMPLETED, FAILED | Worker updates progress and handles completion |
| `JobStatus.COMPLETED` | Job finished successfully with outputs | None (terminal state) | Worker publishes results and cleans up |
| `JobStatus.FAILED` | Job failed permanently after all retries | None (terminal state) | Worker logs error and triggers notifications |

> **Decision: Single Job Entity vs Separate Task Breakdown**
> - **Context**: Complex processing jobs (like video transcoding) involve multiple discrete steps that could be modeled as separate task entities
> - **Options Considered**: 
>   1. Single `ProcessingJob` entity with stage-based progress
>   2. `ProcessingJob` with separate `Task` entities for each processing step
>   3. Hierarchical job structure with parent/child relationships
> - **Decision**: Single `ProcessingJob` entity with stage-based progress tracking
> - **Rationale**: Simplifies data model and reduces coordination complexity while still providing detailed progress information through stage reporting. Most media processing operations are sequential pipelines that don't benefit from independent task scheduling.
> - **Consequences**: Enables simpler queue management and progress tracking, but limits ability to pause/resume individual processing steps or optimize resource allocation per step

### Media Metadata Structures

Media metadata structures capture the essential characteristics of input files and processed outputs, enabling format-aware processing decisions and preserving important file attributes through the processing pipeline. These structures must handle the diversity of media formats while providing a consistent interface for processing components.

**Media metadata** serves two critical functions in our processing pipeline: informing processing decisions based on source file characteristics, and preserving important metadata through format conversions. Different media types require different metadata structures, but all follow common patterns for basic file information, format specifications, and technical parameters that affect processing operations.

The metadata extraction process occurs early in job processing, with results cached to avoid repeated parsing of the same source files. This approach enables processing components to make informed decisions about optimal algorithms, quality settings, and output formats based on source file characteristics rather than relying on generic processing parameters.

| Field Name | Type | Description |
|------------|------|-------------|
| `file_path` | `str` | Absolute path to the media file in storage |
| `file_size` | `int` | File size in bytes for resource planning |
| `mime_type` | `str` | MIME type determined by content inspection |
| `format_name` | `str` | Human-readable format name (JPEG, PNG, MP4, etc.) |
| `created_at` | `datetime` | File creation timestamp from filesystem |
| `checksum` | `str` | SHA-256 hash for integrity verification |
| `width` | `int` | Media width in pixels (images) or video resolution |
| `height` | `int` | Media height in pixels (images) or video resolution |

**Image metadata** extends the base media metadata with photography-specific information including color profiles, compression settings, and embedded metadata that affects processing quality and client compatibility.

| Field Name | Type | Description |
|------------|------|-------------|
| `color_mode` | `str` | Color space representation (RGB, CMYK, Grayscale, Palette) |
| `bit_depth` | `int` | Bits per channel for color precision (8, 16, 32) |
| `compression` | `str` | Compression algorithm used in source file |
| `quality` | `int` | Original quality setting for lossy formats (1-100) |
| `has_transparency` | `bool` | Whether image contains alpha channel information |
| `orientation` | `int` | EXIF orientation value for rotation correction |
| `dpi` | `Tuple[int, int]` | Horizontal and vertical dots per inch |
| `icc_profile` | `bytes` | Embedded ICC color profile data |
| `exif_data` | `Dict[str, Any]` | Complete EXIF metadata dictionary |

**Video metadata** captures motion picture characteristics including codecs, frame rates, and stream information necessary for transcoding decisions and quality preservation.

| Field Name | Type | Description |
|------------|------|-------------|
| `duration` | `float` | Video length in seconds with millisecond precision |
| `frame_rate` | `float` | Frames per second as decimal value |
| `video_codec` | `str` | Video compression codec (H.264, H.265, VP9, etc.) |
| `audio_codec` | `str` | Audio compression codec (AAC, MP3, Opus, etc.) |
| `bitrate` | `int` | Average bitrate in bits per second |
| `keyframe_interval` | `int` | GOP size in frames for seeking optimization |
| `pixel_format` | `str` | Pixel format specification (yuv420p, rgb24, etc.) |
| `aspect_ratio` | `str` | Display aspect ratio (16:9, 4:3, etc.) |
| `audio_channels` | `int` | Number of audio channels (1=mono, 2=stereo, etc.) |
| `audio_sample_rate` | `int` | Audio sample rate in Hz (44100, 48000, etc.) |
| `subtitle_tracks` | `List[str]` | Available subtitle languages and formats |

> **Decision: Embedded EXIF vs Separate EXIF Entity**
> - **Context**: EXIF metadata can be extensive and includes nested structures like GPS coordinates, camera settings, and thumbnail images
> - **Options Considered**:
>   1. Store complete EXIF as JSON blob in image metadata
>   2. Create separate EXIF entity with structured fields
>   3. Extract only commonly-used EXIF fields into image metadata
> - **Decision**: Store complete EXIF as JSON dictionary with extracted key fields in image metadata
> - **Rationale**: Preserves complete EXIF data for applications that need it while providing fast access to commonly-used fields like orientation and GPS coordinates. JSON storage handles the nested and variable structure of EXIF data.
> - **Consequences**: Enables flexible EXIF handling and preservation while maintaining query performance for common fields, but requires JSON parsing for detailed EXIF access

### Processing Configuration Types

Processing configuration structures define the parameters and constraints that control media processing operations, enabling fine-tuned control over quality, performance, and output characteristics. These configurations must balance processing quality with resource consumption while providing sensible defaults for common use cases.

**Processing configurations** serve as the bridge between high-level client requirements (resize to thumbnail, transcode for web playback) and low-level processing parameters (interpolation algorithms, codec settings, bitrate targets). The configuration system enables both simple preset-based processing and detailed parameter control for advanced use cases.

Configuration structures follow a hierarchical pattern where general processing parameters are inherited by specific operation types, allowing shared settings like quality levels and resource limits while enabling specialized parameters for image resizing, video transcoding, and thumbnail generation operations.

| Field Name | Type | Description |
|------------|------|-------------|
| `output_path` | `str` | Target file path for processed output |
| `format` | `str` | Output format specification (JPEG, PNG, WebP, MP4, WebM) |
| `quality` | `int` | Quality level from 1-100 for lossy compression formats |
| `overwrite_existing` | `bool` | Whether to replace existing files at output path |
| `preserve_metadata` | `bool` | Whether to copy source metadata to output file |
| `strip_private_data` | `bool` | Whether to remove GPS and personal information from metadata |

**Image processing configurations** specify parameters for resize operations, format conversions, and optimization techniques that balance output quality with file size and processing performance.

| Field Name | Type | Description |
|------------|------|-------------|
| `target_width` | `int` | Desired output width in pixels, null to preserve aspect ratio |
| `target_height` | `int` | Desired output height in pixels, null to preserve aspect ratio |
| `resize_mode` | `str` | Scaling behavior: 'fit', 'fill', 'crop', 'stretch' |
| `interpolation` | `str` | Resampling algorithm: 'nearest', 'bilinear', 'bicubic', 'lanczos' |
| `crop_position` | `str` | Crop anchor point: 'center', 'top', 'bottom', 'left', 'right' |
| `background_color` | `str` | Fill color for letterboxing in hex format (#FFFFFF) |
| `sharpen_amount` | `float` | Post-resize sharpening intensity (0.0-2.0) |
| `optimize_for_web` | `bool` | Apply web-specific optimizations (progressive JPEG, etc.) |
| `max_colors` | `int` | Color palette size for PNG optimization |

**Video transcoding configurations** control codec selection, quality settings, and streaming optimizations for different playback scenarios and device capabilities.

| Field Name | Type | Description |
|------------|------|-------------|
| `video_codec` | `str` | Output video codec: 'h264', 'h265', 'vp9', 'av1' |
| `audio_codec` | `str` | Output audio codec: 'aac', 'mp3', 'opus', 'vorbis' |
| `target_bitrate` | `int` | Average bitrate in kbps, null for CRF encoding |
| `crf_value` | `int` | Constant Rate Factor for quality-based encoding (18-28) |
| `frame_rate` | `float` | Output frame rate, null to preserve source frame rate |
| `keyframe_interval` | `int` | GOP size in seconds for streaming optimization |
| `preset` | `str` | Encoding speed preset: 'ultrafast', 'fast', 'medium', 'slow' |
| `profile` | `str` | Codec profile: 'baseline', 'main', 'high' for H.264 compatibility |
| `level` | `str` | Codec level constraint for device compatibility |
| `two_pass_encoding` | `bool` | Use two-pass encoding for optimal bitrate distribution |

**Adaptive bitrate configurations** define multiple quality variants for streaming applications, enabling clients to select appropriate quality based on network conditions and device capabilities.

| Field Name | Type | Description |
|------------|------|-------------|
| `variant_name` | `str` | Human-readable quality identifier ('720p', '1080p', etc.) |
| `width` | `int` | Video width for this quality variant |
| `height` | `int` | Video height for this quality variant |
| `bitrate` | `int` | Target bitrate in kbps for this variant |
| `max_bitrate` | `int` | Maximum bitrate ceiling in kbps |
| `buffer_size` | `int` | Rate control buffer size in kbits |
| `audio_bitrate` | `int` | Audio bitrate in kbps for this variant |
| `segment_duration` | `int` | HLS/DASH segment length in seconds |

The configuration system uses a preset-based approach for common scenarios while allowing parameter overrides for specialized requirements. This design enables simple integration for typical use cases while maintaining flexibility for advanced applications.

| Preset Name | Use Case | Default Parameters | Typical Output Size |
|-------------|----------|-------------------|-------------------|
| `thumbnail` | Small preview images | 150x150 crop, JPEG quality 80 | 5-15 KB |
| `web_optimized` | Website display images | Max 1920px, WebP quality 85 | 50-200 KB |
| `mobile_video` | Mobile streaming | 720p H.264, 1.5 Mbps | 10-20% of source |
| `web_video` | Desktop streaming | 1080p H.264, 3 Mbps | 20-30% of source |
| `archive_quality` | Long-term storage | Lossless or very high quality | 80-95% of source |

> **Decision: Preset-Based vs Parameter-Based Configuration**
> - **Context**: Clients need both simple configuration for common cases and detailed control for specialized requirements
> - **Options Considered**:
>   1. Pure preset system with fixed parameters
>   2. Pure parameter system requiring all settings
>   3. Hybrid system with presets and parameter overrides
> - **Decision**: Hybrid system with presets as starting points and parameter overrides
> - **Rationale**: Provides ease of use for common scenarios while maintaining flexibility for advanced requirements. Presets encode best practices and tested parameter combinations.
> - **Consequences**: Enables rapid integration and consistent results for typical use cases while supporting advanced customization, but requires careful preset design and parameter validation

⚠️ **Pitfall: Quality Parameter Confusion**

A common mistake is assuming that quality parameters work the same way across all formats and codecs. JPEG quality of 85 produces very different file sizes and visual quality compared to WebP quality 85 or H.264 CRF 23. Each format has its own quality scale, compression characteristics, and optimal parameter ranges.

**Why this is wrong**: Using the same quality value across different formats can result in unnecessarily large files (over-encoding) or poor visual quality (under-encoding). For example, WebP typically achieves the same visual quality as JPEG at 15-20% lower quality settings.

**How to fix it**: Use format-specific quality mapping in your configuration system. Define quality levels semantically ('low', 'medium', 'high', 'maximum') and map them to appropriate numerical values for each format. Provide format-specific presets that encode best practices for each codec and container combination.

⚠️ **Pitfall: Missing Resource Constraints**

Processing configurations often omit resource limits, leading to worker processes consuming excessive memory or CPU time when processing large files or using computationally expensive settings like very slow encoding presets or high-quality interpolation algorithms.

**Why this is wrong**: Without resource constraints, a single large video transcoding job can consume all available system memory, causing the worker process to crash or the entire system to become unresponsive. High-quality processing settings can extend job execution time from minutes to hours.

**How to fix it**: Include resource constraint fields in your configuration types: `max_memory_mb`, `max_processing_time_seconds`, `cpu_threads`. Implement resource monitoring in your worker processes to enforce these limits and fail jobs gracefully when constraints are exceeded.

### Implementation Guidance

The data model implementation focuses on creating clean, serializable data structures that support both SQL database storage and Redis job queue operations. Python's dataclasses provide an excellent foundation for these structures while maintaining compatibility with JSON serialization for API communication and queue message formats.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| Data Validation | Python dataclasses with type hints | Pydantic models with validation |
| Serialization | Built-in `json` module | `msgpack` for binary efficiency |
| Configuration Loading | YAML with `pyyaml` | TOML with `tomli` for type safety |
| Enum Management | Python `enum.Enum` | `enum.IntEnum` for database compatibility |

**Recommended File Structure:**

```
media_processing/
  models/
    __init__.py                 ← export all public types
    base.py                     ← base classes and common fields
    jobs.py                     ← ProcessingJob and related types  
    media.py                    ← media metadata structures
    config.py                   ← processing configuration types
    enums.py                    ← status, priority, format enums
  serializers/
    __init__.py
    json_serializer.py          ← JSON serialization helpers
    queue_serializer.py         ← queue message format handling
```

**Base Data Structure Infrastructure:**

```python
# models/base.py
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional
import json
import uuid

@dataclass
class BaseModel:
    """Base class for all data models with common serialization support."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary with datetime serialization."""
        # TODO 1: Use dataclasses.asdict to get base dictionary
        # TODO 2: Iterate through values and convert datetime objects to ISO strings
        # TODO 3: Convert enums to their string/int values
        # TODO 4: Handle None values appropriately
        # TODO 5: Return clean dictionary ready for JSON serialization
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create model instance from dictionary with type conversion."""
        # TODO 1: Get class field annotations for type information
        # TODO 2: Parse datetime strings back to datetime objects
        # TODO 3: Convert enum string/int values back to enum instances
        # TODO 4: Handle optional fields with None defaults
        # TODO 5: Create and return class instance with converted values
        pass
    
    def to_json(self) -> str:
        """Serialize model to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str):
        """Deserialize model from JSON string."""
        return cls.from_dict(json.loads(json_str))

def generate_job_id() -> str:
    """Generate unique job identifier with timestamp prefix for sorting."""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}"
```

**Enumeration Definitions:**

```python
# models/enums.py
from enum import Enum, IntEnum

class JobStatus(Enum):
    """Job lifecycle status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    
    def is_terminal(self) -> bool:
        """Check if status represents a finished job."""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED)
    
    def can_transition_to(self, new_status: 'JobStatus') -> bool:
        """Validate state transition rules."""
        # TODO 1: Define valid transitions from current status
        # TODO 2: Return True only for legal state changes
        # TODO 3: Prevent transitions from terminal states
        pass

class JobPriority(IntEnum):
    """Job priority levels with numeric values for queue ordering."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20
    
    @classmethod
    def from_string(cls, priority_str: str) -> 'JobPriority':
        """Parse priority from string representation."""
        # TODO 1: Handle case-insensitive string matching
        # TODO 2: Return appropriate JobPriority enum value
        # TODO 3: Raise ValueError for invalid priority strings
        pass

class MediaFormat(Enum):
    """Supported media format enumeration."""
    JPEG = "jpeg"
    PNG = "png" 
    WEBP = "webp"
    AVIF = "avif"
    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"
    
    def is_image_format(self) -> bool:
        """Check if format is for still images."""
        return self in (MediaFormat.JPEG, MediaFormat.PNG, 
                       MediaFormat.WEBP, MediaFormat.AVIF, MediaFormat.GIF)
    
    def is_video_format(self) -> bool:
        """Check if format is for video content."""
        return self in (MediaFormat.MP4, MediaFormat.WEBM)
```

**Core Job Entity Implementation:**

```python
# models/jobs.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from .base import BaseModel, generate_job_id
from .enums import JobStatus, JobPriority

@dataclass
class OutputSpecification(BaseModel):
    """Specification for a single output file from processing job."""
    output_path: str
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    quality: Optional[int] = None
    # TODO: Add additional format-specific parameters as needed

@dataclass 
class ProcessingJob(BaseModel):
    """Core job entity representing a media processing request."""
    job_id: str = field(default_factory=generate_job_id)
    input_file_path: str = ""
    output_specifications: List[OutputSpecification] = field(default_factory=list)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    webhook_url: Optional[str] = None
    progress_percentage: float = 0.0
    estimated_duration: Optional[int] = None
    
    def mark_started(self) -> None:
        """Mark job as started and update status."""
        # TODO 1: Set started_at to current UTC time
        # TODO 2: Change status to PROCESSING
        # TODO 3: Validate that job was in PENDING status
        pass
    
    def update_progress(self, percentage: float, stage: str = "") -> None:
        """Update job progress with validation."""
        # TODO 1: Validate percentage is between 0.0 and 100.0
        # TODO 2: Ensure progress only increases (no going backwards)
        # TODO 3: Update progress_percentage field
        # TODO 4: Log progress update with stage information
        pass
    
    def mark_completed(self, output_files: List[str]) -> None:
        """Mark job as successfully completed."""
        # TODO 1: Set completed_at to current UTC time
        # TODO 2: Change status to COMPLETED  
        # TODO 3: Set progress_percentage to 100.0
        # TODO 4: Validate that output files exist and are readable
        pass
    
    def mark_failed(self, error_msg: str, retryable: bool = True) -> None:
        """Mark job as failed with error details."""
        # TODO 1: Set error_message field
        # TODO 2: Set completed_at to current UTC time
        # TODO 3: Increment retry_count
        # TODO 4: Set status to FAILED only if max retries exceeded
        # TODO 5: Keep status as PENDING if retryable and under retry limit
        pass
```

**Milestone Checkpoint:**

After implementing the data model structures:

1. **Run the data model tests:** `python -m pytest tests/models/ -v`
2. **Expected output:** All serialization and validation tests should pass
3. **Manual verification:** Create a `ProcessingJob` instance, serialize it to JSON, deserialize it back, and verify all fields match
4. **Test job state transitions:** Verify that status transitions follow the defined state machine rules
5. **Signs of problems:** 
   - Serialization errors indicate missing type conversion handling
   - State transition failures suggest incomplete validation logic
   - Datetime handling issues point to timezone or format problems

**Language-Specific Python Tips:**

- Use `dataclasses.field(default_factory=list)` for mutable default values to avoid shared state bugs
- Implement `__post_init__` method for complex validation that requires multiple fields
- Use `typing.Optional[T]` for nullable fields and handle None values in serialization
- Consider `pydantic` models instead of dataclasses for production systems requiring extensive validation
- Use `datetime.utcnow()` consistently for all timestamps to avoid timezone confusion
- Implement custom JSON serialization for complex types like `datetime` and enums


## Image Processing Component

> **Milestone(s):** Milestone 1 (Image Processing) - this section covers the core image manipulation, format conversion, and metadata handling capabilities

### Mental Model: Digital Darkroom

Think of the image processing component as a sophisticated **digital darkroom** with automated equipment. In a traditional film darkroom, photographers would manipulate exposure, contrast, and cropping using enlargers, filters, and chemical baths. Our digital darkroom operates on the same principles but with pixel-perfect precision and unlimited repeatability.

Just as a darkroom technician follows a recipe card specifying exposure time, paper grade, and chemical concentrations, our image processor follows an `ImageProcessingConfig` that specifies target dimensions, quality settings, and output formats. The "enlarger" is our resize algorithm that can scale images up or down with various interpolation methods. The "cropping easel" becomes our intelligent crop positioning that can center-crop, smart-crop, or preserve aspect ratios. The "chemical baths" for different paper types become our format converters that transform JPEG to WebP or PNG to AVIF while optimizing for web delivery.

The critical difference is that our digital darkroom can process hundreds of images simultaneously across multiple "stations" (worker processes) and automatically handle the complex chemistry of color spaces, bit depths, and compression algorithms that would require expert knowledge in the physical world. Each processing job represents a batch of prints from a single negative, where we might produce thumbnails (contact sheet), medium-resolution previews (work prints), and high-quality finals (exhibition prints) all from the same source image.

### Core Image Operations

The image processing component implements four fundamental operations that mirror real-world image manipulation workflows but leverage computational precision for optimal results.

**Image Loading and Format Detection**

The image loader serves as the component's entry point, responsible for reading diverse image formats into a standardized in-memory representation. This process involves more complexity than simply opening a file because images arrive with varying color spaces, bit depths, compression schemes, and metadata structures.

The loader first performs **format detection** by examining file headers rather than relying on file extensions, which can be misleading or absent. JPEG files begin with the hex signature `FFD8FF`, PNG files start with `89504E47`, and WebP files contain `52494646` followed by `57454250`. This signature-based detection prevents processing failures when files have incorrect extensions or no extensions at all.

Once the format is identified, the loader delegates to format-specific decoders that handle the intricacies of each format. JPEG decoding involves Huffman decompression and DCT (Discrete Cosine Transform) coefficient reconstruction. PNG decoding requires LZ77 decompression and filter row processing. WebP can contain either lossy VP8 or lossless VP8L data, requiring different decoder paths.

| Operation | Input | Output | Critical Considerations |
|-----------|-------|--------|------------------------|
| Format Detection | Raw file bytes | Format enum + confidence | Must handle corrupted headers, ambiguous signatures |
| JPEG Decoding | JPEG bitstream | RGB pixel array | EXIF orientation, CMYK→RGB conversion, progressive vs baseline |
| PNG Decoding | PNG bitstream | RGBA pixel array | Transparency handling, gamma correction, color profile application |
| WebP Decoding | WebP container | RGB/RGBA array | Lossy vs lossless detection, animation frame extraction |
| GIF Decoding | GIF bitstream | RGB array + palette | First frame extraction, transparency index handling |

**Resize Algorithms and Interpolation**

Image resizing is mathematically equivalent to **resampling a discrete signal** in two dimensions. When scaling an image, we must calculate new pixel values at positions that don't correspond exactly to original pixel locations. The interpolation algorithm determines how we estimate these new values, directly impacting visual quality and processing time.

**Nearest-neighbor interpolation** simply copies the value of the closest original pixel. This preserves sharp edges and is computationally fastest, making it suitable for pixel art or when preserving exact color values is critical. However, it produces jaggy, blocky results for photographic content.

**Bilinear interpolation** computes each new pixel as a weighted average of the four nearest original pixels. The weights are based on distance, creating smooth gradients but potentially causing slight blurriness. This strikes a good balance for general-purpose resizing.

**Bicubic interpolation** considers a 4×4 grid of surrounding pixels, using cubic polynomial curves to estimate new values. This produces sharper results than bilinear but requires significantly more computation. Bicubic is particularly effective for upscaling operations where detail preservation is paramount.

**Lanczos resampling** applies a sophisticated windowed sinc function that minimizes aliasing artifacts while preserving edge sharpness. This is the gold standard for high-quality downscaling, especially when reducing images to small thumbnail sizes where detail loss would otherwise be severe.

> **Key Insight**: The optimal interpolation algorithm depends on the scaling direction and ratio. Lanczos excels for downscaling by factors of 2× or more, while bicubic is preferred for modest upscaling. For real-time previews, bilinear provides acceptable quality with minimal latency.

| Algorithm | Quality Score | Speed Score | Best Use Cases | Computational Complexity |
|-----------|---------------|-------------|----------------|-------------------------|
| Nearest-Neighbor | 2/10 | 10/10 | Pixel art, exact color preservation | O(1) per pixel |
| Bilinear | 6/10 | 8/10 | General-purpose, real-time previews | O(4) per pixel |
| Bicubic | 8/10 | 5/10 | High-quality upscaling, print preparation | O(16) per pixel |
| Lanczos | 9/10 | 3/10 | Thumbnail generation, dramatic downscaling | O(64) per pixel |

**Aspect Ratio Handling and Smart Cropping**

When target dimensions don't match the source image's aspect ratio, the system must choose between **distortion, letterboxing, or cropping**. Each approach serves different use cases and requires careful implementation to avoid common pitfalls.

**Preserve aspect ratio** scaling calculates the maximum scale factor that fits the image within target bounds without distortion. For a 1600×1200 source targeting 800×400, the limiting factor is height (1200→400 = 3×), so the result becomes 800×267 with potential letterboxing to reach exactly 800×400.

**Center cropping** first scales the image so its smallest dimension matches the target, then crops from the center. This ensures the target dimensions are met exactly but may remove important content from image edges. The algorithm scales to fill the larger target dimension, then crops the excess from the perpendicular dimension.

**Smart cropping** attempts to identify the most visually interesting region before cropping. Simple implementations use edge detection to find high-contrast areas. More sophisticated versions analyze face detection, rule-of-thirds positioning, or entropy maps to preserve the most important content.

> **Decision: Smart Cropping Implementation**
> - **Context**: Users upload images with varying compositions, and center cropping often removes important subjects
> - **Options Considered**: Center crop only, edge-detection based smart crop, ML-based composition analysis
> - **Decision**: Implement entropy-based smart cropping with center crop fallback
> - **Rationale**: Entropy-based cropping identifies visually complex regions without requiring ML models, providing better results than center crop while remaining computationally efficient
> - **Consequences**: Enables better automatic thumbnails but requires additional processing time; some images may still benefit from manual crop positioning

**Format Conversion and Quality Optimization**

Format conversion involves more than changing file extensions—it requires understanding the capabilities and limitations of each target format, then optimizing encoding parameters for the intended use case.

**JPEG optimization** centers on the quality parameter (1-100) which controls quantization table scaling. Quality 85-95 provides excellent visual results for most photographic content while achieving significant compression. The `optimize` flag enables Huffman table optimization, typically reducing file size by 5-10% with minimal processing overhead.

**PNG optimization** focuses on compression level (0-9) and filter selection. PNG uses DEFLATE compression after applying prediction filters to reduce entropy. Filter selection can be automatic (Pillow chooses per-row) or fixed. For images with large solid areas, higher compression levels (7-9) provide substantial size reduction. For images with high-frequency detail, the computational cost of level 9 may not justify the modest size improvement.

**WebP encoding** offers both lossy and lossless modes. Lossy WebP typically achieves 25-30% smaller file sizes than equivalent-quality JPEG while supporting alpha transparency. The quality parameter (0-100) behaves similarly to JPEG. Lossless WebP compresses better than PNG for photographic content but worse for simple graphics with few colors.

| Format | Compression Type | Alpha Support | Animation Support | Browser Support | Typical Use Cases |
|--------|-----------------|---------------|-------------------|-----------------|-------------------|
| JPEG | Lossy only | No | No | Universal | Photographs, complex images |
| PNG | Lossless only | Yes | No | Universal | Graphics, logos, screenshots |
| WebP | Lossy + Lossless | Yes | Yes | Modern browsers | Web-optimized photos and graphics |
| AVIF | Lossy + Lossless | Yes | Yes | Very modern | Next-gen web images, highest compression |

### EXIF and Metadata Management

Image metadata encompasses technical camera settings, geolocation data, copyright information, and processing history. Proper metadata handling is crucial for legal compliance, user privacy, and image display accuracy.

**EXIF Data Structure and Extraction**

EXIF (Exchangeable Image File Format) data is embedded within JPEG and TIFF files as a series of **IFD (Image File Directory) entries**. Each entry contains a tag ID, data type, count, and value or value offset. The structure is hierarchical, with primary IFD containing basic image information and sub-IFDs containing specialized data like GPS coordinates or camera-specific settings.

The most critical EXIF tag for image processing is **Orientation** (tag 0x0112), which indicates how the camera was rotated when the photo was taken. Digital cameras often save images in their sensor's native orientation, then record the intended viewing orientation in EXIF. Failing to handle this correctly results in sideways or upside-down images after processing.

| EXIF Tag | Tag ID | Data Type | Critical for Processing | Privacy Implications |
|----------|--------|-----------|------------------------|---------------------|
| Orientation | 0x0112 | SHORT | Yes - affects display rotation | None |
| DateTime | 0x0132 | ASCII | No - metadata only | Medium - reveals when photo taken |
| GPS Latitude | 0x0002 | RATIONAL | No - metadata only | High - reveals location |
| GPS Longitude | 0x0004 | RATIONAL | No - metadata only | High - reveals location |
| Camera Make | 0x010F | ASCII | No - metadata only | Low - device information |
| Camera Model | 0x0110 | ASCII | No - metadata only | Low - device information |
| Software | 0x0131 | ASCII | No - metadata only | Low - processing software |

**Orientation Handling and Rotation**

EXIF orientation values range from 1-8, representing different combinations of rotation and mirroring. The standard defines these transformations relative to the top-left corner of the image as stored in the file.

| Orientation Value | Transformation Required | Description |
|-------------------|------------------------|-------------|
| 1 | None | Normal (top-left) |
| 2 | Horizontal flip | Mirrored |
| 3 | 180° rotation | Upside down |
| 4 | Vertical flip | Mirrored upside down |
| 5 | 90° CCW + horizontal flip | Mirrored + rotated left |
| 6 | 90° CW | Rotated right |
| 7 | 90° CW + horizontal flip | Mirrored + rotated right |
| 8 | 90° CCW | Rotated left |

The critical processing step is applying the orientation transformation **before** any resize or crop operations. This ensures that user-specified dimensions refer to the correctly oriented image. After applying the transformation, the orientation tag should be reset to 1 (normal) or removed entirely to prevent downstream applications from re-applying the correction.

**Privacy and Metadata Stripping**

For web-facing applications, metadata stripping is essential for user privacy protection. GPS coordinates can reveal home addresses, timestamps can establish presence at specific locations, and camera serial numbers can link images to specific devices across platforms.

The system implements **configurable metadata preservation** based on output usage. Internal processing might preserve technical metadata for quality analysis, while public-facing outputs strip all potentially sensitive information.

> **Decision: Selective Metadata Preservation**
> - **Context**: Different use cases require different metadata handling—internal analysis needs technical data while public sharing requires privacy protection
> - **Options Considered**: Strip all metadata, preserve all metadata, configurable per-output-type
> - **Decision**: Implement three metadata modes: preserve_all, preserve_technical, strip_all
> - **Rationale**: Provides flexibility for different use cases while defaulting to privacy-safe stripping for web outputs
> - **Consequences**: Enables compliance with privacy regulations but requires careful configuration management to prevent accidental data exposure

| Metadata Category | preserve_all | preserve_technical | strip_all |
|-------------------|--------------|-------------------|-----------|
| Camera settings (ISO, aperture, etc.) | ✓ | ✓ | ✗ |
| GPS coordinates | ✓ | ✗ | ✗ |
| Timestamps | ✓ | ✗ | ✗ |
| Device serial numbers | ✓ | ✗ | ✗ |
| Color profile (ICC) | ✓ | ✓ | ✗ |
| Processing software | ✓ | ✓ | ✗ |

### Image Processing Architecture Decisions

The image processing component's architecture must balance performance, quality, memory usage, and format compatibility while remaining maintainable and extensible.

> **Decision: Pillow vs OpenCV vs ImageIO**
> - **Context**: Python offers multiple image processing libraries with different performance characteristics and feature sets
> - **Options Considered**: Pillow (PIL fork), OpenCV, ImageIO + scikit-image
> - **Decision**: Use Pillow as primary library with ImageIO for AVIF support
> - **Rationale**: Pillow provides excellent format support, straightforward API, and robust EXIF handling. OpenCV is overkill for basic operations and has complex deployment requirements. ImageIO fills format gaps.
> - **Consequences**: Enables rapid development and broad format support but may require performance optimization for high-throughput scenarios

**Memory Management Strategy**

Large images can quickly exhaust available memory, especially when processing multiple images concurrently. A 24-megapixel image (6000×4000 pixels) requires approximately 72MB of RAM when loaded as uncompressed RGB data (6000 × 4000 × 3 bytes). Processing operations often require additional temporary buffers, potentially doubling or tripling memory usage.

The system implements **streaming processing** for operations that support it, processing image tiles rather than loading entire images into memory. For operations requiring global image access (like smart cropping), the system enforces **memory limits per worker process** and queues large images for processing by workers with sufficient available memory.

| Image Size | Uncompressed RGB | Peak Processing Memory | Recommended Worker Limit |
|------------|------------------|----------------------|-------------------------|
| 1MP (1024×1024) | 3MB | 9MB | 16 workers per 1GB RAM |
| 6MP (3000×2000) | 18MB | 54MB | 4 workers per 1GB RAM |
| 24MP (6000×4000) | 72MB | 216MB | 1 worker per 1GB RAM |
| 50MP+ (8000×6000) | 144MB+ | 432MB+ | Queue for dedicated high-memory workers |

> **Decision: Tile-Based vs Full-Image Processing**
> - **Context**: Large images can exhaust worker process memory and impact overall system stability
> - **Options Considered**: Always load full image, tile-based streaming, hybrid approach based on image size
> - **Decision**: Implement hybrid approach—tile-based for resize/conversion, full-image for smart cropping
> - **Rationale**: Resize and format conversion can be done on tiles, reducing memory usage. Smart cropping requires global image analysis.
> - **Consequences**: Reduces memory footprint for large images but adds complexity for operations requiring global analysis

**Color Space and Profile Management**

Digital images can exist in various color spaces (RGB, CMYK, LAB, etc.) and may include ICC color profiles that define how colors should be interpreted for display or printing. Naive processing can result in significant color shifts or incorrect color reproduction.

The system standardizes on **sRGB as the working color space** for web delivery, as it's supported universally by browsers and provides predictable color reproduction on most displays. Images with embedded ICC profiles undergo color space conversion during loading to ensure consistent color handling.

CMYK images (common in print workflows) require special handling because they cannot be directly displayed on RGB monitors. The conversion process involves ICC profile-based transformation or, when profiles are unavailable, approximation using standard CMYK-to-RGB conversion matrices.

**Thread Safety and Concurrent Processing**

Pillow's underlying image objects are **not thread-safe**, meaning the same image instance cannot be safely accessed from multiple threads simultaneously. However, separate image instances can be processed concurrently without issues.

The system architecture ensures thread safety by creating **separate image instances per processing task** and never sharing image objects between worker threads. Each processing job operates on its own memory space and file handles, eliminating race conditions and allowing full parallel processing.

### Common Image Processing Pitfalls

Understanding and avoiding these common mistakes is crucial for building a robust image processing system that handles real-world image diversity.

⚠️ **Pitfall: Ignoring EXIF Orientation**

Many developers load images and immediately begin processing without checking EXIF orientation data. This results in images that appear correctly in photo viewers (which honor EXIF orientation) but display incorrectly after processing (when EXIF orientation is lost or ignored).

**Why it's wrong**: Modern smartphones and digital cameras often store images in their sensor's native orientation (landscape) and record the intended viewing orientation in EXIF metadata. Processing the raw pixel data without applying the orientation transformation produces sideways or upside-down results.

**How to fix**: Always read and apply EXIF orientation before any other processing operations. Use Pillow's `ImageOps.exif_transpose()` function which automatically handles all 8 orientation cases. After applying the transformation, either remove the orientation EXIF tag or set it to 1 (normal orientation).

```python
# Correct approach - handle orientation first
image = Image.open(input_path)
image = ImageOps.exif_transpose(image)  # Apply EXIF rotation
# Now proceed with resize, crop, format conversion
```

⚠️ **Pitfall: Inappropriate Interpolation Algorithm Selection**

Using the same interpolation algorithm for all resize operations leads to suboptimal quality. Many developers default to bilinear interpolation for everything, missing opportunities for better quality or faster processing.

**Why it's wrong**: Different scaling factors and image types benefit from different algorithms. Nearest-neighbor preserves sharp pixel boundaries for sprites and pixel art. Lanczos minimizes aliasing for dramatic downscaling. Bicubic provides better upscaling quality than bilinear.

**How to fix**: Implement algorithm selection logic based on scaling factor and image characteristics. Use Lanczos for downscaling by 2× or more, bicubic for upscaling, and bilinear for real-time previews or modest scaling.

⚠️ **Pitfall: Memory Exhaustion with Large Images**

Loading very large images (50+ megapixels) into memory without size checks can crash worker processes or exhaust system memory, affecting other concurrent operations.

**Why it's wrong**: A 100-megapixel image requires 300MB just to store the RGB pixel data, before any processing operations that might require additional temporary buffers. Multiple workers processing large images simultaneously can quickly exhaust available RAM.

**How to fix**: Implement image dimension checks before loading. For images exceeding memory thresholds, either reject them with appropriate error messages, queue them for specialized high-memory workers, or implement tile-based processing for supported operations.

⚠️ **Pitfall: Ignoring Color Space Conversion**

Processing CMYK images as if they were RGB, or failing to handle embedded ICC color profiles, results in dramatic color shifts and unprofessional output quality.

**Why it's wrong**: CMYK uses subtractive color mixing (like printing inks) while RGB uses additive color mixing (like display pixels). Direct interpretation of CMYK values as RGB produces inverted, muddy colors. Similarly, images with wide-gamut color profiles may appear oversaturated or color-shifted when viewed in sRGB.

**How to fix**: Always check the image's color mode and convert to RGB using proper color management. Use ICC profile-based conversion when profiles are available, or standard conversion matrices as fallback. Pillow's `Image.convert('RGB')` handles basic cases, but professional workflows may require more sophisticated color management.

⚠️ **Pitfall: Lossy Recompression Cascades**

Repeatedly loading and saving JPEG images during multi-step processing operations accumulates compression artifacts and degrades image quality with each iteration.

**Why it's wrong**: JPEG uses lossy compression, meaning some image information is permanently discarded during each save operation. Processing workflows that load JPEG → process → save JPEG → load JPEG → process → save JPEG accumulate generational loss, similar to making photocopies of photocopies.

**How to fix**: Maintain images in uncompressed format (PNG or in-memory bitmap) throughout multi-step processing pipelines. Only apply final compression when generating the ultimate output files. If intermediate files must be saved, use lossless PNG for temporary storage.

⚠️ **Pitfall: Inconsistent Quality Parameter Interpretation**

Different image formats interpret quality parameters differently, leading to inconsistent file sizes and visual quality when converting between formats.

**Why it's wrong**: JPEG quality 85 and WebP quality 85 produce different visual quality levels and file sizes because they use different compression algorithms and quality scales. Users expecting consistent results across formats may be surprised by significant variations.

**How to fix**: Implement format-specific quality mapping or target file size algorithms. Document quality parameter behavior clearly, or provide high-level quality settings (like "web_optimized", "high_quality", "maximum_compression") that map to appropriate format-specific parameters.

### Implementation Guidance

#### Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Image Library | Pillow (PIL) with basic operations | Pillow + ImageIO + pyheif for extended format support |
| EXIF Handling | Pillow's built-in EXIF support | ExifRead library for detailed metadata parsing |
| Color Management | Pillow's basic color conversion | colour-science library for professional color workflows |
| Performance Optimization | Sequential processing with memory limits | Wand (ImageMagick binding) for high-performance operations |
| Format Support | JPEG, PNG, WebP via Pillow | Add AVIF, HEIF via ImageIO plugins |

#### Recommended File/Module Structure

```
media_processing/
  image_processing/
    __init__.py                    ← component exports
    core/
      processor.py                 ← ImageProcessor main class
      operations.py                ← resize, crop, convert functions
      metadata.py                  ← EXIF handling and metadata management
      formats.py                   ← format detection and loader registry
    config/
      processing_config.py         ← ImageProcessingConfig and related types
      quality_profiles.py          ← predefined quality settings
    utils/
      color_space.py              ← color conversion utilities
      interpolation.py            ← algorithm selection logic
      memory.py                   ← memory management helpers
    tests/
      test_processor.py           ← main processor tests
      test_operations.py          ← operation-specific tests
      fixtures/                   ← test images in various formats
        sample.jpg
        sample.png
        sample_with_exif.jpg
```

#### Infrastructure Starter Code

**Format Detection and Image Loader** (complete implementation):

```python
# image_processing/core/formats.py
import io
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ExifTags
from PIL.ExifTags import ORIENTATION

class SupportedFormat(Enum):
    JPEG = "JPEG"
    PNG = "PNG" 
    WEBP = "WEBP"
    GIF = "GIF"
    AVIF = "AVIF"

class FormatDetector:
    """Detects image formats by file signature rather than extension."""
    
    SIGNATURES = {
        b'\xff\xd8\xff': SupportedFormat.JPEG,
        b'\x89PNG\r\n\x1a\n': SupportedFormat.PNG,
        b'RIFF': SupportedFormat.WEBP,  # Needs additional validation
        b'GIF87a': SupportedFormat.GIF,
        b'GIF89a': SupportedFormat.GIF,
    }
    
    @classmethod
    def detect_format(cls, file_path: str) -> Optional[SupportedFormat]:
        """Detect image format by reading file signature."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)
                
            # Check standard signatures
            for sig, fmt in cls.SIGNATURES.items():
                if header.startswith(sig):
                    if fmt == SupportedFormat.WEBP:
                        # Verify WebP by checking for WEBP signature at offset 8
                        return fmt if header[8:12] == b'WEBP' else None
                    return fmt
                    
            # Check for AVIF (more complex signature)
            if cls._is_avif(header):
                return SupportedFormat.AVIF
                
            return None
        except (IOError, IndexError):
            return None
    
    @staticmethod
    def _is_avif(header: bytes) -> bool:
        """Check for AVIF format signature."""
        return (len(header) >= 12 and 
                header[4:8] == b'ftyp' and 
                header[8:12] in [b'avif', b'avis'])

class ImageLoader:
    """Loads images from various formats into standardized PIL Image objects."""
    
    def __init__(self):
        self.detector = FormatDetector()
    
    def load_image(self, file_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """Load image and extract metadata."""
        # Detect format
        detected_format = self.detector.detect_format(file_path)
        if not detected_format:
            raise ValueError(f"Unsupported image format: {file_path}")
        
        # Load with PIL
        image = Image.open(file_path)
        
        # Extract metadata
        metadata = self._extract_metadata(image, detected_format)
        
        return image, metadata
    
    def _extract_metadata(self, image: Image.Image, fmt: SupportedFormat) -> Dict[str, Any]:
        """Extract comprehensive metadata from loaded image."""
        metadata = {
            'format': fmt.value,
            'size': image.size,
            'mode': image.mode,
            'has_transparency': self._has_transparency(image),
        }
        
        # Extract EXIF if available
        if hasattr(image, '_getexif') and image._getexif():
            exif_dict = {}
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_dict[tag] = value
            metadata['exif'] = exif_dict
            
            # Extract orientation specifically
            if ORIENTATION in exif:
                metadata['orientation'] = exif[ORIENTATION]
        
        return metadata
    
    @staticmethod
    def _has_transparency(image: Image.Image) -> bool:
        """Check if image has transparency channel."""
        return (image.mode in ('RGBA', 'LA') or 
                'transparency' in image.info)
```

**Memory Management Helper** (complete implementation):

```python
# image_processing/utils/memory.py
import psutil
import os
from typing import Tuple, Optional
from PIL import Image

class MemoryManager:
    """Manages memory usage for image processing operations."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
    
    def check_image_memory_requirements(self, width: int, height: int, 
                                      channels: int = 3) -> int:
        """Calculate memory required for image in bytes."""
        # Base memory for pixel data
        pixel_memory = width * height * channels
        
        # Add overhead for processing (temporary buffers, etc.)
        # Typical operations need 2-3x the base memory
        total_memory = pixel_memory * 3
        
        return total_memory
    
    def can_process_image(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """Check if image can be safely processed within memory limits."""
        try:
            # Get image dimensions without fully loading
            with Image.open(image_path) as img:
                width, height = img.size
                channels = len(img.getbands())
            
            required_memory = self.check_image_memory_requirements(
                width, height, channels)
            
            if required_memory > self.max_memory_bytes:
                return False, f"Image requires {required_memory // 1024 // 1024}MB, limit is {self.max_memory_mb}MB"
            
            # Check current system memory
            available_memory = psutil.virtual_memory().available
            if required_memory > available_memory * 0.8:  # Leave 20% buffer
                return False, "Insufficient system memory available"
            
            return True, None
            
        except Exception as e:
            return False, f"Cannot analyze image: {str(e)}"
    
    def get_optimal_tile_size(self, image_width: int, image_height: int, 
                             target_memory_mb: int = 64) -> Tuple[int, int]:
        """Calculate optimal tile size for streaming processing."""
        target_pixels = (target_memory_mb * 1024 * 1024) // (3 * 3)  # RGB + overhead
        
        # Try to maintain aspect ratio while staying under memory limit
        aspect_ratio = image_width / image_height
        
        if target_pixels >= image_width * image_height:
            return image_width, image_height  # Can process whole image
        
        # Calculate tile dimensions maintaining aspect ratio
        tile_height = int((target_pixels / aspect_ratio) ** 0.5)
        tile_width = int(tile_height * aspect_ratio)
        
        # Ensure minimum tile size for quality
        tile_width = max(tile_width, 256)
        tile_height = max(tile_height, 256)
        
        return tile_width, tile_height
```

#### Core Logic Skeleton Code

**Main Image Processor** (signatures + TODOs):

```python
# image_processing/core/processor.py
from typing import List, Dict, Any, Optional
from PIL import Image, ImageOps
from ..config.processing_config import ImageProcessingConfig
from ..utils.memory import MemoryManager

class ImageProcessor:
    """Main image processing coordinator that orchestrates operations."""
    
    def __init__(self, memory_limit_mb: int = 512):
        self.memory_manager = MemoryManager(memory_limit_mb)
        self.format_loader = ImageLoader()
    
    def process_image(self, input_path: str, output_specs: List[OutputSpecification]) -> List[str]:
        """
        Process a single image according to multiple output specifications.
        Returns list of generated output file paths.
        """
        # TODO 1: Load and validate input image using ImageLoader
        # TODO 2: Check memory requirements using MemoryManager
        # TODO 3: Apply EXIF orientation correction using ImageOps.exif_transpose()
        # TODO 4: For each output specification:
        #   TODO 4a: Calculate resize parameters (target size, interpolation method)
        #   TODO 4b: Apply resize operation with appropriate algorithm
        #   TODO 4c: Apply cropping if aspect ratios don't match
        #   TODO 4d: Convert to target format with quality settings
        #   TODO 4e: Handle metadata according to privacy settings
        #   TODO 4f: Save optimized output file
        # TODO 5: Return list of successfully generated output paths
        # Hint: Use try/except around each output spec to continue processing others on failure
        pass
    
    def resize_image(self, image: Image.Image, target_width: int, target_height: int,
                    interpolation: str = "lanczos", resize_mode: str = "fit") -> Image.Image:
        """
        Resize image to target dimensions with specified algorithm and mode.
        
        Args:
            image: Source PIL Image
            target_width, target_height: Target dimensions
            interpolation: Algorithm ("nearest", "bilinear", "bicubic", "lanczos")  
            resize_mode: How to handle aspect ratio ("fit", "fill", "crop")
        """
        # TODO 1: Select PIL resampling filter based on interpolation parameter
        # TODO 2: Calculate scaling factors for width and height
        # TODO 3: If resize_mode is "fit": scale to fit within bounds, preserve aspect ratio
        # TODO 4: If resize_mode is "fill": scale to fill bounds exactly (may distort)
        # TODO 5: If resize_mode is "crop": scale to fill bounds, crop excess
        # TODO 6: Apply the resize operation using PIL Image.resize()
        # TODO 7: Return resized image
        # Hint: Use Image.Resampling.LANCZOS etc for PIL resampling constants
        pass
    
    def convert_format(self, image: Image.Image, target_format: str, quality: int = 85,
                      optimize: bool = True) -> bytes:
        """
        Convert image to target format with specified quality settings.
        Returns the converted image as bytes.
        """
        # TODO 1: Validate target format against supported formats
        # TODO 2: Handle color mode conversions (RGBA->RGB for JPEG, etc.)
        # TODO 3: Set format-specific parameters (quality for JPEG/WebP, compression for PNG)
        # TODO 4: Create BytesIO buffer for output
        # TODO 5: Save image to buffer with format and parameters
        # TODO 6: Return buffer contents as bytes
        # Hint: JPEG doesn't support transparency, convert RGBA to RGB with background
        pass
    
    def generate_thumbnail(self, image: Image.Image, size: int = 256, 
                          crop_mode: str = "center") -> Image.Image:
        """
        Generate square thumbnail with smart cropping.
        
        Args:
            image: Source PIL Image
            size: Target thumbnail size (square)
            crop_mode: Cropping strategy ("center", "smart", "top", "bottom")
        """
        # TODO 1: Calculate current aspect ratio
        # TODO 2: If crop_mode is "center": crop from center to square aspect ratio
        # TODO 3: If crop_mode is "smart": use entropy-based cropping to find interesting region
        # TODO 4: If crop_mode is "top"/"bottom": crop from specified edge
        # TODO 5: Resize the cropped square image to target size using appropriate interpolation
        # TODO 6: Apply sharpening filter if thumbnail is significantly smaller than source
        # TODO 7: Return thumbnail image
        # Hint: For smart cropping, analyze image entropy in sliding windows
        pass

    def extract_and_handle_metadata(self, image: Image.Image, 
                                   metadata_mode: str = "strip_all") -> Dict[str, Any]:
        """
        Extract metadata and filter according to privacy settings.
        """
        # TODO 1: Extract EXIF data using PIL image.getexif()
        # TODO 2: Parse GPS coordinates if present
        # TODO 3: Extract camera information (make, model, settings)
        # TODO 4: Based on metadata_mode, filter what to preserve:
        #   - "preserve_all": keep everything
        #   - "preserve_technical": keep camera settings, strip GPS/timestamps
        #   - "strip_all": remove all metadata
        # TODO 5: Return filtered metadata dictionary
        # Hint: Use ExifTags.TAGS to convert numeric tag IDs to readable names
        pass
```

#### Milestone Checkpoint

After implementing the image processing component, verify functionality with these tests:

**Expected Behavior After Implementation:**
1. **Format Detection**: Run `python -m image_processing.test_formats` - should correctly identify JPEG, PNG, WebP files by signature
2. **EXIF Orientation**: Process a rotated smartphone photo - output should be correctly oriented even if input appears sideways
3. **Memory Management**: Attempt to process a very large image (>100MB) - should either process successfully or fail gracefully with memory limit message
4. **Quality Settings**: Convert the same image to JPEG at quality 50, 85, and 95 - file sizes should vary significantly with visible quality differences
5. **Thumbnail Generation**: Generate thumbnails from landscape and portrait images - all should be properly cropped squares at specified size

**Commands to Test:**
```bash
# Test basic functionality
python -c "
from image_processing.core.processor import ImageProcessor
processor = ImageProcessor()
outputs = processor.process_image('test.jpg', [
    OutputSpecification('thumb.jpg', 'JPEG', 256, 256, 85),
    OutputSpecification('web.webp', 'WEBP', 1200, 800, 80)
])
print(f'Generated: {outputs}')
"

# Test memory limits
python -c "
from image_processing.utils.memory import MemoryManager
mem = MemoryManager(max_memory_mb=100)
can_process, reason = mem.can_process_image('large_image.jpg') 
print(f'Can process: {can_process}, Reason: {reason}')
"
```

**Signs Something Is Wrong:**
- Images appear rotated after processing → Check EXIF orientation handling
- Process crashes with large images → Verify memory management implementation
- Poor thumbnail quality → Check interpolation algorithm selection
- Wrong colors in output → Verify color space conversion (CMYK→RGB)
- Metadata leaking in outputs → Check metadata stripping implementation


## Video Transcoding Component

> **Milestone(s):** Milestone 2 (Video Transcoding) - this section covers video format conversion, adaptive bitrate encoding, thumbnail extraction, and FFmpeg integration

### Mental Model: Film Studio Pipeline

Think of the video transcoding component as a **modern film studio's post-production pipeline**. Just as a film studio receives raw footage from cameras and transforms it into multiple distribution formats for theaters, streaming services, and home video, our video transcoding component takes input video files and produces multiple optimized versions for different playback scenarios.

In a film studio, the workflow follows a structured pipeline. First, the **dailies** (raw footage) are ingested and cataloged with metadata about format, resolution, and quality. Then, the **color correction** and **audio sync** departments work their magic to optimize the content. Next, the **mastering** department creates different versions - a high-quality theatrical print, compressed versions for streaming, and lower-bitrate versions for mobile devices. Finally, the **quality control** department validates each output before distribution.

Our video transcoding component mirrors this workflow exactly. The **FFmpeg integration layer** acts as our sophisticated film processing equipment - it can handle virtually any input format and codec, just like professional film equipment can work with different film stocks and digital formats. The **adaptive bitrate streaming** component functions as our distribution mastering department, creating multiple quality variants from a single source. The **progress tracking** system serves as our production coordinator, keeping stakeholders informed about which films are in what stage of processing.

This mental model is crucial because video transcoding involves the same core challenges as film production: **quality versus file size trade-offs**, **format compatibility** across different playback devices, **processing time management** for large files, and **workflow coordination** across multiple processing stages. Understanding this parallel helps developers appreciate why video transcoding requires sophisticated orchestration rather than simple format conversion.

### FFmpeg Integration Layer

The **FFmpeg integration layer** serves as the foundational component that wraps the powerful FFmpeg command-line tools in a Python-friendly interface. FFmpeg is the industry-standard Swiss Army knife for multimedia processing, capable of handling hundreds of video codecs, container formats, and processing operations. However, integrating FFmpeg into a production system requires careful handling of process management, progress parsing, error interpretation, and resource monitoring.

The integration layer abstracts FFmpeg's complexity behind clean Python interfaces while preserving access to its full capabilities. This approach allows the media processing pipeline to leverage FFmpeg's battle-tested encoding algorithms and format support without requiring developers to become experts in FFmpeg's extensive command-line syntax.

**Core FFmpeg Wrapper Architecture**

The wrapper architecture centers around a `FFmpegProcessor` class that manages FFmpeg subprocess execution with sophisticated error handling and progress monitoring. The processor maintains a **command builder pattern** that constructs FFmpeg commands from high-level configuration objects, ensuring type safety and preventing common command-line construction errors.

| Component | Responsibility | Key Methods |
|-----------|---------------|-------------|
| `FFmpegProcessor` | Process lifecycle management | `execute_command`, `parse_progress`, `handle_errors` |
| `CommandBuilder` | FFmpeg command construction | `build_transcode_command`, `build_thumbnail_command`, `add_codec_options` |
| `ProgressParser` | Real-time progress extraction | `parse_ffmpeg_output`, `calculate_percentage`, `estimate_remaining_time` |
| `ErrorHandler` | FFmpeg error interpretation | `classify_error_type`, `suggest_recovery_action`, `extract_error_details` |

The `FFmpegProcessor` implements a **non-blocking execution model** that spawns FFmpeg as a subprocess while continuously monitoring its stderr output for progress updates and error conditions. This design prevents long-running video transcoding operations from blocking the worker process and enables real-time progress reporting to end users.

**Command Construction and Validation**

The command builder component translates high-level `VideoTranscodingConfig` objects into properly formatted FFmpeg command lines. This translation layer is critical because FFmpeg's command-line interface is extremely flexible but also prone to subtle errors that can result in failed transcoding jobs or suboptimal output quality.

| FFmpeg Parameter Category | Configuration Fields | Validation Rules |
|---------------------------|---------------------|------------------|
| Input Handling | `input_file_path`, `input_format` | File existence, format detection, stream analysis |
| Video Encoding | `video_codec`, `crf_value`, `preset`, `profile` | Codec compatibility, CRF range (0-51), preset validation |
| Audio Processing | `audio_codec`, `audio_bitrate`, `audio_channels` | Channel count limits, bitrate ranges, codec support |
| Output Options | `output_path`, `container_format`, `optimize_flags` | Path writability, format/codec compatibility |
| Advanced Settings | `two_pass_encoding`, `keyframe_interval`, `pixel_format` | GOP structure validation, pixel format support |

The command builder implements **defensive validation** at multiple levels. First, it validates individual parameter values against known ranges and supported options. Then, it performs **cross-parameter validation** to ensure codec and container format compatibility. Finally, it adds **optimization flags** specific to the target use case, such as web streaming optimization or file size minimization.

**Progress Parsing and Monitoring**

FFmpeg outputs detailed progress information to stderr in a semi-structured format that requires careful parsing to extract meaningful progress metrics. The progress parser component implements a **state machine** that tracks FFmpeg's processing phases and translates raw output into standardized progress updates.

| Processing Phase | FFmpeg Output Indicators | Progress Calculation Method |
|-----------------|-------------------------|----------------------------|
| Input Analysis | "Input #0", stream detection | 5% of total progress |
| Video Processing | frame count, time position | (current_time / total_duration) * 0.85 |
| Audio Processing | audio frame processing | Combined with video progress |
| Output Finalization | muxing, index writing | Final 10% of progress |

The progress calculation uses **duration-based estimation** rather than frame-based counting because frame processing rates vary significantly based on video complexity, encoding settings, and system performance. The parser maintains a **rolling average** of processing speed to provide increasingly accurate time estimates as transcoding progresses.

> **Key Insight**: FFmpeg progress parsing is inherently imprecise because encoding complexity varies throughout a video file. Scene changes, motion complexity, and audio dynamics all affect processing speed. The progress parser focuses on providing directionally accurate estimates rather than precise percentages.

**Error Classification and Recovery**

FFmpeg can fail for dozens of different reasons, from unsupported codecs to insufficient disk space to corrupted input files. The error handler component implements a **hierarchical error classification system** that categorizes failures by their root cause and suggests appropriate recovery strategies.

| Error Category | Example Causes | Recovery Strategy | Retry Eligible |
|---------------|---------------|------------------|----------------|
| Input Errors | Corrupted file, unsupported format | Validate input, suggest format conversion | No |
| Configuration Errors | Invalid codec combination, bad parameters | Adjust encoding settings, use fallback profile | Yes |
| Resource Errors | Insufficient memory, disk full | Wait and retry, reduce quality settings | Yes |
| System Errors | FFmpeg binary missing, permission denied | Check system configuration, adjust paths | No |

The error classification logic uses **pattern matching** against FFmpeg's stderr output combined with **exit code analysis** to determine the failure category. This dual approach is necessary because FFmpeg's error reporting is inconsistent - some errors produce specific exit codes while others require parsing text output for diagnostic information.

> **Decision: Process Isolation Strategy**
> - **Context**: FFmpeg processing can consume significant system resources and occasionally crash or hang, potentially affecting other jobs
> - **Options Considered**: 
>   1. Run FFmpeg in same process with threading
>   2. Spawn FFmpeg as subprocess with resource limits
>   3. Use containerized FFmpeg execution
> - **Decision**: Subprocess execution with resource monitoring and timeouts
> - **Rationale**: Provides process isolation while maintaining simplicity. Resource limits prevent runaway processes, and subprocess cleanup ensures system stability
> - **Consequences**: Enables robust error handling and resource management at the cost of slightly increased system overhead

### Adaptive Bitrate Streaming

**Adaptive Bitrate Streaming (ABR)** represents one of the most sophisticated aspects of modern video delivery. ABR enables video players to dynamically adjust quality based on network conditions, device capabilities, and user preferences by providing multiple quality variants of the same content. The video transcoding component must generate these variants efficiently while ensuring smooth transitions between quality levels during playback.

The ABR implementation centers around creating **quality ladders** - sets of video renditions at different resolutions and bitrates that cover the full spectrum from mobile networks to high-speed broadband. Each quality ladder must be carefully designed to provide meaningful quality improvements while avoiding wasteful bandwidth usage.

**Quality Ladder Generation**

The quality ladder generation algorithm analyzes the source video's characteristics to determine optimal encoding targets. This analysis considers **source resolution**, **content complexity**, **motion characteristics**, and **target delivery scenarios** to create a customized quality ladder rather than applying a one-size-fits-all approach.

| Resolution Tier | Typical Use Case | Bitrate Range | Audio Bitrate | Target Devices |
|-----------------|-----------------|---------------|---------------|----------------|
| 240p (426×240) | Very low bandwidth | 300-500 kbps | 64 kbps | Feature phones, poor connections |
| 360p (640×360) | Mobile networks | 500-800 kbps | 96 kbps | Smartphones, tablet cellular |
| 480p (854×480) | Standard definition | 800-1200 kbps | 128 kbps | Laptops, tablets, smart TVs |
| 720p (1280×720) | High definition | 1500-3000 kbps | 128 kbps | Desktop, modern mobile devices |
| 1080p (1920×1080) | Full HD | 3000-6000 kbps | 192 kbps | High-end devices, good connections |
| 1440p (2560×1440) | 2K/QHD | 6000-12000 kbps | 256 kbps | Premium displays, excellent bandwidth |

The quality ladder algorithm implements **content-aware encoding** that adjusts bitrate targets based on video complexity analysis. Simple content like presentations or screencasts can achieve acceptable quality at lower bitrates, while complex content with rapid motion requires higher bitrates to maintain visual fidelity.

The encoding process follows a **parallel generation strategy** where multiple quality variants are produced simultaneously rather than sequentially. This approach reduces total processing time by leveraging multiple CPU cores and optimizing I/O patterns across the encoding pipeline.

**HLS (HTTP Live Streaming) Implementation**

HLS represents Apple's adaptive streaming standard that segments video content into small chunks (typically 6-10 seconds) and creates manifest files that describe available quality variants. The HLS implementation must generate properly formatted segments and manifests that ensure compatibility across a wide range of devices and players.

The HLS generation process involves several critical steps that must be executed with precise timing and format adherence:

1. **Segment Duration Calculation**: Analyze source video keyframe intervals to determine optimal segment boundaries that align with GOP (Group of Pictures) structures
2. **Multi-Variant Playlist Creation**: Generate the master manifest that lists all available quality variants with their technical specifications
3. **Individual Stream Processing**: Create separate media playlists for each quality variant with segment references and timing information
4. **Encryption Key Management**: Generate and rotate AES-128 encryption keys for content protection when required
5. **Subtitle and Audio Track Handling**: Process additional streams for multi-language support and accessibility features

| HLS Component | File Extension | Purpose | Update Frequency |
|---------------|---------------|---------|------------------|
| Master Playlist | `.m3u8` | Lists all variant streams | Static after creation |
| Media Playlist | `.m3u8` | Segment list for specific quality | Dynamic for live streams |
| Video Segments | `.ts` or `.m4s` | Actual video content chunks | Static after creation |
| Key Files | `.key` | Encryption keys for protected content | Rotated periodically |

The HLS encoder implements **segment alignment** across quality variants to ensure smooth switching between streams. This alignment requires careful keyframe placement and GOP structure coordination across all encoding passes.

**DASH (Dynamic Adaptive Streaming over HTTP) Support**

DASH provides an alternative adaptive streaming standard that offers more flexibility than HLS through its XML-based manifest format and support for multiple container formats. The DASH implementation complements the HLS support to ensure broad client compatibility across different platforms and players.

DASH manifest generation requires creating **Media Presentation Description (MPD)** files that describe the temporal structure of content, available representations, and addressing schemes for segment retrieval. The MPD format is more complex than HLS playlists but provides greater expressiveness for advanced streaming scenarios.

| DASH Element | XML Structure | Content Description |
|--------------|---------------|-------------------|
| Period | `<Period>` | Temporal division of content |
| AdaptationSet | `<AdaptationSet>` | Group of representations with same content |
| Representation | `<Representation>` | Specific encoding variant |
| SegmentTemplate | `<SegmentTemplate>` | URL pattern for segment addressing |

The DASH encoder supports both **SegmentTemplate** and **SegmentList** addressing modes, with template-based addressing preferred for its efficiency and scalability. Template addressing allows clients to construct segment URLs mathematically rather than downloading explicit segment lists.

> **Decision: Dual ABR Format Support**
> - **Context**: Different platforms and clients prefer different adaptive streaming formats (HLS vs DASH)
> - **Options Considered**:
>   1. HLS only (Apple/Safari focus)
>   2. DASH only (open standard focus)
>   3. Dual format support with shared segments
> - **Decision**: Generate both HLS and DASH manifests from common segment files
> - **Rationale**: Maximizes client compatibility while minimizing storage overhead by reusing segment files across formats
> - **Consequences**: Increased processing complexity but broader device support and reduced storage costs

### Video Transcoding Architecture Decisions

The video transcoding component requires numerous architectural decisions that significantly impact performance, quality, compatibility, and resource utilization. These decisions form the foundation for reliable, scalable video processing that meets diverse client requirements while maintaining operational efficiency.

**Codec Selection Strategy**

Video codec selection represents one of the most impactful architectural decisions because it directly affects output quality, file sizes, encoding performance, and client compatibility. The transcoding component implements a **multi-codec strategy** that selects optimal codecs based on content characteristics, target platforms, and performance requirements.

| Codec | Primary Use Case | Encoding Speed | Quality Efficiency | Browser Support | Hardware Acceleration |
|-------|-----------------|----------------|-------------------|----------------|----------------------|
| H.264 (AVC) | Universal compatibility | Fast | Good | Excellent (99%+) | Widely available |
| H.265 (HEVC) | High efficiency, 4K content | Slow | Excellent | Limited (60%) | Growing support |
| VP9 | Open source, web focus | Medium | Very Good | Good (80%) | Limited hardware support |
| AV1 | Future standard, efficiency | Very Slow | Excellent | Growing (40%) | Emerging hardware support |

The codec selection algorithm implements **content analysis heuristics** that examine source video characteristics to determine the most appropriate encoding approach. High-resolution content with complex motion benefits from advanced codecs like H.265 or VP9, while simple content prioritizes H.264 for universal compatibility.

The system supports **fallback codec chains** where encoding attempts with advanced codecs automatically fall back to more compatible options if hardware acceleration is unavailable or encoding times exceed acceptable thresholds. This approach ensures job completion while optimizing for available resources.

**Encoding Parameter Optimization**

Video encoding involves dozens of parameters that control the trade-offs between quality, file size, and processing time. The transcoding component implements **profile-based parameter management** that groups related settings into coherent configurations optimized for specific use cases.

| Encoding Profile | Target Scenario | CRF Range | Preset | Special Flags |
|------------------|----------------|-----------|---------|---------------|
| Web Optimized | Streaming delivery | 20-24 | medium | `-movflags +faststart` |
| High Quality | Archive/premium content | 16-20 | slow | `-tune film` |
| Mobile Optimized | Cellular networks | 24-28 | fast | `-profile:v baseline` |
| Low Latency | Live streaming | 22-26 | ultrafast | `-tune zerolatency` |

The **Constant Rate Factor (CRF)** approach provides superior quality control compared to target bitrate encoding because it maintains consistent perceptual quality across varying content complexity. The system automatically adjusts CRF values based on source resolution and target quality requirements.

**Two-pass encoding** is selectively applied for scenarios where precise bitrate control is required, such as broadcast delivery or bandwidth-constrained environments. The first pass analyzes content complexity while the second pass optimizes bit allocation based on the analysis results.

> **Decision: CRF-Based Quality Control**
> - **Context**: Need consistent quality across diverse content while maintaining reasonable file sizes
> - **Options Considered**:
>   1. Target bitrate encoding (CBR/VBR)
>   2. Constant Rate Factor (CRF) encoding
>   3. Constrained Quality (CQ) encoding
> - **Decision**: CRF as primary method with bitrate constraints for ABR
> - **Rationale**: CRF provides consistent perceptual quality regardless of content complexity, resulting in better user experience and more predictable quality
> - **Consequences**: File sizes vary based on content complexity, but quality remains consistent across all content types

**Container Format and Streaming Optimization**

Container format selection affects compatibility, streaming performance, and metadata handling capabilities. The transcoding component supports multiple container formats with optimization flags tailored to specific delivery scenarios.

| Container Format | Primary Use Case | Streaming Support | Metadata Capacity | Browser Compatibility |
|------------------|-----------------|------------------|-------------------|---------------------|
| MP4 | Web delivery, mobile | Excellent | Good | Universal |
| WebM | Open web standards | Good | Limited | Modern browsers |
| MKV | High-quality archival | Limited | Extensive | Desktop players |
| HLS/TS | Adaptive streaming | Native | Limited | iOS, modern browsers |

The MP4 container receives **web streaming optimizations** including the `faststart` flag that relocates metadata to the file beginning, enabling progressive download and immediate playback initiation. This optimization is crucial for user experience in web-based video players.

**Hardware Acceleration Integration**

Modern systems provide hardware-accelerated video encoding through GPU-based encoders like NVIDIA's NVENC, Intel's Quick Sync, and AMD's VCE. The transcoding component implements **intelligent hardware acceleration** that automatically detects available acceleration capabilities and falls back gracefully to software encoding when hardware acceleration is unavailable or insufficient.

| Hardware Platform | Acceleration Technology | Supported Codecs | Performance Gain | Quality Trade-off |
|-------------------|------------------------|-----------------|------------------|-------------------|
| NVIDIA GPU | NVENC | H.264, H.265 | 5-10x faster | Slight quality reduction |
| Intel CPU | Quick Sync | H.264, H.265, AV1 | 3-5x faster | Minimal quality impact |
| AMD GPU | VCE/AMF | H.264, H.265 | 4-8x faster | Slight quality reduction |
| Apple Silicon | VideoToolbox | H.264, H.265, ProRes | 6-12x faster | High quality maintained |

The hardware acceleration detection implements **capability probing** during system initialization to build a matrix of available acceleration options. This probe includes performance benchmarking to ensure hardware encoders provide meaningful speed improvements without unacceptable quality degradation.

Hardware encoding failures trigger **automatic fallback** to software encoding to ensure job completion. The system logs hardware failures for monitoring and analysis to identify patterns that might indicate driver issues or hardware problems.

### Common Video Processing Pitfalls

Video processing introduces numerous opportunities for subtle bugs and performance issues that can severely impact system reliability and user experience. Understanding these common pitfalls helps developers build robust video transcoding systems that handle edge cases gracefully and maintain consistent performance under diverse conditions.

⚠️ **Pitfall: Memory Exhaustion with High-Resolution Video**

Video processing requires substantial memory for frame buffering, especially when handling 4K or 8K content. A single uncompressed 4K frame requires approximately 33MB of memory (3840×2160×4 bytes per pixel), and FFmpeg typically buffers multiple frames simultaneously for encoding efficiency.

The primary mistake developers make is **not implementing memory monitoring** during video processing operations. Long-running transcoding jobs can gradually consume system memory until they trigger out-of-memory conditions that crash worker processes or destabilize the entire system.

**Detection**: Monitor process memory usage during transcoding operations. Memory consumption that grows continuously rather than stabilizing indicates a potential memory leak or insufficient memory availability for the current job.

**Prevention**: Implement **resource-aware job scheduling** that estimates memory requirements based on input resolution and rejects jobs that would exceed available system memory. Use FFmpeg's buffer size controls to limit memory usage during processing.

```python
# Memory estimation for job scheduling
def estimate_memory_requirements(video_metadata):
    width = video_metadata.width
    height = video_metadata.height
    # Estimate memory for frame buffers (assuming 4 bytes per pixel, 8 frame buffer)
    frame_memory = width * height * 4 * 8
    # Add overhead for FFmpeg processing
    total_memory = frame_memory * 1.5
    return int(total_memory)
```

⚠️ **Pitfall: Inaccurate Progress Estimation**

Video transcoding progress is notoriously difficult to estimate accurately because encoding complexity varies dramatically throughout a video file. Scene changes, motion complexity, and content characteristics all affect processing speed in unpredictable ways.

Many developers implement **linear progress estimation** based on processed time or frame count, leading to progress indicators that jump around unpredictably or stall for extended periods. This creates poor user experience and makes it difficult to detect genuinely stalled processing jobs.

**Detection**: Progress updates that move backward, remain stalled for extended periods, or jump by large amounts indicate flawed progress calculation logic.

**Solution**: Implement **stage-based progress reporting** that divides transcoding into distinct phases with predictable duration ranges. Use historical processing data to improve time estimates for similar content types.

| Processing Stage | Typical Duration % | Progress Range | Key Indicators |
|-----------------|-------------------|----------------|----------------|
| Input Analysis | 2-5% | 0-5% | Stream detection, metadata extraction |
| Video Transcoding | 80-90% | 5-90% | Frame processing, bitrate varies by content |
| Audio Processing | 5-10% | 90-95% | Audio encoding, typically consistent speed |
| Output Finalization | 2-5% | 95-100% | Container muxing, index generation |

⚠️ **Pitfall: Codec Compatibility Issues**

Different combinations of video codecs, audio codecs, and container formats create complex compatibility matrices that can result in files that encode successfully but fail to play on certain devices or browsers. This is particularly problematic for web delivery where broad compatibility is essential.

The most common mistake is **not validating codec/container combinations** before starting transcoding operations. Invalid combinations can waste significant processing time and produce unusable output files.

**Detection**: Files that encode without errors but fail validation testing on target playback platforms indicate codec compatibility problems.

**Prevention**: Implement **compatibility validation matrices** that verify codec and container combinations before job execution. Test output files on representative target platforms during development.

| Target Platform | Recommended Video Codec | Audio Codec | Container | Profile Constraints |
|------------------|------------------------|-------------|-----------|-------------------|
| iOS Safari | H.264 | AAC | MP4 | Baseline/Main profile, Level 4.0 max |
| Android Chrome | H.264/VP9 | AAC/Opus | MP4/WebM | High profile acceptable |
| Desktop Browsers | H.264/VP9/AV1 | AAC/Opus | MP4/WebM | All profiles supported |
| Smart TV | H.264 | AAC | MP4 | Main profile, conservative levels |

⚠️ **Pitfall: Keyframe Alignment in ABR Streams**

Adaptive bitrate streaming requires keyframes to be aligned across all quality variants to enable smooth switching between streams during playback. Misaligned keyframes cause visual artifacts, rebuffering, or playback failures when clients attempt to switch quality levels.

Many implementations **generate ABR variants independently** without coordinating keyframe placement, resulting in keyframes that occur at different timestamps across quality levels. This breaks the fundamental requirement for seamless quality switching.

**Detection**: ABR streams with playback stuttering or visual artifacts during quality changes indicate keyframe alignment problems. Analysis tools can detect timestamp mismatches in segment boundaries.

**Solution**: Use **synchronized encoding** with fixed GOP (Group of Pictures) structure across all quality variants. Force keyframes at identical timestamps for all encoding passes.

```python
# Keyframe synchronization for ABR encoding
def generate_abr_variants(input_file, quality_configs):
    # Calculate keyframe intervals based on segment duration
    segment_duration = 6  # seconds
    keyframe_interval = segment_duration * frame_rate
    
    common_flags = [
        '-force_key_frames', f'expr:gte(t,n_forced*{segment_duration})',
        '-g', str(keyframe_interval),
        '-keyint_min', str(keyframe_interval),
    ]
    
    # Apply common keyframe settings to all variants
    for config in quality_configs:
        config.encoding_flags.extend(common_flags)
```

⚠️ **Pitfall: Temporary File Management and Cleanup**

Video processing generates numerous temporary files including intermediate encode passes, segment files, and working directories. Poor temporary file management leads to disk space exhaustion, permission issues, and security vulnerabilities from abandoned sensitive content.

The most critical mistake is **not implementing comprehensive cleanup logic** that handles both successful completion and error scenarios. Temporary files can accumulate rapidly during high-volume processing, eventually exhausting available disk space.

**Detection**: Disk usage that grows continuously over time, even after job completion, indicates temporary file cleanup problems. Monitor temporary directory sizes and file ages.

**Prevention**: Implement **context managers** for temporary file handling that guarantee cleanup regardless of how processing completes. Use unique temporary directories for each job to prevent file conflicts.

```python
import tempfile
import shutil
from contextlib import contextmanager

@contextmanager
def processing_workspace(job_id):
    """Create and cleanup temporary workspace for video processing"""
    temp_dir = tempfile.mkdtemp(prefix=f"video_job_{job_id}_")
    try:
        yield temp_dir
    finally:
        # Cleanup regardless of success or failure
        shutil.rmtree(temp_dir, ignore_errors=True)

# Usage ensures cleanup
with processing_workspace(job.job_id) as workspace:
    # All temporary files created in workspace
    process_video(input_file, workspace)
    # Automatic cleanup when context exits
```

![Video Transcoding Sequence](./diagrams/video-processing-sequence.svg)

### Implementation Guidance

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|--------------|-----------------|
| FFmpeg Integration | `subprocess` with `Popen` | `asyncio.subprocess` with process pools |
| Progress Parsing | Regex pattern matching | State machine with tokenizer |
| Error Handling | Exit code checking | ML-based error classification |
| Temporary Files | `tempfile.mkdtemp()` | Memory-mapped temporary storage |
| Resource Monitoring | `psutil` memory checking | cgroup-based resource limits |

**Recommended Module Structure**

The video transcoding component integrates into the larger media processing pipeline through well-defined interfaces and module boundaries:

```
media_processing/
  video/
    __init__.py                 ← Public interface exports
    transcoder.py              ← Main VideoTranscoder class
    ffmpeg_wrapper.py          ← FFmpeg process management
    progress_parser.py         ← Progress monitoring logic  
    quality_ladder.py          ← ABR variant generation
    codecs.py                  ← Codec compatibility matrices
    containers.py              ← Container format handling
    errors.py                  ← Video-specific error types
  tests/
    test_transcoder.py         ← Core transcoding tests
    test_ffmpeg_wrapper.py     ← FFmpeg integration tests
    fixtures/                  ← Sample video files for testing
      sample_video.mp4
      sample_audio.wav
```

**Complete FFmpeg Wrapper Implementation**

This wrapper provides production-ready FFmpeg integration that handles process management, progress parsing, and error recovery:

```python
import subprocess
import re
import json
import logging
from typing import Dict, List, Optional, Callable
from pathlib import Path
import asyncio
import signal
import psutil

class FFmpegError(Exception):
    """FFmpeg processing error with detailed context"""
    def __init__(self, message: str, exit_code: int = None, stderr: str = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr

class FFmpegProgressParser:
    """Parses FFmpeg stderr output for progress information"""
    
    def __init__(self):
        self.duration_pattern = re.compile(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})')
        self.progress_pattern = re.compile(r'time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})')
        self.total_duration = 0.0
    
    def parse_duration(self, line: str) -> Optional[float]:
        """Extract total duration from FFmpeg output"""
        match = self.duration_pattern.search(line)
        if match:
            hours, minutes, seconds, centiseconds = map(int, match.groups())
            self.total_duration = hours * 3600 + minutes * 60 + seconds + centiseconds / 100
            return self.total_duration
        return None
    
    def parse_progress(self, line: str) -> Optional[float]:
        """Extract current progress from FFmpeg output"""
        match = self.progress_pattern.search(line)
        if match and self.total_duration > 0:
            hours, minutes, seconds, centiseconds = map(int, match.groups())
            current_time = hours * 3600 + minutes * 60 + seconds + centiseconds / 100
            return min(current_time / self.total_duration, 1.0)
        return None

class FFmpegWrapper:
    """Production-ready FFmpeg process wrapper"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.logger = logging.getLogger(__name__)
    
    async def execute_command(
        self,
        command: List[str],
        progress_callback: Optional[Callable[[float], None]] = None,
        timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """Execute FFmpeg command with progress monitoring and timeout"""
        
        full_command = [self.ffmpeg_path] + command
        self.logger.info(f"Executing FFmpeg: {' '.join(full_command)}")
        
        parser = FFmpegProgressParser()
        process = None
        
        try:
            # Start FFmpeg process with stderr capture
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=None if os.name == 'nt' else os.setsid
            )
            
            # Monitor progress in real-time
            stderr_lines = []
            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stderr.readline(), 
                        timeout=30.0  # Line timeout
                    )
                    if not line:
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    stderr_lines.append(line_str)
                    
                    # Parse duration on first encounter
                    parser.parse_duration(line_str)
                    
                    # Report progress updates
                    if progress_callback:
                        progress = parser.parse_progress(line_str)
                        if progress is not None:
                            progress_callback(progress)
                
                except asyncio.TimeoutError:
                    # Check if process is still alive
                    if process.returncode is not None:
                        break
                    self.logger.warning("FFmpeg output timeout, continuing...")
            
            # Wait for process completion
            await asyncio.wait_for(process.wait(), timeout=timeout)
            
            stderr_output = '\n'.join(stderr_lines)
            
            if process.returncode != 0:
                raise FFmpegError(
                    f"FFmpeg failed with exit code {process.returncode}",
                    exit_code=process.returncode,
                    stderr=stderr_output
                )
            
            return subprocess.CompletedProcess(
                args=full_command,
                returncode=process.returncode,
                stdout=None,  # We don't capture stdout
                stderr=stderr_output
            )
        
        except asyncio.TimeoutError:
            if process:
                await self._terminate_process(process)
            raise FFmpegError(f"FFmpeg timed out after {timeout} seconds")
        
        except Exception as e:
            if process:
                await self._terminate_process(process)
            raise FFmpegError(f"FFmpeg execution failed: {str(e)}")
    
    async def _terminate_process(self, process: asyncio.subprocess.Process):
        """Gracefully terminate FFmpeg process"""
        try:
            if os.name != 'nt':
                # Send SIGTERM to process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill if needed
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                await process.wait()
        
        except (ProcessLookupError, OSError):
            pass  # Process already terminated
```

**Core Video Transcoder Skeleton**

This skeleton provides the main transcoding interface that builds on the FFmpeg wrapper:

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import asyncio
import json
from pathlib import Path

@dataclass
class VideoTranscodingConfig:
    """Configuration for video transcoding operation"""
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf_value: int = 23
    preset: str = "medium"
    profile: str = "high"
    level: str = "4.0"
    max_bitrate: Optional[int] = None
    target_resolution: Optional[tuple] = None
    segment_duration: Optional[int] = None  # For HLS/DASH
    two_pass: bool = False

class VideoTranscoder:
    """Main video transcoding orchestrator"""
    
    def __init__(self, ffmpeg_wrapper: FFmpegWrapper):
        self.ffmpeg = ffmpeg_wrapper
        self.logger = logging.getLogger(__name__)
    
    async def transcode_video(
        self,
        input_path: str,
        output_path: str,
        config: VideoTranscodingConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, any]:
        """
        Transcode video file according to configuration.
        
        Args:
            input_path: Source video file path
            output_path: Destination file path
            config: Transcoding parameters
            progress_callback: Function called with (progress_percentage, stage_name)
        
        Returns:
            Dictionary with transcoding results and metadata
        """
        # TODO 1: Validate input file exists and is readable
        # TODO 2: Analyze input video to extract metadata (resolution, duration, codecs)
        # TODO 3: Build FFmpeg command based on config and input characteristics
        # TODO 4: Execute transcoding with progress monitoring
        # TODO 5: Validate output file was created successfully
        # TODO 6: Extract output metadata and return results
        pass
    
    def _build_transcode_command(
        self, 
        input_path: str, 
        output_path: str, 
        config: VideoTranscodingConfig,
        input_metadata: Dict
    ) -> List[str]:
        """
        Build FFmpeg command for transcoding operation.
        
        This method constructs the complete FFmpeg command line based on:
        - Input file characteristics
        - Target configuration parameters  
        - Container format requirements
        - Web optimization flags
        """
        # TODO 1: Start with basic input/output file parameters
        # TODO 2: Add video codec and encoding parameters (CRF, preset, profile)
        # TODO 3: Add audio codec and quality settings
        # TODO 4: Add resolution scaling if target_resolution specified
        # TODO 5: Add container-specific optimization flags (faststart for MP4)
        # TODO 6: Add any advanced encoding options (two-pass, keyframe intervals)
        # Hint: Use list concatenation to build command parts: cmd += ['-c:v', config.video_codec]
        pass
    
    async def generate_abr_variants(
        self,
        input_path: str,
        output_dir: str,
        variant_configs: List[AdaptiveBitrateConfig],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, List[str]]:
        """
        Generate multiple quality variants for adaptive bitrate streaming.
        
        Returns:
            Dictionary mapping variant names to lists of segment files
        """
        # TODO 1: Create output directory structure for variants
        # TODO 2: Process variants in parallel using asyncio.gather()
        # TODO 3: Ensure keyframe alignment across all variants
        # TODO 4: Generate HLS and/or DASH manifests
        # TODO 5: Validate all segments were created successfully
        # Hint: Use asyncio.gather(*tasks) to process multiple variants concurrently
        pass
    
    async def extract_thumbnail(
        self,
        input_path: str,
        output_path: str,
        timestamp: float = 10.0,
        resolution: tuple = (320, 180)
    ) -> bool:
        """
        Extract thumbnail frame from video at specified timestamp.
        
        Args:
            timestamp: Time in seconds to extract frame
            resolution: Target thumbnail resolution (width, height)
        
        Returns:
            True if thumbnail was extracted successfully
        """
        # TODO 1: Build FFmpeg command for single frame extraction
        # TODO 2: Add timestamp seeking and resolution scaling
        # TODO 3: Execute extraction command
        # TODO 4: Verify output image file was created
        # Hint: Use -ss for seeking, -vframes 1 for single frame, -s for resolution
        pass
```

**Milestone Checkpoint**

After implementing the video transcoding component, verify the following behaviors:

1. **Basic Transcoding**: Convert a sample MP4 file to different resolutions and formats
   ```bash
   python -m media_processing.video.test_basic_transcoding
   ```
   Expected: Output files created with correct resolutions and playable in VLC

2. **Progress Monitoring**: Watch progress updates during a long transcoding job  
   Expected: Progress increases from 0% to 100% with reasonable time estimates

3. **Error Handling**: Attempt to process a corrupted or missing video file
   Expected: Clear error message with appropriate error classification

4. **ABR Generation**: Create HLS segments from a sample video
   Expected: Master playlist and variant playlists with properly segmented content

5. **Resource Management**: Monitor memory usage during 4K video processing
   Expected: Memory usage remains stable without continuous growth

**Performance Debugging Tips**

| Symptom | Likely Cause | Diagnosis Method | Solution |
|---------|-------------|------------------|----------|
| Transcoding hangs indefinitely | FFmpeg process deadlock | Check process tree, stderr output | Add timeout, restart worker |
| Memory usage grows continuously | Frame buffer leaks | Monitor RSS memory over time | Implement buffer limits, restart workers |
| Poor output quality | Wrong CRF/bitrate settings | Compare source vs output bitrates | Adjust quality parameters |
| Slow encoding performance | Software encoding on server | Check for hardware acceleration | Enable NVENC/QSV if available |
| Segment alignment issues | Keyframe misalignment | Analyze GOP structure in outputs | Force keyframes at segment boundaries |


## Job Queue and Scheduling Component

> **Milestone(s):** Milestone 3 (Processing Queue & Progress) - this section covers asynchronous job processing, priority queuing, worker management, and distributed task execution

### Mental Model: Restaurant Kitchen

Think of the media processing pipeline as a busy restaurant kitchen during dinner rush. When customers place orders (submit processing jobs), those orders don't go directly to the chefs—they go to an **order management system** that prioritizes them, assigns them to available cooks based on specialization and workload, and tracks progress from preparation through completion.

The **order tickets** represent processing jobs, each containing specific instructions: "Table 12 wants a medium steak with truffle sauce, allergic to shellfish." Similarly, a processing job contains the input file path, desired output specifications, priority level, and callback information for notifications.

The **kitchen manager** acts like our job queue system, deciding which orders get processed first. Rush orders (high priority) jump ahead of regular dinner orders (normal priority), and simple appetizers (image resizing) might be handled by different stations than complex entrees requiring specialized equipment (video transcoding with FFmpeg).

Each **cooking station** represents a worker process—some specialize in grilling (image processing), others handle sauce preparation (video encoding), and some can do general prep work. The kitchen manager knows each station's current workload and capabilities, distributing orders accordingly. When a dish is ready, the expediter (webhook notification system) alerts the wait staff and updates the order status.

Just as restaurants handle kitchen fires, equipment breakdowns, and ingredient shortages without losing orders, our job queue system must gracefully handle worker crashes, resource exhaustion, and processing failures while ensuring no jobs disappear into the void.

### Queue Operations and Priority

The job queue serves as the central coordination hub for all media processing operations, managing the entire lifecycle from job submission through completion notification. At its core, the queue implements a **priority-based work distribution system** that ensures time-sensitive jobs receive immediate attention while maintaining fair processing of routine work.

When a client submits a processing job through the `submit_job` function, the system first validates the request parameters and generates a unique job identifier using `generate_job_id()`. This identifier includes a timestamp prefix to enable chronological sorting and debugging. The job gets wrapped in a `ProcessingJob` structure containing all necessary execution metadata: input file path, output specifications, priority level, webhook notification URL, and tracking timestamps.

> **Decision: Priority Queue with Redis Sorted Sets**
> - **Context**: Need to handle jobs with different urgency levels while maintaining FIFO order within each priority band
> - **Options Considered**: Simple FIFO queue, weighted round-robin, Redis sorted sets with priority scores
> - **Decision**: Redis sorted sets with numeric priority values
> - **Rationale**: Sorted sets provide atomic priority insertion, efficient range queries for worker polling, and built-in deduplication. Priority scores allow fine-grained control over job ordering.
> - **Consequences**: Enables responsive handling of urgent jobs while preventing starvation of lower-priority work. Requires careful priority score design to avoid edge cases.

The priority system uses enumerated levels that map to numeric scores for queue ordering. The priority-to-score mapping ensures that urgent jobs always process before high-priority jobs, which process before normal and low-priority jobs respectively.

| Priority Level | Numeric Score | Use Case | Typical SLA |
|---------------|---------------|----------|-------------|
| `JobPriority.URGENT` | 20 | Real-time processing requests, critical system operations | < 30 seconds |
| `JobPriority.HIGH` | 10 | User-facing operations, thumbnail generation | < 2 minutes |
| `JobPriority.NORMAL` | 5 | Standard processing requests, background transcoding | < 10 minutes |
| `JobPriority.LOW` | 1 | Batch operations, archive processing, non-critical tasks | Best effort |

Worker processes continuously poll the queue using blocking pop operations that respect priority ordering. The polling mechanism implements a **hybrid pull model** where workers request jobs based on their current capacity and specialization. This approach prevents overwhelming individual workers while ensuring efficient resource utilization across the worker pool.

The job distribution algorithm considers several factors beyond simple priority ordering. Workers maintain capability flags indicating which media formats and processing types they support. For example, a worker running on a GPU-enabled machine might advertise video transcoding capabilities, while CPU-only workers focus on image processing tasks. The queue system matches job requirements to worker capabilities during the assignment process.

> **Critical Insight**: Job priority affects scheduling order, but it doesn't preempt running jobs. Once a worker begins processing a job, it continues to completion regardless of higher-priority jobs arriving in the queue. This design prevents resource waste from partially completed work while still providing prioritization benefits for queued jobs.

The queue implements several advanced features to handle real-world operational challenges. **Job deduplication** prevents duplicate processing when clients accidentally submit the same request multiple times. The system generates a content-based hash from the input file checksum and output specifications, rejecting duplicate submissions within a configurable time window.

**Batch job submission** allows clients to submit multiple related jobs atomically, ensuring they're processed as a cohesive unit. This feature proves particularly valuable for generating multiple video quality variants or processing image galleries where partial completion would be problematic.

The queue system maintains comprehensive metrics and monitoring data to support operational visibility. Key metrics include queue depth by priority level, average job processing time, worker utilization rates, and failure rates categorized by error type. These metrics feed into alerting systems that notify operators of queue backup, worker failures, or other operational issues requiring attention.

### Worker Process Management

Worker process management represents one of the most complex aspects of the job queue system, requiring careful coordination of resource allocation, process isolation, health monitoring, and failure recovery. Each worker operates as an independent process with clearly defined responsibilities and resource constraints.

The **worker lifecycle** follows a standardized pattern designed to ensure reliable operation under various failure scenarios. During startup, each worker registers with the job queue system, advertising its capabilities, resource limits, and processing specializations. This registration process includes a health check sequence that verifies the worker can access required dependencies like FFmpeg, storage systems, and temporary workspace directories.

| Worker State | Description | Actions Permitted | Monitoring |
|--------------|-------------|-------------------|------------|
| `STARTING` | Worker initializing, loading configuration | Registration, dependency checks | Startup timeout |
| `IDLE` | Worker available, polling for jobs | Accept jobs, health checks | Heartbeat monitoring |
| `PROCESSING` | Worker executing job | Progress updates, resource monitoring | Job timeout, memory limits |
| `DRAINING` | Worker finishing current job before shutdown | Complete current job only | Graceful shutdown timer |
| `FAILED` | Worker encountered unrecoverable error | Cleanup, restart procedures | Error reporting |

Each worker maintains a **resource budget** that prevents individual jobs from consuming excessive system resources. The budget includes memory limits, processing time constraints, temporary disk space allocation, and network bandwidth quotas. Before accepting a job, workers verify they have sufficient resources to complete the processing without exceeding their allocated limits.

Memory management requires particular attention given the potentially large size of media files. Workers implement a **staged processing approach** where they estimate memory requirements before loading media files into memory. For video processing, this might involve analyzing file headers to determine resolution and codec information, then calculating expected memory usage based on frame buffer requirements and transcoding parameters.

```
Memory Estimation Process:
1. Analyze input file headers to extract resolution and format information
2. Calculate raw frame buffer size: width × height × channels × bit_depth
3. Estimate transcoding memory overhead based on codec complexity
4. Add safety margin for temporary buffers and processing overhead
5. Compare total estimate to available worker memory budget
6. Reject job if memory requirements exceed budget, otherwise proceed
```

The worker pool implements **horizontal scaling** through a coordinator process that monitors overall system load and can spawn additional workers when queue depth exceeds configurable thresholds. Conversely, during low-demand periods, workers can be gracefully terminated to reduce resource consumption. This auto-scaling mechanism responds to both queue depth metrics and historical load patterns to anticipate demand fluctuations.

**Process isolation** ensures that failures in one worker don't affect others or compromise system stability. Each worker runs in a separate process with restricted file system access, limited network permissions, and resource quotas enforced at the operating system level. Workers communicate with the job queue and storage systems through well-defined APIs that validate all inputs and sanitize file paths to prevent directory traversal attacks.

Health monitoring operates at multiple levels to detect various failure modes. **Heartbeat monitoring** requires workers to send periodic status updates to the queue system, including current job progress, resource utilization, and processing stage information. Workers that fail to send heartbeats within the configured timeout period are marked as failed and their current jobs are rescheduled for processing by healthy workers.

> **Decision: Process-Based Worker Isolation**
> - **Context**: Need to isolate media processing operations to prevent failures from affecting other jobs
> - **Options Considered**: Thread-based workers, process-based workers, containerized workers
> - **Decision**: Separate processes with resource limits
> - **Rationale**: Processes provide strong isolation boundaries, prevent memory leaks from affecting other jobs, and enable independent restart of failed workers. Operating system process limits provide reliable resource enforcement.
> - **Consequences**: Higher memory overhead compared to threads, but significantly improved fault tolerance and resource management. Simplified debugging since each worker has independent memory space.

Worker coordination handles several challenging scenarios that arise in distributed processing environments. **Split-brain prevention** ensures that only one worker processes a given job, even during network partitions or queue system failures. This protection uses distributed locking with expiration timeouts to prevent jobs from being permanently orphaned.

The system implements **graceful degradation** when worker capacity becomes insufficient for demand. Rather than rejecting new jobs outright, the queue system can temporarily increase job timeout limits, reduce quality settings for non-critical processing, or defer low-priority jobs until capacity improves. These degradation strategies allow the system to continue operating during peak load periods without complete service disruption.

### Job Queue Architecture Decisions

The architecture of the job queue system required careful evaluation of multiple design alternatives, each with significant implications for scalability, reliability, and operational complexity. These decisions form the foundation that enables the entire media processing pipeline to operate reliably under production workloads.

**Message Broker Selection** represents the most fundamental architectural choice, as it determines the queue's performance characteristics, durability guarantees, and operational requirements. The decision between Redis, RabbitMQ, Apache Kafka, and cloud-native solutions like AWS SQS required analysis of both technical capabilities and operational constraints.

> **Decision: Redis with Persistence for Message Broker**
> - **Context**: Need reliable job queuing with priority support, reasonable throughput, and operational simplicity
> - **Options Considered**: Redis (in-memory), RabbitMQ (AMQP), Apache Kafka (distributed log), AWS SQS (managed service)
> - **Decision**: Redis with AOF persistence enabled
> - **Rationale**: Redis sorted sets provide native priority queue functionality with atomic operations. AOF persistence ensures job durability across restarts. Simpler operational model than RabbitMQ or Kafka for media processing workloads. Single-node deployment sufficient for moderate scale.
> - **Consequences**: Excellent performance for job queuing patterns, but requires careful memory management for large job volumes. Limited horizontal scaling compared to Kafka. Simplified deployment and monitoring.

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Redis** | Native priority queues, atomic operations, simple deployment | Memory-limited storage, single-node bottleneck | ✅ **Chosen** |
| **RabbitMQ** | Mature AMQP implementation, flexible routing, persistent queues | Complex configuration, requires queue management | Considered |
| **Apache Kafka** | High throughput, horizontal scaling, replay capability | Over-engineered for job queuing, complex operations | Rejected |
| **AWS SQS** | Fully managed, automatic scaling, pay-per-use | Vendor lock-in, limited priority support, higher latency | Rejected |

**Serialization Format** affects both performance and system evolution capability. The choice between JSON, MessagePack, Protocol Buffers, and other serialization approaches required balancing human readability, parsing performance, schema evolution support, and cross-language compatibility.

> **Decision: JSON with Schema Validation**
> - **Context**: Need human-readable job data for debugging with reasonable parsing performance
> - **Options Considered**: JSON (human-readable), MessagePack (compact binary), Protocol Buffers (schema evolution)
> - **Decision**: JSON with JSON Schema validation for job structures
> - **Rationale**: Human readability crucial for debugging media processing issues. JSON parsing performance adequate for job queue throughput. Schema validation prevents malformed job data. Wide language support for future extensions.
> - **Consequences**: Larger message sizes than binary formats, but acceptable for job queue usage patterns. Easier debugging and system monitoring. Simplified client integration.

The **concurrency model** determines how the system handles multiple simultaneous jobs and coordinates access to shared resources. The design choice between actor-based concurrency, shared-memory threading, and process-based isolation has far-reaching implications for system reliability and debugging complexity.

Process-based concurrency was selected over threading or actor models primarily due to the resource-intensive nature of media processing operations. FFmpeg and image processing libraries can consume significant memory and CPU resources, making process isolation essential for preventing resource leaks from affecting other jobs. Additionally, many media processing libraries have complex memory management patterns that are difficult to coordinate safely in multi-threaded environments.

**Job Persistence Strategy** required balancing durability guarantees with system performance. The choice between storing complete job data in the message queue versus using lightweight job references with external storage affects both system complexity and failure recovery capabilities.

| Persistence Approach | Description | Durability | Performance | Complexity |
|---------------------|-------------|------------|-------------|------------|
| **In-Queue Job Data** | Complete job specifications stored in Redis | High | Medium | Low |
| **Reference + Database** | Job IDs in queue, full data in PostgreSQL | Very High | Lower | Medium |
| **Hybrid Approach** | Essential data in queue, full specs in storage | High | High | Medium |

The system implements a **hybrid persistence approach** where the job queue contains essential execution metadata (job ID, priority, worker assignment) while complete job specifications and processing history are stored in external persistent storage. This design provides the performance benefits of in-memory job queuing while ensuring that job data survives system restarts and can be analyzed for historical reporting.

**Worker Discovery and Load Balancing** mechanisms determine how jobs are distributed across available workers and how the system adapts to changing worker capacity. The choice between push-based job assignment and pull-based worker polling affects system responsiveness and load distribution fairness.

The implemented pull-based model allows workers to request jobs based on their current capacity and capabilities, preventing overloading of slower workers while ensuring that fast workers remain fully utilized. Workers include capability advertisements in their job requests, allowing the queue system to match job requirements with worker specializations automatically.

**Error Handling and Recovery Architecture** establishes how the system responds to various failure scenarios, from temporary network issues to permanent worker failures. The decision between immediate retry, exponential backoff, dead letter queues, and manual intervention affects both system reliability and operational burden.

The multi-tiered error handling approach categorizes failures based on their likelihood of recovery and implements appropriate response strategies for each category. Transient failures (network timeouts, temporary resource exhaustion) trigger automatic retry with exponential backoff. Persistent failures (corrupted input files, unsupported formats) are moved to dead letter queues for manual investigation. System failures (worker crashes, storage unavailability) trigger immediate job rescheduling to healthy workers.

### Common Queue Implementation Pitfalls

Implementing a robust job queue system involves navigating numerous subtle pitfalls that can compromise reliability, performance, or data consistency. Understanding these common mistakes helps avoid production incidents and ensures the system operates correctly under all conditions.

⚠️ **Pitfall: Job Deduplication Race Conditions**

Many implementations fail to properly handle the race condition where multiple workers might pick up duplicate jobs before the deduplication check completes. This occurs when clients submit the same job multiple times in rapid succession, and the queue system hasn't completed processing the first submission before subsequent duplicates arrive.

The incorrect approach uses a simple "check then insert" pattern that's vulnerable to race conditions:
```
1. Check if job with same hash exists in queue
2. If not found, insert new job
3. Return job ID
```

Between steps 1 and 2, another thread might insert the same job, resulting in duplicate processing. The correct implementation uses atomic operations provided by Redis to ensure deduplication works even under high concurrency. The system generates a content-based hash from the input file and output specifications, then uses Redis's SET with NX (not exists) flag to atomically insert the job only if no duplicate exists.

⚠️ **Pitfall: Improper Dead Letter Queue Management**

A common mistake is implementing dead letter queues without proper categorization of failure types, leading to either excessive manual intervention or silent data loss. Not all job failures should go to the dead letter queue—some represent permanent errors that should never be retried, while others might be temporary issues that resolve automatically.

The system should categorize failures into distinct types with appropriate handling strategies:
- **Transient failures** (network timeouts, temporary resource exhaustion): Retry with exponential backoff up to maximum retry limit
- **Permanent failures** (corrupted input files, unsupported formats): Move to dead letter queue with detailed error information for manual review
- **System failures** (worker crashes, storage unavailability): Reschedule on different worker without counting against retry limit

Additionally, dead letter queues require active monitoring and periodic cleanup to prevent indefinite accumulation of failed jobs. The system should implement automated alerts when dead letter queue depth exceeds thresholds and provide tools for bulk job analysis and cleanup.

⚠️ **Pitfall: Resource Cleanup on Worker Failure**

Worker processes that crash or are forcibly terminated often leave behind temporary files, partially processed media, and locked resources. Without proper cleanup mechanisms, these artifacts accumulate over time and can exhaust disk space or cause resource conflicts for subsequent jobs.

The solution requires implementing cleanup procedures at multiple levels:
1. **Graceful shutdown handlers** that cleanup temporary files when workers receive termination signals
2. **Periodic cleanup processes** that identify and remove orphaned temporary files based on age and worker heartbeat status
3. **Job reschedule logic** that ensures crashed jobs are properly reset and rescheduled rather than left in a permanent "processing" state
4. **Resource lock recovery** that releases file locks and other resources held by failed workers

⚠️ **Pitfall: Progress Estimation Accuracy**

Many implementations provide progress estimates that are wildly inaccurate, either showing 90% completion for hours or jumping from 10% to 100% instantly. This occurs because developers base progress estimates on simple metrics like file size processed rather than actual processing complexity.

Media processing operations have highly variable computational requirements depending on content characteristics. A one-hour video with static scenes processes much faster than a one-hour video with rapid motion and complex visual effects, even though both files might be similar in size. The system should implement **stage-based progress reporting** rather than attempting to provide precise percentage estimates.

The correct approach divides processing into discrete stages (validation, preprocessing, encoding, postprocessing) and reports progress as stage completion rather than attempting granular percentage tracking. This provides users with meaningful progress information while avoiding the false precision of inaccurate time estimates.

⚠️ **Pitfall: Worker Capacity Estimation**

Incorrectly estimating worker processing capacity leads to either resource over-allocation (causing memory exhaustion and crashes) or under-utilization (leaving workers idle while jobs queue up). The challenge lies in predicting resource requirements for media processing jobs before loading the actual media files.

The system should implement **conservative capacity estimation** that analyzes input file metadata to predict memory and CPU requirements. For images, this involves calculating raw pixel buffer size based on resolution and bit depth. For videos, estimation includes frame buffer requirements for the target resolution plus transcoding overhead for the selected codec.

Workers should maintain resource budgets that account for peak memory usage during processing, not just average usage. A safety margin prevents edge cases from causing worker crashes, and jobs that exceed estimated resource requirements should be gracefully rejected rather than causing worker failures.

⚠️ **Pitfall: Queue Monitoring Blind Spots**

Production issues often arise from monitoring gaps that miss critical failure modes. Common blind spots include gradual memory leaks in workers, increasing job processing times due to resource contention, and cascade failures where problems in one component affect others.

Comprehensive monitoring should track both technical metrics (queue depth, processing times, error rates) and business metrics (job completion rates, customer-facing latency, output quality measures). Alert thresholds should account for normal operational variance while detecting genuine issues early enough for proactive intervention.

The monitoring system should implement **trend analysis** that detects gradual degradation in addition to sudden failures. For example, slowly increasing job processing times might indicate resource exhaustion or performance regression that requires attention before it becomes a customer-impacting outage.

### Implementation Guidance

The job queue and scheduling component serves as the coordination backbone for the entire media processing system, managing job distribution, worker coordination, and progress tracking across the processing pipeline. This implementation guidance provides the infrastructure foundation and core logic structure needed to build a production-ready job queue system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Message Broker** | Redis with `redis-py` (in-memory sorted sets) | Redis Cluster with `redis-py-cluster` (distributed) |
| **Worker Framework** | Custom worker processes with multiprocessing | Celery with Redis broker (full-featured task queue) |
| **Job Persistence** | Redis with AOF persistence | PostgreSQL with Redis cache layer |
| **Process Monitoring** | Python multiprocessing with custom health checks | Supervisor process management with monitoring |
| **Configuration** | YAML files with `pyyaml` | Consul/etcd distributed configuration |
| **Logging** | Python `logging` with structured JSON output | ELK stack (Elasticsearch, Logstash, Kibana) |

#### Recommended File Structure

```
media_pipeline/
  queue/
    __init__.py
    models.py                 ← Job, Priority, Status data structures
    queue_manager.py          ← Job submission, priority handling
    worker_pool.py           ← Worker process management and coordination
    progress_tracker.py      ← Progress updates and webhook notifications
    redis_backend.py         ← Redis operations and connection management
    worker_process.py        ← Individual worker implementation
  config/
    queue_config.py          ← Queue configuration and validation
    worker_config.py         ← Worker resource limits and capabilities
  tests/
    test_queue_operations.py ← Queue submission and priority tests
    test_worker_management.py ← Worker lifecycle and coordination tests
    integration/
      test_job_processing.py ← End-to-end job processing tests
  scripts/
    start_workers.py         ← Worker startup script
    queue_monitor.py         ← Queue monitoring and alerting
    cleanup_jobs.py          ← Maintenance and cleanup utilities
```

#### Infrastructure Starter Code

**Redis Backend Connection Management** (`redis_backend.py`):
```python
import redis
import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RedisBackend:
    """Redis backend for job queue with connection pooling and error handling."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.pool = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            decode_responses=True,
            max_connections=20
        )
        self.redis = redis.Redis(connection_pool=self.pool)
        
        # Queue and status key patterns
        self.job_queue_key = "media:jobs:queue"
        self.job_data_key = "media:jobs:data:{}"
        self.worker_status_key = "media:workers:status"
        self.progress_key = "media:jobs:progress:{}"
    
    def health_check(self) -> bool:
        """Verify Redis connection and basic operations."""
        try:
            return self.redis.ping()
        except redis.RedisError as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def submit_job_atomic(self, job: ProcessingJob) -> bool:
        """Atomically submit job with deduplication check."""
        pipe = self.redis.pipeline()
        try:
            # Use job content hash for deduplication
            dedup_key = f"media:jobs:dedup:{job.content_hash()}"
            job_data_key = self.job_data_key.format(job.job_id)
            
            # Watch deduplication key for atomic operation
            pipe.watch(dedup_key)
            
            if pipe.get(dedup_key):
                logger.info(f"Duplicate job detected for hash {job.content_hash()}")
                return False
            
            # Atomic job submission
            pipe.multi()
            pipe.setex(dedup_key, timedelta(hours=1), job.job_id)
            pipe.hset(job_data_key, mapping=job.to_dict())
            pipe.zadd(self.job_queue_key, {job.job_id: job.priority.value})
            pipe.execute()
            
            logger.info(f"Job {job.job_id} submitted successfully")
            return True
            
        except redis.WatchError:
            logger.warning("Job submission failed due to concurrent modification")
            return False
        except redis.RedisError as e:
            logger.error(f"Redis error during job submission: {e}")
            return False
        finally:
            pipe.reset()
    
    def pop_highest_priority_job(self, worker_id: str, timeout: int = 10) -> Optional[ProcessingJob]:
        """Blocking pop of highest priority job with worker assignment."""
        try:
            # Use blocking pop on sorted set (highest score first)
            result = self.redis.bzpopmax(self.job_queue_key, timeout=timeout)
            if not result:
                return None
            
            _, job_id, priority = result
            job_data_key = self.job_data_key.format(job_id)
            
            # Atomically assign worker and update status
            pipe = self.redis.pipeline()
            job_data = pipe.hgetall(job_data_key)
            pipe.hset(job_data_key, 'assigned_worker', worker_id)
            pipe.hset(job_data_key, 'status', JobStatus.PROCESSING.value)
            pipe.hset(job_data_key, 'started_at', datetime.utcnow().isoformat())
            pipe.execute()
            
            if job_data:
                job = ProcessingJob.from_dict(job_data)
                logger.info(f"Job {job_id} assigned to worker {worker_id}")
                return job
            
            logger.error(f"Job data not found for job {job_id}")
            return None
            
        except redis.RedisError as e:
            logger.error(f"Redis error during job pop: {e}")
            return None
    
    def update_job_progress(self, job_id: str, progress_percentage: float, 
                          stage: str, details: Dict[str, Any] = None) -> bool:
        """Update job progress and timestamp."""
        try:
            progress_data = {
                'job_id': job_id,
                'progress_percentage': progress_percentage,
                'stage': stage,
                'updated_at': datetime.utcnow().isoformat(),
                'details': json.dumps(details or {})
            }
            
            progress_key = self.progress_key.format(job_id)
            job_data_key = self.job_data_key.format(job_id)
            
            pipe = self.redis.pipeline()
            pipe.hset(progress_key, mapping=progress_data)
            pipe.hset(job_data_key, 'progress_percentage', progress_percentage)
            pipe.expire(progress_key, timedelta(days=1))  # Auto-expire progress data
            pipe.execute()
            
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to update progress for job {job_id}: {e}")
            return False
    
    def mark_job_completed(self, job_id: str, output_files: List[str]) -> bool:
        """Mark job as completed and store output file paths."""
        try:
            completion_data = {
                'status': JobStatus.COMPLETED.value,
                'completed_at': datetime.utcnow().isoformat(),
                'output_files': json.dumps(output_files),
                'progress_percentage': 100.0
            }
            
            job_data_key = self.job_data_key.format(job_id)
            self.redis.hset(job_data_key, mapping=completion_data)
            
            logger.info(f"Job {job_id} marked as completed")
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to mark job {job_id} as completed: {e}")
            return False
    
    def mark_job_failed(self, job_id: str, error_message: str, retry_eligible: bool = True) -> bool:
        """Mark job as failed and handle retry logic."""
        try:
            job_data_key = self.job_data_key.format(job_id)
            current_data = self.redis.hgetall(job_data_key)
            
            if not current_data:
                logger.error(f"Job data not found for {job_id}")
                return False
            
            retry_count = int(current_data.get('retry_count', 0))
            max_retries = 3  # Should come from config
            
            failure_data = {
                'status': JobStatus.FAILED.value,
                'error_message': error_message,
                'failed_at': datetime.utcnow().isoformat(),
                'retry_count': retry_count + 1
            }
            
            pipe = self.redis.pipeline()
            pipe.hset(job_data_key, mapping=failure_data)
            
            # Retry logic
            if retry_eligible and retry_count < max_retries:
                # Calculate exponential backoff delay
                delay_seconds = min(300, 10 * (2 ** retry_count))  # Cap at 5 minutes
                
                # Reschedule job with delay
                retry_time = datetime.utcnow() + timedelta(seconds=delay_seconds)
                pipe.zadd("media:jobs:retry", {job_id: retry_time.timestamp()})
                
                logger.info(f"Job {job_id} scheduled for retry in {delay_seconds} seconds")
            else:
                # Move to dead letter queue for manual review
                pipe.lpush("media:jobs:dead_letter", job_id)
                logger.error(f"Job {job_id} moved to dead letter queue after {retry_count} retries")
            
            pipe.execute()
            return True
            
        except redis.RedisError as e:
            logger.error(f"Failed to mark job {job_id} as failed: {e}")
            return False
```

**Worker Process Infrastructure** (`worker_process.py`):
```python
import os
import signal
import psutil
import multiprocessing
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WorkerProcess:
    """Individual worker process with resource management and health monitoring."""
    
    def __init__(self, worker_id: str, config: ProcessingConfig, redis_backend: RedisBackend):
        self.worker_id = worker_id
        self.config = config
        self.redis_backend = redis_backend
        self.current_job: Optional[ProcessingJob] = None
        self.start_time = datetime.utcnow()
        self.is_shutting_down = False
        self.resource_monitor = ResourceMonitor()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
    
    def run(self):
        """Main worker loop with job processing and health monitoring."""
        logger.info(f"Worker {self.worker_id} starting up")
        
        try:
            self._register_worker()
            
            while not self.is_shutting_down:
                # TODO 1: Send heartbeat with current status and resource usage
                # TODO 2: Check if worker should shutdown (maintenance mode, resource limits)
                # TODO 3: Poll for available job with timeout
                # TODO 4: If job received, validate worker can process it
                # TODO 5: Execute job processing with progress callbacks
                # TODO 6: Handle job completion or failure
                # TODO 7: Cleanup temporary resources after job
                # TODO 8: Brief sleep to prevent busy polling
                pass
                
        except Exception as e:
            logger.error(f"Worker {self.worker_id} encountered fatal error: {e}")
            self._emergency_cleanup()
        finally:
            self._unregister_worker()
            logger.info(f"Worker {self.worker_id} shutdown complete")
    
    def _register_worker(self):
        """Register worker with queue system and advertise capabilities."""
        # TODO 1: Create worker status record with capabilities
        # TODO 2: Register in Redis worker registry
        # TODO 3: Send initial heartbeat
        # Hint: Include worker_id, start_time, capabilities, resource_limits
        pass
    
    def _send_heartbeat(self) -> bool:
        """Send worker status update to queue system."""
        try:
            status_data = {
                'worker_id': self.worker_id,
                'status': 'processing' if self.current_job else 'idle',
                'current_job': self.current_job.job_id if self.current_job else None,
                'heartbeat_time': datetime.utcnow().isoformat(),
                'cpu_usage': self.resource_monitor.get_cpu_usage(),
                'memory_usage': self.resource_monitor.get_memory_usage(),
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds()
            }
            
            # Store heartbeat with expiration
            key = f"media:workers:heartbeat:{self.worker_id}"
            self.redis_backend.redis.setex(key, timedelta(minutes=5), status_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False
    
    def _handle_shutdown_signal(self, signum, frame):
        """Handle graceful shutdown signal."""
        logger.info(f"Worker {self.worker_id} received shutdown signal {signum}")
        self.is_shutting_down = True
        
        if self.current_job:
            logger.info(f"Completing current job {self.current_job.job_id} before shutdown")
            # Allow current job to complete but don't accept new jobs

class ResourceMonitor:
    """Monitor worker resource usage and enforce limits."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.start_time = datetime.utcnow()
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Track peak memory usage
        self.peak_memory = max(self.peak_memory, memory_info.rss)
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': memory_percent,
            'peak_mb': self.peak_memory / (1024 * 1024)
        }
    
    def check_resource_limits(self, limits: Dict[str, Any]) -> Dict[str, bool]:
        """Check if current resource usage exceeds configured limits."""
        # TODO 1: Compare current memory usage to limits.max_memory_mb
        # TODO 2: Compare current CPU usage to limits.max_cpu_percent
        # TODO 3: Check disk usage in temporary directories
        # TODO 4: Return dict with limit_exceeded: bool for each resource
        # Hint: This helps prevent workers from overwhelming the system
        pass
```

#### Core Logic Skeleton Code

**Job Queue Manager** (`queue_manager.py`):
```python
class QueueManager:
    """Manages job submission, priority queuing, and worker coordination."""
    
    def __init__(self, redis_backend: RedisBackend, config: AppConfig):
        self.redis_backend = redis_backend
        self.config = config
        self.job_stats = JobStatistics()
        
    def submit_job(self, input_file: str, output_specs: List[OutputSpecification], 
                   priority: JobPriority, webhook_url: str = None) -> ProcessingJob:
        """Submit new processing job to queue with priority handling."""
        # TODO 1: Validate input_file exists and is readable
        # TODO 2: Validate output_specs have valid formats and parameters
        # TODO 3: Generate unique job_id using generate_job_id()
        # TODO 4: Create ProcessingJob instance with all parameters
        # TODO 5: Submit job atomically using redis_backend.submit_job_atomic()
        # TODO 6: If submission successful, send webhook notification
        # TODO 7: Update job statistics and metrics
        # TODO 8: Return ProcessingJob instance
        # Hint: Use try/except to handle validation and submission errors
        pass
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get current queue depth and processing statistics by priority."""
        # TODO 1: Query Redis for total jobs in queue
        # TODO 2: Count jobs by priority level using ZRANGEBYSCORE
        # TODO 3: Get count of active workers from heartbeat keys
        # TODO 4: Calculate average job processing time from recent completions
        # TODO 5: Get failed job count from dead letter queue
        # TODO 6: Return comprehensive statistics dict
        # Hint: Use Redis pipeline for efficient multi-query operations
        pass
    
    def cleanup_expired_jobs(self, max_age_hours: int = 24):
        """Remove old job data and cleanup temporary resources."""
        # TODO 1: Find jobs older than max_age_hours in COMPLETED/FAILED status
        # TODO 2: Archive job data to long-term storage if configured
        # TODO 3: Remove job data keys from Redis
        # TODO 4: Cleanup temporary files associated with old jobs
        # TODO 5: Update cleanup metrics and log results
        # Hint: Use SCAN to iterate through job keys efficiently
        pass

class WorkerCoordinator:
    """Coordinates worker processes and manages worker pool scaling."""
    
    def __init__(self, redis_backend: RedisBackend, config: ProcessingConfig):
        self.redis_backend = redis_backend
        self.config = config
        self.worker_processes: Dict[str, multiprocessing.Process] = {}
        self.target_worker_count = config.max_workers
        
    def start_worker_pool(self):
        """Start configured number of worker processes."""
        # TODO 1: Calculate optimal worker count based on CPU cores and memory
        # TODO 2: Create worker processes using multiprocessing.Process
        # TODO 3: Start each worker process and track PIDs
        # TODO 4: Register signal handlers for pool shutdown
        # TODO 5: Start monitoring thread for worker health checks
        # Hint: Store worker PIDs for graceful shutdown management
        pass
    
    def scale_worker_pool(self, target_count: int):
        """Dynamically scale worker pool up or down."""
        # TODO 1: Compare target_count to current active workers
        # TODO 2: If scaling up, start additional worker processes
        # TODO 3: If scaling down, signal workers to shutdown gracefully
        # TODO 4: Wait for workers to complete current jobs before termination
        # TODO 5: Update worker pool tracking and metrics
        # Hint: Always allow current jobs to complete before shutdown
        pass
    
    def monitor_worker_health(self):
        """Monitor worker heartbeats and restart failed workers."""
        # TODO 1: Check heartbeat timestamps for all registered workers
        # TODO 2: Identify workers with stale heartbeats (likely crashed)
        # TODO 3: Restart failed workers and reassign their jobs
        # TODO 4: Send alerts for worker failures and restarts
        # TODO 5: Update worker health metrics
        # Hint: Use heartbeat timeout from config to detect failures
        pass
```

#### Milestone Checkpoints

**Checkpoint 1: Basic Queue Operations**
After implementing the Redis backend and basic job submission:
```bash
python -m pytest tests/test_queue_operations.py -v
```

Expected behavior:
- Jobs submitted with different priorities are retrieved in correct order
- Duplicate job submission is properly rejected with deduplication
- Job status transitions work correctly (PENDING → PROCESSING → COMPLETED)
- Redis connection handling works with proper error recovery

Test manually:
```python
from queue.queue_manager import QueueManager
from queue.models import ProcessingJob, JobPriority

# Submit test job
manager = QueueManager(redis_backend, config)
job = manager.submit_job("test.jpg", [output_spec], JobPriority.HIGH)
print(f"Job submitted: {job.job_id}")

# Check queue statistics
stats = manager.get_queue_statistics()
print(f"Queue depth: {stats['total_jobs']}")
```

**Checkpoint 2: Worker Process Management**
After implementing worker processes and coordination:
```bash
python scripts/start_workers.py --worker-count 2
python -m pytest tests/test_worker_management.py -v
```

Expected behavior:
- Worker processes start and register with queue system
- Heartbeat monitoring detects healthy and failed workers
- Jobs are distributed to available workers based on capabilities
- Graceful shutdown completes current jobs before terminating

**Checkpoint 3: End-to-End Job Processing**
After implementing complete job processing pipeline:
```bash
python -m pytest tests/integration/test_job_processing.py -v
```

Expected behavior:
- Jobs flow from submission through processing to completion
- Progress updates are tracked and reported correctly
- Failed jobs are retried with exponential backoff
- Webhook notifications are sent at appropriate stages
- Resource cleanup occurs after job completion

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| **Jobs stuck in PENDING** | No available workers or worker capability mismatch | Check worker heartbeats in Redis, verify worker capabilities | Start workers or check worker capability registration |
| **Workers not picking up jobs** | Polling timeout too short or Redis connection issues | Check worker logs for Redis errors, verify queue key names | Increase polling timeout, verify Redis connectivity |
| **Memory exhaustion crashes** | Worker memory limits not enforced or estimation incorrect | Monitor worker memory usage, check job size estimation | Implement proper memory limits, improve size estimation |
| **Jobs processed multiple times** | Duplicate job submission or improper deduplication | Check job deduplication keys in Redis, verify atomic submission | Fix deduplication hash calculation, ensure atomic operations |
| **Progress updates missing** | Worker not sending updates or Redis key expiration | Check progress update calls in worker, verify Redis keys | Add progress update calls, adjust Redis key expiration |
| **Dead letter queue growing** | Permanent failures not categorized properly | Analyze dead letter queue jobs, check error categorization | Improve error classification, add manual job review process |


## Progress Tracking and Notification Component

> **Milestone(s):** Milestone 3 (Processing Queue & Progress) - this section covers real-time progress updates, webhook notifications, and job status management across the processing pipeline

### Mental Model: Package Delivery Tracking

Think of progress tracking like a modern package delivery service such as FedEx or UPS. When you ship a package, you receive a tracking number that lets you monitor its journey from pickup to delivery. The package moves through distinct stages: "Package picked up", "Arrived at sorting facility", "Out for delivery", and "Delivered". Each stage transition triggers an update to the tracking system, and you can receive notifications via SMS, email, or app push notifications when important milestones occur.

Similarly, our media processing pipeline treats each processing job like a package moving through a fulfillment system. The job progresses through well-defined stages: "Job queued", "Processing started", "Image resizing complete", "Format conversion in progress", "Thumbnail generation", and "Processing complete". Just as package tracking provides estimated delivery times based on historical data and current conditions, our system estimates completion times based on file size, processing complexity, and current worker load.

The notification system works like delivery alerts - when significant events occur (job completion, failure, or major progress milestones), the system sends webhook notifications to registered callback URLs, similar to how delivery services send status updates to your phone. If a notification fails to deliver (like when your phone is offline), the system retries with exponential backoff, ensuring important updates eventually reach their destination.

This mental model helps us understand three key principles: **stage-based progress** (discrete milestones rather than continuous percentages), **event-driven notifications** (updates triggered by state changes), and **reliable delivery** (guaranteed notification delivery with retry mechanisms).

### Progress Calculation Strategies

Progress tracking in media processing presents unique challenges compared to simple file operations. Unlike copying files where progress correlates linearly with bytes transferred, media processing involves multiple distinct phases with varying computational complexity and duration. Our system employs **stage-based progress** rather than time-based estimates, providing more accurate and meaningful progress updates to users.

#### Stage-Based Progress Architecture

The foundation of our progress system divides each processing job into discrete, measurable stages. Each stage represents a significant computational milestone with clearly defined inputs, outputs, and completion criteria. This approach provides several advantages over percentage-based progress: stages are deterministic (a job always progresses through the same sequence), measurable (each stage has concrete completion criteria), and meaningful to users (stage names communicate actual processing activities).

| Stage Name | Description | Percentage Weight | Duration Factors |
|------------|-------------|------------------|------------------|
| `VALIDATION` | File format detection and metadata extraction | 5% | File size, format complexity |
| `PREPROCESSING` | EXIF handling, orientation correction | 10% | Metadata density, rotation needed |
| `RESIZE_OPERATIONS` | Primary resizing and cropping operations | 40% | Output count, resolution changes |
| `FORMAT_CONVERSION` | Transcoding between image/video formats | 35% | Codec complexity, quality settings |
| `OPTIMIZATION` | Compression and web optimization | 8% | Quality targets, advanced features |
| `FINALIZATION` | File writing and cleanup operations | 2% | Storage latency, temp file cleanup |

Each stage maintains its own progress tracking mechanism. For example, during `RESIZE_OPERATIONS`, progress increments as each output specification completes processing. If a job requires generating five different image sizes, progress within this stage advances by 20% for each completed resize operation. This granular tracking provides meaningful feedback even for complex jobs with dozens of output variants.

> **Design Insight**: Stage-based progress solves the estimation problem that plagues traditional progress bars. Instead of guessing "45% complete in 3.7 minutes", we can accurately report "Format conversion in progress - 3 of 5 outputs complete", which provides actionable information to users.

#### Time Estimation Algorithms

While stage-based progress forms our primary tracking mechanism, users still desire time estimates for planning purposes. Our system combines historical performance data with real-time job characteristics to generate dynamic estimates that improve in accuracy as processing progresses.

The estimation algorithm operates in three phases: **initial estimate** (based on file size and historical averages), **refinement phase** (adjusts estimates as early stages complete), and **final phase** (provides high-accuracy estimates for remaining work). Initial estimates use a regression model trained on historical job completion times, factoring in input file size, output specification count, and current worker load.

```
Initial Estimate Calculation:
1. Extract job characteristics: file_size, output_count, format_complexity_score
2. Query historical database for similar jobs (±20% file size, same format family)
3. Calculate baseline duration using weighted average of historical completion times
4. Apply load factor adjustment based on current queue depth and worker availability
5. Add safety margin (typically 15-25%) to account for estimation uncertainty
```

As processing progresses, the system refines estimates using actual stage completion times. If the `VALIDATION` stage completes faster than expected, the algorithm proportionally adjusts estimates for remaining stages. Conversely, if early stages take longer than predicted, the system increases estimates for subsequent work to maintain accuracy.

> **Decision: Dynamic Estimation Refinement**
> - **Context**: Static time estimates become inaccurate as processing reveals actual complexity
> - **Options Considered**: Fixed estimates, linear interpolation, machine learning models
> - **Decision**: Weighted refinement algorithm that adjusts estimates based on completed stage performance
> - **Rationale**: Balances simplicity with accuracy, provides meaningful updates without complex ML infrastructure
> - **Consequences**: Estimates improve in accuracy over time but may initially fluctuate as refinements apply

#### Progress Update Mechanisms

Progress updates must balance accuracy with performance overhead. Our system implements a **hierarchical update strategy** that provides frequent updates for user-facing operations while minimizing database write load and notification spam. The strategy employs different update frequencies based on the operation type and user visibility requirements.

For long-running operations like video transcoding, the system reports progress updates at regular time intervals (every 30 seconds) rather than after every processed frame. This prevents overwhelming the notification system while maintaining responsive user feedback. Short-duration operations like image resizing trigger updates only at stage boundaries to minimize overhead while still providing meaningful progress visibility.

| Operation Type | Update Frequency | Trigger Events | Notification Threshold |
|----------------|------------------|----------------|------------------------|
| Image Processing | Stage boundaries | Stage complete, error encountered | 25% progress increments |
| Video Transcoding | 30-second intervals | Time elapsed, segment complete | 20% progress increments |
| Batch Operations | Per-item completion | Individual job complete | Per-job completion |
| Format Conversion | Quality milestone | Encoding pass complete | Major milestone only |

The progress update pipeline employs **atomic state transitions** to prevent race conditions between multiple worker processes updating the same job. Each progress update includes a sequence number and timestamp to ensure updates apply in correct order and detect potential concurrency issues.

```
Atomic Progress Update Procedure:
1. Acquire advisory lock on job record using job_id as lock key
2. Read current job state and validate update is newer than stored progress
3. Calculate new progress percentage based on stage completion and weights
4. Update job record with new progress, stage, timestamp, and sequence number
5. Determine if notification threshold reached since last webhook delivery
6. Release advisory lock and trigger async notification if threshold met
7. Log progress update with correlation_id for debugging and audit trails
```

### Webhook Notification System

The webhook notification system provides reliable, real-time communication between the media processing pipeline and external applications. Unlike simple HTTP callbacks, our implementation handles the complexities of distributed systems: network failures, service outages, duplicate detection, and security verification. The system guarantees **at-least-once delivery** for critical notifications while providing **exactly-once semantics** for idempotent operations.

#### Webhook Event Types and Payloads

The notification system supports multiple event types, each with standardized payload structures optimized for different integration patterns. Events follow a consistent schema that includes correlation metadata, security signatures, and extensible payload sections for future enhancement without breaking existing integrations.

| Event Type | Trigger Condition | Payload Contents | Retry Policy |
|------------|-------------------|------------------|--------------|
| `job.started` | Job processing begins | Job metadata, input file info | Standard retry |
| `job.progress` | Major progress milestone | Current stage, percentage, ETA | No retry (informational) |
| `job.completed` | Processing successfully finishes | Output file URLs, metadata | Aggressive retry |
| `job.failed` | Permanent failure occurs | Error details, retry eligibility | Aggressive retry |
| `job.retry` | Job scheduled for retry | Retry count, next attempt time | Standard retry |

Each webhook payload includes a comprehensive event envelope that provides recipients with sufficient context for processing and debugging. The envelope structure promotes consistency across event types while allowing type-specific data in the payload section.

```
Webhook Payload Structure:
{
  "event_id": "unique identifier for deduplication",
  "event_type": "job.completed",
  "event_timestamp": "ISO 8601 timestamp of event occurrence", 
  "correlation_id": "request tracking identifier",
  "api_version": "v1",
  "signature": "HMAC-SHA256 signature for verification",
  "job": {
    "job_id": "processing job identifier",
    "status": "current job status enum value",
    "progress_percentage": "current completion percentage",
    "created_at": "job creation timestamp",
    "started_at": "processing start timestamp", 
    "completed_at": "completion timestamp if applicable"
  },
  "payload": {
    // Event-specific data varies by event_type
    "output_files": ["array of generated file URLs"],
    "processing_duration": "total processing time in seconds",
    "metadata": {"extracted media metadata"}
  }
}
```

The signature field contains an HMAC-SHA256 hash computed over the entire payload body using a shared secret configured per webhook endpoint. This prevents tampering and allows recipients to verify message authenticity and integrity. The signature calculation includes the raw JSON payload to prevent parsing-related security vulnerabilities.

#### Reliable Delivery Implementation

Webhook delivery reliability requires sophisticated retry logic that balances prompt notification delivery with respectful behavior toward recipient services. Our implementation employs **exponential backoff with jitter** to prevent thundering herd effects while ensuring important notifications eventually reach their destinations even during extended outages.

The retry system categorizes delivery failures into **transient** (network timeouts, 5xx responses), **permanent** (4xx client errors, invalid URLs), and **throttling** (429 rate limit responses) categories, each with appropriate retry behavior. Transient failures trigger exponential backoff retries, permanent failures immediately move to dead letter storage, and throttling failures use the `Retry-After` header when provided by the recipient.

| Failure Type | HTTP Status Codes | Retry Behavior | Max Attempts |
|--------------|------------------|----------------|--------------|
| Transient | 408, 500, 502, 503, 504 | Exponential backoff (1, 2, 4, 8, 16 min) | 5 attempts |
| Permanent | 400, 401, 403, 404, 410 | No retry, dead letter | 1 attempt |
| Rate Limiting | 429 | Honor Retry-After header | 3 attempts |
| Network Error | Connection timeout, DNS failure | Exponential backoff | 5 attempts |

```
Webhook Delivery Algorithm:
1. Construct webhook payload with current job state and event-specific data
2. Generate HMAC-SHA256 signature using configured webhook secret
3. Set HTTP headers: Content-Type, User-Agent, X-Webhook-Signature, X-Event-Type
4. Attempt HTTP POST delivery with 30-second timeout
5. Classify response: success (2xx), transient error, permanent error, rate limit
6. For transient errors: schedule retry with exponential backoff plus random jitter
7. For permanent errors: log failure and move to dead letter queue  
8. For rate limits: respect Retry-After header or use default backoff
9. Update delivery attempt metrics for monitoring and alerting
```

The delivery system maintains **idempotency** through unique event identifiers that allow recipients to detect and discard duplicate deliveries. This enables aggressive retry policies without fear of causing duplicate processing in downstream systems.

> **Decision: At-Least-Once Delivery Guarantee**
> - **Context**: Network failures and service outages require robust notification delivery
> - **Options Considered**: At-most-once, at-least-once, exactly-once delivery semantics
> - **Decision**: At-least-once delivery with idempotency keys for duplicate detection
> - **Rationale**: Balances reliability with complexity - easier to detect duplicates than recover lost messages
> - **Consequences**: Recipients must implement idempotency checking but gain strong delivery guarantees

#### Security and Authentication

Webhook security protects against message tampering, replay attacks, and unauthorized webhook injection. Our implementation provides **cryptographic verification** through HMAC signatures, **timestamp validation** to prevent replay attacks, and **endpoint validation** to ensure webhook URLs point to expected domains.

The signature verification process uses HMAC-SHA256 with webhook-specific secrets that rotate periodically for enhanced security. Recipients verify signatures by computing the HMAC over the raw request body and comparing it to the provided signature header. Mismatched signatures result in rejection and security logging for investigation.

```
Signature Verification Process:
1. Extract X-Webhook-Signature header from incoming request
2. Retrieve webhook secret associated with the endpoint
3. Compute HMAC-SHA256(secret, raw_request_body) 
4. Compare computed signature with provided signature using constant-time comparison
5. Verify timestamp is within acceptable window (default: 5 minutes)
6. Accept message if signature valid and timestamp fresh, otherwise reject
```

Timestamp validation prevents replay attacks where attackers capture valid webhook payloads and retransmit them later. The system includes the current timestamp in the signed payload and rejects messages older than a configurable threshold (typically 5-10 minutes).

### Progress Tracking Architecture Decisions

The progress tracking system requires careful architectural decisions that balance real-time responsiveness, storage efficiency, and system reliability. These decisions shape how the system handles concurrent updates, stores progress data, and provides query interfaces for both internal components and external integrations.

#### Storage Architecture for Progress Data

Progress tracking data exhibits unique characteristics that influence storage design: **high write frequency** (frequent progress updates), **temporal access patterns** (recent data accessed most often), **moderate retention requirements** (historical progress needed for debugging), and **real-time query needs** (dashboard and API access). These characteristics favor a hybrid storage approach combining in-memory caching for active jobs with persistent storage for historical data.

> **Decision: Redis + PostgreSQL Hybrid Storage**
> - **Context**: Progress data requires high-frequency writes with real-time read access
> - **Options Considered**: PostgreSQL only, Redis only, hybrid Redis+PostgreSQL approach
> - **Decision**: Active job progress in Redis with periodic persistence to PostgreSQL
> - **Rationale**: Redis provides sub-millisecond updates for active jobs, PostgreSQL ensures durability and complex queries
> - **Consequences**: Adds storage complexity but enables real-time performance with data durability

The hybrid approach stores **active job progress** in Redis using optimized data structures for fast updates and queries. Each active job maintains a Redis hash containing current progress state, stage information, and update timestamps. Redis persistence occurs through periodic snapshots to PostgreSQL, ensuring progress data survives system restarts while maintaining real-time performance.

| Data Category | Primary Storage | Persistence Layer | Access Pattern | Retention |
|---------------|-----------------|-------------------|----------------|-----------|
| Active Progress | Redis Hash | PostgreSQL sync every 30s | High frequency read/write | Until job complete |
| Completed Jobs | PostgreSQL | Archive after 90 days | Occasional queries | 90 days active |
| Progress History | PostgreSQL | Compressed storage | Analytics queries | 1 year archived |
| Notification Log | PostgreSQL | Time-series partitioning | Debugging access | 30 days |

Redis data structures optimize for common access patterns. Job progress uses Redis hashes for atomic field updates, while progress history uses sorted sets ordered by timestamp for efficient range queries. The notification system employs Redis lists for webhook delivery queues with automatic expiration for failed deliveries.

```
Redis Data Structure Design:
progress:job:{job_id} = Hash {
  "percentage": "current completion percentage",
  "stage": "current processing stage name", 
  "stage_progress": "progress within current stage",
  "updated_at": "timestamp of last update",
  "estimated_completion": "ETA timestamp",
  "notification_sent": "last notification percentage"
}

progress:history:{job_id} = Sorted Set {
  "timestamp1": "stage:percentage:details",
  "timestamp2": "stage:percentage:details" 
}

webhooks:pending = List [
  "webhook_delivery_request_1",
  "webhook_delivery_request_2"
]
```

#### Real-Time Update Mechanisms

Real-time progress updates serve multiple consumers with different latency requirements: **user dashboards** (sub-second updates), **API clients** (5-10 second intervals), **webhook notifications** (event-driven), and **monitoring systems** (1-minute aggregates). The system employs a **publish-subscribe pattern** with Redis Pub/Sub to efficiently distribute updates to all interested consumers without coupling update generation to consumption.

Progress updates flow through a **multi-tier notification system** that provides different update frequencies and delivery guarantees based on consumer requirements. Real-time dashboard connections receive immediate updates via WebSocket connections, while API clients can poll for updates at configurable intervals. Webhook notifications trigger only when significant thresholds are crossed to prevent notification spam.

| Consumer Type | Update Frequency | Delivery Method | Filtering |
|---------------|------------------|------------------|-----------|
| WebSocket Dashboard | Immediate | Redis Pub/Sub → WebSocket | All progress changes |
| REST API Clients | On-demand poll | Direct Redis query | Current state only |
| Webhook Notifications | Threshold-based | Async delivery queue | Major milestones |
| Monitoring Systems | 1-minute aggregates | Background collection | Statistical summaries |

The pub/sub architecture decouples progress generation from consumption, allowing the system to scale notification consumers independently of progress update frequency. Workers publish progress updates to Redis channels without waiting for acknowledgment, ensuring update operations remain fast and don't block processing.

```
Real-Time Update Flow:
1. Worker process calculates new progress percentage and stage information
2. Worker atomically updates Redis progress hash with new values
3. Worker publishes update message to Redis channel: progress:updates
4. WebSocket service subscribed to channel receives update immediately
5. WebSocket service filters updates and forwards to connected dashboard clients
6. Background process periodically persists Redis state to PostgreSQL
7. Webhook service checks notification thresholds and queues deliveries as needed
```

#### Concurrency Control and Race Conditions

Multiple worker processes may attempt simultaneous progress updates for the same job, particularly during batch operations or parallel processing stages. The system prevents race conditions through **optimistic concurrency control** using Redis atomic operations and sequence numbers that detect out-of-order updates.

Each progress update includes a **monotonically increasing sequence number** generated by the worker process performing the update. Redis WATCH/MULTI/EXEC transactions ensure progress updates apply atomically and reject out-of-order updates that could cause progress to move backward incorrectly.

> **Decision: Optimistic Concurrency with Sequence Numbers**
> - **Context**: Multiple workers may update progress for the same job simultaneously
> - **Options Considered**: Pessimistic locking, optimistic concurrency, last-writer-wins
> - **Decision**: Optimistic concurrency control with sequence numbers and Redis atomic operations
> - **Rationale**: Avoids lock contention while preventing race conditions and progress reversals
> - **Consequences**: Requires retry logic in workers but provides better scalability than locking

```
Atomic Progress Update Implementation:
1. Worker generates sequence number: current_timestamp_microseconds
2. Redis WATCH command monitors the progress hash for concurrent modifications  
3. Worker reads current sequence number from progress hash
4. If new sequence number <= current sequence, abort update (out of order)
5. Begin Redis MULTI transaction for atomic updates
6. Update progress hash fields: percentage, stage, sequence, timestamp
7. Publish update message to progress channel
8. EXEC transaction - succeeds only if no concurrent modifications detected
9. If transaction fails, retry with exponential backoff up to 3 attempts
```

Sequence number comparison prevents older updates from overwriting newer progress data, which could occur due to network delays or worker process variations in timing. Workers that detect rejected updates log the event for debugging but continue processing since another worker has already reported more recent progress.

### Common Progress Tracking Pitfalls

Progress tracking systems encounter predictable failure modes that can significantly impact user experience and system reliability. Understanding these pitfalls and their solutions helps avoid common implementation mistakes that lead to inaccurate progress reporting, notification failures, and poor system observability.

#### ⚠️ **Pitfall: Progress Reversals and Non-Monotonic Updates**

One of the most confusing user experiences occurs when progress appears to move backward, showing 75% complete followed by 60% complete. This typically happens when multiple worker processes or processing stages report progress concurrently without proper coordination, or when different estimation algorithms provide conflicting progress calculations.

Progress reversals commonly occur during **parallel processing scenarios** where multiple workers handle different output specifications for the same job. Worker A might complete 3 of 4 image resize operations (reporting 75% stage progress), while Worker B starts format conversion and reports early conversion progress (30% of final stage), causing the overall job progress to appear to decrease when calculated incorrectly.

**Why this breaks user trust**: Users expect progress to monotonically increase toward completion. Backward movement suggests system errors or inaccurate estimates, damaging confidence in the processing pipeline. Dashboard interfaces may show confusing progress bar animations that move backward, creating a broken user experience.

**Prevention and fixes**:
- Implement sequence numbers for all progress updates to detect and reject out-of-order updates
- Use atomic Redis operations to ensure progress updates apply consistently without race conditions  
- Design stage-based progress weights that sum to 100% and validate updates maintain monotonic progression
- Add progress validation logic that rejects updates showing backward movement unless explicitly resetting job state
- Log progress reversals as warnings for debugging while maintaining the last valid monotonic progress value

#### ⚠️ **Pitfall: Webhook Delivery Storms and Rate Limiting**

Aggressive progress reporting can overwhelm webhook endpoints with excessive notification traffic, particularly during batch processing operations or when multiple jobs complete simultaneously. This leads to rate limiting responses from recipient services, failed deliveries, and potential service degradation for webhook consumers.

Webhook storms typically occur when the system sends notifications for every progress increment rather than meaningful milestones. A video transcoding job might generate hundreds of progress updates per minute, causing the notification system to attempt hundreds of webhook deliveries that quickly exceed recipient rate limits and cause cascading failures.

**Why this causes problems**: Recipient services implement rate limiting to protect against abuse, and webhook storms can trigger these protections. Failed deliveries require retry logic that can compound the problem. Excessive webhook traffic also increases processing overhead and can impact system performance.

**Prevention and fixes**:
- Implement notification thresholds that trigger webhooks only for significant progress changes (typically 10-25% increments)
- Use webhook consolidation that batches multiple progress updates into single notifications when appropriate
- Add per-endpoint rate limiting in the webhook delivery system to respect recipient service limits
- Implement backoff strategies that honor `Retry-After` headers and reduce delivery frequency for rate-limited endpoints
- Provide webhook filtering configuration that lets recipients specify which event types and progress thresholds they want to receive

#### ⚠️ **Pitfall: Memory Leaks in Progress Storage**

Long-running systems accumulate progress tracking data for completed jobs, leading to memory exhaustion if not properly managed. Redis instances storing active job progress can grow unbounded when cleanup processes fail to remove completed job data, eventually causing out-of-memory errors and system instability.

Memory leaks often occur because progress cleanup depends on external job completion notifications that may be lost due to system failures or race conditions. Completed jobs leave behind Redis hashes, sorted sets, and pub/sub subscriptions that continue consuming memory without providing value.

**Why this causes system failures**: Redis operates primarily in memory, and unbounded growth leads to memory exhaustion. System administrators may not notice gradual memory increases until sudden spikes in job volume cause out-of-memory crashes. Recovery requires manual intervention and potential data loss.

**Prevention and fixes**:
- Implement automatic cleanup processes with Redis TTL (Time To Live) settings on all progress data structures
- Use separate cleanup workers that periodically scan for completed jobs and remove associated progress data
- Add memory monitoring and alerting for Redis instances to detect gradual memory growth trends
- Implement circuit breaker patterns that stop accepting new jobs when Redis memory usage exceeds safe thresholds
- Design progress data structures with explicit expiration policies rather than relying solely on cleanup processes

#### ⚠️ **Pitfall: Inaccurate Time Estimates and User Expectations**

Time estimation algorithms often provide wildly inaccurate completion estimates, particularly for complex media processing operations where actual processing time varies significantly based on content characteristics, system load, and processing complexity that's difficult to predict from input file metadata.

Estimation errors frequently occur when algorithms rely solely on file size without considering content complexity factors like video codec efficiency, image compression characteristics, or the computational cost of specific processing operations. A large but simply-encoded video may process faster than a smaller file with complex motion that requires intensive encoding.

**Why this damages user experience**: Inaccurate estimates lead to user frustration and poor planning. Estimates that consistently overshoot completion times cause users to abandon jobs they believe are stuck. Underestimated completion times cause users to wait expecting imminent completion.

**Prevention and fixes**:
- Use conservative estimation algorithms that tend to overestimate rather than underestimate completion times
- Implement estimation confidence intervals that communicate uncertainty to users ("15-25 minutes remaining")
- Refine estimates dynamically as processing progresses and actual performance data becomes available
- Provide stage-based progress communication ("Converting video format - 3 of 5 quality variants complete") that gives meaningful information without specific time commitments
- Track estimation accuracy over time and adjust algorithms based on historical performance data

#### ⚠️ **Pitfall: Race Conditions in Progress Percentage Calculations**

Complex jobs involving multiple parallel processing stages can produce race conditions when calculating overall progress percentages, leading to inconsistent or impossible progress values (such as percentages exceeding 100% or stage progress that doesn't align with overall job progress).

Race conditions typically emerge when different processing threads update stage-specific progress simultaneously while another thread calculates overall job progress. The calculation may read partially updated stage progress values, producing incorrect overall percentages that don't reflect actual processing state.

**Why this causes confusion**: Inconsistent progress reporting makes the system appear unreliable and can cause client applications to malfunction if they depend on progress values for logic decisions. Users see confusing progress displays that don't match the actual processing state.

**Prevention and fixes**:
- Use atomic read operations when calculating progress percentages to ensure consistent snapshots of all stage progress values
- Implement progress calculation locks or atomic transactions that prevent concurrent modification during percentage calculation
- Add validation logic that verifies calculated progress percentages fall within expected ranges (0-100%) and stage consistency
- Design progress data structures that support atomic updates of multiple related values
- Use eventual consistency patterns where progress displays may lag slightly but always show consistent, valid values

![Webhook Notification Flow](./diagrams/webhook-notification-flow.svg)

![Video Processing Sequence](./diagrams/video-processing-sequence.svg)

![Worker Process Coordination](./diagrams/worker-coordination.svg)

### Implementation Guidance

The progress tracking and notification system requires careful orchestration of real-time updates, persistent storage, and reliable webhook delivery. This implementation guidance provides complete starter code for the infrastructure components while identifying the core logic areas where learners should focus their implementation efforts.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Progress Storage | SQLite with in-memory caching | Redis + PostgreSQL hybrid |
| Real-time Updates | HTTP polling with 5-second intervals | WebSocket connections with Redis Pub/Sub |
| Webhook Delivery | Simple HTTP POST with basic retry | Async queue with exponential backoff |
| Progress Calculation | Fixed percentage increments per stage | Dynamic stage weighting with time estimates |
| Notification Filtering | All progress changes trigger webhooks | Threshold-based filtering with batching |

#### Recommended File Structure

The progress tracking component integrates with job queue operations and external webhook delivery, requiring clean separation between progress calculation, storage management, and notification delivery concerns.

```
media_processing/
  progress/
    __init__.py                 ← component exports
    tracker.py                  ← core ProgressTracker class
    calculator.py               ← stage-based progress calculation
    storage.py                  ← Redis/PostgreSQL hybrid storage
    notifications.py            ← webhook notification system  
    models.py                   ← progress data structures
    webhooks/
      __init__.py
      delivery.py               ← reliable webhook delivery
      retry.py                  ← exponential backoff logic
      security.py               ← signature generation/verification
    tests/
      test_tracker.py           ← progress tracking tests
      test_webhooks.py          ← webhook delivery tests
      test_storage.py           ← storage integration tests
  queue/
    worker.py                   ← imports progress.tracker
  main.py                       ← progress tracking configuration
```

#### Progress Storage Infrastructure (Complete Implementation)

```python
# progress/storage.py
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from dataclasses import asdict

from .models import JobProgress, ProgressUpdate

logger = logging.getLogger(__name__)

class ProgressStorage:
    """
    Hybrid Redis + PostgreSQL storage for job progress tracking.
    Redis provides real-time performance for active jobs.
    PostgreSQL ensures durability and supports complex queries.
    """
    
    def __init__(self, redis_client: redis.Redis, pg_conn_string: str):
        self.redis = redis_client
        self.pg_conn_string = pg_conn_string
        self._setup_schema()
    
    def _setup_schema(self):
        """Initialize PostgreSQL schema for progress persistence."""
        with psycopg2.connect(self.pg_conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS job_progress (
                        job_id VARCHAR(255) PRIMARY KEY,
                        percentage FLOAT NOT NULL DEFAULT 0.0,
                        stage VARCHAR(100) NOT NULL DEFAULT 'pending',
                        stage_progress FLOAT NOT NULL DEFAULT 0.0,
                        updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        estimated_completion TIMESTAMP WITH TIME ZONE,
                        sequence_number BIGINT NOT NULL,
                        details JSONB
                    );
                    
                    CREATE TABLE IF NOT EXISTS progress_history (
                        id SERIAL PRIMARY KEY,
                        job_id VARCHAR(255) NOT NULL,
                        percentage FLOAT NOT NULL,
                        stage VARCHAR(100) NOT NULL,
                        recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        details JSONB
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_progress_history_job_time 
                        ON progress_history(job_id, recorded_at);
                """)
                conn.commit()

    def update_progress(self, job_id: str, progress: JobProgress) -> bool:
        """
        Atomically update job progress in Redis with PostgreSQL backup.
        Returns True if update succeeded, False if rejected (out of order).
        """
        sequence_num = int(time.time() * 1000000)  # microsecond timestamp
        
        # Atomic update with sequence number validation
        pipe = self.redis.pipeline()
        pipe.watch(f"progress:job:{job_id}")
        
        try:
            current_data = pipe.hgetall(f"progress:job:{job_id}")
            current_seq = int(current_data.get('sequence_number', 0))
            
            if sequence_num <= current_seq:
                logger.warning(f"Out of order progress update for job {job_id}: "
                             f"new={sequence_num}, current={current_seq}")
                return False
            
            pipe.multi()
            progress_dict = asdict(progress)
            progress_dict['sequence_number'] = sequence_num
            progress_dict['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            pipe.hset(f"progress:job:{job_id}", mapping=progress_dict)
            pipe.zadd(f"progress:history:{job_id}", 
                     {json.dumps(progress_dict): sequence_num})
            pipe.expire(f"progress:job:{job_id}", 86400)  # 24 hour TTL
            pipe.expire(f"progress:history:{job_id}", 86400)
            pipe.publish(f"progress:updates", 
                        json.dumps({"job_id": job_id, **progress_dict}))
            
            pipe.execute()
            
            # Async backup to PostgreSQL every 10% progress
            if progress.percentage % 10.0 < 1.0:
                self._backup_to_postgres(job_id, progress, sequence_num)
            
            return True
            
        except redis.WatchError:
            logger.info(f"Concurrent update detected for job {job_id}, retrying")
            return False
        finally:
            pipe.reset()

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        """Retrieve current progress for job from Redis or PostgreSQL."""
        # Try Redis first for active jobs
        data = self.redis.hgetall(f"progress:job:{job_id}")
        if data:
            return JobProgress(
                job_id=job_id,
                percentage=float(data['percentage']),
                stage=data['stage'],
                stage_progress=float(data['stage_progress']),
                details=json.loads(data.get('details', '{}'))
            )
        
        # Fallback to PostgreSQL for completed jobs
        with psycopg2.connect(self.pg_conn_string) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT percentage, stage, stage_progress, details, updated_at
                    FROM job_progress WHERE job_id = %s
                """, (job_id,))
                row = cur.fetchone()
                
                if row:
                    return JobProgress(
                        job_id=job_id,
                        percentage=row['percentage'],
                        stage=row['stage'], 
                        stage_progress=row['stage_progress'],
                        details=row['details'] or {}
                    )
        
        return None

    def _backup_to_postgres(self, job_id: str, progress: JobProgress, 
                           sequence_num: int):
        """Asynchronously backup progress to PostgreSQL."""
        try:
            with psycopg2.connect(self.pg_conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO job_progress 
                        (job_id, percentage, stage, stage_progress, updated_at, 
                         sequence_number, details)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (job_id) DO UPDATE SET
                            percentage = EXCLUDED.percentage,
                            stage = EXCLUDED.stage,
                            stage_progress = EXCLUDED.stage_progress,
                            updated_at = EXCLUDED.updated_at,
                            sequence_number = EXCLUDED.sequence_number,
                            details = EXCLUDED.details
                    """, (job_id, progress.percentage, progress.stage,
                          progress.stage_progress, datetime.now(timezone.utc),
                          sequence_num, json.dumps(progress.details)))
                    
                    # Archive progress milestone to history
                    cur.execute("""
                        INSERT INTO progress_history 
                        (job_id, percentage, stage, recorded_at, details)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (job_id, progress.percentage, progress.stage,
                          datetime.now(timezone.utc), json.dumps(progress.details)))
                    
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to backup progress for job {job_id}: {e}")

    def cleanup_completed_job(self, job_id: str):
        """Remove Redis data for completed job to prevent memory leaks."""
        pipe = self.redis.pipeline()
        pipe.delete(f"progress:job:{job_id}")
        pipe.delete(f"progress:history:{job_id}")
        pipe.execute()
```

#### Webhook Security and Signature Verification (Complete Implementation)

```python
# progress/webhooks/security.py
import hmac
import hashlib
import time
import json
from typing import Dict, Any, Optional

class WebhookSecurity:
    """
    Provides HMAC signature generation and verification for webhook security.
    Prevents tampering and replay attacks through cryptographic validation.
    """
    
    @staticmethod
    def generate_signature(payload: str, secret: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload.
        
        Args:
            payload: Raw JSON payload string
            secret: Webhook endpoint secret key
            
        Returns:
            Hex-encoded signature for X-Webhook-Signature header
        """
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str, 
                        max_age: int = 300) -> bool:
        """
        Verify webhook signature and timestamp to prevent replay attacks.
        
        Args:
            payload: Raw request body string
            signature: X-Webhook-Signature header value
            secret: Webhook endpoint secret
            max_age: Maximum message age in seconds (default 5 minutes)
            
        Returns:
            True if signature valid and timestamp fresh
        """
        if not signature.startswith('sha256='):
            return False
            
        expected_signature = WebhookSecurity.generate_signature(payload, secret)
        
        # Constant-time comparison prevents timing attacks
        if not hmac.compare_digest(signature, expected_signature):
            return False
        
        # Verify timestamp to prevent replay attacks
        try:
            data = json.loads(payload)
            event_timestamp = data.get('event_timestamp')
            if event_timestamp:
                event_time = time.mktime(time.strptime(
                    event_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ'
                ))
                if time.time() - event_time > max_age:
                    return False
        except (json.JSONDecodeError, ValueError, KeyError):
            return False
            
        return True

    @staticmethod
    def create_webhook_payload(event_type: str, job_data: Dict[str, Any], 
                              payload_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create standardized webhook payload with security metadata.
        
        Args:
            event_type: Type of event (job.started, job.completed, etc.)
            job_data: Job information dictionary  
            payload_data: Event-specific payload data
            
        Returns:
            Complete webhook payload ready for signing and delivery
        """
        import uuid
        
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "event_timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            "api_version": "v1",
            "job": job_data,
            "payload": payload_data
        }
```

#### Core Progress Tracking Logic (Implementation Skeleton)

```python
# progress/tracker.py
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum

from .storage import ProgressStorage  
from .calculator import ProgressCalculator
from .notifications import WebhookNotifier
from .models import JobProgress, ProcessingStage

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    Central progress tracking coordinator that manages stage-based progress
    calculation, storage persistence, and webhook notification delivery.
    """
    
    def __init__(self, storage: ProgressStorage, calculator: ProgressCalculator,
                 notifier: WebhookNotifier):
        self.storage = storage
        self.calculator = calculator
        self.notifier = notifier
        self.notification_thresholds = [25.0, 50.0, 75.0, 90.0, 100.0]

    def initialize_job_progress(self, job_id: str, total_stages: List[str],
                               webhook_url: Optional[str] = None) -> bool:
        """
        Initialize progress tracking for new processing job.
        
        Args:
            job_id: Unique job identifier
            total_stages: List of processing stages in order
            webhook_url: Optional webhook URL for notifications
            
        Returns:
            True if initialization successful
        """
        # TODO 1: Create initial JobProgress object with 0% completion
        # TODO 2: Set stage to first stage in total_stages list  
        # TODO 3: Initialize stage_progress to 0.0 and details to empty dict
        # TODO 4: Store initial progress using self.storage.update_progress()
        # TODO 5: Register webhook URL with notifier if provided
        # TODO 6: Send job.started webhook notification if webhook_url configured
        # Hint: Use self.calculator.initialize_stage_weights(total_stages)
        pass

    def update_stage_progress(self, job_id: str, stage: str, 
                             stage_percentage: float, 
                             details: Optional[Dict] = None) -> bool:
        """
        Update progress within current processing stage.
        
        Args:
            job_id: Job identifier
            stage: Current processing stage name
            stage_percentage: Completion percentage within stage (0-100)
            details: Optional additional progress details
            
        Returns:
            True if update successful and within monotonic constraints
        """
        # TODO 1: Validate stage_percentage is between 0.0 and 100.0
        # TODO 2: Get current progress from storage to check stage consistency
        # TODO 3: Use calculator to compute overall job percentage from stage progress
        # TODO 4: Validate new percentage >= current percentage (monotonic progression)
        # TODO 5: Create updated JobProgress object with new values
        # TODO 6: Store updated progress atomically using storage layer
        # TODO 7: Check if notification threshold crossed since last webhook
        # TODO 8: Trigger async webhook delivery if threshold met
        # Hint: self.calculator.calculate_overall_progress(stage, stage_percentage)
        pass

    def advance_to_next_stage(self, job_id: str, next_stage: str,
                             completion_details: Optional[Dict] = None) -> bool:
        """
        Transition job to next processing stage with 100% completion of current stage.
        
        Args:
            job_id: Job identifier  
            next_stage: Name of next processing stage
            completion_details: Optional details about completed stage
            
        Returns:
            True if stage transition successful
        """
        # TODO 1: Get current progress and validate job exists
        # TODO 2: Mark current stage as 100% complete with stage completion details
        # TODO 3: Validate next_stage is valid in the configured stage sequence
        # TODO 4: Calculate new overall percentage with current stage complete
        # TODO 5: Create JobProgress for new stage with 0% stage progress
        # TODO 6: Update storage with new stage and overall progress
        # TODO 7: Check for notification thresholds and send webhooks if needed
        # Hint: Use self.calculator.get_stage_transition_percentage(next_stage)
        pass

    def mark_job_completed(self, job_id: str, output_files: List[str],
                          processing_duration: float) -> bool:
        """
        Mark job as 100% complete and send final webhook notification.
        
        Args:
            job_id: Job identifier
            output_files: List of generated output file paths
            processing_duration: Total processing time in seconds
            
        Returns:
            True if completion marking successful
        """
        # TODO 1: Set job progress to 100% with 'completed' stage
        # TODO 2: Include output_files and processing_duration in progress details  
        # TODO 3: Store final progress state with completion timestamp
        # TODO 4: Send job.completed webhook with output file information
        # TODO 5: Clean up Redis progress data to prevent memory leaks
        # TODO 6: Log completion event with duration and output file count
        # Hint: self.storage.cleanup_completed_job(job_id) for memory management
        pass

    def mark_job_failed(self, job_id: str, error_message: str,
                       retry_eligible: bool = False) -> bool:
        """
        Mark job as permanently failed or eligible for retry.
        
        Args:
            job_id: Job identifier
            error_message: Detailed error description
            retry_eligible: Whether job can be retried automatically
            
        Returns:
            True if failure marking successful
        """
        # TODO 1: Get current progress to preserve stage and percentage information
        # TODO 2: Create failed JobProgress with error details and retry status
        # TODO 3: Store failure state with error message and timestamp
        # TODO 4: Send job.failed webhook with error details and retry information
        # TODO 5: Clean up Redis data if failure is permanent (not retryable)
        # TODO 6: Log failure event with error classification for monitoring
        # Hint: Include current stage in failure details for debugging context
        pass
```

#### Stage-Based Progress Calculation (Implementation Skeleton)

```python
# progress/calculator.py
from typing import Dict, List, Optional
from enum import Enum

class ProcessingStage(Enum):
    """Standard processing stages with predefined weight allocations."""
    VALIDATION = ("validation", 5.0)
    PREPROCESSING = ("preprocessing", 10.0) 
    RESIZE_OPERATIONS = ("resize_operations", 40.0)
    FORMAT_CONVERSION = ("format_conversion", 35.0)
    OPTIMIZATION = ("optimization", 8.0)
    FINALIZATION = ("finalization", 2.0)

class ProgressCalculator:
    """
    Calculates overall job progress based on stage completion and weights.
    Ensures monotonic progress and accurate percentage computation.
    """
    
    def __init__(self):
        self.stage_weights: Dict[str, float] = {}
        self.stage_order: List[str] = []

    def initialize_stage_weights(self, stages: List[str], 
                                custom_weights: Optional[Dict[str, float]] = None) -> bool:
        """
        Configure stage weights for progress calculation.
        
        Args:
            stages: Ordered list of processing stages
            custom_weights: Optional custom weight allocation per stage
            
        Returns:
            True if weights total 100% and are valid
        """
        # TODO 1: Use custom_weights if provided, otherwise use default ProcessingStage weights
        # TODO 2: Validate that all stages in list have corresponding weight values
        # TODO 3: Verify total weights sum to exactly 100.0 (within floating point tolerance)
        # TODO 4: Store stage_order list to track stage progression sequence  
        # TODO 5: Store stage_weights dict for percentage calculations
        # TODO 6: Return False if weight validation fails, True on success
        # Hint: sum(weights.values()) should equal 100.0 ± 0.01 for floating point comparison
        pass

    def calculate_overall_progress(self, current_stage: str, 
                                  stage_progress: float) -> float:
        """
        Calculate overall job progress percentage based on current stage position.
        
        Args:
            current_stage: Name of currently executing stage
            stage_progress: Completion percentage within current stage (0-100)
            
        Returns:
            Overall job completion percentage (0-100)
        """
        # TODO 1: Validate current_stage exists in configured stage_order
        # TODO 2: Calculate completed percentage from all previous stages
        # TODO 3: Add weighted progress from current stage: (stage_weight * stage_progress / 100)
        # TODO 4: Ensure result stays within 0-100 bounds
        # TODO 5: Return total percentage with appropriate precision (1 decimal place)
        # Hint: Use self.stage_order.index(current_stage) to find position
        pass

    def get_stage_transition_percentage(self, next_stage: str) -> float:
        """
        Calculate progress percentage when transitioning to next stage.
        
        Args:
            next_stage: Stage being transitioned to
            
        Returns:
            Progress percentage with all previous stages 100% complete
        """
        # TODO 1: Find index of next_stage in stage_order list
        # TODO 2: Sum weights of all stages before next_stage (these are 100% complete)
        # TODO 3: Return cumulative percentage for completed stages
        # TODO 4: Validate result is reasonable (should be < 100% unless final stage)
        # Hint: This shows progress at start of next_stage with previous stages done
        pass

    def estimate_remaining_time(self, current_progress: float,
                               elapsed_time: float) -> Optional[float]:
        """
        Estimate remaining processing time based on current progress rate.
        
        Args:
            current_progress: Current job completion percentage  
            elapsed_time: Time elapsed since job started (seconds)
            
        Returns:
            Estimated remaining time in seconds, or None if insufficient data
        """
        # TODO 1: Validate current_progress > 0 to avoid division by zero
        # TODO 2: Calculate processing rate: elapsed_time / current_progress
        # TODO 3: Estimate total time: processing_rate * 100.0
        # TODO 4: Calculate remaining time: total_time - elapsed_time
        # TODO 5: Return None if estimate seems unreasonable (negative or > 24 hours)
        # Hint: Add safety margin (15-25%) to account for varying stage complexity
        pass
```

#### Milestone Checkpoints

After implementing the progress tracking system, verify functionality through these checkpoints:

**Checkpoint 1: Basic Progress Updates**
```bash
# Test progress tracking with sample job
python -m pytest progress/tests/test_tracker.py::test_stage_progression -v

# Expected behavior:
# - Job initializes with 0% progress at first stage
# - Stage progress updates increment overall percentage correctly  
# - Stage transitions advance to next stage with proper percentage
# - Progress updates maintain monotonic progression (never go backward)
```

**Checkpoint 2: Webhook Notification Delivery**
```bash
# Start webhook test server to receive notifications
python progress/tests/webhook_test_server.py --port 8080 &

# Run webhook delivery tests
python -m pytest progress/tests/test_webhooks.py::test_delivery_with_retry -v

# Verify webhook server receives:
# - job.started notification when job begins
# - job.progress notifications at 25%, 50%, 75% thresholds  
# - job.completed notification with output file URLs
# - Valid HMAC signatures on all webhook payloads
```

**Checkpoint 3: Concurrent Progress Updates**
```bash
# Test concurrent worker progress updates
python progress/tests/test_concurrency.py --workers 5 --job-count 10

# Expected behavior:
# - Multiple workers update same job without race conditions
# - Progress values remain monotonic despite concurrent updates
# - No progress reversals or inconsistent percentages
# - Redis sequence numbers prevent out-of-order updates
```

**Signs of Implementation Issues:**
- **Progress jumps backward**: Check sequence number implementation and Redis atomic operations
- **Webhook delivery failures**: Verify signature generation and retry logic configuration
- **Memory usage growth**: Confirm cleanup processes remove completed job data from Redis
- **Inaccurate percentages**: Validate stage weight calculations sum to exactly 100%


## Component Interactions and Data Flow

> **Milestone(s):** All milestones (1-3) as this section details the communication patterns between image processing, video transcoding, job queue, and progress tracking components

### Mental Model: Orchestra Performance

Think of the media processing pipeline as a symphony orchestra performing a complex musical piece. The **API Gateway** acts as the conductor, receiving requests from the audience (clients) and coordinating the entire performance. The **job queue** serves as the sheet music distribution system, ensuring each musician (worker process) knows what to play and when. Individual **worker processes** are like specialized musicians - some excel at string instruments (image processing), others at brass (video transcoding), each following their part while contributing to the overall performance.

The **progress tracking system** functions like the concert program that audience members follow, providing real-time updates about which movement is currently playing and how much remains. **Webhook notifications** are like the applause cues in the program - they signal important moments to the audience at precisely the right time. Just as musicians must stay synchronized through visual cues and timing, our components communicate through well-defined message formats and sequenced interactions to produce a harmonious result.

This orchestration requires precise timing, clear communication channels, and graceful recovery when a musician (component) encounters difficulties. The conductor doesn't need to know how to play every instrument, but must understand how each section contributes to the whole performance and coordinate their interactions seamlessly.

### API Interfaces and Contracts

The API layer serves as the primary interface between external clients and the internal media processing system, providing REST endpoints that abstract the complexity of job submission, status monitoring, and result retrieval. These interfaces follow RESTful principles while accommodating the asynchronous nature of media processing operations.

#### Job Submission Endpoint

The job submission endpoint receives media processing requests and returns immediately with a job identifier, enabling clients to track processing asynchronously. This endpoint accepts multipart form data containing the input file and processing specifications, validating inputs before queue submission.

| Method | Endpoint | Request Format | Response Format | Status Codes |
|--------|----------|---------------|-----------------|--------------|
| POST | `/jobs` | `multipart/form-data` with file and JSON config | JSON with `job_id`, `status`, `estimated_duration` | 201 Created, 400 Bad Request, 413 Payload Too Large, 422 Unprocessable Entity |
| GET | `/jobs/{job_id}` | None | JSON with `ProcessingJob` details | 200 OK, 404 Not Found |
| GET | `/jobs/{job_id}/progress` | None | JSON with `JobProgress` details | 200 OK, 404 Not Found |
| DELETE | `/jobs/{job_id}` | None | JSON confirmation | 200 OK, 404 Not Found, 409 Conflict if processing |

The request payload structure accommodates multiple output specifications within a single job, allowing clients to request multiple formats, resolutions, or quality variants in one submission:

| Field Name | Type | Required | Description | Validation Rules |
|------------|------|----------|-------------|------------------|
| `input_file` | File | Yes | Media file to process | Max 500MB, supported MIME types only |
| `output_specifications` | JSON Array | Yes | List of desired outputs | 1-10 specifications per job |
| `priority` | String | No | Job priority level | One of: low, normal, high, urgent |
| `webhook_url` | URL | No | Callback URL for notifications | Must be HTTPS, reachable endpoint |
| `metadata_handling` | String | No | EXIF/metadata preservation mode | strip, preserve, selective |
| `processing_options` | JSON Object | No | Advanced processing parameters | Format-specific validation |

#### Response Data Structures

API responses provide comprehensive job information while maintaining consistency across different endpoint variations. The response format adapts to include relevant details based on the endpoint and job state.

| Response Field | Type | Always Present | Description | Example Values |
|----------------|------|---------------|-------------|----------------|
| `job_id` | String | Yes | Unique job identifier | `img_20231215_143022_abc123` |
| `status` | Enum | Yes | Current job status | `pending`, `processing`, `completed`, `failed` |
| `created_at` | ISO8601 | Yes | Job submission timestamp | `2023-12-15T14:30:22Z` |
| `started_at` | ISO8601 | If started | Processing start time | `2023-12-15T14:30:25Z` |
| `completed_at` | ISO8601 | If terminal | Processing completion time | `2023-12-15T14:32:18Z` |
| `progress_percentage` | Float | If processing | Overall completion percentage | `67.5` |
| `current_stage` | String | If processing | Current processing phase | `resize_operations` |
| `estimated_duration` | Integer | If processing | Remaining seconds estimate | `45` |
| `output_files` | Array | If completed | Generated file download URLs | `["/downloads/thumb.jpg"]` |
| `error_message` | String | If failed | Human-readable error description | `Unsupported codec in input video` |
| `retry_count` | Integer | Yes | Number of retry attempts made | `2` |

> **Design Insight**: The API maintains idempotency by accepting optional `idempotency_key` headers. If a client submits the same key within 24 hours, the system returns the existing job rather than creating a duplicate. This prevents accidental duplicate processing when clients retry failed requests.

#### Content Negotiation and Format Support

The API supports multiple response formats and content negotiation to accommodate different client requirements and integration patterns. Clients can request JSON, XML, or abbreviated formats based on their capabilities.

| Accept Header | Response Format | Use Case | Content Type |
|---------------|----------------|----------|--------------|
| `application/json` | Full JSON with all fields | Web applications, modern APIs | `application/json; charset=utf-8` |
| `application/json; compact=true` | Minimal JSON with essential fields | Mobile apps, bandwidth-limited | `application/json; charset=utf-8` |
| `application/xml` | XML format for legacy systems | Enterprise integration | `application/xml; charset=utf-8` |
| `text/plain` | Simple status text | Command-line tools, monitoring | `text/plain; charset=utf-8` |

**Decision: REST over GraphQL for API Design**
- **Context**: Need to choose API paradigm for client communication with varying complexity requirements
- **Options Considered**: REST with JSON, GraphQL, gRPC with Protocol Buffers
- **Decision**: REST with JSON as primary interface
- **Rationale**: Media processing APIs have simple, resource-oriented operations (submit job, check status, retrieve results). REST provides excellent caching, is universally supported, and matches the async processing model naturally. GraphQL adds complexity without significant benefit for this domain.
- **Consequences**: Enables standard HTTP caching, simple client integration, but requires multiple requests for complex queries. Webhook notifications complement REST's limitations for real-time updates.

#### Error Response Format

Error responses follow RFC 7807 Problem Details format, providing structured error information that clients can programmatically handle while remaining human-readable for debugging purposes.

| Error Field | Type | Description | Example |
|-------------|------|-------------|---------|
| `type` | URI | Problem type identifier | `/errors/unsupported-format` |
| `title` | String | Short error summary | `Unsupported Media Format` |
| `status` | Integer | HTTP status code | `422` |
| `detail` | String | Detailed error explanation | `WebM container with AV1 codec not supported` |
| `instance` | URI | Specific error instance | `/jobs/video_20231215_143022_xyz789` |
| `timestamp` | ISO8601 | Error occurrence time | `2023-12-15T14:30:23Z` |
| `request_id` | String | Request correlation ID | `req_abc123def456` |

### Internal Message Formats

Internal communication between components uses structured message formats that ensure reliable delivery, enable monitoring, and support system evolution. These messages flow through Redis-based queues with JSON serialization for cross-language compatibility.

#### Job Queue Message Structure

Job queue messages contain all information necessary for worker processes to execute media processing tasks independently. The message format balances completeness with serialization efficiency, supporting both immediate processing and delayed execution scenarios.

| Message Field | Type | Description | Serialization Notes |
|---------------|------|-------------|-------------------|
| `job_id` | String | Unique job identifier | Used for correlation across all systems |
| `message_type` | Enum | Message type discriminator | `PROCESS_MEDIA`, `CANCEL_JOB`, `HEARTBEAT` |
| `priority` | Integer | Numeric priority for queue ordering | Higher numbers processed first |
| `input_file_path` | String | Absolute path to input media file | Must be accessible to all workers |
| `output_specifications` | Array | List of `OutputSpecification` objects | Serialized as nested JSON |
| `processing_config` | Object | Processing parameters and constraints | Contains format-specific settings |
| `webhook_url` | String | Callback URL for progress notifications | Optional, null if no notifications |
| `metadata_options` | Object | EXIF and metadata handling preferences | Controls privacy and compatibility |
| `retry_policy` | Object | Retry behavior configuration | Max attempts, backoff strategy |
| `resource_constraints` | Object | Memory and CPU limits for processing | Helps with worker assignment |
| `correlation_id` | String | Request tracking across components | Links API request to internal operations |
| `created_at` | Integer | Unix timestamp of message creation | Used for TTL and aging policies |
| `timeout_at` | Integer | Unix timestamp when job expires | Prevents infinite queue retention |

The `OutputSpecification` objects within job messages contain format-specific parameters that workers use to configure processing operations:

| Output Spec Field | Type | Description | Format Applicability |
|-------------------|------|-------------|---------------------|
| `output_path` | String | Target file path for generated output | All formats |
| `format` | String | Target format identifier | `jpeg`, `png`, `webp`, `mp4`, `webm` |
| `width` | Integer | Target width in pixels | Images and videos |
| `height` | Integer | Target height in pixels | Images and videos |
| `quality` | Integer | Quality level (1-100) | Lossy formats only |
| `optimization_level` | Integer | Processing effort vs speed tradeoff | Format-dependent scale |
| `codec_settings` | Object | Format-specific encoding parameters | Videos: bitrate, CRF, preset |
| `color_profile` | String | ICC color profile to apply | Images with color management |

> **Architecture Decision**: Messages include complete processing specifications rather than references to external configuration. This design ensures workers can process jobs independently without additional database lookups, improving reliability and reducing latency. The tradeoff is larger message sizes, but media processing jobs are inherently heavy operations where message overhead is negligible.

#### Progress Update Messages

Progress update messages flow from worker processes to the progress tracking system, providing real-time visibility into job execution. These messages support both incremental updates within processing stages and major stage transitions.

| Progress Field | Type | Description | Update Frequency |
|----------------|------|-------------|-----------------|
| `job_id` | String | Job identifier for correlation | Every message |
| `message_type` | String | Update type discriminator | `STAGE_PROGRESS`, `STAGE_COMPLETE`, `JOB_COMPLETE` |
| `overall_percentage` | Float | Total job completion percentage (0-100) | Every update |
| `current_stage` | String | Processing stage name | Changes on stage transitions |
| `stage_percentage` | Float | Completion within current stage (0-100) | Frequent updates during processing |
| `stage_details` | Object | Stage-specific progress information | Contains operation-specific data |
| `estimated_remaining` | Integer | Estimated seconds to completion | Recalculated on each update |
| `worker_id` | String | Identifier of processing worker | For debugging and load analysis |
| `timestamp` | Integer | Unix timestamp of progress measurement | Enables progress rate calculation |
| `sequence_number` | Integer | Monotonically increasing sequence | Prevents out-of-order updates |
| `resource_usage` | Object | Current memory and CPU utilization | Optional, for monitoring |

The `stage_details` object provides operation-specific progress information that varies by processing type:

| Stage Detail Field | Type | Image Processing | Video Transcoding |
|--------------------|------|------------------|-------------------|
| `current_operation` | String | `resize`, `format_convert` | `encode_video`, `extract_audio` |
| `processed_items` | Integer | Number of output variants completed | Number of segments processed |
| `total_items` | Integer | Total output variants requested | Total segments in video |
| `current_resolution` | String | `1920x1080` | `1280x720` |
| `bitrate_kbps` | Integer | N/A (images) | Current encoding bitrate |
| `fps_processed` | Float | N/A (images) | Frames processed per second |
| `temp_file_size` | Integer | Intermediate file size in bytes | Current output file size |

#### Error and Notification Messages

Error messages provide detailed failure information for debugging and recovery decisions. The message format distinguishes between transient errors (suitable for retry) and permanent failures (requiring human intervention or job cancellation).

| Error Message Field | Type | Description | Recovery Usage |
|---------------------|------|-------------|----------------|
| `job_id` | String | Failed job identifier | Links error to specific job |
| `error_type` | String | Error category | `INPUT_ERROR`, `PROCESSING_ERROR`, `RESOURCE_ERROR` |
| `error_code` | String | Specific error identifier | `UNSUPPORTED_CODEC`, `MEMORY_LIMIT_EXCEEDED` |
| `error_message` | String | Human-readable error description | Displayed to end users |
| `technical_details` | String | Detailed technical information | For debugging and troubleshooting |
| `retry_eligible` | Boolean | Whether error is potentially transient | Controls automatic retry behavior |
| `failed_stage` | String | Processing stage where error occurred | Helps isolate problem area |
| `worker_id` | String | Worker that encountered the error | For worker health monitoring |
| `input_file_info` | Object | Information about input file | Helps identify problematic inputs |
| `resource_state` | Object | System resource state at failure | Memory, disk space, CPU load |
| `stack_trace` | String | Exception stack trace if available | Development and debugging |
| `suggested_action` | String | Recommended resolution steps | Guides manual intervention |

**Decision: JSON Message Serialization**
- **Context**: Need serialization format for inter-component communication supporting multiple languages
- **Options Considered**: JSON, Protocol Buffers, MessagePack, Apache Avro
- **Decision**: JSON with schema validation
- **Rationale**: JSON provides human readability for debugging, universal language support, and flexible schema evolution. Media processing messages are relatively infrequent compared to processing time, so serialization performance is not critical. Schema validation catches format errors early.
- **Consequences**: Enables easy debugging and monitoring, but larger message sizes than binary formats. Schema evolution requires careful compatibility management.

### Processing Sequence Scenarios

The media processing pipeline orchestrates complex sequences of operations across multiple components, with different workflows for image processing, video transcoding, and batch operations. These scenarios demonstrate how components interact and coordinate through the message passing and state management systems.

#### Image Processing Workflow Sequence

Image processing jobs follow a predictable sequence from submission through completion, with progress tracking and error handling at each stage. This workflow optimizes for quality while maintaining reasonable processing times for typical web and mobile image requirements.

![Image Processing Workflow](./diagrams/image-processing-flow.svg)

**Stage 1: Job Submission and Validation (5% of total progress)**

1. **Client submits** image processing request to API Gateway with input file and output specifications
2. **API Gateway validates** file format, size limits, and output specification parameters
3. **Job identifier generation** creates unique job ID using timestamp and random suffix format
4. **Input file storage** moves uploaded file to processing storage location with secure naming
5. **Job record creation** initializes `ProcessingJob` with status `PENDING` in Redis and PostgreSQL
6. **Queue submission** publishes job message to high-priority image processing queue
7. **Response return** sends job ID and initial status to client immediately
8. **Progress initialization** sets up progress tracking with stage weights and webhook configuration

**Stage 2: Worker Assignment and Preprocessing (10% of total progress)**

1. **Worker selection** occurs when available image processing worker pops job from queue
2. **Resource reservation** worker claims memory and CPU resources based on job requirements
3. **Input validation** worker verifies file accessibility, format support, and corruption checks
4. **Metadata extraction** reads EXIF data, color profiles, dimensions, and technical specifications
5. **Processing plan** creation determines optimal resize algorithms, format conversion steps, and quality settings
6. **Progress notification** updates job status to `PROCESSING` and notifies webhook if configured
7. **Temporary workspace** allocation creates isolated directory for intermediate processing files

**Stage 3: Resize Operations (40% of total progress)**

1. **Image loading** into memory with automatic EXIF orientation correction and color space handling
2. **Memory optimization** calculates processing chunks if image exceeds available memory limits
3. **Interpolation algorithm** selection based on resize ratio (upscaling vs downscaling requirements)
4. **Resize execution** for each output specification with aspect ratio preservation or cropping
5. **Quality assessment** validates output dimensions and visual quality against requirements
6. **Progress updates** sent incrementally as each resize variant completes processing
7. **Intermediate storage** saves resized images to temporary files before format conversion

**Stage 4: Format Conversion and Optimization (35% of total progress)**

1. **Format compatibility** checking ensures target formats support required features (transparency, color depth)
2. **Compression optimization** applies format-specific encoding with quality vs file size balancing
3. **Color profile** management preserves or converts ICC profiles based on target format capabilities
4. **Metadata handling** strips or preserves EXIF data according to privacy and compatibility requirements
5. **Quality validation** verifies output files meet specified quality levels and size constraints
6. **Progressive encoding** setup for JPEG files to enable progressive download capabilities
7. **Format-specific optimization** applies WebP/AVIF advanced features or PNG palette optimization

**Stage 5: Finalization and Cleanup (10% of total progress)**

1. **Output validation** confirms all requested variants were generated successfully with correct specifications
2. **File verification** performs integrity checks and validates that files can be opened by standard readers
3. **Secure storage** moves completed files to final output location with proper permissions and access controls
4. **Database updates** marks job as `COMPLETED` with output file paths and processing statistics
5. **Progress completion** sets overall percentage to 100% and sends final webhook notification
6. **Resource cleanup** removes temporary files, releases memory, and frees worker for next job
7. **Metrics recording** logs processing duration, resource usage, and quality metrics for monitoring

> **Critical Timing Consideration**: Image processing typically completes within 5-30 seconds depending on input size and output variants. Progress updates occur every 2-3 seconds during active processing stages, balancing real-time visibility with system overhead. Workers batch multiple small resize operations to minimize progress update frequency.

#### Video Transcoding Workflow Sequence

Video transcoding represents the most complex and resource-intensive processing scenario, requiring careful coordination between FFmpeg processes, adaptive bitrate variant generation, and long-running progress tracking. This workflow can span minutes to hours depending on video duration and output requirements.

**Stage 1: Video Analysis and Planning (5% of total progress)**

1. **Input validation** verifies video file accessibility, format support, and basic integrity through FFmpeg probe
2. **Metadata extraction** analyzes video streams, audio tracks, subtitle tracks, and container properties
3. **Codec compatibility** assessment determines optimal transcoding paths and identifies unsupported features
4. **Resource estimation** calculates memory requirements, processing time, and temporary storage needs
5. **Quality ladder** generation creates adaptive bitrate variant specifications based on input resolution
6. **Processing strategy** selection chooses single-pass vs two-pass encoding based on quality requirements
7. **Worker assignment** reserves video processing worker with sufficient resources for estimated job duration

**Stage 2: Audio Stream Processing (15% of total progress)**

1. **Audio stream** extraction from input video using FFmpeg with format preservation or conversion
2. **Audio normalization** applies volume leveling and dynamic range compression for consistent playback
3. **Multi-bitrate** audio encoding generates variants at different quality levels for adaptive streaming
4. **Audio sync** verification ensures audio timing alignment is maintained throughout processing
5. **Codec optimization** applies audio codec-specific settings for bandwidth efficiency and compatibility
6. **Quality validation** verifies audio output meets bitrate and quality specifications through automated analysis

**Stage 3: Video Stream Transcoding (65% of total progress)**

1. **Keyframe analysis** identifies scene changes and optimal keyframe placement for streaming compatibility
2. **Encoding initialization** starts FFmpeg processes for each required video quality variant simultaneously
3. **Progress parsing** extracts completion percentage from FFmpeg output streams using regex pattern matching
4. **Frame processing** monitoring tracks encoding speed and adjusts resource allocation dynamically
5. **Quality control** sampling periodically validates output quality against CRF settings and bitrate targets
6. **Segment generation** for HLS/DASH creates properly aligned segments across all quality variants
7. **Parallel processing** coordination ensures keyframe alignment across multiple quality streams

**Stage 4: Manifest Generation and Packaging (10% of total progress)**

1. **Stream analysis** examines completed video and audio variants to extract technical specifications
2. **HLS manifest** creation generates M3U8 playlists with proper segment references and bandwidth declarations
3. **DASH manifest** generation creates MPD files with adaptation sets and representation metadata
4. **Segment validation** verifies all media segments are properly formatted and accessible
5. **Bandwidth testing** confirms actual bitrates match declared values in manifests
6. **Compatibility verification** ensures manifests work with standard players and CDN requirements

**Stage 5: Thumbnail Extraction and Cleanup (5% of total progress)**

1. **Thumbnail timestamps** calculation identifies optimal frames representing video content variety
2. **Frame extraction** uses FFmpeg to generate high-quality still images at specified time offsets
3. **Thumbnail optimization** applies image processing techniques for consistent sizing and quality
4. **Output verification** confirms all transcoded variants, manifests, and thumbnails are accessible
5. **Cleanup operations** remove temporary files while preserving final outputs in secure storage
6. **Completion notification** sends webhook with comprehensive job results and download URLs

> **Resource Management**: Video transcoding can consume 2-8 GB RAM and utilize multiple CPU cores for hours. Workers implement memory monitoring and can pause/resume processing if system resources become constrained. Progress updates occur every 10-15 seconds due to the longer processing duration.

#### Batch Processing Coordination Scenario

Batch processing handles multiple related media files as a single logical operation, requiring coordination across multiple workers while maintaining consistent progress reporting and error handling. This scenario is common for photo album processing or video playlist transcoding.

**Stage 1: Batch Job Decomposition**

1. **Batch validation** confirms all input files are accessible and supported formats
2. **Dependency analysis** identifies files that can be processed in parallel vs those requiring sequential processing  
3. **Resource planning** estimates total processing time and memory requirements across all batch items
4. **Work distribution** creates individual processing jobs for each file while maintaining batch coordination
5. **Progress aggregation** setup configures batch-level progress calculation from individual job progress
6. **Error handling** strategy defines batch behavior when individual items fail (fail-fast vs continue)

**Stage 2: Parallel Processing Coordination**

1. **Worker allocation** distributes batch items across available workers based on file types and resource requirements
2. **Progress consolidation** aggregates progress from individual jobs into overall batch completion percentage
3. **Error monitoring** tracks failed items and makes retry decisions based on batch error tolerance policy
4. **Resource balancing** dynamically reassigns work if some workers complete tasks significantly faster than others
5. **Checkpoint creation** periodically saves batch progress state to enable recovery from system failures
6. **Completion detection** recognizes when all batch items reach terminal states (completed or permanently failed)

**Stage 3: Batch Finalization**

1. **Results aggregation** collects output files from all successful processing jobs into batch results
2. **Error reporting** compiles detailed failure information for any items that could not be processed
3. **Consistency verification** ensures all related outputs (thumbnails, variants) are properly linked
4. **Batch manifest** generation creates index files linking all generated outputs for easy client consumption
5. **Cleanup coordination** removes temporary files from all workers that participated in batch processing
6. **Final notification** delivers comprehensive batch results including partial success scenarios

⚠️ **Pitfall: Race Conditions in Progress Aggregation**
Batch processing can create race conditions when multiple workers update progress simultaneously. Without proper synchronization, progress percentages may appear to move backwards or skip values. Implement sequence numbers in progress updates and use atomic Redis operations for progress consolidation. Each individual job maintains its own progress sequence, and the batch coordinator uses the latest sequence number from each job to calculate accurate batch progress.

⚠️ **Pitfall: Memory Exhaustion During Video Processing**
Video transcoding jobs can easily consume all available system memory, especially when processing multiple large videos simultaneously. Workers must implement memory monitoring and refuse new jobs when memory usage exceeds safe thresholds (typically 80% of available RAM). Use FFmpeg memory limits and temporary file streaming to prevent out-of-memory crashes that corrupt partially processed outputs.

⚠️ **Pitfall: Webhook Delivery Failures**
Webhook notifications can fail due to network issues, endpoint unavailability, or client-side errors. Without proper retry logic, clients lose visibility into job progress and completion. Implement exponential backoff for webhook retries (1s, 2s, 4s, 8s intervals) with circuit breaker patterns. Store failed webhooks in a dead letter queue for manual review, and provide alternative mechanisms like polling for clients with unreliable webhook endpoints.

⚠️ **Pitfall: Progress Estimation Inaccuracy**
Video processing progress is notoriously difficult to estimate accurately because encoding complexity varies dramatically based on video content (static scenes encode faster than high-motion scenes). Avoid time-based progress estimates and instead use stage-based progress with frame counts when possible. For FFmpeg integration, parse the total frame count from initial analysis and calculate progress based on frames processed rather than elapsed time.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| API Framework | Flask with Flask-RESTful | FastAPI with async/await |
| Message Queue | Redis with basic pub/sub | Redis with Celery task queue |
| HTTP Client | requests library | aiohttp for async requests |
| JSON Validation | jsonschema library | Pydantic models with validation |
| File Upload | werkzeug FileStorage | streaming upload with progress |
| Authentication | API key headers | JWT tokens with scope validation |
| Rate Limiting | Flask-Limiter | Redis-based sliding window |
| Monitoring | Basic logging | Prometheus metrics + Grafana |

#### Recommended Project Structure

```
media-processor/
  api/
    __init__.py
    routes.py              ← REST endpoint definitions
    models.py              ← Request/response models
    validation.py          ← Input validation logic
    middleware.py          ← Authentication, rate limiting
  messaging/
    __init__.py
    queue_manager.py       ← Job queue operations
    message_formats.py     ← Message serialization
    progress_notifier.py   ← Webhook delivery system
  workers/
    __init__.py
    base_worker.py         ← Common worker functionality
    image_worker.py        ← Image processing worker
    video_worker.py        ← Video transcoding worker
  storage/
    __init__.py
    file_manager.py        ← File I/O and cleanup
    metadata_store.py      ← Job status persistence
  config/
    __init__.py
    settings.py            ← Configuration management
    logging.py             ← Logging setup
  tests/
    api/                   ← API endpoint tests
    workers/               ← Worker process tests
    integration/           ← End-to-end tests
  docker/
    api.Dockerfile         ← API server container
    worker.Dockerfile      ← Worker process container
  requirements.txt
  main.py                  ← Application entry point
```

#### API Server Infrastructure Code

```python
# api/routes.py - Complete REST API implementation
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import uuid
import os
from datetime import datetime
import json

from messaging.queue_manager import QueueManager
from storage.metadata_store import MetadataStore
from config.settings import AppConfig

app = Flask(__name__)
config = AppConfig()
queue_manager = QueueManager(config.redis)
metadata_store = MetadataStore(config.storage)

@app.route('/jobs', methods=['POST'])
def submit_job():
    """Submit new media processing job with file upload and specifications."""
    # TODO 1: Validate multipart form data contains required fields
    # TODO 2: Check file size against MAX_FILE_SIZE limit
    # TODO 3: Validate file format using magic number detection
    # TODO 4: Parse and validate output_specifications JSON
    # TODO 5: Generate unique job_id using generate_job_id()
    # TODO 6: Store uploaded file in secure storage location
    # TODO 7: Create ProcessingJob object with validated parameters
    # TODO 8: Submit job to appropriate queue based on media type
    # TODO 9: Initialize progress tracking for the new job
    # TODO 10: Return job details with 201 Created status
    pass

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Retrieve current job status and details."""
    # TODO 1: Validate job_id format and existence
    # TODO 2: Load ProcessingJob from metadata store
    # TODO 3: Get current progress from progress tracker
    # TODO 4: Format response with job details and progress
    # TODO 5: Return 404 if job not found
    pass

@app.route('/jobs/<job_id>/progress', methods=['GET'])
def get_job_progress(job_id):
    """Get detailed progress information for active job."""
    # TODO 1: Verify job exists and is in processing state
    # TODO 2: Retrieve JobProgress from progress tracking system
    # TODO 3: Calculate stage-specific progress details
    # TODO 4: Include estimated remaining time if available
    # TODO 5: Format response with progress breakdown
    pass

@app.route('/jobs/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel pending or processing job."""
    # TODO 1: Load job and verify it can be cancelled
    # TODO 2: Send cancellation message to worker if processing
    # TODO 3: Update job status to cancelled in metadata store
    # TODO 4: Clean up any temporary files
    # TODO 5: Return confirmation or error if cannot cancel
    pass

@app.errorhandler(413)
def file_too_large(error):
    """Handle file upload size limit exceeded."""
    return jsonify({
        'type': '/errors/file-too-large',
        'title': 'File Too Large',
        'status': 413,
        'detail': f'File size exceeds maximum limit of {config.storage.max_file_size} bytes',
        'timestamp': datetime.utcnow().isoformat()
    }), 413

@app.errorhandler(422)
def validation_error(error):
    """Handle request validation failures."""
    return jsonify({
        'type': '/errors/validation-failed',
        'title': 'Validation Error',
        'status': 422,
        'detail': str(error),
        'timestamp': datetime.utcnow().isoformat()
    }), 422
```

```python
# messaging/message_formats.py - Message serialization utilities
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class MessageType(Enum):
    PROCESS_MEDIA = "process_media"
    PROGRESS_UPDATE = "progress_update"
    JOB_COMPLETE = "job_complete"
    JOB_FAILED = "job_failed"
    CANCEL_JOB = "cancel_job"

@dataclass
class JobMessage:
    """Complete job message format for queue communication."""
    job_id: str
    message_type: MessageType
    priority: int
    input_file_path: str
    output_specifications: List[Dict[str, Any]]
    processing_config: Dict[str, Any]
    webhook_url: Optional[str] = None
    metadata_options: Optional[Dict[str, Any]] = None
    retry_policy: Optional[Dict[str, Any]] = None
    resource_constraints: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    created_at: Optional[int] = None
    timeout_at: Optional[int] = None

    def to_json(self) -> str:
        """Serialize message to JSON string for queue storage."""
        # TODO 1: Convert dataclass to dictionary with enum handling
        # TODO 2: Format timestamps as Unix timestamps
        # TODO 3: Handle None values appropriately
        # TODO 4: Return JSON string with consistent formatting
        pass

    @classmethod
    def from_json(cls, json_str: str) -> 'JobMessage':
        """Deserialize message from JSON string."""
        # TODO 1: Parse JSON string to dictionary
        # TODO 2: Convert string enums back to enum objects
        # TODO 3: Handle timestamp conversion from Unix format
        # TODO 4: Create and return JobMessage instance
        pass

@dataclass
class ProgressMessage:
    """Progress update message format."""
    job_id: str
    message_type: MessageType
    overall_percentage: float
    current_stage: str
    stage_percentage: float
    stage_details: Dict[str, Any]
    estimated_remaining: Optional[int] = None
    worker_id: Optional[str] = None
    timestamp: Optional[int] = None
    sequence_number: Optional[int] = None
    resource_usage: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Serialize progress message to JSON."""
        # TODO: Implement similar to JobMessage.to_json()
        pass

    @classmethod
    def from_json(cls, json_str: str) -> 'ProgressMessage':
        """Deserialize progress message from JSON."""
        # TODO: Implement similar to JobMessage.from_json()
        pass
```

#### Queue Management Infrastructure Code

```python
# messaging/queue_manager.py - Redis-based job queue implementation
import redis
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from config.settings import RedisConfig
from messaging.message_formats import JobMessage, MessageType

class QueueManager:
    """Manages job submission and worker coordination through Redis queues."""
    
    def __init__(self, redis_config: RedisConfig):
        self.redis_client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            decode_responses=True
        )
        self.job_queue_key = "media_jobs"
        self.progress_channel = "job_progress"
        self.worker_heartbeat_key = "worker_heartbeats"
    
    def submit_job(self, job_message: JobMessage) -> bool:
        """Submit job to priority queue with deduplication check."""
        # TODO 1: Check for duplicate job_id in active jobs set
        # TODO 2: Serialize job_message to JSON
        # TODO 3: Add job to priority queue using ZADD with priority score
        # TODO 4: Add job_id to active jobs set for deduplication
        # TODO 5: Set job TTL based on timeout_at value
        # TODO 6: Return True if successful, False if duplicate
        pass
    
    def get_next_job(self, worker_id: str, timeout: int = 30) -> Optional[JobMessage]:
        """Blocking pop highest priority job from queue."""
        # TODO 1: Use BZPOPMAX to get highest priority job with timeout
        # TODO 2: If job received, parse JSON to JobMessage
        # TODO 3: Register worker assignment in Redis
        # TODO 4: Update worker heartbeat timestamp
        # TODO 5: Return JobMessage or None if timeout
        pass
    
    def mark_job_completed(self, job_id: str, worker_id: str) -> bool:
        """Mark job as completed and clean up queue state."""
        # TODO 1: Remove job from active jobs set
        # TODO 2: Remove worker assignment
        # TODO 3: Update completion statistics
        # TODO 4: Clean up any job-specific temporary data
        # TODO 5: Return success status
        pass
    
    def publish_progress(self, progress_message) -> bool:
        """Publish progress update to subscribers."""
        # TODO 1: Serialize progress_message to JSON
        # TODO 2: Publish to progress channel using PUBLISH
        # TODO 3: Store latest progress in Redis for polling clients
        # TODO 4: Return publication success status
        pass
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics for monitoring."""
        # TODO 1: Count jobs in priority queue
        # TODO 2: Count active worker assignments
        # TODO 3: Get average job processing time
        # TODO 4: Return comprehensive statistics dictionary
        pass
```

#### Core Processing Logic Skeleton

```python
# workers/base_worker.py - Common worker functionality
import os
import time
import signal
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from messaging.queue_manager import QueueManager
from messaging.message_formats import JobMessage, ProgressMessage
from storage.metadata_store import MetadataStore

class BaseWorker(ABC):
    """Base class for all media processing workers."""
    
    def __init__(self, worker_id: str, queue_manager: QueueManager, 
                 metadata_store: MetadataStore):
        self.worker_id = worker_id
        self.queue_manager = queue_manager
        self.metadata_store = metadata_store
        self.shutdown_requested = False
        self.current_job = None
        self.logger = logging.getLogger(f"worker.{worker_id}")
    
    def run(self):
        """Main worker loop - gets jobs and processes them."""
        # TODO 1: Set up signal handlers for graceful shutdown
        # TODO 2: Start heartbeat thread for worker health monitoring
        # TODO 3: Enter main processing loop until shutdown requested
        # TODO 4: Get next job from queue with timeout
        # TODO 5: Process job using subclass implementation
        # TODO 6: Handle job completion or failure
        # TODO 7: Clean up resources and update worker status
        pass
    
    def update_progress(self, percentage: float, stage: str, 
                       stage_percentage: float, details: Dict[str, Any]):
        """Send progress update for current job."""
        # TODO 1: Validate progress values (0-100, stage exists)
        # TODO 2: Create ProgressMessage with current job details
        # TODO 3: Include worker resource usage information
        # TODO 4: Publish progress message to queue manager
        # TODO 5: Log progress update for debugging
        pass
    
    @abstractmethod
    def process_job(self, job: JobMessage) -> bool:
        """Process specific job type - implemented by subclasses."""
        # TODO: Subclasses implement specific processing logic
        pass
    
    def handle_job_error(self, job: JobMessage, error: Exception) -> bool:
        """Handle job processing errors with retry logic."""
        # TODO 1: Determine if error is transient or permanent
        # TODO 2: Check retry count against policy limits
        # TODO 3: Apply exponential backoff for transient errors
        # TODO 4: Mark job as failed if retry limit exceeded
        # TODO 5: Clean up any partial processing results
        # TODO 6: Return True if job should be retried
        pass
```

#### Milestone Checkpoints

**After API Implementation:**
- Run `python -m pytest tests/api/` - all endpoint tests should pass
- Start API server: `python main.py`
- Test job submission: `curl -X POST -F "file=@test.jpg" -F "output_specs=[{\"format\":\"webp\",\"width\":800}]" http://localhost:5000/jobs`
- Verify response contains job_id and status "pending"
- Test status endpoint: `curl http://localhost:5000/jobs/{job_id}`

**After Queue Implementation:**
- Run Redis server: `redis-server`
- Test queue operations: `python -c "from messaging.queue_manager import QueueManager; q=QueueManager(); print('Queue connected')"`
- Submit test job and verify it appears in Redis: `redis-cli ZRANGE media_jobs 0 -1 WITHSCORES`
- Check job message format: `redis-cli ZRANGE media_jobs 0 0` should show valid JSON

**After Worker Implementation:**
- Start worker process: `python -m workers.image_worker`
- Worker should connect to queue and wait for jobs
- Submit job through API - worker should pick it up and process
- Check progress updates in Redis: `redis-cli SUBSCRIBE job_progress`
- Verify completed job produces output files in storage directory

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Jobs stuck in pending | Worker not running or crashed | Check worker process status, Redis connection | Restart workers, verify Redis connectivity |
| Progress updates missing | Worker not publishing or Redis pub/sub issue | Monitor Redis PUBSUB channels, check worker logs | Verify Redis PUBLISH permissions, restart progress tracking |
| File upload fails | Storage permissions or disk space | Check file system permissions, disk usage | Fix directory permissions, free up storage space |
| Memory errors during processing | Large files exceeding worker memory limits | Monitor worker memory usage during processing | Implement streaming processing, increase worker memory |
| Webhook delivery fails | Network issues or client endpoint problems | Check webhook endpoint availability, review retry logs | Implement circuit breaker, verify endpoint certificates |
| Race conditions in progress | Multiple workers updating same job | Check for duplicate job assignments in Redis | Implement atomic job assignment with Redis locks |


## Error Handling and Edge Cases

> **Milestone(s):** All milestones (1-3) as robust error handling underlies image processing, video transcoding, and job queue reliability across the entire system

### Mental Model: Hospital Emergency Response System

Think of error handling in our media processing pipeline like a hospital's emergency response system. Just as a hospital has protocols for different types of medical emergencies — cardiac arrest, trauma, poisoning — each requiring different response procedures, our media processing system must handle different types of failures with appropriate recovery strategies.

In a hospital, triage nurses quickly categorize incoming patients by severity: critical (immediate attention), urgent (can wait briefly), less urgent (stable for now). Similarly, our system must rapidly classify errors as transient (retry immediately), resource-constrained (retry with backoff), or permanent failures (move to dead letter queue). The hospital maintains detailed logs of every emergency response to improve future care, just as our system logs every failure and recovery attempt to enable debugging and system improvement.

The hospital's emergency protocols ensure that even when multiple crises occur simultaneously, resources are allocated appropriately, staff coordinate effectively, and no patient is forgotten. Our error handling system provides the same guarantee: even when multiple jobs fail across different worker processes, failures are detected quickly, resources are cleaned up properly, and no job is lost or left in an inconsistent state.

![Job Lifecycle State Machine](./diagrams/job-state-machine.svg)

## Failure Modes and Detection

Comprehensive error detection requires understanding the different categories of failures that can occur in a distributed media processing system. Each failure mode has distinct characteristics, detection methods, and recovery strategies that must be implemented systematically.

### Mental Model: Weather Monitoring System

Think of failure detection like a weather monitoring system with multiple sensors detecting different types of atmospheric disturbances. Just as meteorologists use temperature sensors, pressure gauges, humidity detectors, and wind speed meters to identify incoming storms, our system deploys multiple detection mechanisms to identify different failure patterns before they cascade into larger system outages.

Weather systems provide early warning alerts with different severity levels — watches, warnings, and emergencies — each triggering different response protocols. Our failure detection system similarly provides graduated alert levels that trigger increasingly aggressive recovery measures as failure severity escalates.

### System-Level Failure Categories

The media processing pipeline encounters failures across multiple system layers, each requiring specialized detection and handling approaches.

**Infrastructure Failures** represent the foundational layer of potential system disruption. Redis connection failures manifest when the job queue becomes unreachable due to network partitions, Redis server crashes, or configuration errors. These failures are detected through connection timeout exceptions and failed ping operations during health checks. Storage system failures occur when temporary file writes fail due to disk space exhaustion, permission errors, or filesystem corruption. Worker process crashes happen when FFmpeg consumes excessive memory, encounters segmentation faults, or is terminated by the operating system's out-of-memory killer.

**Resource Constraint Failures** emerge when system resources become insufficient to process jobs effectively. Memory exhaustion occurs when large video files or high-resolution images exceed available RAM during processing operations. CPU throttling manifests as dramatically increased processing times when system load exceeds capacity. Disk space exhaustion prevents temporary file creation and output file writing. Network bandwidth limitations cause timeout failures when downloading input files or uploading processed results to external storage systems.

**Application Logic Failures** stem from processing-specific errors and business rule violations. Invalid input files include corrupted media files, unsupported formats, or files that claim to be one format but contain different data. Configuration errors occur when output specifications request impossible operations like upscaling to extreme resolutions or using incompatible codec combinations. Processing timeout failures happen when jobs exceed configured time limits due to unexpectedly large input files or complex processing requirements.

**External Dependency Failures** involve third-party systems and services that the processing pipeline relies upon. Webhook delivery failures occur when recipient endpoints are unreachable, return error status codes, or fail to respond within timeout windows. Cloud storage API failures prevent input file downloads or processed output uploads. FFmpeg execution failures happen when the underlying binary encounters unsupported codec combinations, corrupted input streams, or internal processing errors.

### Failure Detection Mechanisms

Early and accurate failure detection requires implementing multiple monitoring layers that work together to provide comprehensive system visibility.

| Detection Method | Monitors | Detection Time | Granularity | Implementation |
|-----------------|----------|----------------|-------------|----------------|
| Health Check Probes | Service availability | 30-60 seconds | Component level | Redis ping, filesystem write test |
| Process Monitoring | Worker process status | 5-10 seconds | Process level | PID monitoring, heartbeat signals |
| Resource Monitoring | CPU, memory, disk usage | 10-30 seconds | System level | Resource usage thresholds |
| Job Timeout Detection | Processing duration | Real-time | Job level | Timer-based cancellation |
| Progress Stall Detection | Progress update frequency | 60-120 seconds | Job level | Progress timestamp monitoring |
| Exception Handling | Application errors | Immediate | Operation level | Try-catch blocks with logging |
| Log Analysis | Error patterns | 1-5 minutes | System level | Log aggregation and pattern matching |
| External Service Monitoring | API response codes | Real-time | Request level | HTTP status code analysis |

The `ResourceMonitor` component implements proactive resource constraint detection by continuously tracking system utilization and predicting when thresholds will be exceeded.

| Resource Type | Warning Threshold | Critical Threshold | Detection Method | Response Action |
|---------------|-------------------|-------------------|------------------|-----------------|
| Memory Usage | 80% of available RAM | 90% of available RAM | Process memory monitoring | Pause new job acceptance |
| Disk Space | 85% of temp directory | 95% of temp directory | Filesystem usage check | Clean up old temporary files |
| CPU Load | 80% sustained over 5 minutes | 95% sustained over 2 minutes | Load average monitoring | Scale down concurrent jobs |
| Network I/O | 80% of bandwidth limit | 90% of bandwidth limit | Network traffic analysis | Queue bandwidth-heavy jobs |

Progress stall detection identifies jobs that appear to be running but have stopped making meaningful progress. This situation occurs when FFmpeg processes hang, infinite loops occur in image processing algorithms, or worker processes become unresponsive while maintaining their connection to the job queue.

**Progress Stall Detection Algorithm:**

1. The `ProgressTracker` maintains a timestamp for each job's most recent progress update
2. A background monitor process scans all active jobs every 60 seconds
3. For each job, calculate the time since the last progress update
4. If the stall duration exceeds the job type's expected maximum interval, mark the job as potentially stalled
5. Send a health check request to the worker process handling the stalled job
6. If the worker fails to respond within 30 seconds, initiate job recovery procedures
7. Log the stall event with job context for debugging analysis

### Error Classification and Triage

Once failures are detected, they must be rapidly classified to determine the appropriate recovery strategy. This classification process mirrors medical triage protocols where patients are quickly categorized by severity and treatment urgency.

**Transient Errors** are temporary failures that typically resolve themselves or can be resolved through immediate retry. Network connection timeouts to Redis often succeed on retry due to temporary network congestion. File system busy errors during temporary file operations usually succeed when retried after a brief delay. FFmpeg initialization failures sometimes resolve when retried due to transient resource contention.

**Resource Errors** indicate system resource constraints that require managed backoff and resource cleanup before retry attempts. Memory allocation failures for large images require waiting for other jobs to complete and free memory. Disk space exhaustion requires cleanup of temporary files before retry. CPU throttling requires reducing concurrent job processing to allow system recovery.

**Permanent Errors** represent fundamental problems that will not resolve through retry attempts and require human intervention or job rejection. Corrupted input files will never process successfully regardless of retry count. Invalid codec combinations in video transcoding specifications cannot be resolved automatically. Malformed webhook URLs will always fail delivery attempts.

**Dependency Errors** involve external system failures that may be transient or permanent depending on the root cause. Database connection failures to PostgreSQL may be temporary network issues or permanent service outages. Cloud storage API errors might be temporary rate limiting or permanent authentication problems. Webhook delivery failures could be temporary recipient downtime or permanent endpoint changes.

The error classification algorithm evaluates multiple factors to determine the appropriate error category:

| Error Type | Detection Criteria | Retry Eligible | Backoff Strategy | Max Retries |
|------------|-------------------|----------------|------------------|-------------|
| Network Timeout | Connection timeout exception | Yes | Exponential with jitter | 5 attempts |
| File I/O Error | Permission denied, disk full | Depends on error code | Linear backoff | 3 attempts |
| Memory Error | Out of memory exception | Yes | Resource-aware backoff | 2 attempts |
| Format Error | Invalid file format | No | None | 0 attempts |
| Configuration Error | Invalid parameters | No | None | 0 attempts |
| FFmpeg Error | Exit code analysis | Depends on exit code | Job-specific backoff | 3 attempts |
| Webhook Failure | HTTP status code | Depends on status code | Exponential | 7 attempts |

> **Decision: Error Classification Strategy**
> - **Context**: Failures need rapid classification to determine retry eligibility and recovery strategies without human intervention
> - **Options Considered**: Rule-based classification, machine learning classification, hybrid approach
> - **Decision**: Rule-based classification with exit code and exception type analysis
> - **Rationale**: Rule-based systems provide deterministic behavior, easier debugging, and immediate classification without training data requirements
> - **Consequences**: Enables automated retry decisions but requires manual rule updates as new error patterns emerge

![Error Recovery Decision Tree](./diagrams/error-recovery-paths.svg)

## Retry and Backoff Strategies

Effective retry mechanisms must balance rapid error recovery with system stability, preventing retry storms that can overwhelm already-stressed system components. The retry strategy varies significantly based on error type, system load, and historical failure patterns.

### Mental Model: Air Traffic Control During Storms

Think of retry strategies like air traffic control managing flight delays during severe weather. When storms hit an airport, controllers don't immediately reroute all flights to the same alternate airport — that would overwhelm the backup facility. Instead, they implement graduated delays: some flights wait briefly for weather to clear, others are sent to different airports with spacing to prevent congestion, and some are cancelled entirely when conditions are too dangerous.

Similarly, our retry system implements intelligent backoff that prevents overwhelming system components during failure scenarios. When Redis becomes unavailable, we don't immediately retry all queued operations — instead, we implement exponential backoff with jitter so that retry attempts are distributed over time, giving the system opportunity to recover without being bombarded with retry traffic.

Air traffic controllers also prioritize based on urgency: emergency landings get immediate priority, while routine flights can be delayed longer. Our retry system similarly prioritizes high-priority jobs for earlier retry attempts while allowing lower-priority jobs to wait longer between retries.

### Exponential Backoff Implementation

Exponential backoff prevents retry storms by progressively increasing delays between retry attempts, giving system components time to recover from failure conditions.

The base retry calculation uses the formula: `delay = base_delay * (2^attempt_number) + random_jitter` where jitter prevents the thundering herd problem when multiple jobs fail simultaneously.

**Exponential Backoff Configuration:**

| Error Category | Base Delay | Max Delay | Jitter Range | Max Attempts | Backoff Factor |
|---------------|------------|-----------|--------------|--------------|----------------|
| Network Timeout | 1 second | 300 seconds | ±20% | 5 | 2.0 |
| Resource Constraint | 5 seconds | 900 seconds | ±30% | 3 | 2.5 |
| External API Error | 2 seconds | 600 seconds | ±25% | 7 | 2.0 |
| File I/O Error | 0.5 seconds | 60 seconds | ±10% | 3 | 1.5 |
| FFmpeg Error | 3 seconds | 180 seconds | ±15% | 3 | 2.0 |

The jitter component prevents synchronized retry attempts that can overwhelm recovering systems. Random jitter is calculated as: `jitter = delay * (1.0 + random(-jitter_range, +jitter_range))` where random generates values within the specified range.

**Priority-Based Retry Scheduling** ensures that high-priority jobs receive preferential retry treatment while preventing lower-priority jobs from being starved indefinitely.

| Job Priority | Retry Delay Multiplier | Queue Position | Max Concurrent Retries |
|--------------|----------------------|----------------|----------------------|
| `JobPriority.URGENT` | 0.5x | Front of retry queue | 10 |
| `JobPriority.HIGH` | 0.75x | Priority section | 7 |
| `JobPriority.NORMAL` | 1.0x | Standard section | 5 |
| `JobPriority.LOW` | 1.5x | End of retry queue | 2 |

### Circuit Breaker Pattern for External Dependencies

Circuit breakers protect the system from cascade failures when external dependencies become unreliable. Like electrical circuit breakers that prevent electrical fires by cutting power during overload conditions, software circuit breakers prevent retry storms by temporarily stopping calls to failing services.

The circuit breaker maintains three states: **Closed** (normal operation), **Open** (blocking calls to failed service), and **Half-Open** (testing if service has recovered).

**Circuit Breaker State Transitions:**

| Current State | Condition | Next State | Action Taken |
|---------------|-----------|------------|--------------|
| Closed | Failure rate < threshold | Closed | Allow all requests |
| Closed | Failure rate ≥ threshold | Open | Block all requests, start timeout |
| Open | Timeout not expired | Open | Continue blocking requests |
| Open | Timeout expired | Half-Open | Allow single test request |
| Half-Open | Test request succeeds | Closed | Resume normal operation |
| Half-Open | Test request fails | Open | Reset timeout, continue blocking |

Circuit breaker configuration varies based on the criticality and typical failure patterns of different external dependencies:

| Dependency Type | Failure Threshold | Timeout Duration | Test Interval | Impact |
|----------------|-------------------|------------------|---------------|--------|
| Redis Job Queue | 5 failures in 60s | 30 seconds | 10 seconds | Job processing stops |
| Webhook Delivery | 10 failures in 300s | 120 seconds | 30 seconds | Notifications delayed |
| Cloud Storage | 3 failures in 30s | 60 seconds | 15 seconds | Upload/download blocked |
| PostgreSQL Progress | 3 failures in 45s | 45 seconds | 15 seconds | Progress tracking disabled |

> **Decision: Circuit Breaker vs Simple Retry**
> - **Context**: External dependencies like webhooks and cloud storage can become unreliable, causing job failures to cascade
> - **Options Considered**: Simple retry with exponential backoff, circuit breaker pattern, hybrid approach with circuit breaker for critical paths
> - **Decision**: Circuit breaker for external dependencies, simple retry for internal operations
> - **Rationale**: Circuit breakers prevent cascade failures and retry storms during external service outages, while simple retry is sufficient for transient internal errors
> - **Consequences**: Adds complexity but prevents system-wide outages when external services fail, improves overall system stability

### Retry Queue Management

Failed jobs requiring retry must be managed separately from the main processing queue to prevent blocking new job processing during system recovery periods. The retry queue implements priority-based scheduling and resource-aware retry timing.

The `QueueManager` maintains separate Redis lists for different retry categories, allowing independent processing and monitoring of retry workloads:

| Queue Name | Purpose | Priority | Worker Pool |
|------------|---------|----------|-------------|
| `retry:immediate` | Network timeouts, transient errors | High | Dedicated retry workers |
| `retry:resource` | Memory/CPU constraint errors | Medium | Resource-aware workers |
| `retry:dependency` | External API, webhook failures | Low | Limited concurrency workers |
| `dead_letter` | Permanently failed jobs | N/A | Manual review queue |

**Retry Scheduling Algorithm:**

1. Failed jobs are classified by error type and assigned to appropriate retry queue
2. Retry delay is calculated based on error type, attempt number, job priority, and current system load
3. Jobs are scheduled for retry using Redis sorted sets with retry timestamp as score
4. Background scheduler process scans sorted sets every 10 seconds for jobs ready to retry
5. Ready jobs are moved back to main processing queue with retry context preserved
6. Retry metrics are updated for monitoring and alerting

Resource-aware retry scheduling monitors system health before processing retry jobs to prevent overwhelming an already-stressed system:

| System Condition | Retry Processing | Queue Behavior | Worker Action |
|------------------|------------------|----------------|---------------|
| Normal load (<70%) | Process all retry types | Standard scheduling | Full retry processing |
| High load (70-85%) | Pause resource-intensive retries | Delay resource retries | Process only network retries |
| Critical load (>85%) | Pause all retries | Queue all retries | Focus on current jobs |
| Recovery mode | Gradual retry resumption | Priority-based restart | Slowly increase retry rate |

### Dead Letter Queue Management

Jobs that exceed maximum retry attempts or encounter permanent errors are moved to a dead letter queue for manual investigation and potential reprocessing. The dead letter queue serves as both a safety net preventing data loss and a debugging resource for system improvement.

**Dead Letter Queue Structure:**

| Field Name | Type | Description |
|------------|------|-------------|
| `original_job` | `ProcessingJob` | Complete original job specification |
| `failure_history` | `List[FailureRecord]` | Chronological list of all retry attempts |
| `final_error` | `str` | Final error message that caused permanent failure |
| `classification` | `str` | Error classification (permanent, retry_exhausted, configuration) |
| `moved_at` | `datetime` | Timestamp when moved to dead letter queue |
| `investigation_notes` | `str` | Manual notes from debugging investigation |
| `resolution_action` | `str` | Action taken (reprocessed, cancelled, fixed) |

Dead letter queue monitoring provides insights into system reliability and helps identify recurring failure patterns:

| Metric | Calculation | Alert Threshold | Action Required |
|--------|-------------|-----------------|-----------------|
| Dead letter rate | Dead letter jobs / total jobs | > 2% | Investigate error patterns |
| Error pattern frequency | Count by error classification | > 10 same errors/hour | Fix underlying issue |
| Resolution time | Time from dead letter to resolution | > 24 hours | Improve support process |
| Reprocess success rate | Successful reprocessed jobs / total reprocessed | < 80% | Review error classification |

⚠️ **Pitfall: Retry Storm Prevention**

**The Problem:** When a shared dependency like Redis fails, all worker processes simultaneously retry their operations, creating a retry storm that prevents the dependency from recovering and may cause additional system failures.

**Why It's Wrong:** Synchronized retries can overwhelm recovering systems, extend outage duration, and cause cascade failures to other system components.

**How to Fix:** Implement jittered exponential backoff with circuit breakers, spread retry attempts over time, and monitor retry rates to detect storm conditions before they cause damage.

## Resource Cleanup and Recovery

Comprehensive resource cleanup ensures that failed jobs do not consume system resources indefinitely and that worker processes can recover gracefully from various failure scenarios. Resource management must handle both normal cleanup during successful job completion and emergency cleanup during various failure modes.

### Mental Model: Restaurant Kitchen Cleanup

Think of resource cleanup like the closing procedures at a busy restaurant kitchen. Throughout the day, cooks use various workstations, utensils, and ingredients. When a dish is completed successfully, the cook cleans their station, puts away ingredients, and washes utensils. But when something goes wrong — a pot burns, a sauce spills, or a cook gets sick mid-service — there are emergency cleanup protocols.

The kitchen manager doesn't just abandon the messy workstation; they have systematic procedures: contain any spills to prevent spread, safely dispose of any ruined food, sanitize contaminated surfaces, and return the workstation to usable condition. Similarly, our media processing system must clean up temporary files, release memory allocations, terminate background processes, and reset worker state when jobs fail.

Just as restaurants have health inspectors who ensure cleanup procedures are followed correctly, our system includes resource monitors that verify cleanup completion and alert operators when resources are not properly released.

### Temporary File Management

Media processing operations create numerous temporary files during processing: extracted video frames, intermediate processing results, format conversion buffers, and FFmpeg output segments. These temporary files must be tracked and cleaned up reliably even when processing fails unexpectedly.

The `ResourceManager` maintains a registry of all temporary files associated with each processing job, enabling comprehensive cleanup regardless of how processing terminates.

**Temporary File Lifecycle Management:**

| Stage | File Types Created | Cleanup Trigger | Cleanup Method |
|-------|-------------------|-----------------|----------------|
| Job Initialization | Working directory, lock files | Job start failure | Immediate deletion |
| Image Processing | Resized variants, format conversions | Processing completion/failure | Staged cleanup |
| Video Transcoding | FFmpeg segments, audio extracts | Transcoding completion/failure | Background cleanup |
| Output Generation | Final outputs, compressed archives | Job completion/timeout | Retention policy cleanup |
| Error Recovery | Debug dumps, partial results | Recovery completion | Delayed cleanup |

The temporary file registry tracks file metadata and cleanup requirements for each job:

| Registry Field | Type | Purpose |
|----------------|------|---------|
| `job_id` | `str` | Associate files with processing job |
| `file_path` | `str` | Absolute path to temporary file |
| `file_type` | `str` | Category for cleanup prioritization |
| `size_bytes` | `int` | File size for cleanup planning |
| `created_at` | `datetime` | Creation timestamp for age-based cleanup |
| `cleanup_priority` | `int` | Cleanup order priority (1=immediate, 5=delayed) |
| `retention_hours` | `int` | How long to keep file after job completion |

**Cleanup Priority Algorithm:**

1. **Immediate Priority (1)**: Lock files, PID files, and job control structures that prevent worker restart
2. **High Priority (2)**: Large temporary files, intermediate processing results consuming significant disk space  
3. **Medium Priority (3)**: Debug output files, processing logs, and diagnostic information
4. **Low Priority (4)**: Backup files, cached intermediate results that might be useful for retry
5. **Delayed Priority (5)**: Output files, final results that may need manual verification

Background cleanup processes run every 60 seconds to handle orphaned temporary files from crashed worker processes:

```
Cleanup Algorithm:
1. Scan temporary directory for files older than configured age threshold
2. Cross-reference found files against active job registry
3. Identify orphaned files not associated with running jobs
4. Group files by cleanup priority for efficient processing
5. Delete immediate priority files first, then work down priority levels
6. Log cleanup statistics for monitoring and alerting
7. Update disk space metrics after cleanup completion
```

### Memory Management and Resource Limits

Image and video processing operations can consume substantial memory, particularly when handling high-resolution content or multiple concurrent jobs. Effective memory management prevents out-of-memory conditions and ensures fair resource allocation across worker processes.

**Memory Allocation Tracking:**

The `ResourceMonitor` tracks memory usage at multiple levels to provide comprehensive memory management:

| Memory Category | Tracking Method | Limit Enforcement | Cleanup Trigger |
|----------------|-----------------|-------------------|-----------------|
| Process Memory | RSS monitoring | Process limits | Memory threshold exceeded |
| Image Buffers | Allocation tracking | Buffer size limits | Processing completion |
| Video Frames | Frame buffer counting | Frame limit enforcement | Frame processing completion |
| FFmpeg Processes | Child process monitoring | Process memory limits | FFmpeg completion/timeout |
| Cache Memory | Cache size tracking | LRU eviction | Cache size limits |

Pre-processing memory estimation prevents jobs from starting when insufficient memory is available:

**Memory Estimation Algorithm:**

1. Analyze input file metadata to estimate processing requirements
2. Calculate memory needed for image processing: `width * height * channels * bytes_per_channel * processing_factor`
3. Estimate video processing memory: `frame_width * frame_height * fps * processing_duration * memory_factor`
4. Add overhead estimates for format conversion, compression, and temporary buffers
5. Compare total estimate against available system memory minus safety margin
6. Reject job submission if estimated memory exceeds available resources
7. Queue job with resource requirements for later processing when memory becomes available

| Content Type | Base Memory Formula | Processing Factor | Safety Margin |
|--------------|-------------------|-------------------|---------------|
| JPEG Images | width × height × 3 bytes | 2.5x (for processing buffers) | 20% of available RAM |
| PNG Images | width × height × 4 bytes | 3.0x (for transparency handling) | 20% of available RAM |
| 4K Video | 3840 × 2160 × 3 × fps | 5.0x (for frame buffers) | 30% of available RAM |
| HD Video | 1920 × 1080 × 3 × fps | 4.0x (for processing) | 25% of available RAM |

### Worker Process Recovery

Worker processes may fail due to memory exhaustion, segmentation faults, infinite loops, or external process termination. The worker recovery system detects failed processes and restarts them while preserving job queue integrity.

**Worker Health Monitoring:**

| Health Check Type | Frequency | Timeout | Failure Action |
|------------------|-----------|---------|----------------|
| Heartbeat Signal | 30 seconds | 45 seconds | Mark worker unhealthy |
| Job Progress Update | Based on job type | 2x expected duration | Investigate job status |
| Memory Usage Check | 60 seconds | N/A | Warn if approaching limits |
| Process Existence | 15 seconds | N/A | Restart if process missing |
| Queue Connection | 60 seconds | 10 seconds | Restart worker |

The `WorkerCoordinator` manages worker lifecycle and implements recovery procedures:

**Worker Recovery Procedure:**

1. **Failure Detection**: Monitor detects worker process has failed or become unresponsive
2. **Job Status Assessment**: Determine status of job being processed by failed worker
3. **Resource Cleanup**: Clean up temporary files, release memory, terminate child processes
4. **Job State Recovery**: Mark in-progress job as failed and eligible for retry if appropriate
5. **Worker Process Restart**: Launch new worker process with same configuration
6. **Queue Reconnection**: Re-establish Redis connection and resume job processing
7. **Health Verification**: Verify new worker process is healthy before accepting jobs
8. **Monitoring Update**: Update worker registry and notify monitoring systems

**Job Recovery State Machine:**

| Job State at Worker Failure | Recovery Action | Job Disposition |
|----------------------------|-----------------|-----------------|
| `JobStatus.PENDING` | No action needed | Remains in queue for other worker |
| `JobStatus.PROCESSING` (< 10% complete) | Mark as failed | Eligible for immediate retry |
| `JobStatus.PROCESSING` (10-90% complete) | Mark as failed with progress | Eligible for retry with backoff |
| `JobStatus.PROCESSING` (> 90% complete) | Attempt recovery | Check for partial outputs |
| Partially completed | Validate outputs | Complete if outputs valid, retry if not |

### Database Transaction Cleanup

Progress tracking and job status updates use database transactions that must be properly cleaned up when worker processes fail unexpectedly. Abandoned transactions can lock database resources and prevent other workers from updating job status.

The `MetadataStore` implements transaction cleanup procedures:

**Transaction Management:**

| Transaction Type | Timeout | Cleanup Method | Recovery Action |
|-----------------|---------|----------------|-----------------|
| Job Status Update | 30 seconds | Auto-rollback | Retry with fresh transaction |
| Progress Update | 15 seconds | Auto-rollback | Use cached progress value |
| Batch Progress Update | 60 seconds | Partial commit | Commit successful updates |
| Job Completion | 45 seconds | Manual cleanup | Verify final state |

Connection pool management prevents resource exhaustion from failed transactions:

```
Transaction Cleanup Algorithm:
1. Monitor active database connections for each worker process
2. Track transaction duration and identify long-running transactions
3. When worker process fails, identify orphaned database connections
4. Rollback any active transactions associated with failed worker
5. Close database connections cleanly to return them to pool
6. Update connection pool statistics and alert on connection leaks
7. Verify database consistency after cleanup completion
```

⚠️ **Pitfall: Incomplete Resource Cleanup**

**The Problem:** When worker processes crash or are terminated forcefully, temporary files, database connections, and child processes may not be cleaned up properly, leading to resource exhaustion and system instability.

**Why It's Wrong:** Accumulated resource leaks can cause disk space exhaustion, database connection pool exhaustion, zombie processes, and memory leaks that degrade system performance over time.

**How to Fix:** Implement comprehensive resource tracking with cleanup verification, use signal handlers for graceful shutdown, implement resource monitors that detect and clean up orphaned resources, and test cleanup procedures under various failure scenarios.

![Webhook Notification Flow](./diagrams/webhook-notification-flow.svg)

### Implementation Guidance

This section provides practical implementation details for building robust error handling and recovery mechanisms in the media processing pipeline.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Classification | Exception type matching | Rule engine with configurable decision trees |
| Retry Management | Redis sorted sets | Dedicated retry service with complex scheduling |
| Circuit Breaker | Simple state tracking | Library like `pybreaker` with metrics |
| Resource Monitoring | Basic system calls | APM tools like Prometheus with custom metrics |
| Cleanup Scheduling | Cron-based cleanup | Event-driven cleanup with dependency tracking |

**Recommended File Structure:**

```
internal/
  error_handling/
    __init__.py              ← Error handling exports
    error_classifier.py      ← Error classification logic  
    retry_manager.py         ← Retry and backoff implementation
    circuit_breaker.py       ← Circuit breaker for external deps
    resource_cleanup.py      ← Cleanup and recovery procedures
    recovery_coordinator.py  ← Overall recovery orchestration
  monitoring/
    resource_monitor.py      ← System resource monitoring
    health_checker.py        ← Worker and service health checks
  storage/
    temp_file_manager.py     ← Temporary file lifecycle management
    cleanup_scheduler.py     ← Background cleanup processes
```

**Infrastructure Starter Code:**

Here's a complete error classification system that categorizes failures for appropriate retry handling:

```python
import logging
import re
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import subprocess

class ErrorCategory(Enum):
    TRANSIENT = "transient"
    RESOURCE = "resource" 
    PERMANENT = "permanent"
    DEPENDENCY = "dependency"

@dataclass
class ErrorClassification:
    category: ErrorCategory
    retry_eligible: bool
    max_retries: int
    backoff_factor: float
    description: str

class ErrorClassifier:
    """
    Classifies errors into categories for appropriate retry handling.
    Uses exception types, error codes, and message patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_classification_rules()
    
    def _init_classification_rules(self):
        """Initialize error classification rules."""
        # Network and connectivity errors
        self.network_errors = {
            'ConnectionError', 'TimeoutError', 'ConnectionResetError',
            'redis.exceptions.ConnectionError', 'requests.exceptions.Timeout'
        }
        
        # Resource constraint errors  
        self.resource_errors = {
            'MemoryError', 'OSError', 'IOError'
        }
        
        # FFmpeg error code patterns
        self.ffmpeg_exit_codes = {
            # Transient errors
            1: ErrorClassification(ErrorCategory.RESOURCE, True, 3, 2.0, "Generic FFmpeg error"),
            # Permanent errors  
            69: ErrorClassification(ErrorCategory.PERMANENT, False, 0, 0, "Invalid input format"),
            # Add more as needed
        }
        
        # Error message patterns
        self.error_patterns = [
            (re.compile(r'disk.*full|no space left', re.I), ErrorCategory.RESOURCE),
            (re.compile(r'permission denied|access denied', re.I), ErrorCategory.PERMANENT),
            (re.compile(r'invalid.*format|corrupt.*file', re.I), ErrorCategory.PERMANENT),
            (re.compile(r'network.*unreachable|connection.*refused', re.I), ErrorCategory.DEPENDENCY),
        ]
    
    def classify_error(self, exception: Exception, context: Dict = None) -> ErrorClassification:
        """
        Classify an error for retry handling.
        
        Args:
            exception: The exception that occurred
            context: Additional context (exit codes, job info, etc.)
            
        Returns:
            ErrorClassification with retry strategy
        """
        exception_name = type(exception).__name__
        error_message = str(exception)
        
        # Check for network/connectivity errors
        if exception_name in self.network_errors:
            return ErrorClassification(
                ErrorCategory.TRANSIENT, True, 5, 2.0, 
                "Network connectivity error"
            )
        
        # Check for resource errors
        if exception_name in self.resource_errors:
            return self._classify_resource_error(exception, error_message)
        
        # Check FFmpeg exit codes
        if context and 'exit_code' in context:
            if context['exit_code'] in self.ffmpeg_exit_codes:
                return self.ffmpeg_exit_codes[context['exit_code']]
        
        # Check error message patterns
        for pattern, category in self.error_patterns:
            if pattern.search(error_message):
                return self._get_default_classification(category)
        
        # Default to transient with limited retries
        return ErrorClassification(
            ErrorCategory.TRANSIENT, True, 2, 1.5,
            "Unclassified error - limited retry"
        )
    
    def _classify_resource_error(self, exception: Exception, message: str) -> ErrorClassification:
        """Classify resource-related errors more specifically."""
        if 'memory' in message.lower() or isinstance(exception, MemoryError):
            return ErrorClassification(
                ErrorCategory.RESOURCE, True, 2, 2.5,
                "Memory exhaustion error"  
            )
        elif 'disk' in message.lower() or 'space' in message.lower():
            return ErrorClassification(
                ErrorCategory.RESOURCE, True, 3, 2.0,
                "Disk space error"
            )
        else:
            return ErrorClassification(
                ErrorCategory.RESOURCE, True, 3, 2.0, 
                "General resource error"
            )
    
    def _get_default_classification(self, category: ErrorCategory) -> ErrorClassification:
        """Get default classification for error category."""
        defaults = {
            ErrorCategory.TRANSIENT: ErrorClassification(category, True, 5, 2.0, "Transient error"),
            ErrorCategory.RESOURCE: ErrorClassification(category, True, 3, 2.5, "Resource error"), 
            ErrorCategory.PERMANENT: ErrorClassification(category, False, 0, 0, "Permanent error"),
            ErrorCategory.DEPENDENCY: ErrorClassification(category, True, 7, 2.0, "External dependency error")
        }
        return defaults[category]
```

**Core Logic Skeleton Code:**

Here's the main retry manager that needs to be implemented:

```python
import asyncio
import time
import random
from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict
import redis
import json
import logging

@dataclass 
class RetryContext:
    job_id: str
    attempt_number: int
    last_error: str
    classification: ErrorClassification
    scheduled_at: float
    priority: int

class RetryManager:
    """
    Manages job retries with exponential backoff and priority scheduling.
    Implements circuit breaker pattern for external dependencies.
    """
    
    def __init__(self, redis_client: redis.Redis, config: ProcessingConfig):
        self.redis = redis_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_classifier = ErrorClassifier()
        
        # Circuit breaker state tracking
        self.circuit_breakers = {}
    
    async def schedule_retry(self, job: ProcessingJob, error: Exception, context: Dict = None) -> bool:
        """
        Schedule a job for retry based on error classification.
        
        Args:
            job: The failed processing job
            error: Exception that caused failure  
            context: Additional error context
            
        Returns:
            True if retry scheduled, False if job moved to dead letter queue
        """
        # TODO 1: Use error_classifier to classify the error and get retry strategy
        # TODO 2: Check if job has exceeded maximum retry attempts for its error type
        # TODO 3: If retry limit exceeded, move job to dead letter queue and return False
        # TODO 4: Calculate retry delay using exponential backoff with jitter
        # TODO 5: Create RetryContext with job info, attempt number, and scheduled time
        # TODO 6: Store retry context in Redis sorted set with schedule time as score  
        # TODO 7: Update job status to indicate retry scheduled
        # TODO 8: Log retry scheduling with job context and delay information
        # TODO 9: Update retry metrics for monitoring
        # TODO 10: Return True to indicate retry was scheduled
        
        classification = self.error_classifier.classify_error(error, context)
        
        if not classification.retry_eligible:
            return await self._move_to_dead_letter_queue(job, error, "Non-retryable error")
            
        if job.retry_count >= classification.max_retries:
            return await self._move_to_dead_letter_queue(job, error, "Retry limit exceeded")
            
        # Implementation continues...
        pass
    
    def calculate_retry_delay(self, attempt: int, base_delay: float, backoff_factor: float, 
                            max_delay: float, jitter_range: float = 0.2) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.
        
        Args:
            attempt: Current retry attempt number (1-based)
            base_delay: Base delay in seconds
            backoff_factor: Exponential backoff multiplier  
            max_delay: Maximum delay cap in seconds
            jitter_range: Jitter as fraction of delay (0.0-1.0)
            
        Returns:
            Calculated delay in seconds with jitter applied
        """
        # TODO 1: Calculate exponential delay: base_delay * (backoff_factor ** (attempt - 1))
        # TODO 2: Apply maximum delay cap to prevent excessive wait times
        # TODO 3: Calculate jitter amount: delay * random value in [-jitter_range, +jitter_range]  
        # TODO 4: Add jitter to delay to prevent thundering herd
        # TODO 5: Ensure final delay is not negative
        # TODO 6: Log calculated delay for debugging
        # TODO 7: Return final delay value
        pass
    
    async def process_retry_queue(self, worker_id: str) -> Optional[ProcessingJob]:
        """
        Check retry queue for jobs ready to process.
        
        Args:
            worker_id: ID of worker requesting retry job
            
        Returns:
            Job ready for retry processing, or None if no jobs ready
        """
        # TODO 1: Get current timestamp for comparing against scheduled retry times
        # TODO 2: Query Redis sorted set for jobs with score <= current time (ready to retry)
        # TODO 3: Use ZPOPMIN to atomically get and remove highest priority retry job
        # TODO 4: If no retry jobs ready, return None
        # TODO 5: Deserialize retry context from Redis data
        # TODO 6: Load original ProcessingJob from retry context
        # TODO 7: Increment job retry count and update retry context
        # TODO 8: Mark job status as processing and assign to worker
        # TODO 9: Log retry job assignment with worker and attempt information
        # TODO 10: Return ProcessingJob ready for retry processing
        pass
    
    async def _move_to_dead_letter_queue(self, job: ProcessingJob, error: Exception, 
                                       reason: str) -> bool:
        """Move job to dead letter queue for manual review."""
        # TODO 1: Create dead letter record with job, error history, and failure reason
        # TODO 2: Store record in dead letter queue (Redis list or database)
        # TODO 3: Update job status to FAILED with reason
        # TODO 4: Send webhook notification about job failure if webhook_url configured
        # TODO 5: Log dead letter queue movement with job ID and reason
        # TODO 6: Update dead letter queue metrics
        # TODO 7: Clean up any retry queue entries for this job
        # TODO 8: Return True to indicate successful dead letter processing
        pass
    
    def _get_circuit_breaker_key(self, service: str) -> str:
        """Generate Redis key for circuit breaker state."""
        return f"circuit_breaker:{service}"
    
    async def check_circuit_breaker(self, service: str) -> bool:
        """
        Check if circuit breaker allows requests to service.
        
        Args:
            service: Service name (e.g., 'webhooks', 'cloud_storage')
            
        Returns:
            True if requests allowed, False if circuit is open
        """  
        # TODO 1: Get circuit breaker state from Redis for the service
        # TODO 2: If no state exists, initialize as CLOSED (allow requests)
        # TODO 3: If state is CLOSED, allow request and return True
        # TODO 4: If state is OPEN, check if timeout period has expired
        # TODO 5: If timeout expired, transition to HALF_OPEN and allow single test request
        # TODO 6: If still in timeout period, block request and return False  
        # TODO 7: Log circuit breaker decision for monitoring
        # TODO 8: Return decision (True = allow, False = block)
        pass
    
    async def record_service_result(self, service: str, success: bool) -> None:
        """Record result of service call for circuit breaker logic."""
        # TODO 1: Get current circuit breaker state for service
        # TODO 2: If state is HALF_OPEN and request succeeded, transition to CLOSED
        # TODO 3: If state is HALF_OPEN and request failed, transition back to OPEN  
        # TODO 4: If state is CLOSED, track failure rate over time window
        # TODO 5: If failure rate exceeds threshold, transition to OPEN
        # TODO 6: Update circuit breaker metrics and state in Redis
        # TODO 7: Log state transitions for monitoring and alerting
        pass
```

**Milestone Checkpoint:**

After implementing the error handling system:

1. **Test Error Classification**: Run `python -m pytest tests/test_error_classifier.py -v` to verify error classification logic
2. **Test Retry Logic**: Simulate job failures and verify exponential backoff calculations
3. **Test Circuit Breaker**: Simulate external service failures and verify circuit breaker state transitions  
4. **Test Resource Cleanup**: Create temporary files, kill worker processes, verify cleanup completion
5. **Manual Verification**: Submit a job that will fail (invalid input file), verify it moves through retry attempts and eventually reaches dead letter queue

**Expected Behaviors:**
- Transient network errors should retry immediately with exponential backoff
- Resource errors should delay retry until system resources are available  
- Permanent errors (invalid file format) should move directly to dead letter queue
- Circuit breakers should prevent retry storms when external services fail
- Temporary files should be cleaned up within 60 seconds of job failure
- Worker processes should restart automatically and resume processing jobs


## Testing Strategy and Milestones

> **Milestone(s):** All milestones (1-3) as comprehensive testing strategy ensures reliable image processing, video transcoding, and job queue functionality across the entire media processing pipeline

### Mental Model: Quality Assurance Laboratory

Think of our testing strategy as a comprehensive quality assurance laboratory for a manufacturing facility. Just as a QA lab tests components individually (unit testing), validates the entire assembly line with real products (integration testing), and maintains strict checkpoints at each production stage (milestone validation), our media processing pipeline requires the same systematic approach to ensure reliability.

In a manufacturing QA lab, you have three distinct testing phases: component testing with synthetic materials to verify individual part specifications, full production line testing with actual products to validate end-to-end workflows, and stage-gate reviews at each manufacturing milestone to ensure quality before proceeding to the next phase. Our media processing testing strategy follows this exact pattern, ensuring each component works correctly in isolation, the entire pipeline processes real media files reliably, and each development milestone delivers measurable functionality.

The critical insight here is that media processing testing requires specialized approaches because we're dealing with binary data, time-consuming operations, and external dependencies. Unlike simple web applications where you can mock everything, media processing demands testing with actual image and video files, realistic processing times, and real queue behavior to catch the subtle bugs that only emerge under production conditions.

### Unit Testing Approach

**Component Isolation Strategy**

Unit testing in media processing systems requires a sophisticated approach to isolating components while maintaining realistic test conditions. Each processing component—image manipulation, video transcoding, job queue management, and progress tracking—must be testable independently without requiring external dependencies or lengthy processing operations.

The fundamental challenge lies in creating test conditions that mirror production behavior without the overhead of processing large media files or waiting for complex transcoding operations. This demands carefully crafted test fixtures, mock implementations that preserve behavioral accuracy, and testing strategies that validate both happy path operations and edge case scenarios.

> **Decision: Synthetic Test Media Generation**
> - **Context**: Unit tests need realistic media files but cannot rely on large binary fixtures or external file dependencies that slow down test execution and complicate CI/CD pipelines
> - **Options Considered**: Real media file fixtures, procedurally generated test media, mocked media objects with metadata simulation
> - **Decision**: Implement procedurally generated test media with configurable characteristics
> - **Rationale**: Generated media provides precise control over test conditions (specific resolutions, formats, corruption patterns) while maintaining fast test execution and avoiding binary repository bloat
> - **Consequences**: Tests run quickly and reliably but require additional infrastructure to generate realistic media characteristics and edge cases

| Test Media Type | Generation Method | Characteristics | Test Scenarios |
|-----------------|-------------------|-----------------|-----------------|
| Minimal Valid Image | PIL/Pillow solid color generation | 1x1 to 100x100 pixels, RGB/RGBA | Format detection, basic operations |
| Edge Case Image | Programmatic creation with specific properties | Extreme dimensions, unusual aspect ratios | Memory limit validation, resize edge cases |
| Corrupted Image | Binary manipulation of valid images | Truncated headers, invalid metadata | Error handling, graceful degradation |
| Video Test Files | FFmpeg synthetic generation | Short duration, known characteristics | Transcoding logic, progress calculation |
| Metadata Rich Media | Embedded EXIF/metadata injection | GPS, orientation, camera data | Metadata extraction, privacy stripping |

**Image Processing Unit Tests**

Image processing unit tests focus on validating individual operations like resizing algorithms, format conversion accuracy, and metadata handling without requiring large image files or lengthy processing times. Each test verifies specific algorithmic behavior using minimal synthetic images that exercise the same code paths as production media.

The testing strategy emphasizes boundary condition validation, ensuring resize operations handle edge cases like 1-pixel images, extreme aspect ratios, and memory-constrained scenarios. Format conversion tests verify color space preservation, quality setting behavior, and lossy compression characteristics using programmatically generated test images with known pixel values.

| Test Category | Test Methods | Validation Focus | Expected Outcomes |
|---------------|--------------|------------------|-------------------|
| Resize Operations | `test_resize_preserve_aspect_ratio()` | Aspect ratio mathematics | Output dimensions match calculated ratios |
| Interpolation | `test_lanczos_vs_bilinear_quality()` | Algorithm quality differences | Measurable quality metrics comparison |
| Format Conversion | `test_jpeg_quality_settings()` | Lossy compression behavior | File size decreases with lower quality |
| Metadata Handling | `test_exif_orientation_rotation()` | EXIF-based rotation accuracy | Rotated image matches expected orientation |
| Memory Management | `test_large_image_memory_calculation()` | Resource requirement estimation | Memory calculations match actual usage |
| Error Conditions | `test_corrupted_image_graceful_failure()` | Exception handling patterns | Specific exceptions with descriptive messages |

**Video Processing Unit Tests**

Video processing unit tests require a different approach because video operations typically involve external FFmpeg processes and longer execution times. The testing strategy focuses on command generation validation, progress parsing accuracy, and metadata extraction without executing full transcoding operations.

FFmpeg integration tests use synthetic video files generated on-demand with specific characteristics like resolution, duration, and codec combinations. These minimal test videos allow validation of transcoding logic, adaptive bitrate variant generation, and thumbnail extraction without the overhead of processing production-sized media files.

```python
# Example test structure for FFmpeg command generation validation
def test_build_transcode_command_h264():
    # Validates FFmpeg command construction without executing
    config = VideoTranscodingConfig(
        video_codec='h264',
        target_bitrate=2000000,
        crf_value=23
    )
    command = wrapper._build_transcode_command(
        'input.mp4', 'output.mp4', config, test_metadata
    )
    assert '-c:v libx264' in command
    assert '-b:v 2M' in command
    assert '-crf 23' in command
```

| Test Category | Test Methods | Validation Focus | Mock Strategy |
|---------------|--------------|------------------|---------------|
| Command Generation | `test_ffmpeg_command_building()` | Proper FFmpeg argument construction | Mock metadata, validate command arrays |
| Progress Parsing | `test_ffmpeg_progress_extraction()` | Progress percentage calculation | Mock FFmpeg output streams |
| Error Detection | `test_ffmpeg_error_classification()` | Exit code and stderr interpretation | Simulated FFmpeg failure scenarios |
| Codec Selection | `test_optimal_codec_selection()` | Input format to output codec mapping | Mock format detection results |
| ABR Variant Logic | `test_adaptive_bitrate_calculation()` | Multi-variant parameter generation | Mock input video characteristics |

**Job Queue Unit Tests**

Job queue unit tests validate queue operations, priority handling, and worker coordination logic using in-memory implementations that preserve behavioral accuracy while eliminating Redis dependencies. These tests focus on race condition prevention, atomic operation behavior, and state transition validation.

The testing approach uses dependency injection to replace Redis connections with thread-safe in-memory implementations that simulate Redis behavior including atomic operations, blocking pops, and pub/sub messaging. This allows comprehensive testing of queue logic without external infrastructure dependencies.

| Test Category | Test Methods | Validation Focus | Mock Implementation |
|---------------|--------------|------------------|-------------------|
| Job Submission | `test_atomic_job_submission()` | Deduplication and atomic operations | In-memory job store with threading locks |
| Priority Queuing | `test_priority_order_preservation()` | Higher priority jobs processed first | Mock Redis sorted set behavior |
| Worker Coordination | `test_multiple_worker_job_distribution()` | Jobs distributed fairly across workers | Simulated worker pool with job assignment |
| State Transitions | `test_job_lifecycle_state_machine()` | Valid state transitions only | State machine validation with invalid transition attempts |
| Dead Letter Queue | `test_failed_job_dead_letter_routing()` | Permanent failures routed correctly | Mock failure classification and routing logic |

**Progress Tracking Unit Tests**

Progress tracking unit tests validate stage-based progress calculation, notification threshold logic, and webhook delivery mechanisms using mock HTTP endpoints and in-memory progress storage. These tests ensure accurate progress reporting and reliable notification delivery without external webhook dependencies.

The testing strategy focuses on progress calculation accuracy, race condition prevention in concurrent updates, and webhook delivery retry logic. Mock HTTP servers simulate various webhook endpoint behaviors including success responses, timeout conditions, and authentication failures.

| Test Category | Test Methods | Validation Focus | Mock Strategy |
|---------------|--------------|------------------|---------------|
| Progress Calculation | `test_stage_weighted_progress()` | Accurate percentage calculation | Mock stage completion data |
| Concurrent Updates | `test_progress_race_condition_prevention()` | Sequence number validation | Multithreaded progress updates |
| Webhook Delivery | `test_webhook_retry_exponential_backoff()` | Retry logic and backoff timing | Mock HTTP server with failure simulation |
| Signature Verification | `test_webhook_hmac_signature_validation()` | Cryptographic signature accuracy | Known test vectors with expected signatures |

### Integration Testing Strategy

**End-to-End Media Processing Workflows**

Integration testing validates complete media processing workflows using real media files and the full technology stack including Redis job queues, worker processes, and webhook notification delivery. These tests ensure all components work together correctly under realistic conditions and catch integration issues that unit tests cannot detect.

The integration testing environment requires careful setup to provide realistic test conditions while maintaining fast execution and reliable cleanup. This includes containerized dependencies, realistic media file fixtures, and automated environment provisioning that matches production characteristics.

> **Decision: Containerized Integration Test Environment**
> - **Context**: Integration tests require Redis, potentially FFmpeg, webhook endpoint simulation, and file system storage that must be consistent across development machines and CI/CD pipelines
> - **Options Considered**: Local dependency installation, Docker Compose test environment, cloud-based test infrastructure
> - **Decision**: Docker Compose with service containers for Redis, mock webhook server, and shared volume for media files
> - **Rationale**: Containerized approach ensures consistent test environment, easy cleanup, parallel test execution without conflicts, and matches production deployment patterns
> - **Consequences**: Tests require Docker but provide reliable, reproducible results with easy local development and CI/CD integration

**Real Media File Test Fixtures**

Integration tests require carefully curated media file fixtures that represent realistic production scenarios without consuming excessive storage or processing time. The test fixture strategy balances comprehensive format coverage with practical test execution requirements.

The fixture collection includes representative samples of each supported format with varying characteristics: different resolutions, aspect ratios, metadata complexity, and file sizes. Additionally, it includes problematic files that test edge cases like corrupted headers, unusual metadata, and format variants that require special handling.

| File Category | Format Examples | Characteristics | Test Scenarios |
|---------------|-----------------|-----------------|------------------|
| Standard Images | JPEG, PNG, WebP | Common resolutions, normal metadata | Basic processing workflows |
| High Resolution | Large JPEG, PNG | 4K+ resolution, substantial file size | Memory management, processing time |
| Unusual Formats | GIF, TIFF, BMP | Less common formats | Format detection, conversion support |
| Metadata Rich | JPEG with GPS/EXIF | Comprehensive metadata sets | Privacy stripping, metadata preservation |
| Problematic Files | Corrupted headers, invalid metadata | Various corruption types | Error handling, graceful degradation |
| Video Samples | MP4 (H.264), WebM, MOV | Short duration, various codecs | Transcoding workflows, ABR generation |

**Queue Processing Integration Tests**

Queue processing integration tests validate the complete job lifecycle from submission through worker processing to completion notification. These tests use real Redis instances and actual worker processes to ensure proper job distribution, progress tracking, and error handling under concurrent conditions.

The testing approach spawns multiple worker processes and submits various job types to validate queue behavior under realistic load conditions. Tests verify proper job prioritization, worker failure recovery, and resource cleanup after job completion or failure.

```python
# Example integration test structure for complete job processing
@pytest.mark.integration
def test_complete_image_processing_workflow():
    # Submit image processing job with webhook
    job = submit_job(
        input_file='fixtures/test_image.jpg',
        output_specs=[
            OutputSpecification(format='webp', width=800, quality=85),
            OutputSpecification(format='png', width=400, quality=None)
        ],
        webhook_url='http://mock-server:8080/webhook'
    )
    
    # Wait for processing completion
    final_status = wait_for_job_completion(job.job_id, timeout=30)
    
    # Validate results
    assert final_status == JobStatus.COMPLETED
    assert len(job.output_files) == 2
    assert all(os.path.exists(f) for f in job.output_files)
    assert webhook_received_completion_notification(job.job_id)
```

| Test Scenario | Components Involved | Validation Points | Expected Behavior |
|---------------|-------------------|-------------------|-------------------|
| Single Image Job | API, Queue, Worker, Storage | Job completion, output file creation | Files created with correct formats and dimensions |
| Multiple Output Specs | Worker, Format Converter | All output specifications processed | Each output meets specified requirements |
| Video Transcoding | Worker, FFmpeg, Progress Tracker | Progress updates, final transcoding | ABR variants created with proper manifests |
| Webhook Notifications | Progress Tracker, HTTP Client | Notification delivery, retry logic | Webhooks delivered with correct signatures |
| Worker Failure Recovery | Queue, Worker Coordinator | Job reassignment, cleanup | Jobs complete despite worker failures |
| Priority Job Processing | Queue, Multiple Workers | Job ordering, execution priority | High priority jobs processed first |

**Webhook Integration Testing**

Webhook integration tests validate notification delivery under various network conditions and endpoint behaviors. These tests use mock HTTP servers that simulate different webhook endpoint responses to ensure robust delivery logic and proper error handling.

The testing infrastructure includes webhook endpoints that simulate success responses, timeout conditions, authentication failures, and temporary unavailability. Tests verify exponential backoff retry logic, signature verification, and eventual delivery guarantees.

| Webhook Scenario | Mock Server Behavior | Test Validation | Expected Outcome |
|------------------|---------------------|-----------------|------------------|
| Successful Delivery | HTTP 200 response | Single delivery attempt | Webhook marked as delivered |
| Temporary Failure | HTTP 503, then 200 | Retry with backoff | Eventually delivered after retries |
| Permanent Failure | Consistent HTTP 404 | Retry limit reached | Marked as permanently failed |
| Timeout Condition | Connection timeout | Timeout handling | Retry with exponential backoff |
| Invalid Signature | Signature verification failure | Security validation | Webhook rejected by endpoint |

**Performance and Load Testing**

Integration tests include performance validation to ensure the system handles realistic workloads without degradation. These tests process multiple concurrent jobs with various media types and sizes to validate system behavior under load conditions.

Performance tests measure processing throughput, queue latency, and resource utilization patterns. They establish baseline performance characteristics and detect performance regressions during development. The tests use realistic media files and job mixes that represent expected production workloads.

| Performance Test Type | Load Characteristics | Metrics Measured | Acceptance Criteria |
|----------------------|---------------------|------------------|-------------------|
| Concurrent Image Processing | 50 simultaneous image jobs | Jobs per second, memory usage | >10 images/sec, <2GB memory per worker |
| Mixed Workload | Images and videos combined | Queue latency, completion time | <5 second queue latency, predictable completion times |
| Large File Handling | High resolution images/videos | Memory efficiency, processing time | Linear scaling with file size |
| Queue Saturation | More jobs than worker capacity | Queue depth, job assignment fairness | Fair job distribution, no worker starvation |

### Milestone Checkpoints and Validation

**Milestone 1: Image Processing Validation**

The first milestone checkpoint validates core image processing functionality including format conversion, resizing operations, metadata handling, and thumbnail generation. This milestone establishes the foundation for media processing capabilities and must demonstrate reliable operation with various image formats and processing parameters.

Validation focuses on algorithmic correctness, output quality, and proper handling of edge cases like extreme aspect ratios, unusual metadata, and format-specific requirements. The checkpoint ensures image processing operations produce consistent, high-quality results that meet specification requirements.

> **Milestone 1 Success Criteria**: Image processing component handles all supported formats, produces outputs matching specifications, preserves or strips metadata as configured, and gracefully handles error conditions without crashing or corrupting data

| Validation Area | Test Procedures | Success Indicators | Troubleshooting Steps |
|-----------------|-----------------|-------------------|----------------------|
| Format Detection | Load images of each supported format | Correct format identification for all files | Check PIL/Pillow installation, verify file signatures |
| Resize Operations | Process images with various target dimensions | Output dimensions match specifications with proper aspect ratio handling | Validate resize algorithm selection, check interpolation settings |
| Quality Settings | Convert to lossy formats with different quality levels | File size decreases appropriately, visual quality preserved | Verify quality parameter mapping, check codec-specific settings |
| Metadata Handling | Process images with rich EXIF data | Metadata preserved/stripped as configured, orientation handled correctly | Test EXIF parsing library, validate rotation logic |
| Error Handling | Process corrupted or invalid image files | Graceful failure with descriptive error messages | Check exception handling, validate error classification |

**Checkpoint Commands and Expected Outputs**

The milestone checkpoint includes specific commands to validate functionality and expected outputs that indicate successful implementation. These commands provide concrete validation steps that can be executed manually or integrated into automated testing pipelines.

```bash
# Milestone 1 validation commands
python -m pytest tests/test_image_processing.py -v
python scripts/process_test_image.py --input fixtures/test.jpg --output /tmp/resized.webp --width 800 --quality 85
python scripts/extract_metadata.py fixtures/test_with_exif.jpg

# Expected outputs:
# - All image processing tests pass
# - Resized image created with correct dimensions and format
# - Metadata extracted and displayed properly formatted
```

| Command | Expected Output | Success Validation | Common Issues |
|---------|-----------------|-------------------|---------------|
| `pytest tests/test_image_processing.py` | All tests pass, <30 second execution | Green test results, no failures | Missing dependencies, incorrect test fixtures |
| Image resize command | Output file created with target dimensions | File exists, correct size via `identify` | Memory errors, incorrect aspect ratio calculation |
| Metadata extraction | JSON output with EXIF fields | GPS, orientation, camera data displayed | EXIF library issues, binary data handling errors |

**Milestone 2: Video Transcoding Validation**

The second milestone validates video transcoding capabilities including format conversion, adaptive bitrate generation, thumbnail extraction, and FFmpeg integration. This milestone demonstrates the system's ability to handle complex video processing workflows with proper progress tracking and error recovery.

Video processing validation requires longer execution times and more complex success criteria due to the computational complexity of video transcoding operations. The checkpoint ensures reliable FFmpeg integration, accurate progress reporting, and proper handling of various video formats and codecs.

| Validation Area | Test Procedures | Success Indicators | Performance Expectations |
|-----------------|-----------------|-------------------|-------------------------|
| Basic Transcoding | Convert MP4 to WebM, various resolutions | Output plays correctly, maintains quality | <2x realtime for simple conversions |
| ABR Generation | Create HLS/DASH variants from source video | Manifest files created, segments playable | Multiple variants with proper bitrate ladder |
| Progress Tracking | Monitor transcoding progress updates | Accurate percentage reporting throughout process | Progress updates at least every 5 seconds |
| Thumbnail Extraction | Extract frames at various timestamps | Thumbnail images created at correct times | Frame accuracy within 1 second |
| Error Recovery | Process corrupted/unsupported video files | Graceful failure, proper error classification | Clear error messages, no worker crashes |

**Checkpoint Validation Procedures**

```bash
# Milestone 2 validation commands
python -m pytest tests/test_video_transcoding.py -v
python scripts/transcode_video.py --input fixtures/test_video.mp4 --output /tmp/transcoded.webm --resolution 720p
python scripts/generate_abr.py fixtures/test_video.mp4 /tmp/abr_output/
python scripts/extract_thumbnails.py fixtures/test_video.mp4 /tmp/thumbnails/ --timestamps 10,30,60

# Expected behavior:
# - Video transcoding completes successfully
# - ABR variants created with manifest files
# - Thumbnail images extracted at correct timestamps
# - Progress updates displayed during processing
```

| Validation Step | Command | Expected Result | Troubleshooting |
|-----------------|---------|-----------------|-----------------|
| Basic Transcoding | `python scripts/transcode_video.py` | Output video file created, playable | Check FFmpeg installation, validate input file |
| ABR Generation | `python scripts/generate_abr.py` | Multiple resolution variants + manifest | Verify HLS/DASH configuration, check segment creation |
| Progress Monitoring | Watch console during transcoding | Regular progress updates, accurate percentages | Check FFmpeg output parsing, validate progress calculation |
| Thumbnail Extraction | `python scripts/extract_thumbnails.py` | Image files at specified timestamps | Verify timestamp parsing, check frame extraction accuracy |

**Milestone 3: Job Queue and Progress Validation**

The final milestone validates complete asynchronous processing workflows including job queuing, worker coordination, progress tracking, and webhook notifications. This milestone demonstrates the system's ability to handle production workloads with reliable job processing and real-time progress updates.

Validation encompasses the entire system integration including Redis queue operations, multiple worker processes, webhook delivery, and error recovery mechanisms. The checkpoint ensures the system can handle concurrent processing requests while maintaining job ordering, progress accuracy, and notification delivery reliability.

| Validation Area | Test Procedures | Success Indicators | Scale Requirements |
|-----------------|-----------------|-------------------|--------------------|
| Job Submission | Submit multiple jobs with different priorities | Jobs queued and processed in priority order | Handle 100+ concurrent job submissions |
| Worker Coordination | Start multiple worker processes | Jobs distributed fairly, no duplicate processing | 5+ workers processing simultaneously |
| Progress Tracking | Monitor real-time progress updates | Accurate stage-based progress reporting | Progress updates within 5-second intervals |
| Webhook Delivery | Configure webhook endpoints | Notifications delivered reliably with retries | 99%+ delivery success rate |
| Error Recovery | Simulate worker failures, network issues | Jobs complete despite infrastructure problems | Automatic recovery within 30 seconds |

**Complete System Validation**

The final validation demonstrates end-to-end system functionality with realistic workloads and production-like conditions. This validation proves the system meets all functional and performance requirements established in the goals and non-goals section.

```bash
# Milestone 3 complete system validation
docker-compose up -d redis webhook-server
python -m pytest tests/integration/ -v
python scripts/submit_batch_jobs.py --count 20 --mixed-media fixtures/
python scripts/monitor_queue_health.py --duration 300

# Expected system behavior:
# - All integration tests pass
# - Batch jobs processed successfully with progress tracking
# - Queue health monitoring shows stable operations
# - Webhook notifications delivered reliably
```

| System Validation | Command | Success Criteria | Performance Targets |
|-------------------|---------|------------------|-------------------|
| Integration Test Suite | `pytest tests/integration/` | All tests pass, <5 minutes execution | No test failures, consistent timing |
| Batch Job Processing | `python scripts/submit_batch_jobs.py` | All jobs complete successfully | >10 concurrent jobs, fair processing |
| Queue Health Monitoring | `python scripts/monitor_queue_health.py` | Stable metrics, no error accumulation | <100ms average queue latency |
| Webhook Reliability | Monitor webhook delivery logs | >99% delivery success, proper retries | <5 second delivery latency |

**⚠️ Common Milestone Pitfalls**

**Pitfall: Insufficient Test Media Coverage**
Many implementations fail milestone validation because test fixtures don't cover edge cases like unusual aspect ratios, extreme file sizes, or format-specific quirks. Always include problematic media files that exercise error handling paths and boundary conditions.

**Pitfall: Race Conditions in Progress Updates**
Progress tracking often fails under concurrent load due to race conditions between progress updates and job completion. Implement proper sequence number validation and atomic update operations to prevent progress reversals and inconsistent state.

**Pitfall: Webhook Delivery Assumptions**
Webhook notification testing frequently assumes reliable network conditions and responsive endpoints. Production environments have unreliable webhooks, so test with flaky network conditions, slow endpoints, and authentication failures to validate retry logic.

**Pitfall: Resource Cleanup Failures**
Failed tests often leave temporary files, processes, or queue entries that interfere with subsequent test runs. Implement comprehensive cleanup procedures in test teardown methods and use isolated test environments to prevent cross-test contamination.

### Implementation Guidance

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Test Framework | pytest with basic fixtures | pytest with custom plugins, parameterized tests |
| Mock Library | unittest.mock for basic mocking | responses library for HTTP mocking, fakeredis for Redis |
| Media Generation | PIL solid color images | FFmpeg synthetic video generation |
| Test Orchestration | Manual test execution | Docker Compose with service containers |
| Assertion Library | Basic assert statements | Custom media validation assertions |
| Coverage Analysis | pytest-cov for basic coverage | Coverage with branch analysis and media-specific metrics |

**B. Recommended File Structure**

```
media-processing-pipeline/
├── tests/
│   ├── unit/
│   │   ├── test_image_processing.py      ← Image component unit tests
│   │   ├── test_video_transcoding.py     ← Video component unit tests
│   │   ├── test_job_queue.py             ← Queue operations unit tests
│   │   └── test_progress_tracking.py     ← Progress component unit tests
│   ├── integration/
│   │   ├── test_end_to_end_workflows.py  ← Complete processing workflows
│   │   ├── test_queue_integration.py     ← Queue with real Redis
│   │   └── test_webhook_delivery.py      ← Webhook notification tests
│   ├── fixtures/
│   │   ├── images/                       ← Test image files
│   │   ├── videos/                       ← Test video files
│   │   └── generate_fixtures.py          ← Synthetic media generation
│   ├── helpers/
│   │   ├── test_media_utils.py           ← Media validation helpers
│   │   ├── mock_servers.py               ← Mock webhook/API servers
│   │   └── redis_test_utils.py           ← Redis testing utilities
│   └── conftest.py                       ← pytest configuration and shared fixtures
├── scripts/
│   ├── validate_milestone_1.py           ← Milestone 1 validation script
│   ├── validate_milestone_2.py           ← Milestone 2 validation script
│   ├── validate_milestone_3.py           ← Milestone 3 validation script
│   └── performance_benchmark.py          ← Performance testing script
└── docker-compose.test.yml               ← Test environment services
```

**C. Infrastructure Starter Code**

**Test Media Generation Utilities**

```python
"""Test media generation utilities for creating synthetic test fixtures."""

import os
import tempfile
from PIL import Image, ImageDraw
from typing import Tuple, Dict, Any
import json

class TestMediaGenerator:
    """Generates synthetic media files for testing purposes."""
    
    @staticmethod
    def create_test_image(width: int, height: int, format: str = 'JPEG', 
                         color: Tuple[int, int, int] = (255, 0, 0)) -> bytes:
        """Create a simple solid color test image in memory.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels  
            format: Output format (JPEG, PNG, WebP)
            color: RGB color tuple
            
        Returns:
            Image data as bytes
        """
        image = Image.new('RGB', (width, height), color)
        
        # Add some visual pattern for resize testing
        draw = ImageDraw.Draw(image)
        draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], 
                      outline=(255, 255, 255), width=2)
        
        # Save to bytes buffer
        import io
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=85 if format == 'JPEG' else None)
        return buffer.getvalue()
    
    @staticmethod
    def create_image_with_metadata(width: int, height: int, exif_data: Dict[str, Any]) -> bytes:
        """Create test image with embedded EXIF metadata."""
        from PIL.ExifTags import TAGS
        import piexif
        
        image = Image.new('RGB', (width, height), (100, 150, 200))
        
        # Convert metadata dict to EXIF format
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
        
        if 'orientation' in exif_data:
            exif_dict["0th"][piexif.ImageIFD.Orientation] = exif_data['orientation']
        if 'camera_make' in exif_data:
            exif_dict["0th"][piexif.ImageIFD.Make] = exif_data['camera_make']
        if 'gps_latitude' in exif_data and 'gps_longitude' in exif_data:
            exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = exif_data['gps_latitude'] 
            exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = exif_data['gps_longitude']
            
        exif_bytes = piexif.dump(exif_dict)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', exif=exif_bytes)
        return buffer.getvalue()

class MockWebhookServer:
    """Simple HTTP server for testing webhook delivery."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.received_webhooks = []
        self.response_status = 200
        self.response_delay = 0
        
    def start_server(self):
        """Start mock webhook server in background thread."""
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                # Record received webhook
                self.server.webhook_server.received_webhooks.append({
                    'path': self.path,
                    'headers': dict(self.headers),
                    'body': body.decode('utf-8'),
                    'timestamp': time.time()
                })
                
                # Simulate response delay
                if self.server.webhook_server.response_delay > 0:
                    time.sleep(self.server.webhook_server.response_delay)
                
                # Send configured response
                self.send_response(self.server.webhook_server.response_status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "received"}')
        
        server = HTTPServer(('localhost', self.port), WebhookHandler)
        server.webhook_server = self
        
        def run_server():
            server.serve_forever()
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        return server
        
    def get_received_webhooks(self):
        """Get list of received webhook requests."""
        return self.received_webhooks.copy()
        
    def clear_webhooks(self):
        """Clear received webhook history."""
        self.received_webhooks.clear()
```

**Redis Testing Utilities**

```python
"""Redis testing utilities for queue integration tests."""

import redis
import time
import json
from typing import Optional, Dict, Any
from contextlib import contextmanager

class RedisTestManager:
    """Manages Redis connections and cleanup for testing."""
    
    def __init__(self, host='localhost', port=6379, db=15):  # Use high DB number for tests
        self.redis_config = {'host': host, 'port': port, 'db': db}
        self.client = None
        
    @contextmanager
    def redis_connection(self):
        """Context manager providing clean Redis connection."""
        self.client = redis.Redis(**self.redis_config, decode_responses=True)
        try:
            # Clear test database
            self.client.flushdb()
            yield self.client
        finally:
            # Cleanup
            if self.client:
                self.client.flushdb()
                self.client.close()
                
    def wait_for_job_completion(self, job_id: str, timeout: int = 30) -> Optional[str]:
        """Wait for job to reach terminal state."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job_data = self.client.hgetall(f"job:{job_id}")
            if job_data and job_data.get('status') in ['completed', 'failed']:
                return job_data.get('status')
            time.sleep(0.5)
            
        return None  # Timeout
        
    def simulate_worker_processing(self, job_message: Dict[str, Any]) -> bool:
        """Simulate worker processing a job for testing."""
        job_id = job_message['job_id']
        
        # Mark job as processing
        self.client.hset(f"job:{job_id}", "status", "processing")
        self.client.hset(f"job:{job_id}", "started_at", time.time())
        
        # Simulate progress updates
        for progress in [25, 50, 75, 100]:
            time.sleep(0.1)  # Brief processing simulation
            self.client.hset(f"job:{job_id}", "progress_percentage", progress)
            
        # Mark as completed
        self.client.hset(f"job:{job_id}", "status", "completed") 
        self.client.hset(f"job:{job_id}", "completed_at", time.time())
        
        return True
```

**D. Core Logic Skeleton Code**

**Image Processing Test Structure**

```python
"""Image processing unit test skeletons."""

import pytest
from PIL import Image
from media_processing.image_processor import ImageProcessor, ImageProcessingConfig

class TestImageProcessing:
    """Unit tests for image processing operations."""
    
    @pytest.fixture
    def image_processor(self):
        """Create ImageProcessor instance for testing."""
        return ImageProcessor()
        
    @pytest.fixture  
    def test_image_data(self):
        """Generate test image data for consistent testing."""
        # TODO: Use TestMediaGenerator to create 100x100 RGB test image
        # TODO: Return image data as bytes for loading tests
        pass
        
    def test_resize_preserve_aspect_ratio(self, image_processor, test_image_data):
        """Test image resizing with aspect ratio preservation."""
        # TODO: Load test image from bytes
        # TODO: Create ImageProcessingConfig with target dimensions 200x100
        # TODO: Call resize_image() with PRESERVE_ASPECT resize mode
        # TODO: Verify output dimensions maintain original aspect ratio
        # TODO: Check that image quality is preserved after resize
        pass
        
    def test_format_conversion_quality_settings(self, image_processor, test_image_data):
        """Test format conversion with different quality settings."""
        # TODO: Load test image as PIL Image object
        # TODO: Convert to JPEG with quality=95, measure file size
        # TODO: Convert to JPEG with quality=50, measure file size  
        # TODO: Assert that lower quality produces smaller file size
        # TODO: Verify both outputs are valid JPEG images
        pass
        
    def test_exif_orientation_handling(self, image_processor):
        """Test proper EXIF orientation rotation.""" 
        # TODO: Create test image with EXIF orientation=6 (90° rotation)
        # TODO: Process image with metadata handling enabled
        # TODO: Verify image is rotated correctly based on EXIF data
        # TODO: Check that output image orientation is normalized
        # Hint: Use TestMediaGenerator.create_image_with_metadata()
        pass
        
    def test_metadata_privacy_stripping(self, image_processor):
        """Test removal of privacy-sensitive metadata."""
        # TODO: Create image with GPS coordinates and camera data
        # TODO: Process with metadata_mode='strip_privacy'
        # TODO: Verify GPS data is removed from output
        # TODO: Check that basic image data (dimensions, format) is preserved
        pass
        
    def test_memory_limit_validation(self, image_processor):
        """Test memory requirement calculation for large images."""
        # TODO: Calculate memory needed for 8000x8000 RGB image
        # TODO: Call check_image_memory_requirements()  
        # TODO: Verify calculation matches expected bytes (width * height * channels)
        # TODO: Test with different color modes (RGB, RGBA, CMYK)
        pass
        
    def test_corrupted_image_error_handling(self, image_processor):
        """Test graceful handling of corrupted image files."""
        # TODO: Create corrupted image data (truncated JPEG header)
        # TODO: Attempt to load corrupted image
        # TODO: Verify specific exception type is raised
        # TODO: Check error message contains useful diagnostic information
        pass
```

**Integration Test Structure**

```python
"""Integration test skeletons for complete workflows."""

import pytest
import asyncio
from media_processing.api import submit_job, ProcessingJob
from media_processing.queue import JobStatus

class TestEndToEndWorkflows:
    """Integration tests for complete media processing workflows."""
    
    @pytest.fixture
    def redis_test_manager(self):
        """Set up Redis test environment."""
        manager = RedisTestManager()
        return manager
        
    @pytest.fixture
    def mock_webhook_server(self):
        """Start mock webhook server for notification testing."""
        # TODO: Initialize MockWebhookServer on port 8080
        # TODO: Start server in background thread
        # TODO: Return server instance for test assertions
        pass
        
    @pytest.mark.integration
    def test_image_processing_complete_workflow(self, redis_test_manager, mock_webhook_server):
        """Test complete image processing from submission to completion."""
        with redis_test_manager.redis_connection() as redis:
            # TODO: Create test image file in temporary directory
            # TODO: Submit job with multiple output specifications  
            # TODO: Start worker process to handle job
            # TODO: Wait for job completion using redis_test_manager.wait_for_job_completion()
            # TODO: Verify all output files exist with correct formats
            # TODO: Check webhook notification was delivered
            # TODO: Validate job status is COMPLETED
            pass
            
    @pytest.mark.integration  
    def test_video_transcoding_with_progress(self, redis_test_manager):
        """Test video transcoding with progress tracking."""
        with redis_test_manager.redis_connection() as redis:
            # TODO: Create short test video using FFmpeg
            # TODO: Submit transcoding job with ABR variants
            # TODO: Monitor progress updates during processing
            # TODO: Verify progress moves through expected stages
            # TODO: Check final output includes HLS manifest and segments
            # TODO: Validate video segments are playable
            pass
            
    @pytest.mark.integration
    def test_concurrent_job_processing(self, redis_test_manager):
        """Test multiple jobs processed concurrently."""
        with redis_test_manager.redis_connection() as redis:
            # TODO: Submit 10 image processing jobs simultaneously
            # TODO: Start 3 worker processes
            # TODO: Monitor job completion across all workers
            # TODO: Verify all jobs complete successfully
            # TODO: Check jobs are distributed fairly across workers
            # TODO: Validate no duplicate processing occurs
            pass
            
    @pytest.mark.integration
    def test_webhook_delivery_retry_logic(self, mock_webhook_server):
        """Test webhook notification retry on delivery failures."""
        # TODO: Configure webhook server to return HTTP 503 initially
        # TODO: Submit job with webhook notification
        # TODO: Wait for first webhook delivery attempt (should fail)
        # TODO: Configure server to return HTTP 200
        # TODO: Verify webhook is eventually delivered via retry
        # TODO: Check exponential backoff timing between retries
        pass
```

**E. Language-Specific Hints**

- **PIL/Pillow**: Use `Image.save()` with `optimize=True` for smaller file sizes
- **pytest fixtures**: Use `@pytest.fixture(scope="session")` for expensive setup like Redis connections
- **Temporary files**: Use `tempfile.TemporaryDirectory()` context manager for automatic cleanup
- **Redis testing**: Use high database numbers (10-15) to avoid conflicts with development data
- **FFmpeg subprocess**: Use `subprocess.Popen()` with `PIPE` for stdout/stderr capture
- **Threading**: Use `threading.Event()` for coordinating between test threads and background workers
- **Mock HTTP**: `responses` library provides easier HTTP mocking than `unittest.mock`
- **Binary data**: Use `bytes()` and `io.BytesIO()` for in-memory binary file simulation

**F. Milestone Checkpoints**

**Milestone 1 Checkpoint**
```bash
# Run these commands to validate Milestone 1 completion
cd media-processing-pipeline
python -m pytest tests/unit/test_image_processing.py::TestImageProcessing -v
python scripts/validate_milestone_1.py

# Expected: All image processing tests pass, validation script reports success
# Signs of problems: PIL import errors, memory issues with large images, EXIF orientation bugs
```

**Milestone 2 Checkpoint** 
```bash
# Run these commands to validate Milestone 2 completion
python -m pytest tests/unit/test_video_transcoding.py::TestVideoTranscoding -v  
python scripts/validate_milestone_2.py --input fixtures/test_video.mp4

# Expected: Video transcoding tests pass, ABR variants generated correctly
# Signs of problems: FFmpeg not found, progress parsing failures, codec errors
```

**Milestone 3 Checkpoint**
```bash
# Run these commands to validate Milestone 3 completion
docker-compose -f docker-compose.test.yml up -d
python -m pytest tests/integration/ -v
python scripts/validate_milestone_3.py --concurrent-jobs 10

# Expected: All integration tests pass, concurrent processing works correctly  
# Signs of problems: Redis connection failures, webhook delivery issues, race conditions in progress tracking

```


## Debugging Guide

> **Milestone(s):** All milestones (1-3) as debugging skills are essential for troubleshooting image processing, video transcoding, and job queue issues across the entire system

### Mental Model: Hospital Emergency Room

Think of debugging a media processing pipeline as running a hospital emergency room. When a patient (processing job) arrives with symptoms (failed processing, stuck progress, resource exhaustion), you need to quickly diagnose the root cause from observable symptoms, apply the right diagnostic tools, and implement targeted treatment while preventing the problem from affecting other patients.

Just as emergency room doctors follow systematic triage protocols—checking vital signs, running targeted tests, consulting specialists—effective media processing debugging requires structured approaches to symptom identification, diagnostic tool usage, and performance analysis. The goal is rapid diagnosis followed by precise intervention that resolves the immediate issue while preventing future occurrences.

The emergency room analogy extends to resource management: you must balance immediate patient needs with overall system capacity, handle multiple concurrent cases without cross-contamination, and maintain detailed records for pattern recognition and quality improvement.

### Common Symptoms and Diagnosis

Effective debugging starts with systematic symptom-to-cause mapping that enables rapid problem identification and resolution. The media processing pipeline exhibits distinct failure patterns that correspond to specific underlying issues across image processing, video transcoding, and job queue management.

**Job Queue and Worker Coordination Issues**

The job queue represents the central nervous system of the media processing pipeline. When queue operations fail, the symptoms manifest in predictable patterns that reveal the underlying coordination problems.

| Symptom | Observable Behavior | Likely Root Cause | Diagnostic Steps |
|---------|-------------------|-------------------|-----------------|
| Jobs stuck in PENDING forever | `ProcessingJob.status` remains PENDING, no worker picks up jobs | Worker processes crashed or not starting | Check worker process status, verify Redis connectivity, examine worker logs for startup errors |
| Progress updates stop mid-processing | Job shows 45% complete but no further updates for >5 minutes | Worker crashed during processing without cleanup | Search logs for worker PID, check for OOM kills, verify temp file cleanup |
| Duplicate job processing | Same input file processed multiple times concurrently | Race condition in job deduplication logic | Check Redis atomic operations, verify `job_id` generation uniqueness |
| High-priority jobs processed after low-priority | Urgent jobs wait behind normal priority jobs | Priority queue implementation bug | Examine Redis ZADD/ZPOP operations, verify priority score calculation |
| Memory exhaustion crashes | Worker processes killed with exit code 137 (SIGKILL) | Large media files exceeding worker memory limits | Monitor memory usage patterns, check file size validation |

**Image Processing Failures**

Image processing failures typically stem from format incompatibilities, memory constraints, or metadata handling issues. The failure patterns reveal specific problems in the image manipulation pipeline.

| Symptom | Observable Behavior | Likely Root Cause | Diagnostic Steps |
|---------|-------------------|-------------------|-----------------|
| Images rotated incorrectly | Output images appear sideways or upside-down | EXIF orientation tag not processed before resize | Check input EXIF data, verify `ImageMetadata.orientation` handling |
| Corrupted output files | Generated images cannot be opened or display artifacts | Memory corruption during processing or invalid format conversion | Test with smaller images, check memory allocation patterns |
| WebP conversion fails silently | WebP format specified but JPEG output produced | Pillow WebP support not installed or misconfigured | Test WebP encoding capability, verify Pillow installation |
| Thumbnail generation produces black images | Thumbnails created but contain no visible content | Crop coordinates exceed image boundaries | Log crop calculations, verify smart cropping algorithm |
| EXIF privacy stripping incomplete | Output images still contain GPS coordinates | Metadata handling configuration error | Audit metadata extraction, test privacy stripping modes |

**Video Transcoding and FFmpeg Issues**

Video transcoding problems often involve FFmpeg integration issues, codec compatibility, or resource management failures during long-running processing operations.

| Symptom | Observable Behavior | Likely Root Cause | Diagnostic Steps |
|---------|-------------------|-------------------|-----------------|
| FFmpeg process hangs indefinitely | Video transcoding never completes, no progress updates | FFmpeg waiting for input or misconfigured parameters | Check FFmpeg command construction, verify input file accessibility |
| Transcoding fails with "Codec not found" | FFmpeg exits with codec-related error messages | Target codec not supported in FFmpeg build | Test codec availability, check FFmpeg compilation flags |
| HLS segments not playable | Individual .ts files generated but playlist broken | Segment alignment or manifest generation issues | Validate HLS manifest syntax, test segment playback individually |
| Audio sync issues in output | Video and audio streams misaligned in transcoded output | Stream mapping or timing configuration problems | Examine input metadata, verify audio/video synchronization settings |
| Progress calculation wildly inaccurate | Progress jumps from 10% to 90% instantly | FFmpeg duration parsing failed or progress regex incorrect | Test duration extraction, validate progress parsing logic |

**Webhook and Notification Problems**

Webhook delivery failures create visibility gaps that make it difficult to track job completion and handle errors appropriately in client applications.

| Symptom | Observable Behavior | Likely Root Cause | Diagnostic Steps |
|---------|-------------------|-------------------|-----------------|
| Webhooks never delivered | Client never receives job completion notifications | Network connectivity or webhook URL configuration issues | Test webhook URL accessibility, verify HMAC signature generation |
| Webhook delivery storms | Client receives hundreds of duplicate notifications | Retry logic not tracking successful deliveries | Check webhook delivery tracking, audit retry backoff implementation |
| Authentication failures | Client rejects webhooks with signature errors | HMAC signature mismatch or clock skew | Verify shared secret configuration, test signature generation |
| Progress webhooks out of order | Client receives 75% progress after 100% completion | Race conditions in progress update delivery | Check sequence number implementation, verify atomic progress updates |
| Webhook timeouts | Webhook delivery fails with timeout errors | Client webhook endpoint slow or unresponsive | Monitor webhook response times, implement client-side timeout handling |

> **Decision: Structured Diagnostic Approach**
> - **Context**: Media processing involves complex interactions between multiple components, making ad-hoc debugging ineffective and time-consuming
> - **Options Considered**: 
>   1. Reactive debugging responding to individual symptoms
>   2. Comprehensive logging with manual correlation
>   3. Structured symptom-to-cause mapping with standardized diagnostic procedures
> - **Decision**: Implement structured diagnostic approach with symptom classification
> - **Rationale**: Systematic diagnosis reduces mean-time-to-resolution and enables junior developers to handle complex debugging scenarios effectively
> - **Consequences**: Requires upfront investment in diagnostic procedures but dramatically improves debugging efficiency and reduces escalation requirements

![Error Recovery Decision Tree](./diagrams/error-recovery-paths.svg)

**Redis and Storage Layer Issues**

The storage layer provides persistence for job state, progress tracking, and metadata. Storage failures cascade through the entire system and require careful diagnosis to prevent data loss.

| Symptom | Observable Behavior | Likely Root Cause | Diagnostic Steps |
|---------|-------------------|-------------------|-----------------|
| Job state inconsistencies | Job shows COMPLETED in Redis but PROCESSING in PostgreSQL | Hybrid storage synchronization failure | Compare Redis and PostgreSQL state, check transaction boundaries |
| Progress updates rejected | Progress tracking returns false from update operations | Sequence number conflicts or atomic operation failures | Check Redis transaction logs, verify sequence number generation |
| File cleanup failures | Temporary files accumulate without deletion | Error handling not cleaning up resources properly | Audit file lifecycle management, check error handler cleanup paths |
| Storage quota exceeded | Jobs fail with disk space errors | Temporary file cleanup not working or storage monitoring absent | Check disk usage patterns, verify cleanup job execution |
| Redis connection pool exhausted | New jobs cannot be queued with connection timeout errors | Connection leaks or insufficient pool sizing | Monitor Redis connection usage, check connection cleanup |

### Debugging Tools and Techniques

Effective debugging requires a comprehensive toolkit that provides visibility into system behavior, enables rapid problem isolation, and supports root cause analysis across distributed components.

### Mental Model: Detective Investigation Kit

Think of debugging tools as a detective's investigation kit. Just as detectives use fingerprints for identity verification, surveillance cameras for timeline reconstruction, and forensic analysis for evidence correlation, media processing debugging requires specialized tools for different types of evidence collection and analysis.

Each tool serves a specific investigative purpose: logs provide timeline evidence, metrics reveal behavioral patterns, tracing shows interaction sequences, and profiling exposes performance bottlenecks. The key is knowing which tool to use for each type of investigation and how to correlate evidence across multiple sources.

**Structured Logging Implementation**

Structured logging forms the foundation of effective debugging by providing consistent, searchable, and correlatable log data across all system components. The logging strategy must balance information richness with performance impact.

| Component | Log Level | Required Fields | Sample Message Structure |
|-----------|-----------|----------------|------------------------|
| API Gateway | INFO, WARN, ERROR | request_id, job_id, user_id, endpoint, duration_ms | `{"timestamp": "2024-01-15T10:30:45Z", "level": "INFO", "component": "api", "request_id": "req_123", "job_id": "job_456", "message": "Job submitted successfully"}` |
| Job Queue | DEBUG, INFO, ERROR | job_id, worker_id, queue_depth, processing_time | `{"timestamp": "2024-01-15T10:31:00Z", "level": "DEBUG", "component": "queue", "job_id": "job_456", "worker_id": "worker_01", "queue_depth": 23, "message": "Job dequeued for processing"}` |
| Image Processor | INFO, WARN, ERROR | job_id, input_file, output_specs, memory_usage_mb | `{"timestamp": "2024-01-15T10:31:15Z", "level": "INFO", "component": "image", "job_id": "job_456", "input_file": "photo.jpg", "memory_usage_mb": 450, "message": "Image resize completed"}` |
| Video Processor | INFO, WARN, ERROR | job_id, ffmpeg_command, progress_percentage, estimated_remaining | `{"timestamp": "2024-01-15T10:32:30Z", "level": "INFO", "component": "video", "job_id": "job_456", "progress_percentage": 45.2, "estimated_remaining": "2m30s", "message": "Transcoding progress update"}` |
| Progress Tracker | DEBUG, INFO, ERROR | job_id, stage, webhook_url, delivery_attempt | `{"timestamp": "2024-01-15T10:35:45Z", "level": "INFO", "component": "progress", "job_id": "job_456", "stage": "FORMAT_CONVERSION", "webhook_url": "https://client.example.com/webhooks", "delivery_attempt": 1, "message": "Webhook delivered successfully"}` |

The `job_logging_context` function provides correlation across all log messages for a specific job, enabling end-to-end request tracing through the entire processing pipeline. This correlation proves essential when debugging complex failure scenarios that span multiple components.

**Monitoring and Metrics Collection**

Real-time metrics provide early warning systems for performance degradation and resource constraints before they escalate to critical failures.

| Metric Category | Key Metrics | Collection Frequency | Alert Thresholds |
|----------------|-------------|---------------------|------------------|
| Queue Health | Queue depth, processing rate, worker utilization | Every 30 seconds | Queue depth > 1000, processing rate < 10 jobs/minute |
| Resource Usage | Memory consumption, CPU utilization, disk usage | Every 10 seconds | Memory > 80%, CPU > 90%, disk > 85% |
| Processing Performance | Job completion time, error rates, format-specific metrics | Per job completion | Error rate > 5%, completion time > 2x average |
| Webhook Delivery | Delivery success rate, retry frequency, response times | Per webhook attempt | Success rate < 95%, response time > 5 seconds |
| Storage Operations | Redis operations/second, PostgreSQL query time, file I/O rates | Every 15 seconds | Query time > 100ms, Redis ops/sec > 10000 |

> **Decision: Metrics vs Logs Balance**
> - **Context**: Need visibility into system behavior without overwhelming storage or affecting performance
> - **Options Considered**:
>   1. Comprehensive logging of all operations
>   2. Metrics-only approach with minimal logging
>   3. Balanced approach with structured logs and targeted metrics
> - **Decision**: Balanced approach with job-level tracing and system-level metrics
> - **Rationale**: Logs provide detailed debugging context while metrics enable proactive monitoring and alerting
> - **Consequences**: Requires careful log level management and metrics aggregation but provides comprehensive observability

**Diagnostic Command-Line Tools**

Command-line diagnostic tools enable rapid system inspection during incident response and provide detailed component-level debugging capabilities.

| Tool Purpose | Command Examples | Expected Output | Usage Scenarios |
|--------------|------------------|------------------|-----------------|
| Queue Status | `redis-cli LLEN media:queue:high` | `23` (queue depth) | Check job backlog during performance issues |
| Worker Health | `ps aux \| grep media_worker` | Process list with memory usage | Verify worker processes running correctly |
| Job Inspection | `redis-cli HGETALL job:job_456` | Job status and metadata | Debug specific job failure |
| Progress Check | `redis-cli GET progress:job_456` | JSON progress object | Verify progress tracking accuracy |
| File Validation | `ffprobe -v quiet -print_format json input.mp4` | Video metadata JSON | Validate input file before processing |
| Storage Usage | `du -sh /tmp/media_processing/*` | Directory size breakdown | Check temporary file cleanup |

**Log Analysis and Correlation Techniques**

Effective log analysis requires tools and techniques that can correlate events across multiple components and identify patterns in large log volumes.

The correlation strategy relies on consistent `job_id` and `request_id` fields that enable end-to-end tracing through the processing pipeline. Advanced log analysis uses tools like `grep`, `jq`, and `awk` for command-line investigation combined with log aggregation platforms for pattern recognition.

Sample correlation commands demonstrate how to trace a specific job through the entire system:

```bash
# Extract all log entries for specific job
grep "job_456" /var/log/media_processing/*.log

# Analyze webhook delivery patterns
jq 'select(.component == "progress" and .delivery_attempt > 1)' /var/log/media_processing/webhooks.log

# Find memory-related errors
grep -E "(OOM|memory|killed)" /var/log/media_processing/*.log | head -20
```

**Integration with External Monitoring**

External monitoring tools provide centralized visibility across distributed deployments and enable automated alerting based on system behavior patterns.

| Integration Type | Tool Examples | Configuration Requirements | Benefits |
|------------------|---------------|---------------------------|----------|
| Log Aggregation | ELK Stack, Splunk, CloudWatch | Structured JSON output, log shipping | Centralized search, pattern analysis |
| Metrics Collection | Prometheus + Grafana, DataDog | StatsD or HTTP endpoints | Real-time dashboards, historical analysis |
| Alerting Systems | PagerDuty, Slack notifications | Webhook integration | Automated incident response |
| APM Tools | Jaeger, Zipkin, New Relic | Distributed tracing headers | End-to-end request visibility |
| Health Checks | Consul, etcd, custom endpoints | HTTP health check endpoints | Service discovery and load balancing |

⚠️ **Pitfall: Log Volume Overwhelming Storage**
Many implementations generate excessive log volume that fills disk space and makes analysis difficult. The solution requires log level configuration based on environment (DEBUG in development, INFO in production) and log rotation policies that balance retention with storage constraints. Use sampling for high-volume DEBUG messages and ensure log rotation prevents disk exhaustion.

### Performance and Resource Debugging

Performance debugging in media processing systems requires specialized techniques that account for the unique characteristics of multimedia workloads, including variable processing times, large memory requirements, and I/O-intensive operations.

### Mental Model: Formula One Pit Crew

Think of performance debugging as running a Formula One pit crew during a race. The pit crew must quickly diagnose performance issues (slow lap times, tire wear, fuel consumption) while the race continues, identify bottlenecks that prevent optimal performance, and implement targeted optimizations without disrupting ongoing operations.

Just as pit crews use telemetry data to understand car performance, real-time metrics reveal system bottlenecks. Like mechanics who know that tire temperature affects grip and aerodynamics impact fuel efficiency, media processing performance debugging requires understanding how memory usage affects processing speed, how file size impacts I/O patterns, and how codec selection influences CPU utilization.

**Memory Usage Analysis and Optimization**

Memory management presents the primary performance challenge in media processing due to the large working sets required for high-resolution images and video frames. Memory debugging must identify both peak usage patterns and memory leaks that accumulate over time.

| Memory Issue Type | Symptoms | Diagnostic Approach | Optimization Strategies |
|-------------------|----------|-------------------|------------------------|
| Peak Memory Spikes | OOM kills during large file processing | Monitor memory usage per job, profile allocation patterns | Implement streaming processing, reduce buffer sizes |
| Memory Leaks | Gradual memory growth over time | Track memory usage trends, identify allocation without deallocation | Audit resource cleanup, use memory profiling tools |
| Fragmentation | Available memory but allocation failures | Monitor memory fragmentation metrics | Implement object pooling, use consistent buffer sizes |
| Buffer Overflow | Crashes or corruption during processing | Memory sanitizers, bounds checking | Validate input sizes, implement memory limits |
| Swap Thrashing | Extremely slow processing with high disk I/O | Monitor swap usage and page faults | Reduce memory footprint, increase available RAM |

Memory profiling for image processing requires understanding the relationship between image dimensions and memory requirements. A 4K image (3840x2160) in RGBA format requires approximately 33MB of uncompressed memory, while video processing may require multiple frame buffers simultaneously.

**Processing Time Optimization Strategies**

Processing time optimization requires identifying bottlenecks in the media processing pipeline and implementing targeted improvements that maintain quality while reducing latency.

| Processing Stage | Common Bottlenecks | Profiling Techniques | Optimization Options |
|------------------|-------------------|---------------------|---------------------|
| Image Loading | File I/O and format parsing | Time file read operations, profile decoder performance | Implement format-specific optimizations, use memory-mapped files |
| Resize Operations | Interpolation algorithm complexity | Compare algorithm performance, measure CPU usage | Choose optimal interpolation method per use case |
| Format Conversion | Encoding complexity and quality settings | Profile encoder performance, measure compression ratios | Optimize quality settings, use hardware acceleration |
| Video Transcoding | CPU-intensive encoding operations | Monitor CPU utilization, measure encoding speed | Implement hardware acceleration, optimize FFmpeg parameters |
| I/O Operations | Disk bandwidth and latency | Monitor disk usage patterns, measure throughput | Use SSD storage, implement async I/O |

> **Decision: Processing Time vs Quality Trade-offs**
> - **Context**: Media processing involves inherent trade-offs between processing speed, output quality, and resource consumption
> - **Options Considered**:
>   1. Optimize for maximum quality regardless of processing time
>   2. Optimize for minimum processing time with acceptable quality loss
>   3. Implement configurable quality/speed profiles based on use case
> - **Decision**: Configurable quality profiles with performance-oriented defaults
> - **Rationale**: Different use cases have different requirements; thumbnails need speed while archival processing needs quality
> - **Consequences**: Requires careful profile configuration and testing but provides optimal performance for each use case

**Worker Bottleneck Identification**

Worker bottleneck analysis identifies coordination issues that prevent optimal resource utilization and job throughput across the distributed processing system.

| Bottleneck Type | Observable Symptoms | Root Cause Analysis | Resolution Approaches |
|-----------------|-------------------|---------------------|----------------------|
| Queue Starvation | Workers idle while jobs remain queued | Examine job distribution logic, check priority queue implementation | Improve job distribution algorithm, rebalance worker assignments |
| Resource Contention | Multiple workers competing for shared resources | Monitor file system locks, database connections, memory usage | Implement resource pooling, use worker-specific resources |
| Coordination Overhead | High messaging overhead relative to processing time | Measure message queue latency, profile worker communication | Optimize message formats, reduce coordination frequency |
| Uneven Load Distribution | Some workers overloaded while others idle | Analyze job assignment patterns, monitor worker utilization | Implement load-aware scheduling, use worker capability matching |
| Cascading Failures | Worker failures causing additional workers to fail | Trace failure propagation, examine error handling | Implement circuit breakers, improve error isolation |

Worker performance profiling requires measuring both individual worker efficiency and collective system throughput. Key metrics include jobs processed per hour per worker, average memory usage per job type, and worker restart frequency.

**Storage and I/O Performance Analysis**

Storage performance directly impacts media processing throughput due to the large file sizes involved in video transcoding and high-resolution image processing.

| Storage Metric | Measurement Technique | Performance Targets | Optimization Strategies |
|---------------|----------------------|-------------------|-------------------------|
| Read Throughput | Monitor bytes/second during file loading | >500 MB/s for video processing | Use SSD storage, implement read-ahead caching |
| Write Throughput | Measure output file creation speed | >200 MB/s for encoded output | Optimize write buffer sizes, use async writes |
| IOPS (Input/Output Operations Per Second) | Count file operations per second | >1000 IOPS for thumbnail generation | Use NVMe storage, batch small file operations |
| Latency | Measure file open/close times | <10ms for file access operations | Implement file handle pooling, reduce metadata operations |
| Temporary File Management | Track temp file lifecycle | Zero leaked files after job completion | Implement proper cleanup, use defer/finally patterns |

Database performance for job queue operations requires optimization for high-frequency read/write patterns with strong consistency requirements.

| Database Operation | Performance Target | Optimization Approach |
|-------------------|-------------------|----------------------|
| Job Queue Operations | <5ms per operation | Use Redis for queue state, optimize data structures |
| Progress Updates | <2ms per update | Implement atomic operations, use appropriate data types |
| Metadata Queries | <10ms per query | Index frequently queried fields, use read replicas |
| Job History | <50ms per complex query | Implement data archiving, optimize query patterns |

**Resource Monitoring and Alerting**

Comprehensive resource monitoring enables proactive performance management and prevents resource exhaustion before it impacts job processing.

| Resource Type | Monitoring Metrics | Alert Thresholds | Automated Responses |
|---------------|-------------------|------------------|-------------------|
| CPU Utilization | Per-core usage, load average, process CPU time | >80% sustained for 5+ minutes | Scale worker pool, throttle job intake |
| Memory Usage | RSS, virtual memory, swap usage, memory growth rate | >85% physical memory, any swap usage | Reduce concurrent jobs, restart workers |
| Disk Space | Used space, free space, growth rate, inode usage | >90% used space, <1GB free | Clean temporary files, archive old jobs |
| Network Bandwidth | Bytes sent/received, connection count, latency | >80% bandwidth utilization | Throttle uploads, implement traffic shaping |
| File Handles | Open file count, handle leaks | >80% of system limit | Audit file handle usage, implement handle pooling |

⚠️ **Pitfall: Premature Optimization**
Many developers optimize components that aren't actual bottlenecks, wasting effort and potentially reducing code maintainability. The solution requires measurement-driven optimization: profile the system under realistic load, identify the true bottlenecks using data, and optimize only the components that measurably impact performance. Use profiling tools consistently and maintain performance benchmarks to validate optimization effectiveness.

⚠️ **Pitfall: Resource Monitoring Overhead**
Excessive monitoring can itself become a performance bottleneck, consuming CPU and memory resources needed for media processing. The solution requires careful monitoring configuration that balances visibility with overhead. Use sampling for high-frequency metrics, implement monitoring on/off switches for debugging, and ensure monitoring tools don't compete with processing workloads for resources.

> The critical insight for performance debugging is understanding that media processing workloads have fundamentally different characteristics than typical web applications. Video transcoding may legitimately consume 100% CPU for extended periods, image processing may require gigabytes of memory for single operations, and file I/O patterns involve large sequential reads rather than small random accesses. Performance optimization must account for these unique requirements rather than applying generic optimization techniques.

### Implementation Guidance

This subsection provides practical tools and code examples for implementing comprehensive debugging capabilities across the media processing pipeline.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Logging Framework | Python `logging` with JSON formatter | Structured logging with `structlog` + ELK Stack |
| Metrics Collection | Custom StatsD client | Prometheus client with Grafana dashboards |
| Performance Profiling | `cProfile` and `memory_profiler` | `py-spy` for production profiling |
| Log Analysis | `grep` + `jq` command line tools | Centralized logging with Elasticsearch |
| Resource Monitoring | `psutil` for system metrics | Full observability with DataDog/New Relic |

**B. Recommended File/Module Structure:**

```
media_processing/
  debug/
    __init__.py                    ← debug utilities exports
    logging_config.py              ← structured logging setup
    metrics.py                     ← metrics collection and reporting
    profiling.py                   ← performance profiling utilities
    diagnostics.py                 ← diagnostic command implementations
    monitoring.py                  ← resource monitoring and alerting
    health_checks.py               ← system health validation
  tools/
    job_inspector.py               ← command-line job debugging tool
    queue_analyzer.py              ← queue performance analysis
    resource_monitor.py            ← real-time resource monitoring
    log_analyzer.py                ← log correlation and analysis
  tests/
    test_debugging.py              ← debugging utility tests
    fixtures/                      ← test data for debugging scenarios
```

**C. Infrastructure Starter Code:**

```python
# debug/logging_config.py - Complete structured logging setup
import logging
import json
import sys
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any, Optional
import threading

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with media processing context."""
    
    def __init__(self):
        super().__init__()
        self.hostname = self._get_hostname()
        self.process_id = os.getpid()
    
    def format(self, record: logging.LogRecord) -> str:
        """Convert log record to structured JSON format."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "hostname": self.hostname,
            "process_id": self.process_id,
            "thread_id": threading.current_thread().ident
        }
        
        # Add job context if available
        job_context = getattr(threading.current_thread(), 'job_context', None)
        if job_context:
            log_entry.update(job_context)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from log record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)
    
    def _get_hostname(self) -> str:
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

def setup_logging(debug: bool = False, log_file: Optional[str] = None):
    """Configure structured logging for the media processing system."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

@contextmanager
def job_logging_context(job_id: str, correlation_id: Optional[str] = None):
    """Context manager for adding job information to all log messages."""
    thread = threading.current_thread()
    old_context = getattr(thread, 'job_context', {})
    
    new_context = old_context.copy()
    new_context.update({
        'job_id': job_id,
        'correlation_id': correlation_id or job_id
    })
    
    thread.job_context = new_context
    try:
        yield
    finally:
        thread.job_context = old_context

# debug/metrics.py - Metrics collection infrastructure
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from threading import Lock
import json

@dataclass
class MetricPoint:
    """Individual metric measurement."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
class MetricsCollector:
    """Thread-safe metrics collection with configurable backends."""
    
    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._lock = Lock()
        self._enabled = True
    
    def counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        if not self._enabled:
            return
        
        with self._lock:
            key = self._metric_key(name, tags)
            self._counters[key] = self._counters.get(key, 0) + value
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric to specific value."""
        if not self._enabled:
            return
        
        with self._lock:
            key = self._metric_key(name, tags)
            self._gauges[key] = value
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Add value to histogram metric."""
        if not self._enabled:
            return
        
        with self._lock:
            key = self._metric_key(name, tags)
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
    
    def timing(self, name: str, duration_seconds: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric in seconds."""
        self.histogram(f"{name}.duration", duration_seconds, tags)
    
    def _metric_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                'counters': self._counters.copy(),
                'gauges': self._gauges.copy(),
                'histograms': {k: v.copy() for k, v in self._histograms.items()},
                'timestamp': time.time()
            }
    
    def reset(self):
        """Reset all metrics (for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

# Global metrics instance
metrics = MetricsCollector()

@contextmanager
def timed_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing operations."""
    start_time = time.time()
    success = False
    try:
        yield
        success = True
    finally:
        duration = time.time() - start_time
        result_tags = (tags or {}).copy()
        result_tags['success'] = str(success)
        metrics.timing(operation_name, duration, result_tags)

# debug/diagnostics.py - Diagnostic utilities
import psutil
import redis
from typing import Dict, Any, List, Optional
import subprocess
import json

class SystemDiagnostics:
    """System-level diagnostic utilities."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    def check_redis_health(self) -> Dict[str, Any]:
        """Comprehensive Redis health check."""
        try:
            info = self.redis_client.info()
            return {
                'connected': True,
                'memory_used_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'operations_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            return {'connected': False, 'error': str(e)}
    
    def check_worker_processes(self) -> List[Dict[str, Any]]:
        """Check status of worker processes."""
        workers = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'media_worker' in proc.info['name']:
                    workers.append({
                        'pid': proc.info['pid'],
                        'memory_mb': proc.info['memory_info'].rss / (1024 * 1024),
                        'cpu_percent': proc.info['cpu_percent'],
                        'status': proc.status()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return workers
    
    def check_disk_space(self, paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """Check disk space for critical paths."""
        disk_info = {}
        for path in paths:
            try:
                usage = psutil.disk_usage(path)
                disk_info[path] = {
                    'total_gb': usage.total / (1024**3),
                    'used_gb': usage.used / (1024**3),
                    'free_gb': usage.free / (1024**3),
                    'percent_used': (usage.used / usage.total) * 100
                }
            except Exception as e:
                disk_info[path] = {'error': str(e)}
        return disk_info
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        try:
            stats = {}
            for priority in ['urgent', 'high', 'normal', 'low']:
                queue_name = f"media:queue:{priority}"
                depth = self.redis_client.llen(queue_name)
                stats[f"{priority}_queue_depth"] = depth
            
            # Get processing job count
            processing_jobs = self.redis_client.hlen("jobs:processing")
            stats['jobs_processing'] = processing_jobs
            
            # Get failed job count
            failed_jobs = self.redis_client.llen("media:queue:failed")
            stats['jobs_failed'] = failed_jobs
            
            return stats
        except Exception as e:
            return {'error': str(e)}
```

**D. Core Logic Skeleton Code:**

```python
# debug/profiling.py - Performance profiling utilities
import cProfile
import pstats
import io
import memory_profiler
from typing import Any, Callable, Dict, Optional
import time

class PerformanceProfiler:
    """Comprehensive performance profiling for media processing operations."""
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self._profiles: Dict[str, Any] = {}
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """Profile function execution with CPU and memory analysis."""
        # TODO 1: Create cProfile.Profile instance for CPU profiling
        # TODO 2: Start memory monitoring if enabled using memory_profiler
        # TODO 3: Execute function with profiling enabled
        # TODO 4: Stop profiling and collect statistics
        # TODO 5: Store profile results with timestamp and function name
        # TODO 6: Return original function result
        # Hint: Use memory_profiler.LineProfiler for detailed memory analysis
        pass
    
    def get_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Get detailed memory usage during function execution."""
        # TODO 1: Record baseline memory usage before execution
        # TODO 2: Execute function while monitoring memory
        # TODO 3: Record peak memory usage during execution
        # TODO 4: Calculate memory delta and peak usage
        # TODO 5: Return comprehensive memory statistics
        # Hint: Use memory_profiler.memory_usage with interval parameter
        pass
    
    def analyze_bottlenecks(self, profile_name: str) -> Dict[str, Any]:
        """Analyze CPU profiling results to identify bottlenecks."""
        # TODO 1: Retrieve stored profile data for analysis
        # TODO 2: Extract top functions by cumulative time
        # TODO 3: Identify functions with high call counts
        # TODO 4: Calculate time per call for each function
        # TODO 5: Generate bottleneck report with recommendations
        pass

class ResourceMonitor:
    """Real-time resource monitoring for worker processes."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self._monitoring = False
        self._samples: List[Dict[str, Any]] = []
    
    def start_monitoring(self, job_id: str):
        """Start resource monitoring for specific job."""
        # TODO 1: Initialize monitoring state and sample storage
        # TODO 2: Start background thread for periodic sampling
        # TODO 3: Record initial resource baseline
        # TODO 4: Set monitoring flag to enable sample collection
        # Hint: Use threading.Thread with daemon=True for background monitoring
        pass
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return comprehensive resource report."""
        # TODO 1: Set monitoring flag to false to stop sampling
        # TODO 2: Wait for background thread to complete
        # TODO 3: Analyze collected samples for patterns
        # TODO 4: Calculate peak usage, averages, and trends
        # TODO 5: Generate summary report with recommendations
        pass
    
    def _collect_sample(self) -> Dict[str, Any]:
        """Collect single resource usage sample."""
        # TODO 1: Get current process CPU and memory usage using psutil
        # TODO 2: Check disk I/O statistics for current process
        # TODO 3: Record file handle count and network connections
        # TODO 4: Include timestamp for temporal analysis
        # TODO 5: Return complete sample data structure
        pass

# tools/job_inspector.py - Command-line job debugging tool
class JobInspector:
    """Command-line tool for inspecting job state and debugging issues."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
    
    def inspect_job(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job state and history."""
        # TODO 1: Retrieve job data from Redis using job_id
        # TODO 2: Get current progress information
        # TODO 3: Collect job processing history and state transitions
        # TODO 4: Check for associated temporary files
        # TODO 5: Analyze job for common failure patterns
        # TODO 6: Return complete job inspection report
        pass
    
    def diagnose_stuck_job(self, job_id: str) -> Dict[str, Any]:
        """Diagnose why job appears stuck in processing."""
        # TODO 1: Check job status and last progress update time
        # TODO 2: Verify worker process is still running
        # TODO 3: Check for temporary files and processing artifacts
        # TODO 4: Analyze memory usage of worker process
        # TODO 5: Determine if job should be marked failed or retried
        pass
    
    def analyze_queue_health(self) -> Dict[str, Any]:
        """Analyze overall queue health and performance."""
        # TODO 1: Get queue depths across all priority levels
        # TODO 2: Calculate processing rates and throughput
        # TODO 3: Identify jobs that have been pending too long
        # TODO 4: Check worker utilization and availability
        # TODO 5: Generate queue health report with recommendations
        pass
```

**E. Language-Specific Hints:**

- Use `logging.getLogger(__name__)` in each module to create component-specific loggers
- `psutil.Process().memory_info().rss` provides resident set size for memory usage
- `redis-py` client provides `info()` method for Redis diagnostics
- Use `contextlib.contextmanager` for resource monitoring decorators
- `json.dumps(obj, default=str)` handles datetime serialization in log messages
- `threading.local()` provides thread-safe storage for job context
- `subprocess.run()` with `capture_output=True` captures command output for analysis

**F. Milestone Checkpoint:**

After implementing the debugging infrastructure:

**What to run:**
```bash
python -m tools.job_inspector --job-id job_123 --diagnose
python -m debug.monitoring --check-health
python -c "from debug.logging_config import setup_logging; setup_logging(debug=True)"
```

**Expected behavior:**
- Structured JSON logs appear in console with job correlation IDs
- Job inspector returns comprehensive job state information
- Health checks report Redis connectivity and worker process status
- Metrics collection tracks job processing rates and resource usage

**Signs of success:**
- Log messages include consistent `job_id` fields for correlation
- Performance profiling identifies bottlenecks in media processing functions
- Resource monitoring detects memory spikes during large file processing
- Queue health checks provide actionable insights for performance optimization

**What to verify manually:**
- Submit test job and trace log messages through entire pipeline
- Verify webhook delivery failures trigger appropriate retry logic
- Confirm memory monitoring detects resource exhaustion scenarios
- Test diagnostic tools provide useful information for debugging failures


## Future Extensions and Scalability

> **Milestone(s):** All milestones (1-3) as this section provides architectural patterns and extension points that build upon the foundational image processing, video transcoding, and job queue capabilities

### Mental Model: Growing Entertainment Studio

Think of the media processing pipeline as a small independent film studio that started with basic equipment and a handful of employees. Initially, they could handle simple photo shoots and short video projects in a single office. As their reputation grows and demand increases, they face several expansion challenges: they need multiple production facilities (horizontal scaling), sophisticated equipment like motion capture and AI-powered editing tools (advanced processing features), and partnerships with distributors, streaming platforms, and equipment rental companies (external integrations). Each growth phase requires careful planning to maintain quality while dramatically increasing capacity and capabilities.

The key insight is that successful scaling isn't just about adding more resources—it's about evolving the entire operational model. A studio that simply adds more editing bays without upgrading their project management systems, client communication processes, and delivery pipelines will quickly become chaotic and inefficient. Similarly, our media processing pipeline must evolve its architecture, not just multiply its components.

## Horizontal Scaling Patterns

### Multi-Node Deployment Architecture

**Resource-aware scheduling** becomes critical when distributing processing jobs across multiple nodes. Unlike the single-node deployment where all worker processes share the same hardware resources, multi-node deployments must consider the heterogeneous capabilities of different machines. Some nodes might have GPU acceleration for video transcoding, while others excel at CPU-intensive image processing tasks.

The **job queue** evolution requires sophisticated routing logic that goes beyond simple priority-based distribution. Each processing job must be matched with nodes that have the appropriate hardware capabilities, available memory, and current load levels. This transforms the simple Redis-based queue into a distributed scheduling system with node registration, capability advertising, and intelligent job placement.

> **Decision: Distributed Queue Architecture**
> - **Context**: Single Redis instance becomes bottleneck and single point of failure as worker nodes scale beyond 10-15 machines
> - **Options Considered**: 
>   - Redis Cluster with consistent hashing
>   - RabbitMQ federation across data centers
>   - Apache Kafka with partitioned topics
> - **Decision**: Redis Cluster for job queue with RabbitMQ for inter-node coordination
> - **Rationale**: Redis Cluster provides horizontal scaling for job distribution while maintaining sub-millisecond job pop latency. RabbitMQ handles complex routing patterns for progress updates and administrative messages that don't require Redis-level performance.
> - **Consequences**: Introduces complexity of managing two message broker systems but enables independent scaling of job throughput vs coordination message handling

| Queue Component | Single Node | Multi-Node Cluster |
|-----------------|-------------|-------------------|
| Job Distribution | Simple Redis LPOP | Redis Cluster with hash slots |
| Node Discovery | Static configuration | Dynamic registration with heartbeats |
| Load Balancing | OS process scheduler | Custom weighted round-robin |
| Health Monitoring | Process-level checks | Network-aware health probes |
| Failure Recovery | Process restart | Node failover with job reassignment |

**Worker coordination** across nodes requires a fundamentally different approach than local process management. The system must track which nodes are healthy, what their current capacity utilization looks like, and how to redistribute work when nodes fail or new ones join the cluster. This involves implementing a **distributed consensus mechanism** for cluster membership and a **gossip protocol** for sharing load information.

The coordination layer maintains several critical data structures across the cluster:

| Data Structure | Purpose | Replication Strategy |
|----------------|---------|---------------------|
| Node Registry | Track active nodes and capabilities | Raft consensus with leader election |
| Job Assignments | Map active jobs to processing nodes | Replicated state machine |
| Load Metrics | CPU, memory, queue depth per node | Eventually consistent gossip |
| Health Status | Node availability and performance | Heartbeat with failure detection |

### Load Balancing Strategies

**Geographic distribution** adds another layer of complexity to load balancing. Media files are often large, and network transfer costs become significant when moving multi-gigabyte video files between data centers. The ideal architecture processes media files on nodes that are network-close to the storage location, but this creates uneven load distribution challenges.

Smart load balancing must consider multiple factors simultaneously:

1. **Network locality**: Prefer nodes in the same data center as the source media file
2. **Specialized hardware**: Route video transcoding jobs to GPU-enabled nodes
3. **Current load**: Avoid overloading nodes that are already processing intensive jobs
4. **Historical performance**: Learn which nodes complete similar jobs fastest
5. **Cost optimization**: Balance processing speed against infrastructure costs

The load balancer maintains a **capability matrix** that tracks each node's hardware specifications, current resource utilization, and recent performance metrics. When a new job arrives, the balancer scores potential nodes using a weighted algorithm that considers all these factors.

> **Decision: Adaptive Load Balancing Algorithm**
> - **Context**: Static load balancing wastes GPU resources and creates uneven processing times across different media types
> - **Options Considered**:
>   - Round-robin with manual node tagging
>   - Least-connections with capability filtering  
>   - Machine learning-based predictive scheduling
> - **Decision**: Weighted scoring algorithm with runtime capability learning
> - **Rationale**: Provides good performance without ML complexity. Scoring weights can be tuned based on observed job completion patterns and resource utilization metrics.
> - **Consequences**: Requires maintaining performance history and periodic rebalancing, but achieves 30-40% better resource utilization than simple strategies

### Distributed Storage Integration

**Object storage backends** like Amazon S3, Google Cloud Storage, or Azure Blob Storage become essential for multi-node deployments. Local file storage doesn't work when jobs can be processed on any node in the cluster. However, integrating distributed storage introduces new challenges around data transfer costs, processing locality, and temporary file management.

The storage abstraction layer must handle several scenarios seamlessly:

| Scenario | Challenge | Solution Pattern |
|----------|-----------|------------------|
| Large input files | Network transfer time dominates processing | Stream processing with progressive download |
| Multiple output formats | Parallel generation on different nodes | Distributed fan-out with result aggregation |
| Temporary processing files | Cleanup across node failures | Distributed lease management with TTL |
| Bandwidth optimization | Minimize redundant transfers | Content-addressed caching with deduplication |

**Caching strategies** become critical for performance when dealing with distributed storage. Frequently accessed source files should be cached locally on multiple nodes, while processed outputs might benefit from edge caching closer to end users. The cache invalidation logic must handle scenarios where source files are updated or processing parameters change.

A sophisticated caching layer implements multiple tiers:

1. **Local node cache**: Fast SSD storage for active job files
2. **Cluster shared cache**: High-speed network storage shared across nodes
3. **Edge cache**: CDN integration for processed output delivery
4. **Cold storage**: Long-term archival with slower retrieval times

### Auto-scaling and Resource Management

**Dynamic scaling policies** must react to both queue depth and resource utilization patterns. Unlike web services that scale primarily based on request volume, media processing workloads have highly variable resource requirements. A single 4K video transcoding job might consume more resources than 100 image resize operations.

The auto-scaling system monitors multiple metrics simultaneously:

| Metric Type | Scaling Signal | Response Action |
|-------------|----------------|-----------------|
| Queue Depth | Jobs waiting > 5 minutes | Add general-purpose nodes |
| GPU Utilization | Video queue backlog | Add GPU-enabled nodes |
| Memory Pressure | Failed jobs due to OOM | Add high-memory nodes |
| Network Bandwidth | Transfer times > SLA | Add nodes in new regions |
| Cost Optimization | Low utilization periods | Remove excess capacity |

**Resource quotas and limits** prevent individual jobs from consuming excessive resources and impacting other workloads. The quota system must be sophisticated enough to handle legitimate large media files while preventing abuse or runaway processing jobs.

## Advanced Processing Features

### AI-Powered Content Analysis

**Smart cropping algorithms** represent a significant evolution beyond simple center-crop or face detection. Modern AI-powered cropping uses computer vision models trained on millions of images to understand visual composition, subject importance, and aesthetic principles. These systems can identify the most visually interesting regions of an image and crop accordingly, even for complex scenes with multiple subjects.

The smart cropping pipeline involves several stages:

1. **Content analysis**: Object detection, face recognition, and scene understanding using pre-trained neural networks
2. **Composition scoring**: Evaluate different crop regions using rules of thirds, leading lines, and symmetry principles  
3. **Multi-format optimization**: Generate different crops optimized for square thumbnails, banner images, and mobile screens
4. **A/B testing integration**: Track engagement metrics to continuously improve cropping decisions

| AI Model Component | Input | Output | Computational Cost |
|-------------------|-------|---------|-------------------|
| Object Detection | Full resolution image | Bounding boxes with confidence | 50-200ms GPU time |
| Saliency Mapping | Image regions | Visual importance heatmap | 20-100ms GPU time |
| Composition Analysis | Crop candidates | Aesthetic quality scores | 10-50ms CPU time |
| Face Detection | Image regions | Face locations and orientations | 5-30ms GPU time |

**Content-aware optimization** goes beyond smart cropping to include automated quality adjustments, color correction, and format selection based on image content. For example, images with large areas of solid color compress better as PNG, while photographic content benefits from JPEG compression. AI models can analyze image characteristics and automatically select optimal processing parameters.

The content analysis results feed into processing decisions:

- **Format selection**: Choose JPEG for photos, PNG for graphics, WebP for web delivery
- **Compression settings**: Adjust quality levels based on content complexity and target use case  
- **Color space conversion**: Optimize for target display characteristics (sRGB for web, P3 for modern displays)
- **Sharpening parameters**: Apply appropriate sharpening based on content type and output size

### Automated Video Enhancement

**Scene detection and keyframe extraction** enables sophisticated video processing that goes beyond simple transcoding. AI-powered scene detection can identify shot boundaries, recognize different types of content (talking heads vs action sequences), and extract the most representative frames for thumbnail generation.

Advanced scene analysis provides rich metadata for downstream processing:

| Scene Analysis Output | Use Cases | Implementation Complexity |
|----------------------|-----------|---------------------------|
| Shot boundaries | Chapter markers, thumbnail selection | Medium - temporal analysis |
| Content classification | Encoding parameter optimization | High - requires trained models |
| Motion analysis | Adaptive bitrate tuning | Medium - optical flow calculation |
| Audio classification | Music vs speech detection | High - audio ML models |

**Adaptive transcoding parameters** use content analysis to optimize encoding settings for each video segment. Fast-motion action sequences might benefit from higher bitrates and keyframe frequency, while talking-head segments can use more aggressive compression without quality loss.

The adaptive transcoding system builds a **content fingerprint** for each video:

1. **Motion analysis**: Calculate optical flow vectors to measure scene complexity
2. **Spatial complexity**: Analyze texture and detail levels in each frame
3. **Temporal consistency**: Measure how much content changes between frames
4. **Audio characteristics**: Detect music, speech, silence, and ambient sound

These fingerprints drive encoding decisions:

- **Variable bitrate curves**: Allocate bits based on scene complexity
- **Keyframe placement**: Insert keyframes at scene boundaries for better seeking
- **Encoding speed vs quality**: Use slower, higher-quality encoding for important content
- **Format selection**: Choose codecs optimized for content characteristics

### Batch Processing Optimization

**Pipeline parallelization** enables processing multiple versions of the same media file simultaneously across different nodes. Instead of sequentially generating thumbnail, mobile, and desktop versions, the system can create all variants in parallel and combine the results.

The parallel processing coordinator must handle several challenges:

| Challenge | Impact | Solution Pattern |
|-----------|--------|------------------|
| Resource contention | Multiple jobs competing for same input file | Shared read-only cache with reference counting |
| Result synchronization | Coordinating completion of parallel tasks | Distributed barrier with timeout handling |
| Partial failure handling | Some variants succeed while others fail | Per-variant status tracking with retry logic |
| Progress aggregation | Combining progress from multiple parallel jobs | Weighted progress calculation across variants |

**Dependency-aware scheduling** optimizes processing pipelines by understanding relationships between different processing steps. For example, video thumbnail extraction can begin as soon as transcoding produces the first few seconds of output, rather than waiting for complete transcoding to finish.

The dependency graph represents processing relationships:

```
Input Video → Scene Analysis → Thumbnail Extraction
            → Transcoding → [Mobile, Desktop, Streaming variants]
            → Audio Extraction → Podcast version
```

Each node in the graph can start as soon as its dependencies are satisfied, dramatically reducing overall processing time for complex workflows.

## External Integration Opportunities

### Content Delivery Network Integration

**Multi-CDN strategies** become essential for global media delivery at scale. Different CDNs have varying performance characteristics across geographic regions and content types. A sophisticated system can route video streaming through CDN A while serving image thumbnails through CDN B based on real-time performance monitoring.

The CDN integration layer must handle several responsibilities:

| Integration Aspect | Challenge | Solution Approach |
|-------------------|-----------|-------------------|
| Geographic routing | Optimal CDN selection per region | Real-time latency monitoring with failover |
| Cache invalidation | Coordinating updates across multiple CDNs | Distributed invalidation with retry logic |
| Cost optimization | Minimizing bandwidth and storage costs | Intelligent tier placement and caching policies |
| Format negotiation | Serving optimal formats per client | Edge-side logic with capability detection |

**Edge computing integration** pushes simple processing tasks closer to end users. Image resizing for common thumbnail sizes can happen at CDN edge nodes rather than central processing clusters, reducing latency and bandwidth costs. However, this requires careful orchestration between central processing and edge capabilities.

Edge processing scenarios include:

1. **On-demand resizing**: Generate thumbnail sizes not pre-computed centrally
2. **Format conversion**: Convert between WebP, AVIF, and JPEG based on browser support
3. **Quality adaptation**: Adjust compression based on detected connection speed
4. **Watermark application**: Add region-specific branding at delivery time

### Cloud Storage Backend Integration

**Multi-cloud storage strategies** provide resilience, cost optimization, and geographic distribution capabilities. Different cloud providers offer varying price points and performance characteristics for media storage and processing workloads.

The storage abstraction layer manages complexity across providers:

| Storage Tier | Primary Provider | Backup Provider | Use Case |
|-------------|------------------|-----------------|----------|
| Hot storage | AWS S3 Standard | Google Cloud Storage | Active media files |
| Warm storage | S3 Infrequent Access | Azure Cool Blob Storage | Recently processed files |
| Cold storage | S3 Glacier | Google Archive Storage | Long-term backup |
| Processing cache | Local NVMe SSD | S3 Standard | Active job temporary files |

**Storage lifecycle management** automatically transitions media files between storage tiers based on access patterns and retention policies. Newly uploaded files remain in hot storage for immediate processing, while older processed outputs move to cheaper storage tiers over time.

Lifecycle policies consider multiple factors:

- **Access frequency**: Files not accessed for 30 days move to warm storage
- **Processing requirements**: Source files for ongoing jobs remain in hot storage
- **Compliance requirements**: Legal holds prevent automatic archive transitions
- **Cost optimization**: Balance access time against storage costs

### Third-Party Processing Service Integration

**Hybrid processing architectures** combine on-premise processing with cloud-based specialized services. For example, basic image resizing might happen locally while AI-powered content tagging uses cloud APIs from Google Vision or Amazon Rekognition.

The service integration layer provides a unified interface across providers:

| Processing Type | Local Implementation | Cloud Service Option | Fallback Strategy |
|----------------|---------------------|---------------------|------------------|
| Image resizing | Pillow + custom algorithms | None needed | N/A |
| Object detection | Local ML models | Google Vision API | Disable feature on API failure |
| Video transcoding | FFmpeg cluster | AWS Elemental MediaConvert | Queue jobs until service recovery |
| Content moderation | Rule-based filtering | Amazon Rekognition | Manual review queue |

**API rate limiting and cost management** becomes critical when integrating external services. The system must track usage across different API endpoints and implement circuit breakers to prevent runaway costs from processing spikes or API failures.

Rate limiting strategies include:

1. **Per-service quotas**: Daily/monthly limits for each external service
2. **Priority-based throttling**: Reserve API capacity for high-priority jobs
3. **Cost-based fallbacks**: Switch to local processing when API costs exceed thresholds
4. **Batching optimization**: Combine multiple requests to reduce API call overhead

### Monitoring and Analytics Integration

**Processing analytics** provide insights into system performance, job patterns, and optimization opportunities. Integration with tools like Prometheus, Grafana, and ELK stack enables comprehensive monitoring of the distributed media processing pipeline.

Key metrics to track include:

| Metric Category | Example Metrics | Monitoring Tool | Alert Conditions |
|----------------|-----------------|-----------------|------------------|
| Processing Performance | Jobs/hour, processing time percentiles | Prometheus + Grafana | p95 processing time > SLA |
| Resource Utilization | CPU, memory, GPU usage per node | Node Exporter | Sustained > 90% utilization |
| Quality Metrics | Transcoding quality scores, error rates | Custom collectors | Error rate > 1% |
| Business Metrics | Processing costs, throughput trends | Application metrics | Cost growth > revenue growth |

**Machine learning feedback loops** use processing outcomes to continuously improve system performance. For example, tracking which smart crop selections get the most user engagement can improve the AI model training data over time.

Feedback collection mechanisms:

- **A/B testing frameworks**: Compare different processing parameters and measure outcomes
- **User engagement tracking**: Monitor click-through rates, view duration, and interaction patterns
- **Quality assessment**: Automated quality metrics and occasional human evaluation
- **Performance correlation**: Link processing parameters to user satisfaction metrics

> The key insight for successful scaling is that each integration point becomes a potential failure mode, but also an opportunity for optimization. The system must be designed with graceful degradation in mind—if the AI cropping service fails, fall back to center crop rather than failing the entire job.

### Implementation Guidance

This implementation guidance focuses on the foundational patterns and infrastructure needed to support horizontal scaling and advanced features. The code provides working examples that can be extended for specific scaling scenarios.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Container Orchestration | Docker Compose | Kubernetes with custom operators |
| Service Discovery | Static configuration files | Consul or etcd with DNS integration |
| Load Balancing | HAProxy with static config | Envoy with dynamic configuration |
| Monitoring | Prometheus + Grafana | Datadog or New Relic APM |
| Log Aggregation | File-based logging | ELK Stack or Splunk |
| Secret Management | Environment variables | HashiCorp Vault or AWS Secrets Manager |

#### Recommended File Structure

```
media-pipeline/
├── cmd/
│   ├── coordinator/           ← cluster coordinator service
│   ├── worker/               ← distributed worker nodes
│   └── scheduler/            ← intelligent job scheduler
├── internal/
│   ├── scaling/
│   │   ├── cluster.go        ← node registration and discovery
│   │   ├── balancer.go       ← load balancing algorithms  
│   │   ├── autoscaler.go     ← dynamic scaling policies
│   │   └── storage.go        ← distributed storage abstraction
│   ├── ai/
│   │   ├── cropping.go       ← smart cropping algorithms
│   │   ├── analysis.go       ← content analysis pipelines
│   │   └── models.go         ← AI model integration
│   ├── integration/
│   │   ├── cdn.go           ← CDN provider integrations
│   │   ├── cloud.go         ← multi-cloud storage
│   │   └── external.go      ← third-party API clients
│   └── monitoring/
│       ├── metrics.go       ← performance metrics collection
│       ├── tracing.go       ← distributed tracing
│       └── alerts.go        ← alerting and notification
├── deployments/
│   ├── kubernetes/          ← K8s manifests
│   ├── terraform/           ← infrastructure as code
│   └── docker/              ← container configurations
└── docs/
    ├── scaling-guide.md     ← operational scaling procedures
    ├── integration-api.md   ← external API documentation
    └── monitoring-runbook.md ← troubleshooting procedures
```

#### Infrastructure Starter Code

**Cluster Node Registry** - Complete implementation for node discovery and health tracking:

```python
import asyncio
import json
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import psutil
import GPUtil

class NodeCapability(Enum):
    IMAGE_PROCESSING = "image"
    VIDEO_TRANSCODING = "video" 
    GPU_ACCELERATION = "gpu"
    HIGH_MEMORY = "highmem"

@dataclass
class NodeInfo:
    node_id: str
    hostname: str
    capabilities: List[NodeCapability]
    max_concurrent_jobs: int
    current_load: float
    memory_gb: int
    gpu_count: int
    network_region: str
    last_heartbeat: float
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'capabilities': [c.value for c in self.capabilities],
            'last_heartbeat': self.last_heartbeat
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeInfo':
        return cls(
            node_id=data['node_id'],
            hostname=data['hostname'], 
            capabilities=[NodeCapability(c) for c in data['capabilities']],
            max_concurrent_jobs=data['max_concurrent_jobs'],
            current_load=data['current_load'],
            memory_gb=data['memory_gb'],
            gpu_count=data['gpu_count'],
            network_region=data['network_region'],
            last_heartbeat=data['last_heartbeat']
        )

class ClusterRegistry:
    def __init__(self, redis_url: str, heartbeat_interval: int = 30):
        self.redis_url = redis_url
        self.heartbeat_interval = heartbeat_interval
        self.redis = None
        self.local_node_id = None
        self._heartbeat_task = None
        
    async def initialize(self) -> bool:
        """Initialize Redis connection and start heartbeat."""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            return True
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            return False
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register this node in the cluster registry."""
        self.local_node_id = node_info.node_id
        node_key = f"cluster:nodes:{node_info.node_id}"
        
        try:
            # Store node information with TTL
            await self.redis.hset(
                node_key,
                mapping={
                    "info": json.dumps(node_info.to_dict()),
                    "registered_at": time.time()
                }
            )
            await self.redis.expire(node_key, self.heartbeat_interval * 3)
            
            # Add to active nodes set
            await self.redis.sadd("cluster:active_nodes", node_info.node_id)
            
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(node_info)
            )
            
            print(f"Node {node_info.node_id} registered successfully")
            return True
            
        except Exception as e:
            print(f"Failed to register node: {e}")
            return False
    
    async def _heartbeat_loop(self, node_info: NodeInfo):
        """Continuous heartbeat to maintain node registration."""
        while True:
            try:
                # Update current system metrics
                node_info.current_load = psutil.cpu_percent(interval=1)
                node_info.last_heartbeat = time.time()
                
                # Update GPU utilization if available
                if node_info.gpu_count > 0:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            node_info.current_load = max(
                                node_info.current_load,
                                gpus[0].load * 100
                            )
                    except:
                        pass  # GPU monitoring is optional
                
                node_key = f"cluster:nodes:{node_info.node_id}"
                await self.redis.hset(
                    node_key,
                    mapping={"info": json.dumps(node_info.to_dict())}
                )
                await self.redis.expire(node_key, self.heartbeat_interval * 3)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")
                await asyncio.sleep(5)  # Retry on error
    
    async def get_active_nodes(self) -> List[NodeInfo]:
        """Get list of all currently active nodes."""
        try:
            node_ids = await self.redis.smembers("cluster:active_nodes")
            nodes = []
            
            for node_id in node_ids:
                node_key = f"cluster:nodes:{node_id.decode()}"
                node_data = await self.redis.hget(node_key, "info")
                
                if node_data:
                    node_info = NodeInfo.from_dict(json.loads(node_data))
                    
                    # Check if node is still alive (heartbeat within 3 intervals)
                    if time.time() - node_info.last_heartbeat < self.heartbeat_interval * 3:
                        nodes.append(node_info)
                    else:
                        # Remove stale node
                        await self._cleanup_stale_node(node_id.decode())
            
            return nodes
            
        except Exception as e:
            print(f"Failed to get active nodes: {e}")
            return []
    
    async def _cleanup_stale_node(self, node_id: str):
        """Remove stale node from registry."""
        await self.redis.srem("cluster:active_nodes", node_id)
        await self.redis.delete(f"cluster:nodes:{node_id}")
        print(f"Removed stale node: {node_id}")
    
    async def find_best_nodes(self, 
                            job_requirements: Dict,
                            count: int = 1) -> List[NodeInfo]:
        """Find best nodes for a job based on requirements."""
        active_nodes = await self.get_active_nodes()
        
        if not active_nodes:
            return []
        
        # Filter nodes by capabilities
        required_capabilities = job_requirements.get('capabilities', [])
        if required_capabilities:
            filtered_nodes = []
            for node in active_nodes:
                node_caps = {cap.value for cap in node.capabilities}
                if set(required_capabilities).issubset(node_caps):
                    filtered_nodes.append(node)
            active_nodes = filtered_nodes
        
        # Score nodes based on current load and capabilities
        scored_nodes = []
        for node in active_nodes:
            score = self._calculate_node_score(node, job_requirements)
            scored_nodes.append((score, node))
        
        # Sort by score (higher is better) and return top nodes
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_nodes[:count]]
    
    def _calculate_node_score(self, node: NodeInfo, requirements: Dict) -> float:
        """Calculate suitability score for a node."""
        score = 100.0
        
        # Penalize high load
        score -= node.current_load * 0.5
        
        # Bonus for GPU capability if needed
        if 'gpu' in requirements.get('capabilities', []):
            if NodeCapability.GPU_ACCELERATION in node.capabilities:
                score += 20.0
            else:
                score -= 50.0  # Heavy penalty if GPU required but not available
        
        # Bonus for sufficient memory
        required_memory = requirements.get('min_memory_gb', 4)
        if node.memory_gb >= required_memory:
            score += 10.0
        else:
            score -= 30.0
        
        # Network locality bonus
        preferred_region = requirements.get('network_region')
        if preferred_region and node.network_region == preferred_region:
            score += 15.0
        
        return max(0.0, score)  # Ensure non-negative score
    
    async def shutdown(self):
        """Clean shutdown of registry client."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self.local_node_id and self.redis:
            await self._cleanup_stale_node(self.local_node_id)
        
        if self.redis:
            await self.redis.close()
```

**Intelligent Load Balancer** - Complete implementation with weighted scoring:

```python
import asyncio
import heapq
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class JobType(Enum):
    IMAGE_RESIZE = "image_resize"
    VIDEO_TRANSCODE = "video_transcode"  
    THUMBNAIL_GENERATE = "thumbnail_gen"
    BATCH_PROCESS = "batch_process"

@dataclass
class JobRequirements:
    job_type: JobType
    estimated_duration: int  # seconds
    memory_mb: int
    cpu_cores: int
    requires_gpu: bool
    input_size_mb: int
    priority: int
    network_region: Optional[str] = None
    
class LoadBalancer:
    def __init__(self, cluster_registry: ClusterRegistry):
        self.registry = cluster_registry
        self.node_assignments = {}  # node_id -> list of assigned jobs
        self.performance_history = {}  # (node_id, job_type) -> avg_duration
        self.last_rebalance = time.time()
        self.rebalance_interval = 60  # seconds
        
    async def assign_job(self, job: 'ProcessingJob') -> Optional[str]:
        """Assign job to the most suitable node."""
        requirements = self._extract_job_requirements(job)
        
        # Get candidate nodes
        candidates = await self.registry.find_best_nodes(
            {
                'capabilities': self._job_type_to_capabilities(requirements.job_type),
                'min_memory_gb': requirements.memory_mb // 1024,
                'network_region': requirements.network_region
            },
            count=5  # Get top 5 candidates for detailed scoring
        )
        
        if not candidates:
            print(f"No suitable nodes found for job {job.job_id}")
            return None
        
        # Score candidates with detailed algorithm
        best_node = None
        best_score = -1.0
        
        for node in candidates:
            score = await self._calculate_detailed_score(node, requirements)
            if score > best_score:
                best_score = score
                best_node = node
        
        if best_node:
            # Track assignment for load balancing
            if best_node.node_id not in self.node_assignments:
                self.node_assignments[best_node.node_id] = []
            self.node_assignments[best_node.node_id].append(job.job_id)
            
            print(f"Assigned job {job.job_id} to node {best_node.node_id} (score: {best_score:.2f})")
            return best_node.node_id
        
        return None
    
    async def _calculate_detailed_score(self, 
                                      node: NodeInfo, 
                                      requirements: JobRequirements) -> float:
        """Calculate detailed suitability score for node."""
        base_score = 100.0
        
        # Current load penalty (0-40 point penalty)
        load_penalty = node.current_load * 0.4
        base_score -= load_penalty
        
        # Memory adequacy (bonus for having enough, penalty for insufficient)
        memory_ratio = node.memory_gb * 1024 / requirements.memory_mb
        if memory_ratio >= 2.0:
            base_score += 15.0  # Plenty of memory
        elif memory_ratio >= 1.5:
            base_score += 10.0  # Adequate memory
        elif memory_ratio >= 1.0:
            base_score += 0.0   # Just enough memory
        else:
            base_score -= 30.0  # Insufficient memory
        
        # GPU requirement matching
        if requirements.requires_gpu:
            if NodeCapability.GPU_ACCELERATION in node.capabilities:
                base_score += 25.0  # GPU available
            else:
                return 0.0  # Cannot handle GPU jobs
        
        # Historical performance bonus
        perf_key = (node.node_id, requirements.job_type.value)
        if perf_key in self.performance_history:
            avg_duration = self.performance_history[perf_key]
            expected_duration = requirements.estimated_duration
            
            if avg_duration < expected_duration * 0.8:
                base_score += 10.0  # Consistently fast
            elif avg_duration > expected_duration * 1.2:
                base_score -= 10.0  # Consistently slow
        
        # Current assignment load penalty
        current_assignments = len(self.node_assignments.get(node.node_id, []))
        if current_assignments >= node.max_concurrent_jobs:
            return 0.0  # Node at capacity
        elif current_assignments >= node.max_concurrent_jobs * 0.8:
            base_score -= 20.0  # Node nearly at capacity
        
        # Network locality bonus
        if (requirements.network_region and 
            node.network_region == requirements.network_region):
            base_score += 12.0
        
        # Priority job bonus for less loaded nodes
        if requirements.priority >= 8:  # High priority
            base_score += (100 - node.current_load) * 0.1
        
        return max(0.0, base_score)
    
    def _extract_job_requirements(self, job: 'ProcessingJob') -> JobRequirements:
        """Extract resource requirements from processing job."""
        # This would analyze the job's output specifications to determine requirements
        
        # Default requirements based on job type
        if any('video' in spec.format for spec in job.output_specifications):
            return JobRequirements(
                job_type=JobType.VIDEO_TRANSCODE,
                estimated_duration=300,  # 5 minutes default
                memory_mb=2048,
                cpu_cores=4,
                requires_gpu=True,
                input_size_mb=100,  # Estimated
                priority=job.priority.value
            )
        else:
            return JobRequirements(
                job_type=JobType.IMAGE_RESIZE,
                estimated_duration=30,   # 30 seconds default
                memory_mb=512,
                cpu_cores=1,
                requires_gpu=False,
                input_size_mb=10,   # Estimated
                priority=job.priority.value
            )
    
    def _job_type_to_capabilities(self, job_type: JobType) -> List[str]:
        """Map job types to required node capabilities."""
        capability_map = {
            JobType.IMAGE_RESIZE: ['image'],
            JobType.VIDEO_TRANSCODE: ['video', 'gpu'],
            JobType.THUMBNAIL_GENERATE: ['image'],
            JobType.BATCH_PROCESS: ['image', 'video']
        }
        return capability_map.get(job_type, [])
    
    async def record_job_completion(self, 
                                  node_id: str,
                                  job_id: str, 
                                  job_type: JobType,
                                  duration: int):
        """Record job completion for performance tracking."""
        # Update performance history
        perf_key = (node_id, job_type.value)
        if perf_key not in self.performance_history:
            self.performance_history[perf_key] = duration
        else:
            # Exponential moving average
            self.performance_history[perf_key] = (
                0.7 * self.performance_history[perf_key] + 0.3 * duration
            )
        
        # Remove from current assignments
        if node_id in self.node_assignments:
            try:
                self.node_assignments[node_id].remove(job_id)
            except ValueError:
                pass  # Job not in list
        
        # Check if rebalancing is needed
        if time.time() - self.last_rebalance > self.rebalance_interval:
            await self._rebalance_if_needed()
            self.last_rebalance = time.time()
    
    async def _rebalance_if_needed(self):
        """Rebalance load across nodes if imbalance detected."""
        active_nodes = await self.registry.get_active_nodes()
        if len(active_nodes) < 2:
            return  # No rebalancing needed with < 2 nodes
        
        # Calculate load distribution
        loads = []
        for node in active_nodes:
            current_jobs = len(self.node_assignments.get(node.node_id, []))
            load_ratio = current_jobs / node.max_concurrent_jobs
            loads.append((load_ratio, node.node_id))
        
        loads.sort()
        min_load = loads[0][0]
        max_load = loads[-1][0]
        
        # If imbalance > 30%, log recommendation
        if max_load - min_load > 0.3:
            print(f"Load imbalance detected: {min_load:.2f} to {max_load:.2f}")
            print("Consider implementing job migration for better balance")
    
    async def get_cluster_statistics(self) -> Dict:
        """Get current cluster load statistics."""
        active_nodes = await self.registry.get_active_nodes()
        
        total_capacity = sum(node.max_concurrent_jobs for node in active_nodes)
        total_assigned = sum(len(jobs) for jobs in self.node_assignments.values())
        
        node_stats = []
        for node in active_nodes:
            assigned_jobs = len(self.node_assignments.get(node.node_id, []))
            utilization = assigned_jobs / node.max_concurrent_jobs if node.max_concurrent_jobs > 0 else 0
            
            node_stats.append({
                'node_id': node.node_id,
                'assigned_jobs': assigned_jobs,
                'max_jobs': node.max_concurrent_jobs,
                'utilization': utilization,
                'cpu_load': node.current_load,
                'capabilities': [cap.value for cap in node.capabilities]
            })
        
        return {
            'total_nodes': len(active_nodes),
            'total_capacity': total_capacity,
            'total_assigned': total_assigned,
            'cluster_utilization': total_assigned / total_capacity if total_capacity > 0 else 0,
            'node_statistics': node_stats
        }
```

#### Core Logic Skeleton Code

**Auto-scaling Controller** - Framework for dynamic scaling decisions:

```python
class AutoScaler:
    def __init__(self, cluster_registry: ClusterRegistry, 
                 load_balancer: LoadBalancer,
                 config: Dict):
        self.registry = cluster_registry
        self.load_balancer = load_balancer
        self.config = config
        self.scaling_decisions = []
        
    async def evaluate_scaling_needs(self) -> Dict[str, int]:
        """
        Evaluate current cluster state and return scaling recommendations.
        
        Returns:
            Dict mapping node types to desired count changes
            Example: {'gpu_nodes': +2, 'general_nodes': -1}
        """
        # TODO 1: Get current cluster statistics from load balancer
        # TODO 2: Analyze queue depth by job type from Redis
        # TODO 3: Calculate average wait times for different job types  
        # TODO 4: Check if any resource type is consistently over 80% utilized
        # TODO 5: Check if any resource type is consistently under 30% utilized
        # TODO 6: Apply scaling policies from config (min/max nodes, cooldown periods)
        # TODO 7: Return scaling recommendations with justification
        # Hint: Use exponential moving averages for load metrics to avoid thrashing
        pass
    
    async def execute_scaling_action(self, node_type: str, count_change: int) -> bool:
        """
        Execute scaling action through infrastructure provider.
        
        Args:
            node_type: Type of nodes to scale ('gpu', 'general', 'highmem')
            count_change: Positive to scale up, negative to scale down
        
        Returns:
            True if scaling action was initiated successfully
        """
        # TODO 1: Validate scaling action against configured limits
        # TODO 2: For scale down: ensure nodes to terminate have no active jobs
        # TODO 3: For scale up: determine optimal instance types/sizes
        # TODO 4: Call infrastructure API (AWS Auto Scaling, K8s HPA, etc.)
        # TODO 5: Record scaling action with timestamp and reason
        # TODO 6: Set cooldown period to prevent rapid scaling oscillations
        # Hint: Implement graceful scale-down by draining nodes before termination
        pass
```

**Smart Content Analysis** - Framework for AI-powered processing decisions:

```python
class ContentAnalyzer:
    def __init__(self, model_config: Dict):
        self.models = {}
        self.model_config = model_config
        
    async def analyze_image_content(self, image_path: str) -> Dict:
        """
        Analyze image content to optimize processing parameters.
        
        Returns:
            Analysis results including optimal formats, quality settings, crop regions
        """
        # TODO 1: Load image and extract basic properties (dimensions, format, size)
        # TODO 2: Run object detection to identify main subjects and regions of interest
        # TODO 3: Calculate saliency map to understand visual importance across image
        # TODO 4: Analyze color distribution and complexity for compression optimization
        # TODO 5: Detect faces and important objects for smart cropping recommendations
        # TODO 6: Generate optimal crop coordinates for different aspect ratios
        # TODO 7: Recommend optimal output formats based on content characteristics
        # TODO 8: Calculate quality settings that balance file size and visual quality
        # Hint: Cache analysis results using content hash to avoid recomputation
        pass
    
    async def optimize_video_encoding(self, video_path: str, 
                                    target_formats: List[str]) -> Dict:
        """
        Analyze video content to optimize encoding parameters.
        
        Returns:
            Optimized encoding parameters for each target format
        """
        # TODO 1: Extract video metadata (duration, resolution, bitrate, codec)
        # TODO 2: Perform scene detection to identify different content types
        # TODO 3: Analyze motion vectors and temporal complexity per scene
        # TODO 4: Detect content types (talking head, action, graphics, text)
        # TODO 5: Calculate optimal bitrate curves for variable rate encoding
        # TODO 6: Determine keyframe placement at scene boundaries
        # TODO 7: Select encoder presets based on content complexity
        # TODO 8: Generate separate parameter sets for each target format
        # Hint: Use ffprobe for metadata extraction and scene detection
        pass
```

#### Milestone Checkpoints

**Horizontal Scaling Validation**:
1. Deploy cluster registry with 3 nodes: `python -m pytest tests/test_cluster_registry.py -v`
2. Verify node discovery: Check that all nodes appear in Redis registry within 60 seconds
3. Test load balancer: Submit 10 mixed jobs, verify distribution across nodes based on capabilities
4. Simulate node failure: Stop one node, verify jobs get reassigned within 2 minutes
5. Test auto-scaling: Generate sustained load, verify new nodes join cluster automatically

**Advanced Features Integration**:
1. Deploy AI analysis services: `python -m pytest tests/test_content_analysis.py -v`
2. Upload test images with different characteristics (photos, graphics, text)
3. Verify smart cropping generates different crops for square vs landscape outputs
4. Test adaptive video encoding with sample talking-head vs action footage
5. Validate that content analysis improves processing quality metrics by 15%+

**External Integration Testing**:
1. Configure multi-CDN setup with test accounts
2. Upload processed media and verify delivery through optimal CDN per region
3. Test storage lifecycle: Verify files transition to cheaper storage after 30 days
4. Load test third-party APIs with rate limiting and fallback logic
5. Validate monitoring dashboards show all scaling and processing metrics

Signs of successful implementation:
- **Cluster coordination**: Nodes register/deregister cleanly, load balancing distributes work optimally
- **Intelligent scaling**: System scales up under load and scales down during quiet periods
- **Quality improvements**: AI-powered features measurably improve output quality
- **Cost optimization**: Multi-tier storage and intelligent CDN routing reduce operational costs
- **Reliability**: System gracefully handles node failures, API outages, and traffic spikes


## Glossary and Reference

> **Milestone(s):** All milestones (1-3) as this terminology reference supports understanding across image processing, video transcoding, and job queue components

### Mental Model: Technical Dictionary for Digital Media

Think of this glossary as a technical dictionary specifically curated for digital media processing systems. Just as a medical dictionary helps doctors communicate precisely about anatomical structures and procedures, this reference ensures everyone on the team uses consistent terminology when discussing codecs, queuing systems, and processing workflows. Each term has been carefully chosen to reflect industry standards while maintaining clarity for developers at all experience levels.

The terminology is organized into three categories that mirror the major architectural domains of our media processing pipeline: media processing concepts that deal with the technical aspects of manipulating images and videos, architecture and queue concepts that handle the distributed systems challenges of coordinating work across multiple processes, and acronyms that provide quick reference for the alphabet soup of technical abbreviations common in media processing.

### Media Processing Terms

The media processing domain contains a rich vocabulary of technical terms that describe how digital content is analyzed, transformed, and optimized. These terms form the foundation for understanding how images and videos are manipulated programmatically.

| Term | Definition | Context |
|------|------------|---------|
| **processing job** | Unit of work containing input file and output specifications that defines a complete media transformation task | Core concept used throughout job queue and worker coordination |
| **media metadata** | Technical information extracted from media files including dimensions, format details, creation timestamps, and embedded data | Essential for processing decisions and output optimization |
| **output specification** | Configuration defining desired processing output including format, dimensions, quality settings, and optimization parameters | Drives all media transformation operations |
| **interpolation algorithm** | Mathematical method for calculating new pixel values during resize operations, affecting quality and performance | Critical for image resizing quality - Lanczos for downscaling, bicubic for upscaling |
| **EXIF orientation** | Metadata tag indicating camera rotation when photo was taken, requiring image rotation before processing | Common source of bugs - must rotate before resize/crop operations |
| **color space conversion** | Transformation between different color representation systems like RGB, CMYK, and YUV | Required for format conversion and display optimization |
| **smart cropping** | Automated cropping that preserves visually important image regions using content analysis algorithms | Preserves subject matter better than center-crop for thumbnails |
| **metadata stripping** | Removal of embedded information like GPS coordinates and camera details for privacy protection | Security requirement for public-facing image processing |
| **lossy compression** | Compression that reduces file size by discarding some image data, trading quality for smaller files | JPEG and WebP compression - quality parameter controls data loss |
| **aspect ratio preservation** | Maintaining original width to height ratio during resize operations to prevent image distortion | Default behavior - use letterboxing or cropping when aspect ratios don't match |
| **adaptive bitrate streaming** | Delivery technique with multiple quality variants allowing clients to switch based on network conditions | Core video streaming technology - requires synchronized segments |
| **quality ladder** | Set of video renditions at different resolutions and bitrates optimized for various network conditions | Standard includes 240p, 360p, 480p, 720p, 1080p variants |
| **keyframe alignment** | Synchronized keyframe placement across ABR variants enabling smooth quality switching | Critical for seamless adaptive streaming experience |
| **CRF encoding** | Constant rate factor approach maintaining consistent perceptual quality across video content | Preferred over target bitrate for quality-focused encoding |
| **GOP structure** | Group of pictures organization with keyframes and predicted frames affecting seek performance | Keyframe interval of 2-4 seconds optimal for streaming |
| **container format** | File wrapper that holds video, audio and metadata streams in a standardized structure | MP4 for broad compatibility, WebM for web-optimized delivery |
| **codec compatibility matrix** | Validation table ensuring codec/container combinations work on target platforms | H.264 + MP4 for maximum compatibility, VP9 + WebM for efficiency |
| **hardware acceleration** | GPU-based video encoding for improved performance and reduced CPU usage | Significant speedup but requires compatible hardware and drivers |
| **progressive download** | Streaming technique allowing playback before complete file download using optimized file structure | Requires 'faststart' flag to move metadata to file beginning |
| **segment alignment** | Synchronized segment boundaries across ABR quality variants for smooth switching | Essential for DASH and HLS adaptive streaming |

### Architecture and Queue Terms

The distributed systems architecture of our media processing pipeline introduces concepts related to coordinating work across multiple processes, managing job lifecycles, and ensuring reliable processing under various failure conditions.

| Term | Definition | Context |
|------|------------|---------|
| **job queue** | Message queue system for distributing processing tasks across available worker processes | Central coordination mechanism using Redis or RabbitMQ |
| **worker process** | Background process that executes media processing operations by consuming jobs from the queue | Isolated processes with resource limits and health monitoring |
| **progress tracking** | System for monitoring and reporting job completion status through multiple processing stages | Real-time updates via webhooks and queryable status API |
| **webhook notification** | HTTP callback sent on job status changes including progress updates and completion events | Reliable delivery with retry logic and signature verification |
| **resource-aware scheduling** | Job distribution based on node capabilities and current resource utilization | Prevents overloading workers with memory-intensive video processing |
| **stage-based progress** | Progress reporting based on processing phases rather than simple percentage complete | More accurate than time-based estimation for complex operations |
| **exponential backoff** | Retry strategy with increasing delays between attempts to prevent overwhelming failed services | Standard pattern: base_delay * (backoff_factor ^ attempt_number) |
| **job lifecycle** | Progression of job through pending, processing, completed/failed states with defined transitions | State machine prevents invalid transitions and ensures consistency |
| **serialization** | Converting data structures to/from JSON for storage and transmission between components | Consistent format enables cross-language compatibility |
| **state machine** | Formal model governing valid job status transitions and preventing invalid state changes | Ensures job integrity and proper cleanup of resources |
| **resource constraints** | Limits on memory, CPU, and processing time for jobs to prevent system overload | Video processing limited by memory, image processing by CPU |
| **dead letter queue** | Queue for jobs that failed permanently after exhausting retry attempts | Prevents infinite retry loops while preserving failed jobs for analysis |
| **worker coordination** | Distributed process management including job assignment, health monitoring, and scaling | Maintains optimal worker pool size based on queue depth |
| **heartbeat monitoring** | Health check system for worker processes to detect failures and trigger recovery | Regular status updates enable prompt detection of worker crashes |
| **priority queue** | Queue that processes higher priority jobs first while maintaining fairness for lower priority work | Urgent jobs bypass normal queue while preventing starvation |
| **atomic operation** | Indivisible database operation that prevents race conditions in job status updates | Critical for maintaining consistency in distributed processing |
| **deduplication** | Preventing duplicate job processing through idempotency keys and job fingerprinting | Prevents wasted resources and conflicting outputs |
| **graceful shutdown** | Orderly termination allowing job completion before process exit | Prevents job corruption and resource leaks during deployment |
| **sequence number** | Monotonically increasing identifier for preventing out-of-order progress updates | Ensures progress only moves forward despite network delays |
| **notification threshold** | Progress percentage that triggers webhook delivery to reduce notification spam | Typically 10%, 25%, 50%, 75%, 90%, 100% to balance updates with overhead |
| **HMAC signature** | Cryptographic hash for webhook authentication and integrity verification | Prevents spoofed notifications using shared secret |
| **monotonic progression** | Progress values that only increase, never decrease, preventing confusing status reversals | Enforced through sequence numbers and validation logic |
| **stage transition** | Moving from one processing phase to the next with appropriate progress weight calculation | Enables accurate overall progress despite varying stage complexity |
| **webhook delivery storm** | Excessive notification traffic that overwhelms recipient systems | Prevented through batching, rate limiting, and threshold-based delivery |
| **progress reversal** | Backward movement of progress percentage due to race conditions between workers | Prevented through atomic updates and sequence number validation |
| **hybrid storage** | Combination of Redis for real-time access and PostgreSQL for durability | Redis for active jobs, PostgreSQL for historical data and recovery |
| **idempotency key** | Unique identifier for detecting and preventing duplicate operations | Prevents double-processing when clients retry requests |
| **message serialization** | Converting data structures to JSON for storage and transmission in message queues | Standardized format enables language-agnostic worker processes |
| **circuit breaker** | Pattern to prevent cascade failures when external dependencies become unavailable | Fails fast when FFmpeg or storage systems are unresponsive |
| **retry storm** | Overwhelming retry attempts that prevent system recovery during outages | Prevented through exponential backoff and circuit breakers |
| **resource cleanup** | Systematic cleanup of temporary files and resources after job completion or failure | Essential for preventing disk space exhaustion |
| **worker recovery** | Process of detecting and restarting failed worker processes | Maintains processing capacity despite individual worker failures |
| **error classification** | Categorizing errors to determine appropriate retry strategy | Transient, permanent, and resource errors require different handling |
| **horizontal scaling** | Distributing processing across multiple nodes for increased capacity | Add more worker nodes rather than upgrading individual machines |
| **cluster coordination** | Managing distributed nodes through service discovery and health monitoring | Enables dynamic scaling and load balancing across regions |
| **adaptive load balancing** | Dynamic job assignment using weighted scoring algorithms based on node performance | Considers CPU, memory, and historical processing times |
| **auto-scaling policies** | Rules for automatically adding or removing nodes based on demand metrics | Responds to queue depth and processing latency thresholds |
| **content-aware optimization** | Using AI analysis to optimize processing parameters for specific media content | Adjusts encoding settings based on scene complexity and content type |
| **smart cropping algorithms** | AI-powered cropping that preserves visually important image regions | Identifies faces, text, and salient objects for optimal framing |
| **multi-CDN strategies** | Using multiple content delivery networks for optimal global performance | Reduces latency and provides redundancy for media delivery |
| **storage lifecycle management** | Automatically transitioning files between storage tiers based on access patterns | Hot storage for active processing, cold storage for archives |
| **hybrid processing architectures** | Combining on-premise processing with cloud-based specialized services | Balances cost, latency, and capability requirements |
| **circuit breaker pattern** | Preventing cascade failures when external dependencies become unavailable | Monitors failure rates and temporarily disables failing services |
| **graceful degradation** | Maintaining core functionality when advanced features or dependencies fail | Continues basic processing when AI services or optimization features fail |

### Acronym Quick Reference

The media processing domain makes extensive use of technical acronyms and abbreviations that can be overwhelming for developers new to the field. This reference provides quick definitions and context for the most commonly encountered terms.

| Acronym | Full Form | Definition | Usage Context |
|---------|-----------|------------|---------------|
| **HLS** | HTTP Live Streaming | Apple's adaptive bitrate streaming protocol using M3U8 playlists and TS segments | Primary streaming format for iOS and Safari compatibility |
| **DASH** | Dynamic Adaptive Streaming over HTTP | ISO standard for adaptive bitrate streaming using MPD manifests and fragmented MP4 | Industry standard streaming protocol with broad device support |
| **EXIF** | Exchangeable Image File Format | Metadata standard for digital images including camera settings and GPS coordinates | Contains orientation, timestamp, and privacy-sensitive location data |
| **CRF** | Constant Rate Factor | Video encoding mode that maintains consistent perceptual quality | Preferred over target bitrate for quality-focused encoding (18-23 range) |
| **GOP** | Group of Pictures | Video compression structure organizing keyframes and predicted frames | Affects seek performance and adaptive streaming segment boundaries |
| **ABR** | Adaptive Bitrate | Streaming technique with multiple quality variants for network adaptation | Core technology enabling smooth playback across varying network conditions |
| **API** | Application Programming Interface | Contract defining how software components communicate with each other | REST endpoints for job submission and status queries |
| **JSON** | JavaScript Object Notation | Lightweight data interchange format used for configuration and messaging | Standard serialization format for job definitions and progress updates |
| **HTTP** | Hypertext Transfer Protocol | Application protocol for distributed systems communication | Transport layer for API requests and webhook notifications |
| **HMAC** | Hash-based Message Authentication Code | Cryptographic hash function for message authentication and integrity | Secures webhook notifications against spoofing and tampering |
| **GPU** | Graphics Processing Unit | Specialized processor optimized for parallel computations | Hardware acceleration for video encoding and AI-powered image analysis |
| **CPU** | Central Processing Unit | Primary processor handling general computation tasks | Image processing and job coordination workloads |
| **RAM** | Random Access Memory | Volatile memory for active data storage during processing | Critical resource constraint for large video processing operations |
| **SSD** | Solid State Drive | Flash-based storage providing fast random access for temporary files | Preferred for video transcoding working directory |
| **CDN** | Content Delivery Network | Distributed network of servers for efficient content delivery | Optimizes delivery of processed media files to end users |
| **RGB** | Red Green Blue | Additive color space used for digital displays and image processing | Standard color representation for most image manipulation operations |
| **CMYK** | Cyan Magenta Yellow Black | Subtractive color space used for print media | Requires conversion to RGB for digital processing |
| **YUV** | Luma Chroma | Color space separating brightness from color information | Efficient for video compression and broadcast applications |
| **DPI** | Dots Per Inch | Resolution measurement for print media and high-density displays | Affects image quality for print output and retina displays |
| **IPTC** | International Press Telecommunications Council | Metadata standard for news and media organizations | Contains copyright, caption, and editorial information |
| **XMP** | Extensible Metadata Platform | Adobe's metadata framework for creative files | Comprehensive metadata system supporting custom fields |
| **MIME** | Multipurpose Internet Mail Extensions | Standard for indicating document types on the internet | Content-Type header for proper file handling and validation |
| **UUID** | Universally Unique Identifier | 128-bit identifier guaranteed to be unique across systems | Job IDs and correlation identifiers in distributed processing |
| **TCP** | Transmission Control Protocol | Reliable connection-oriented network protocol | Ensures message delivery for critical job queue operations |
| **UDP** | User Datagram Protocol | Connectionless network protocol for low-latency communication | Used for real-time progress updates and heartbeat monitoring |
| **TLS** | Transport Layer Security | Cryptographic protocol for secure communication over networks | Encrypts webhook notifications and API communication |
| **REST** | Representational State Transfer | Architectural style for web services using HTTP methods | Design pattern for job management and status query APIs |
| **CORS** | Cross-Origin Resource Sharing | Web security feature controlling cross-domain requests | Browser security for web-based media processing interfaces |
| **JWT** | JSON Web Token | Compact token format for securely transmitting information | Authentication mechanism for API access and webhook validation |
| **SQL** | Structured Query Language | Language for managing relational databases | PostgreSQL queries for job history and metadata persistence |
| **ACID** | Atomicity Consistency Isolation Durability | Properties ensuring reliable database transactions | Critical for job status updates and progress tracking |
| **FIFO** | First In First Out | Queue processing order where earliest jobs are processed first | Default job queue behavior for fair processing order |
| **LIFO** | Last In First Out | Stack processing order where newest items are processed first | Not suitable for job queues as it causes starvation |
| **LRU** | Least Recently Used | Cache eviction policy removing oldest unused items | Memory management for image processing cache |
| **TTL** | Time To Live | Expiration mechanism for cached data or temporary resources | Cleanup policy for temporary files and cached thumbnails |

> **Critical Integration Note**: These terms form an interconnected vocabulary where understanding one concept often requires familiarity with several others. For example, **adaptive bitrate streaming** relies on **keyframe alignment**, **segment alignment**, and **quality ladders** to function properly, while **webhook notifications** depend on **HMAC signatures**, **exponential backoff**, and **idempotency keys** for reliable delivery.

### Implementation Guidance

The terminology and concepts defined in this glossary should be consistently used throughout code, documentation, and team communication. Establishing a shared vocabulary prevents miscommunication and ensures that all team members understand system behavior in the same way.

#### Code Documentation Standards

When implementing the media processing pipeline, use these exact terms in code comments, variable names, and function documentation. For example, use `processing_job` rather than variations like `media_task`, `work_item`, or `job_request`. This consistency makes the codebase self-documenting and reduces the learning curve for new team members.

```python
class ProcessingJob:
    """
    Core unit of work containing input file and output specifications.
    
    Represents a complete media transformation task that progresses through
    the job lifecycle from pending to completed/failed states.
    """
    
def submit_job(input_file: str, output_specs: List[OutputSpecification]) -> ProcessingJob:
    """
    Create and queue new processing job for worker execution.
    
    Args:
        input_file: Path to source media file for processing
        output_specs: List of desired output configurations
        
    Returns:
        ProcessingJob with unique job_id and PENDING status
    """
```

#### API Documentation Consistency

REST API endpoints and webhook payloads should use the terminology from this glossary to ensure consistency between the system internals and external interfaces. This makes integration easier for client developers who can reference the same glossary.

| Endpoint | Request Fields | Response Fields |
|----------|---------------|----------------|
| `POST /jobs` | `input_file_path`, `output_specifications`, `webhook_url` | `job_id`, `status`, `estimated_duration` |
| `GET /jobs/{job_id}/progress` | None | `progress_percentage`, `current_stage`, `stage_progress` |
| `GET /jobs/{job_id}/metadata` | None | `media_metadata`, `processing_config`, `resource_constraints` |

#### Error Message Standardization

Error messages and logging should use these standardized terms to make debugging more efficient. Instead of generic messages like "processing failed," use specific terminology like "video transcoding failed during GOP structure analysis" or "image processing failed during EXIF orientation handling."

```python
# Good: Specific terminology
logger.error("CRF encoding failed: invalid GOP structure for adaptive bitrate streaming")

# Bad: Generic terminology  
logger.error("video processing error occurred")
```

#### Team Communication Guidelines

During code reviews, architectural discussions, and incident response, team members should reference these terms consistently. This shared vocabulary accelerates problem diagnosis and solution design by eliminating ambiguity about what each component does and how it behaves.

When discussing system behavior, use the precise term rather than approximations. For example, say "the job queue exhibits head-of-line blocking" rather than "jobs are getting stuck," or "we need exponential backoff for webhook notification retries" rather than "we should try sending webhooks again."

This glossary serves as the definitive reference for terminology throughout the media processing pipeline project, ensuring that everyone from junior developers to senior architects speaks the same technical language.

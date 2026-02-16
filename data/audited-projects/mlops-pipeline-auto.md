# AUDIT & FIX: mlops-pipeline-auto

## CRITIQUE
- **Missing model registry**: The audit is correct. There is no versioning or artifact management between training and serving. Without a registry, you cannot track which model version is deployed, roll back to a previous version, or compare model performance across versions.
- **Missing data validation**: Training on corrupted, schema-drifted, or poisoned data is the #1 cause of ML system failures. No milestone validates incoming data before it enters the feature store or training pipeline.
- **Sub-20ms for 1000+ features is unrealistic without specific architecture**: Fetching 1000+ features from Redis with individual GET commands takes >20ms due to network round trips alone. This requires either multi-get, a client-side sidecar cache, or features packed into a single key. The architecture must be specified.
- **'Automated GPU partitioning and scheduling' is vague**: Does this mean implementing a Kubernetes operator? Using SLURM? Writing a custom scheduler? The scope is unclear.
- **Missing experiment tracking**: No milestone tracks hyperparameters, metrics, or artifacts for individual training runs. Without this, 'comparison between new and old models' in M4 has no data to compare.
- **'Multi-model inference (A/B testing) at the edge' is scope-creeping**: Edge deployment is a massive topic (model compilation, hardware-specific optimization, fleet management). This should be scoped to server-side A/B testing.
- **Missing feature versioning**: Features evolve over time. If a model was trained on feature_v1 but the feature store now serves feature_v2, predictions are silently wrong. No milestone addresses this.
- **Concept drift detection is mentioned but not specified**: What statistical test? What window size? What threshold? This is hand-waved.
- **TensorRT/ONNX integration as a single AC line**: Model optimization and runtime integration is a significant effort, not a bullet point.

## FIXED YAML
```yaml
id: mlops-pipeline-auto
name: Aether MLOps Pipeline
description: "End-to-end ML infrastructure with feature store, model registry, distributed training orchestration, serving with A/B testing, and automated retraining."
difficulty: expert
estimated_hours: "80-110"
essence: >
  End-to-end ML lifecycle management: a real-time feature store with point-in-time
  correctness, data validation at ingestion, a model registry for versioning and
  lineage, distributed training orchestration with experiment tracking, model serving
  with canary deployments and A/B testing, and automated drift detection triggering
  retraining pipelines.
architecture_doc: architecture-docs/mlops-pipeline-auto/index.md
languages:
  recommended:
    - Python
    - Go
    - Rust
  also_possible:
    - Java
resources:
  - type: documentation
    name: Feast Feature Store
    url: https://docs.feast.dev/
  - type: article
    name: "Hidden Technical Debt in Machine Learning Systems (Google)"
    url: https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
  - type: documentation
    name: MLflow Model Registry
    url: https://mlflow.org/docs/latest/model-registry.html
prerequisites:
  - type: skill
    name: Machine learning fundamentals (training, evaluation, metrics)
  - type: skill
    name: Python data ecosystem (pandas, numpy, scikit-learn or PyTorch)
  - type: skill
    name: Docker containers
  - type: skill
    name: REST API development
  - type: skill
    name: Basic distributed systems concepts
skills:
  - Feature store architecture
  - Point-in-time join correctness
  - Data validation and schema enforcement
  - Model versioning and registry
  - Distributed training orchestration
  - Model serving and traffic routing
  - Statistical drift detection
  - ML pipeline automation
tags:
  - build-from-scratch
  - expert
  - feature-store
  - go
  - ml
  - mlops
  - python
  - rust
  - serving
milestones:
  - id: mlops-pipeline-auto-m1
    name: "Feature Store with Data Validation"
    description: >
      Build a feature store with online (low-latency) and offline (batch) retrieval,
      point-in-time correctness for training, data validation at ingestion, and
      feature versioning.
    acceptance_criteria:
      - "Online store retrieves features for a single entity key in <5ms using a batch GET operation against Redis (or equivalent in-memory store) with all features for the entity packed into a single key or retrieved via MGET—NOT individual GETs per feature"
      - "Offline store supports point-in-time joins: given a set of entity keys and timestamps, retrieve the feature values that were valid at each timestamp, preventing data leakage from future data"
      - "Streaming feature computation: features defined as aggregations (e.g., 'count of purchases in last 1 hour') are computed from a streaming data source (Kafka or equivalent) and materialized to both online and offline stores"
      - "Data validation: incoming feature values are validated against a registered schema (data type, value range, null percentage threshold); records failing validation are quarantined and logged, not silently ingested"
      - "Feature versioning: each feature definition has a version; when a feature's computation logic changes, a new version is created and both old and new versions coexist until all consuming models are migrated"
      - "Feature retrieval for 500+ features for a single entity completes in <20ms from the online store, verified by benchmark (specify the architecture: features packed per-entity or sidecar cache)"
      - "Feature registry API: CRUD operations for feature definitions including name, type, description, owner, version, and source"
    pitfalls:
      - "Point-in-time leakage: the most insidious bug in ML. If the join uses the feature value at query time instead of the label timestamp, the model trains on future data and performs unrealistically well in evaluation but fails in production."
      - "Sub-20ms for 1000+ features is only achievable if features are pre-materialized into a single entity-keyed blob. Fetching 1000 individual keys from Redis takes >20ms due to network round trips. Design the storage layout accordingly."
      - "Streaming lag: if the feature computation pipeline falls behind the event stream, online features are stale. Monitor lag and alert when staleness exceeds a threshold."
      - "Schema evolution: changing a feature's data type (e.g., int to float) without versioning silently corrupts downstream models."
    concepts:
      - Feature store architecture (online/offline)
      - Point-in-time correctness
      - Data validation and schema enforcement
      - Stream processing for feature materialization
    skills:
      - Redis or in-memory store optimization
      - Stream processing (Kafka consumer)
      - Schema validation implementation
      - Time-travel join algorithms
    deliverables:
      - Online feature store with <5ms single-entity retrieval
      - Offline feature store with point-in-time join capability
      - Streaming feature computation pipeline
      - Data validation layer with schema enforcement and quarantine
      - Feature versioning and registry API
      - Benchmark demonstrating latency targets
    estimated_hours: "18-24"

  - id: mlops-pipeline-auto-m2
    name: "Distributed Training Orchestrator & Experiment Tracking"
    description: >
      Manage training job submission across GPU resources with fault-tolerant
      checkpointing, experiment tracking for hyperparameters and metrics, and
      integration with a model registry for artifact storage.
    acceptance_criteria:
      - "Training job submission API accepts a training configuration (Docker image, dataset reference, hyperparameters, GPU requirements) and schedules execution on available GPU resources"
      - "GPU resource scheduler allocates requested GPU count to jobs from a pool of available GPUs; jobs waiting for resources are queued with FIFO ordering; scheduler prevents overcommitment"
      - "Fault-tolerant checkpointing: training jobs save periodic checkpoints (model weights, optimizer state, epoch number); if a job crashes, it automatically resumes from the last checkpoint"
      - "Experiment tracking: each training run records hyperparameters, training/validation metrics per epoch, hardware utilization (GPU memory, utilization percentage), and training duration in a queryable store"
      - "Experiment comparison: an API returns a comparison table/view of metrics across multiple training runs, sortable by any metric"
      - "Model registry: completed training runs produce a model artifact; the artifact is versioned (auto-incrementing version per model name), stored in artifact storage (S3/local), and linked to the experiment run that produced it"
      - "Model metadata includes: training dataset hash, feature versions used, hyperparameters, evaluation metrics, and git commit hash of the training code"
    pitfalls:
      - "Checkpoint corruption: writing a checkpoint while the model is being updated produces a corrupt file. Use atomic write (write to temp file, then rename) or PyTorch's recommended save pattern."
      - "GPU memory fragmentation: CUDA memory is not automatically defragmented. Allocating and freeing tensors of varying sizes during training causes OOM errors even with sufficient total memory. Pre-allocate and reuse."
      - "Experiment tracking overhead: logging metrics too frequently (every batch) slows training and floods the tracking store. Default to per-epoch logging with optional per-batch."
      - "Missing model-to-data lineage: without recording the exact dataset version and feature versions used for training, reproducing a model is impossible."
    concepts:
      - Distributed training orchestration
      - Fault-tolerant checkpointing
      - Experiment tracking
      - Model registry and versioning
    skills:
      - GPU resource management
      - Checkpoint save/resume implementation
      - Metadata storage and querying
      - Artifact versioning and storage
    deliverables:
      - Training job submission API with resource requirements
      - GPU scheduler with queuing
      - Automatic checkpointing with crash-resume
      - Experiment tracking store with comparison API
      - Model registry with versioning, artifact storage, and lineage metadata
    estimated_hours: "18-24"

  - id: mlops-pipeline-auto-m3
    name: "Model Serving with Traffic Routing"
    description: >
      Deploy models behind an inference API with canary deployments, A/B testing
      traffic routing, request batching for throughput, and latency-aware
      auto-scaling.
    acceptance_criteria:
      - "Model serving API loads a model artifact from the registry by name and version, exposes a prediction endpoint, and returns predictions within a configurable latency SLO (e.g., p99 <50ms)"
      - "Canary deployment: a new model version can be deployed to receive a configurable percentage of traffic (e.g., 5%) while the current version handles the rest; automatic rollback if error rate or latency of the canary exceeds a threshold"
      - "A/B testing: traffic is split between two model versions based on a consistent hashing of the request key (user ID), ensuring the same user always hits the same model for the duration of the experiment"
      - "Request batching: individual inference requests are collected into batches (configurable max batch size and max wait time) before being sent to the model, improving GPU utilization for batch-capable models"
      - "Auto-scaling: the number of serving replicas scales up when request queue depth or p99 latency exceeds configured thresholds, and scales down during low traffic; scaling decisions are made every 30 seconds"
      - "Model warm-up: when a new model version is loaded, it processes a configurable number of warm-up requests before receiving live traffic, ensuring the first real requests don't suffer cold-start latency"
      - "Health check: each serving instance exposes liveness and readiness endpoints; the traffic router only sends requests to ready instances"
    pitfalls:
      - "Cold-start latency: loading a large model (GBs of weights) takes seconds. Without warm-up, the first requests after deployment timeout."
      - "Canary metrics attribution: if metrics are not correctly tagged with the model version, canary and baseline metrics are mixed, making rollback decisions unreliable."
      - "Request batching latency vs. throughput tradeoff: large batches improve throughput but increase latency for individual requests. The max wait time must be tuned to the latency SLO."
      - "A/B testing with inconsistent hashing: if the hash function changes or a model version is removed and re-added, users get reassigned, polluting experiment results."
    concepts:
      - Model serving and inference
      - Canary and A/B deployment strategies
      - Request batching
      - Auto-scaling
    skills:
      - Model loading and inference API design
      - Traffic routing with consistent hashing
      - Batching with timeout-based flushing
      - Auto-scaling logic with hysteresis
    deliverables:
      - Model serving API loading artifacts from registry
      - Canary deployment with automatic rollback
      - A/B testing traffic router with consistent hashing
      - Request batching engine
      - Latency-aware auto-scaler
      - Health check and readiness endpoints
    estimated_hours: "18-24"

  - id: mlops-pipeline-auto-m4
    name: "Drift Detection & Automated Retraining"
    description: >
      Implement data drift and concept drift detection on live prediction traffic,
      automated retraining triggers, model evaluation with automated promotion,
      and full lineage tracking from data to deployed model.
    acceptance_criteria:
      - "Data drift detection: monitor the statistical distribution of incoming feature values using a defined test (e.g., Population Stability Index, Kolmogorov-Smirnov test, or Jensen-Shannon divergence) compared to the training distribution; alert when drift score exceeds a configurable threshold"
      - "Concept drift detection: monitor model performance metrics (accuracy, precision, recall, AUC) on labeled feedback data; detect when performance degrades below a configurable threshold over a sliding window"
      - "Automated retraining trigger: when drift or performance degradation is detected, automatically submit a new training job using the latest data and the same training configuration as the deployed model"
      - "Automated evaluation: the newly trained model is evaluated against a held-out test set and compared to the currently deployed model on the same test set; the new model is promoted only if it meets all configured quality gates (e.g., accuracy >= current model - 0.5%)"
      - "Shadow deployment: before full promotion, the new model can optionally run in shadow mode (receiving live traffic but not returning predictions to users) to validate performance on real data"
      - "Full lineage tracking: for any deployed model, trace back to the exact training run, dataset version, feature versions, hyperparameters, and source code commit that produced it"
      - "Lineage is queryable: an API endpoint returns the complete lineage chain for a given model version"
    pitfalls:
      - "Data drift without concept drift: feature distributions can shift while model performance remains fine (e.g., seasonal changes that the model handles). Don't retrain on drift alone; combine with performance metrics."
      - "Feedback loop delay: ground-truth labels for predictions may arrive hours or days later. Concept drift detection is delayed by this labeling latency. Use proxy metrics (prediction confidence, distribution shift) for early warning."
      - "Automated retraining on bad data: if a data pipeline bug causes drift, retraining on the bad data makes the model worse. Data validation (M1) must pass before retraining begins."
      - "Champion/challenger evaluation bias: if the test set is not representative of current production traffic, the comparison is meaningless. Use a recent holdout from production, not a static test set."
    concepts:
      - Statistical drift detection
      - Automated ML pipeline orchestration
      - Champion/challenger model evaluation
      - ML lineage and provenance
    skills:
      - Statistical testing (KS test, PSI, JSD)
      - Pipeline automation with triggers
      - Model evaluation methodology
      - Provenance graph construction
    deliverables:
      - Data drift monitor with configurable statistical tests
      - Concept drift monitor with performance-based detection
      - Automated retraining trigger pipeline
      - Automated evaluation with quality gates
      - Shadow deployment mode
      - Full lineage tracking API
    estimated_hours: "18-24"
```
# AUDIT & FIX: data-quality-framework

## CRITIQUE
- **No Data Lineage / Metadata Propagation**: The audit correctly identifies that quality flags (pass/fail, anomaly detected) are useless if they can't be propagated downstream. There's no AC for tagging records or datasets with quality metadata that downstream consumers can use for filtering or alerting.
- **No Reference Data Validation**: Checking a column's values against an allowed set from another table (e.g., country codes, product IDs) is one of the most common data quality checks in production. Completely missing.
- **Learning Outcomes Are Too Vague**: 'Design data validation frameworks' and 'Create data quality dashboards' are not measurable learning outcomes. They should specify what aspects of design are covered.
- **Anomaly Detection AC for Seasonality is Aspirational**: 'Handle seasonality in time series data to reduce false positive alerts' is extremely complex (requires STL decomposition or similar) and is unrealistic alongside Z-score and IQR in 11 hours. Either scope it down or remove it.
- **Data Contracts Milestone Is Disconnected**: The data contracts milestone describes a full contract lifecycle (producer/consumer registration, CI/CD integration) that is really a separate project. For this scope, focus on schema definition and validation.
- **Profiling Memory Management Not AC'd**: Computing statistics on large datasets requires streaming/sampling algorithms (e.g., HyperLogLog for cardinality, t-digest for percentiles). The pitfall mentions this but no AC requires it.
- **No Integration Point Between Milestones**: The expectation engine (M1), profiling (M2), anomaly detection (M3), and contracts (M4) are described as independent components. There's no AC for integrating them — e.g., using profiling results to auto-generate expectations, or using contracts to drive validation.

## FIXED YAML
```yaml
id: data-quality-framework
name: "Data Quality Framework"
description: >-
  Build a data quality framework with declarative expectation rules, statistical
  profiling, anomaly detection, schema contracts, and quality metadata
  propagation for integration into data pipelines.
difficulty: intermediate
estimated_hours: "40-55"
essence: >-
  Rule-based expectation engines that validate data against declarative contracts,
  combined with statistical profiling methods (Z-scores, IQR, distribution
  comparison) and schema enforcement mechanisms that flag non-conforming data
  while propagating quality metadata through pipeline stages.
why_important: >-
  Building this develops expertise in data reliability engineering — a critical
  skill as organizations increasingly treat data quality as code, using
  validation frameworks that prevent bad data from propagating through pipelines
  and ML systems.
learning_outcomes:
  - Implement declarative validation rules with a composable expectation DSL
  - Compute statistical profiles using streaming-friendly algorithms for large datasets
  - Detect anomalies using Z-score, IQR, and distribution comparison methods
  - Define schema contracts in YAML with versioning and compatibility checking
  - Validate data against reference datasets (allowed value sets from external sources)
  - Propagate quality metadata (pass/fail tags, scores) for downstream consumption
  - Integrate profiling results with expectation auto-generation
skills:
  - Schema Validation
  - Statistical Profiling
  - Anomaly Detection
  - Rule Engine Design
  - Data Contracts
  - Reference Data Validation
  - Quality Metadata Propagation
tags:
  - anomalies
  - expectations
  - framework
  - intermediate
  - profiling
  - validation
  - data-quality
architecture_doc: architecture-docs/data-quality-framework/index.md
languages:
  recommended:
    - Python
    - Scala
    - Java
  also_possible: []
resources:
  - name: "Great Expectations Official Documentation"
    url: https://docs.greatexpectations.io/
    type: documentation
  - name: "Data Contracts for Schema Registry"
    url: https://docs.confluent.io/platform/current/schema-registry/fundamentals/data-contracts.html
    type: documentation
  - name: "Data Drift Detection Guide"
    url: https://www.evidentlyai.com/ml-in-production/data-drift
    type: article
  - name: "HyperLogLog for Cardinality Estimation"
    url: https://engineering.fb.com/2018/12/13/data-infrastructure/hyperloglog/
    type: article
prerequisites:
  - type: skill
    name: SQL and tabular data manipulation
  - type: skill
    name: Statistics basics (mean, std, percentiles, distributions)
  - type: skill
    name: Python programming
milestones:
  - id: data-quality-framework-m1
    name: "Expectation Engine with Reference Validation"
    description: >-
      Build a declarative expectation engine that evaluates data quality rules
      against datasets, including reference data validation against external
      allowed-value sets. Return structured results with quality metadata.
    estimated_hours: "10-14"
    concepts:
      - Declarative validation rules and DSL design
      - Expectation result metadata and structured failures
      - Reference data validation (foreign key style checks)
      - Composable expectation chaining with AND/OR logic
      - Quality metadata tagging for downstream propagation
    skills:
      - Rule definition and evaluation
      - Validation logic implementation
      - Result collection and structuring
      - Reference data lookup
    acceptance_criteria:
      - Built-in expectations include not_null, unique, value_in_set, value_in_range, regex_match, and column_type_check
      - Reference data validation checks column values against an allowed-value set loaded from an external source (CSV file, database table, or in-memory list)
      - Custom expectations are supported via user-provided validation functions that receive a column or row and return pass/fail with a reason string
      - Expectations are composable — an expectation suite groups related expectations and reports per-expectation and aggregate pass/fail
      - Each result includes row_count, failure_count, failure_percentage, and a sample of up to 10 failing rows with their failing values
      - Quality metadata tag (PASS/WARN/FAIL with timestamp and expectation suite name) is attached to the validated dataset for downstream consumption
      - Expectation definitions are serializable to/from JSON or YAML for version control and sharing
      - Performance: evaluating 10 expectations against a 1M-row dataset completes in under 30 seconds on a single core
    pitfalls:
      - Not collecting failing row samples makes debugging impossible; always capture examples
      - Reference data validation can be slow if the allowed-value set is large; use a hash set, not linear scan
      - Expectations that are too strict (e.g., unique on a column with legitimate duplicates) cause false failures; provide warning severity level
      - Forgetting to propagate quality metadata means downstream consumers cannot filter bad data
      - Tightly coupling expectations to specific data formats (e.g., pandas DataFrame) breaks reuse with other frameworks
    deliverables:
      - Expectation base class with evaluate() and serialize() interface
      - Built-in expectations for not_null, unique, value_in_set, value_in_range, regex_match, column_type
      - Reference data validator checking column values against external allowed-value sets
      - Custom expectation support via user-defined validation functions
      - Expectation suite grouping related expectations with aggregate reporting
      - Result object with row counts, failure counts, failure samples, and quality metadata tag
      - JSON/YAML serialization for expectation definitions and results

  - id: data-quality-framework-m2
    name: "Statistical Data Profiling"
    description: >-
      Build an automatic data profiler that computes column statistics, detects
      data types, and generates distribution summaries. Use streaming-friendly
      algorithms for large dataset support.
    estimated_hours: "10-14"
    concepts:
      - Summary statistics (mean, std, min, max, percentiles)
      - Approximate cardinality (HyperLogLog)
      - Approximate percentiles (t-digest or similar)
      - Histogram generation with configurable binning
      - Data type inference from column values
      - Sampling strategies for large datasets
    skills:
      - Descriptive statistics computation
      - Streaming/approximate algorithms
      - Distribution analysis
      - Data type inference heuristics
    acceptance_criteria:
      - Numeric columns report min, max, mean, standard deviation, median, and 5th/95th percentiles
      - Categorical columns report cardinality (distinct count), top-N most frequent values with counts, and null percentage
      - Approximate cardinality estimation uses HyperLogLog or similar sketch for columns with >100K distinct values, with <2% error rate
      - Approximate percentiles use t-digest or similar streaming algorithm, with <1% relative error
      - Data type inference detects integer, float, string, boolean, date, and timestamp types from column values with >95% accuracy on mixed-type columns
      - Histogram generation supports both equal-width and equal-depth (quantile) binning with configurable bin count
      - Null percentage is reported per column; columns with >50% nulls are flagged as potentially problematic
      - Profiling a 10M-row dataset with 20 columns completes in under 5 minutes using streaming algorithms without loading the entire dataset into memory
      - Profile report is generated in both human-readable (text/HTML) and machine-readable (JSON) formats
      - Auto-generation of expectation suggestions based on profiling results (e.g., if a column has 0% nulls, suggest not_null expectation)
    pitfalls:
      - Computing exact percentiles on large datasets requires sorting the entire column; use approximate algorithms
      - HyperLogLog requires careful hash function selection; poor hashing destroys accuracy
      - Data type inference on columns with mixed types (e.g., '123' and 'abc') must choose the most permissive type (string) or report mixed
      - Profiling without sampling on 1B+ row datasets causes OOM; always provide a sampling option with configurable sample size
      - Histogram bin edges must handle edge cases — empty columns, single-value columns, and columns with all-null values
    deliverables:
      - Column statistics calculator for numeric and categorical columns
      - HyperLogLog cardinality estimator for large-cardinality columns
      - Approximate percentile calculator using t-digest or similar
      - Histogram generator with equal-width and equal-depth binning
      - Data type inference engine with confidence scoring
      - Null pattern analyzer reporting null percentage and patterns
      - Profile report generator in JSON and human-readable formats
      - Expectation auto-generator suggesting rules from profiling results

  - id: data-quality-framework-m3
    name: "Anomaly Detection & Drift Monitoring"
    description: >-
      Detect statistical anomalies in data batches using Z-score and IQR methods,
      detect distribution drift between batches using KS test, and monitor volume
      and freshness metrics over time.
    estimated_hours: "10-14"
    concepts:
      - Z-score outlier detection for numeric data
      - IQR method for robust outlier identification
      - Kolmogorov-Smirnov test for distribution comparison
      - Volume anomaly detection (row count deviations)
      - Freshness monitoring (data recency checks)
      - Baseline establishment from historical profiling data
    skills:
      - Statistical test implementation
      - Distribution comparison
      - Time-series baseline management
      - Threshold configuration
    acceptance_criteria:
      - Z-score anomaly detection flags numeric values beyond configurable threshold (default ±3σ) from the column mean
      - IQR method flags values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR as outliers; configurable multiplier
      - Distribution drift detection uses two-sample Kolmogorov-Smirnov test comparing current batch distribution against a baseline; p-value below threshold (default 0.05) triggers drift alert
      - Baseline profiles are stored from previous runs and automatically updated on a configurable rolling window (e.g., last 30 runs)
      - Volume anomaly detection alerts when current row count deviates more than configurable percentage (default ±20%) from the rolling average
      - Freshness check verifies that the most recent timestamp in the data is within a configurable recency window (e.g., last 2 hours); stale data triggers alert
      - Schema drift detection compares current column names and types against baseline schema; added, removed, or changed columns are reported
      - All anomaly results include severity level (INFO, WARNING, CRITICAL), the metric value, the threshold, and a human-readable explanation
      - False positive rate for anomaly detection is measurable — test suite includes known-good data batches that should produce zero alerts
    pitfalls:
      - Setting anomaly thresholds without sufficient baseline data (fewer than 10 historical batches) produces unreliable results; warn the user
      - Z-score assumes normal distribution; highly skewed data produces excessive false positives — IQR is more robust for skewed distributions
      - KS test requires sufficient sample size (>50 values) for meaningful results; warn on small batches
      - Static thresholds on evolving data distributions cause alert fatigue; use rolling baselines that adapt over time
      - Volume anomalies on weekly data have day-of-week patterns; simple rolling average misses this — document as a known limitation
    deliverables:
      - Z-score outlier detector for numeric columns with configurable threshold
      - IQR outlier detector for robust outlier identification
      - KS test drift detector comparing current vs baseline distributions
      - Baseline profile storage with rolling window updates
      - Volume anomaly detector with configurable deviation threshold
      - Freshness checker with configurable recency window
      - Schema drift detector comparing against baseline schema
      - Anomaly result with severity, metric, threshold, and explanation

  - id: data-quality-framework-m4
    name: "Schema Contracts & Integration"
    description: >-
      Define schema contracts in YAML with versioning, validate incoming data
      against contracts, detect breaking changes between versions, and integrate
      all framework components into a unified validation pipeline.
    estimated_hours: "10-13"
    concepts:
      - Schema contract definition (field names, types, constraints)
      - Semantic versioning for contracts (MAJOR.MINOR.PATCH)
      - Breaking vs non-breaking change classification
      - Integration of expectations + profiling + anomaly detection
      - Quality gate (pass/fail decision for pipeline progression)
    skills:
      - Schema definition and validation
      - Semantic versioning
      - Breaking change detection
      - Component integration
    acceptance_criteria:
      - Schema contracts are defined in YAML with field name, data type, nullable flag, and optional constraints (min, max, regex, allowed_values)
      - Contract versions follow semantic versioning — MAJOR for breaking changes, MINOR for additions, PATCH for documentation-only changes
      - Breaking change detection automatically classifies changes — column removal, type narrowing, and nullable→non-nullable are MAJOR; column addition with default is MINOR
      - Incoming data is validated against the active contract; violations are reported per-field with the specific constraint that failed
      - Historical violation metrics are tracked per-contract over time (violation count per run, trend direction)
      - Unified validation pipeline chains expectations (M1), profiling (M2), anomaly detection (M3), and contract validation (M4) into a single invocation
      - Quality gate makes a pass/fail/warn decision based on configurable thresholds across all validation components (e.g., fail if >5% expectation failures OR any critical anomaly)
      - Quality gate result is exported as a machine-readable quality metadata record that downstream pipeline stages can consume to decide whether to proceed
    pitfalls:
      - Not validating contract changes against existing data — a new non-nullable constraint on a column with existing nulls is technically correct but operationally breaking
      - Allowing implicit schema evolution (adding columns without version bump) makes it impossible to track when changes occurred
      - Quality gate thresholds that are too strict cause pipeline stalls; too loose allows bad data through — provide sensible defaults with override capability
      - Integration of all components must not require running them in a specific order — profiling and expectations should be parallelizable
    deliverables:
      - Schema contract definition format in YAML with typed fields and constraints
      - Contract versioning with semantic version numbering
      - Breaking change detector classifying changes by severity
      - Contract validation engine comparing data against active contract
      - Historical violation tracker storing metrics per contract per run
      - Unified validation pipeline integrating expectations, profiling, anomaly detection, and contracts
      - Quality gate with configurable pass/warn/fail thresholds and machine-readable output
```
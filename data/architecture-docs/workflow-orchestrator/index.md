# Workflow Orchestrator: Design Document


## Overview

A DAG-based workflow orchestration system that schedules and executes dependent tasks across distributed workers while providing monitoring and failure recovery. The key architectural challenge is coordinating complex task dependencies, handling partial failures gracefully, and scaling execution across multiple worker nodes while maintaining consistency.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** All milestones (foundational understanding for DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring)

Modern data processing and automation workflows have become increasingly complex, involving multiple dependent tasks that must execute in specific orders, handle failures gracefully, and scale across distributed infrastructure. Organizations need reliable systems to orchestrate these workflows, ensuring data pipelines run on schedule, tasks execute only when their dependencies are satisfied, and failures are detected and handled appropriately. This document outlines the design for building a comprehensive workflow orchestration system that addresses these challenges through a directed acyclic graph (DAG) based approach.

The core challenge in workflow orchestration lies in coordinating complex task dependencies while maintaining system reliability and operational visibility. Consider a typical data processing pipeline: raw data must be extracted from multiple sources, cleaned and validated, transformed according to business rules, loaded into analytical databases, and finally used to generate reports. Each step depends on the successful completion of previous steps, yet some operations can run in parallel to optimize processing time. When failures occur—and they inevitably will—the system must detect problems quickly, provide clear diagnostic information, and enable recovery without corrupting downstream data or duplicate processing.

Traditional approaches to workflow management often fall short in several key areas. Simple cron-based scheduling cannot handle complex dependencies between tasks, leading to race conditions and data inconsistencies. Custom scripting solutions become unmaintainable as workflows grow in complexity. Existing enterprise workflow tools are often heavyweight, expensive, and difficult to customize for specific organizational needs. What organizations need is a lightweight, flexible, and reliable orchestration system that provides the essential features of dependency management, distributed execution, failure handling, and operational monitoring.

### Mental Model: The Factory Assembly Line

Understanding workflow orchestration becomes much clearer when we think of it as managing a sophisticated factory assembly line. In this analogy, each **task** in our workflow corresponds to a **workstation** on the assembly line, where specific operations are performed on the product being manufactured. Just as a car cannot have its wheels installed before the chassis is built, tasks in our workflow have **dependencies**—they cannot begin until their prerequisite tasks have completed successfully.

The **directed acyclic graph (DAG)** represents the factory's production blueprint, showing how materials and partially completed products flow between workstations. The "directed" aspect means the flow has a clear direction—materials move forward through the production process rather than cycling back endlessly. The "acyclic" constraint ensures we never create impossible situations where workstation A waits for workstation B, which waits for workstation C, which waits for workstation A again. Such circular dependencies would freeze the entire production line.

In our factory analogy, the **scheduler** acts as the production manager who decides when to start new production runs based on customer orders (cron schedules) or urgent requests (manual triggers). The scheduler examines the current state of the factory floor, checking which workstations are idle, which have completed their work, and which are ready to begin processing. Just as a production manager wouldn't start assembling cars if the paint shop is backed up or the engine assembly line is broken, our scheduler ensures tasks only begin when their dependencies are satisfied and resources are available.

**Workers** in our system correspond to the actual factory floor personnel and equipment. Each worker has specific capabilities—some can operate welding equipment, others handle painting, and others perform quality inspection. Similarly, our system's workers might specialize in running Python scripts, executing database queries, or processing files. The **executor** acts as the floor supervisor, assigning specific tasks to appropriate workers based on their capabilities and current workload.

The **quality control checkpoints** throughout our factory line mirror the monitoring and alerting capabilities of our workflow orchestrator. Just as factory supervisors need real-time visibility into production status, bottlenecks, and quality issues, operators of our workflow system need dashboards showing which tasks are running, which have failed, and where the system might be experiencing problems. When a workstation breaks down or produces defective output, factory managers need immediate notification and clear information about the impact on downstream operations.

> **Key Insight**: The factory assembly line analogy helps us understand why certain design decisions are critical. Just as a factory needs backup equipment for critical operations, our system needs retry mechanisms for failed tasks. Just as production managers need to track inventory and work-in-progress, our system needs persistent state management to survive crashes and restarts.

This mental model also illuminates the importance of **idempotency** in task design. In a well-designed factory, if a power outage interrupts a workstation, operators can safely restart that operation without corrupting the product or wasting materials. Similarly, our workflow tasks should be designed so they can be safely restarted after failures without causing data duplication or inconsistencies.

The assembly line analogy extends to **resource management** as well. Factory managers must balance production speed with resource constraints—running too many parallel operations can overwhelm equipment or create bottlenecks. Our workflow system faces similar challenges in managing compute resources, database connections, and external API rate limits across concurrent task execution.

### Existing Solutions Comparison

The workflow orchestration space includes several mature solutions, each with distinct architectural approaches and design philosophies. Understanding their strengths and limitations provides crucial context for our design decisions and helps identify opportunities for improvement.

**Apache Airflow** represents the dominant player in the open-source workflow orchestration space. Airflow's architecture centers around a scheduler that reads DAG definitions written in Python and coordinates task execution across multiple workers. Its strength lies in its extensive ecosystem of operators for integrating with databases, cloud services, and analytics tools. Airflow provides rich monitoring capabilities through its web interface and supports multiple execution backends including Kubernetes and Celery.

However, Airflow suffers from several architectural limitations that create operational challenges. The scheduler's polling-based approach can introduce latency and scaling bottlenecks in environments with hundreds of DAGs and frequent schedule intervals. Airflow's heavy reliance on its metadata database creates a single point of failure and can become a performance bottleneck under heavy loads. The system's complex deployment requirements, including separate scheduler, webserver, and worker processes, increase operational overhead and make local development environments challenging to set up.

**Prefect** emerged as a "negative engineering" response to Airflow's complexity, emphasizing simplicity and developer experience. Prefect's core innovation is its functional approach to workflow definition, where tasks are regular Python functions decorated with flow metadata. This approach eliminates many of the DAG definition pitfalls that plague Airflow users. Prefect's hybrid execution model allows workflows to run locally during development and seamlessly transition to distributed execution in production.

Prefect's architecture separates flow registration from execution, enabling better version control and deployment practices. The system's cloud-native design includes built-in support for modern infrastructure patterns like containerization and serverless execution. Prefect also provides superior error handling capabilities with automatic retries, conditional logic, and sophisticated state management.

Despite these advantages, Prefect's relatively recent emergence means it has a smaller ecosystem of integrations compared to Airflow. The system's functional programming paradigm, while elegant, can be challenging for teams more comfortable with declarative configuration approaches. Prefect's commercial cloud offering, while powerful, creates vendor lock-in concerns for organizations preferring self-hosted solutions.

**Luigi**, developed by Spotify, takes a different architectural approach focused on dependency resolution and data pipeline reliability. Luigi's central insight is that workflow orchestration should be primarily concerned with ensuring tasks run in the correct order with proper dependency resolution rather than complex scheduling capabilities. Luigi tasks define their requirements and outputs explicitly, enabling the system to automatically determine execution order and skip unnecessary work when outputs already exist.

Luigi's strength lies in its simplicity and reliability for data processing pipelines. The system's approach to idempotency through output checking helps prevent duplicate processing and enables efficient pipeline recovery after failures. Luigi's minimal resource requirements and straightforward deployment model make it attractive for smaller organizations or teams getting started with workflow orchestration.

However, Luigi's simplicity comes at the cost of advanced features. The system lacks sophisticated scheduling capabilities, requiring external cron jobs for time-based triggers. Luigi's monitoring and alerting capabilities are basic compared to more feature-rich alternatives. The system's file-based approach to dependency tracking can become unwieldy for complex workflows with dynamic dependencies or frequently changing requirements.

The following comparison table summarizes the key architectural differences between these solutions:

| Feature | Apache Airflow | Prefect | Luigi | Our System |
|---------|----------------|---------|--------|------------|
| DAG Definition | Python DSL | Python Functions | Python Classes | YAML Configuration |
| Scheduling | Cron + Polling | Hybrid Cloud/Local | External Cron | Cron + Event-driven |
| Dependency Resolution | Database-driven | Flow-based | Output-based | Graph-based |
| Distributed Execution | Celery/Kubernetes | Cloud Agents | Limited | Message Queue |
| Monitoring | Web UI + REST API | Cloud Dashboard | Basic Web UI | Web UI + REST API |
| Learning Curve | Steep | Moderate | Low | Low |
| Operational Complexity | High | Medium | Low | Medium |
| Ecosystem Maturity | Very High | Medium | Medium | New |

**Temporal** represents a different category of workflow orchestration focused on long-running, stateful workflows that can survive process failures and restarts. Temporal's programming model treats workflow code as deterministic functions that can be interrupted and resumed at any point. This approach excels for business process automation and microservice orchestration but adds complexity for simple data pipeline use cases.

**Argo Workflows** provides Kubernetes-native workflow orchestration using custom resource definitions (CRDs) to define workflows as YAML manifests. This approach integrates naturally with cloud-native infrastructure but limits portability and increases operational complexity for teams not fully committed to Kubernetes.

> **Design Insight**: Our analysis reveals a gap in the market for a workflow orchestrator that combines Airflow's comprehensive feature set with Luigi's operational simplicity and Prefect's developer-friendly approach. Most existing solutions require significant operational expertise or impose specific infrastructure requirements that may not align with all organizational contexts.

Based on this competitive analysis, our workflow orchestrator design prioritizes several key principles that differentiate it from existing solutions:

**Configuration-over-Code Approach**: Unlike Airflow and Prefect, which require workflow definitions in Python, our system uses YAML configuration files for DAG definitions. This approach reduces the barrier to entry for non-programmers while maintaining the flexibility to define complex workflows. YAML configurations are easier to validate, version control, and audit compared to executable Python code.

**Operational Simplicity**: Learning from Luigi's success, our system prioritizes ease of deployment and operation. The core system should run efficiently on a single machine for development and small production workloads while supporting distributed execution for larger scale requirements. This approach contrasts with Airflow's complex multi-process architecture that requires significant operational expertise.

**Pluggable Execution Backends**: Rather than being tightly coupled to a specific execution framework like Kubernetes or Celery, our system provides a clean abstraction layer for different execution backends. This design enables teams to start with simple local execution and upgrade to distributed systems as their requirements evolve.

**Built-in Observability**: Modern workflow orchestration requires comprehensive monitoring and alerting capabilities built into the core system rather than added as an afterthought. Our design includes rich metrics, logging, and notification capabilities that provide operational visibility without requiring additional infrastructure components.

The following table compares architectural decisions across solutions:

| Architectural Decision | Airflow Choice | Prefect Choice | Luigi Choice | Our Choice | Rationale |
|------------------------|----------------|----------------|--------------|------------|-----------|
| Workflow Definition Language | Python DSL | Python Functions | Python Classes | YAML Config | Easier validation, non-programmer friendly |
| State Storage | Database Required | Cloud + Local | File System | Database Optional | Flexibility for different deployment scenarios |
| Execution Model | Pull-based Polling | Push-based Agents | Pull-based Polling | Hybrid Push/Pull | Optimizes for both latency and reliability |
| Dependency Resolution | Database Queries | In-memory Graph | File System Checks | In-memory + Persistent | Balance performance with crash recovery |
| Worker Communication | Database + Message Queue | Cloud API | Direct RPC | Message Queue | Standard distributed systems pattern |
| UI Architecture | Flask + React | Cloud Dashboard | Basic HTML | Go + HTML/JS | Simpler deployment, easier customization |

This analysis informed our architectural decisions and helped identify specific improvements over existing solutions. Our design aims to capture the essential benefits of mature workflow orchestrators while avoiding their operational complexity and vendor lock-in characteristics.

### Implementation Guidance

Understanding the workflow orchestration landscape and architectural patterns provides the foundation for building our system. The following guidance helps translate these concepts into practical implementation decisions.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option | Chosen |
|-----------|---------------|----------------|--------|
| DAG Definition | YAML + filepath scanning | REST API + database storage | YAML + scanning |
| Scheduling Backend | Cron parsing + sleep loops | Event-driven with message queues | Cron + polling |
| Task Distribution | Local goroutines | Redis/RabbitMQ workers | Redis pub/sub |
| State Storage | SQLite embedded | PostgreSQL/MySQL | PostgreSQL |
| Web Framework | net/http + html/template | Gin + React SPA | net/http + templates |
| Configuration | YAML files + viper | Environment variables only | YAML + env overrides |
| Logging | Standard log package | Structured logging (logrus/zap) | Structured with zap |
| Metrics | Basic counters | Prometheus + Grafana | Prometheus metrics |

**B. Recommended File/Module Structure:**

This structure supports the factory assembly line mental model by organizing code around the major "departments" in our workflow factory:

```
workflow-orchestrator/
├── cmd/
│   ├── orchestrator/main.go        ← Main server entry point
│   ├── worker/main.go              ← Worker process entry point
│   └── cli/main.go                 ← Command-line management tool
├── internal/
│   ├── dag/                        ← DAG parsing and validation (blueprint department)
│   │   ├── parser.go               ← YAML parsing and validation
│   │   ├── validator.go            ← Cycle detection and dependency validation
│   │   └── types.go                ← DAG, Task, and related type definitions
│   ├── scheduler/                  ← Production planning department
│   │   ├── scheduler.go            ← Main scheduling logic
│   │   ├── cron.go                 ← Cron expression parsing
│   │   └── triggers.go             ← Manual and event-based triggering
│   ├── executor/                   ← Floor supervisor department
│   │   ├── executor.go             ← Task distribution interface
│   │   ├── local.go                ← Local sequential execution
│   │   └── distributed.go          ← Redis/message queue execution
│   ├── worker/                     ← Factory floor workers
│   │   ├── worker.go               ← Worker process logic
│   │   ├── operators/              ← Task operation implementations
│   │   │   ├── python.go           ← Python script execution
│   │   │   ├── bash.go             ← Shell command execution
│   │   │   └── sql.go              ← Database query execution
│   │   └── runtime.go              ← Task execution context
│   ├── storage/                    ← Inventory and record keeping
│   │   ├── models.go               ← Database schema definitions
│   │   ├── migrations/             ← Database schema migrations
│   │   └── repository.go           ← Data access layer
│   ├── webui/                      ← Management dashboard
│   │   ├── server.go               ← HTTP server and routing
│   │   ├── api/                    ← REST API handlers
│   │   │   ├── dags.go             ← DAG management endpoints
│   │   │   ├── runs.go             ← DAG run monitoring endpoints
│   │   │   └── tasks.go            ← Task instance endpoints
│   │   └── templates/              ← HTML templates for UI
│   └── config/                     ← System configuration
│       ├── config.go               ← Configuration loading and validation
│       └── defaults.go             ← Default configuration values
├── configs/
│   ├── orchestrator.yaml           ← Default system configuration
│   └── examples/                   ← Example DAG definitions
│       ├── simple-pipeline.yaml    ← Basic data processing example
│       └── complex-workflow.yaml   ← Multi-stage workflow example
├── deployments/
│   ├── docker/                     ← Container deployment files
│   └── kubernetes/                 ← Kubernetes manifests
└── docs/
    ├── api.md                      ← REST API documentation
    └── operations.md               ← Deployment and operations guide
```

**C. Infrastructure Starter Code:**

The following provides complete, working infrastructure code that supports the factory assembly line operations but isn't the core learning focus:

**Configuration Management** (`internal/config/config.go`):
```go
package config

import (
    "fmt"
    "time"
    
    "github.com/spf13/viper"
)

type Config struct {
    Server   ServerConfig   `mapstructure:"server"`
    Database DatabaseConfig `mapstructure:"database"`
    Redis    RedisConfig    `mapstructure:"redis"`
    DAGs     DAGConfig      `mapstructure:"dags"`
    Logging  LoggingConfig  `mapstructure:"logging"`
}

type ServerConfig struct {
    Host         string        `mapstructure:"host"`
    Port         int           `mapstructure:"port"`
    ReadTimeout  time.Duration `mapstructure:"read_timeout"`
    WriteTimeout time.Duration `mapstructure:"write_timeout"`
}

type DatabaseConfig struct {
    Driver   string `mapstructure:"driver"`
    Host     string `mapstructure:"host"`
    Port     int    `mapstructure:"port"`
    Database string `mapstructure:"database"`
    Username string `mapstructure:"username"`
    Password string `mapstructure:"password"`
    SSLMode  string `mapstructure:"ssl_mode"`
}

type RedisConfig struct {
    Host     string `mapstructure:"host"`
    Port     int    `mapstructure:"port"`
    Password string `mapstructure:"password"`
    DB       int    `mapstructure:"db"`
}

type DAGConfig struct {
    Directory       string        `mapstructure:"directory"`
    ScanInterval    time.Duration `mapstructure:"scan_interval"`
    FilePattern     string        `mapstructure:"file_pattern"`
    ReloadOnChange  bool          `mapstructure:"reload_on_change"`
}

type LoggingConfig struct {
    Level  string `mapstructure:"level"`
    Format string `mapstructure:"format"`
}

func Load(configPath string) (*Config, error) {
    viper.SetConfigFile(configPath)
    viper.SetEnvPrefix("ORCHESTRATOR")
    viper.AutomaticEnv()
    
    // Set defaults
    viper.SetDefault("server.host", "0.0.0.0")
    viper.SetDefault("server.port", 8080)
    viper.SetDefault("server.read_timeout", "30s")
    viper.SetDefault("server.write_timeout", "30s")
    viper.SetDefault("dags.directory", "./dags")
    viper.SetDefault("dags.scan_interval", "30s")
    viper.SetDefault("dags.file_pattern", "*.yaml")
    viper.SetDefault("logging.level", "info")
    viper.SetDefault("logging.format", "json")
    
    if err := viper.ReadInConfig(); err != nil {
        return nil, fmt.Errorf("failed to read config: %w", err)
    }
    
    var config Config
    if err := viper.Unmarshal(&config); err != nil {
        return nil, fmt.Errorf("failed to unmarshal config: %w", err)
    }
    
    return &config, nil
}
```

**Database Connection and Migration** (`internal/storage/database.go`):
```go
package storage

import (
    "database/sql"
    "fmt"
    
    _ "github.com/lib/pq"
    "github.com/golang-migrate/migrate/v4"
    "github.com/golang-migrate/migrate/v4/database/postgres"
    _ "github.com/golang-migrate/migrate/v4/source/file"
)

type Database struct {
    *sql.DB
}

func NewDatabase(config DatabaseConfig) (*Database, error) {
    dsn := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
        config.Host, config.Port, config.Username, config.Password, 
        config.Database, config.SSLMode)
    
    db, err := sql.Open(config.Driver, dsn)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }
    
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }
    
    return &Database{db}, nil
}

func (db *Database) RunMigrations(migrationsPath string) error {
    driver, err := postgres.WithInstance(db.DB, &postgres.Config{})
    if err != nil {
        return fmt.Errorf("failed to create migration driver: %w", err)
    }
    
    m, err := migrate.NewWithDatabaseInstance(
        fmt.Sprintf("file://%s", migrationsPath),
        "postgres", driver)
    if err != nil {
        return fmt.Errorf("failed to create migration instance: %w", err)
    }
    
    if err := m.Up(); err != nil && err != migrate.ErrNoChange {
        return fmt.Errorf("failed to run migrations: %w", err)
    }
    
    return nil
}
```

**Redis Client Setup** (`internal/storage/redis.go`):
```go
package storage

import (
    "context"
    "fmt"
    "time"
    
    "github.com/go-redis/redis/v8"
)

type RedisClient struct {
    *redis.Client
}

func NewRedisClient(config RedisConfig) (*RedisClient, error) {
    rdb := redis.NewClient(&redis.Options{
        Addr:     fmt.Sprintf("%s:%d", config.Host, config.Port),
        Password: config.Password,
        DB:       config.DB,
    })
    
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    _, err := rdb.Ping(ctx).Result()
    if err != nil {
        return nil, fmt.Errorf("failed to connect to Redis: %w", err)
    }
    
    return &RedisClient{rdb}, nil
}
```

**D. Core Logic Skeleton Code:**

This skeleton maps directly to the factory assembly line concepts and provides the structure for implementing the core orchestration logic:

**DAG Definition Types** (`internal/dag/types.go`):
```go
package dag

import (
    "time"
)

// DAG represents a complete workflow definition - our factory production blueprint
type DAG struct {
    ID          string            `yaml:"id" json:"id"`
    Description string            `yaml:"description" json:"description"`
    Schedule    string            `yaml:"schedule" json:"schedule"` // Cron expression
    Tags        []string          `yaml:"tags" json:"tags"`
    Parameters  map[string]string `yaml:"parameters" json:"parameters"`
    Tasks       []Task            `yaml:"tasks" json:"tasks"`
    
    // TODO: Add fields for DAG-level configuration like max_active_runs, catchup, etc.
}

// Task represents a single unit of work - a workstation in our factory
type Task struct {
    ID           string            `yaml:"id" json:"id"`
    Type         string            `yaml:"type" json:"type"` // python, bash, sql
    Operator     OperatorConfig    `yaml:"operator" json:"operator"`
    Dependencies []string          `yaml:"dependencies" json:"dependencies"`
    Retries      int               `yaml:"retries" json:"retries"`
    Timeout      time.Duration     `yaml:"timeout" json:"timeout"`
    Parameters   map[string]string `yaml:"parameters" json:"parameters"`
    
    // TODO: Add fields for task-specific configuration like pool, priority, etc.
}

// OperatorConfig holds operator-specific configuration
type OperatorConfig struct {
    PythonCommand string            `yaml:"python_command,omitempty" json:"python_command,omitempty"`
    BashCommand   string            `yaml:"bash_command,omitempty" json:"bash_command,omitempty"`
    SQLQuery      string            `yaml:"sql_query,omitempty" json:"sql_query,omitempty"`
    Environment   map[string]string `yaml:"environment,omitempty" json:"environment,omitempty"`
    
    // TODO: Add other operator-specific configurations
}

// ValidateDAG ensures the DAG definition is valid and acyclic
func (d *DAG) ValidateDAG() error {
    // TODO 1: Check that all task IDs are unique within the DAG
    // TODO 2: Verify that all dependencies reference existing task IDs
    // TODO 3: Run cycle detection using topological sort
    // TODO 4: Validate cron expression format if schedule is provided
    // TODO 5: Ensure required operator configurations are present for each task type
    panic("implement me")
}

// GetTaskByID retrieves a task definition by its identifier
func (d *DAG) GetTaskByID(taskID string) (*Task, bool) {
    // TODO: Iterate through tasks and return the matching task, or nil if not found
    panic("implement me")
}

// GetUpstreamTasks returns all tasks that this task depends on
func (d *DAG) GetUpstreamTasks(taskID string) ([]*Task, error) {
    // TODO: Find the task by ID, then look up all tasks listed in its Dependencies field
    panic("implement me")
}

// GetDownstreamTasks returns all tasks that depend on this task
func (d *DAG) GetDownstreamTasks(taskID string) ([]*Task, error) {
    // TODO: Iterate through all tasks and find those that list taskID in their Dependencies
    panic("implement me")
}
```

**E. Language-Specific Hints:**

For Go implementation, leverage these language features effectively:
- Use `gopkg.in/yaml.v3` for robust YAML parsing with struct tags for DAG definitions
- Implement interfaces for executors (`LocalExecutor`, `DistributedExecutor`) to enable pluggable execution backends
- Use `context.Context` throughout for cancellation and timeout handling in long-running operations
- Leverage `sync.RWMutex` for concurrent access to shared state like DAG registry and task queues
- Use `time.Ticker` for scheduled operations and `context.WithTimeout` for task execution limits
- Implement graceful shutdown with `os.Signal` handling and `sync.WaitGroup` for worker coordination

**F. Milestone Checkpoint:**

After completing the context and problem understanding phase, verify your comprehension:

1. **Mental Model Check**: Explain how a complex data processing workflow maps to the factory assembly line analogy. Can you identify the workstations (tasks), dependencies (material flow), quality control points (monitoring), and production manager (scheduler)?

2. **Competitive Analysis**: Compare how you would choose between Airflow, Prefect, and Luigi for these scenarios:
   - Small team, simple data pipelines, minimal operational overhead required
   - Large enterprise, complex workflows, need for extensive integrations
   - Startup, rapid prototyping, developer productivity prioritized

3. **Architecture Decision**: Given the trade-offs discussed, justify why YAML configuration files are preferable to Python DSL for workflow definition in our system.

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Too many existing solutions, why build another?" | Unclear problem definition | Review competitive analysis gaps | Focus on operational simplicity + configuration approach differentiators |
| "Factory analogy doesn't make sense for my use case" | Domain-specific workflow requirements | Map your specific workflow to assembly line concepts | Adapt analogy: data pipeline = assembly line, ML training = research lab, etc. |
| "Don't understand DAG vs other workflow patterns" | Missing foundational concepts | Study directed graph theory basics | Start with simple 3-task linear workflow, then add parallelism |
| "Confused by all the architectural decisions" | Information overload | Focus on one decision at a time | Pick the simplest option for each component initially |

This foundation provides the conceptual framework and practical starting point for building a comprehensive workflow orchestration system that balances feature richness with operational simplicity.


## Goals and Non-Goals

> **Milestone(s):** All milestones (foundational understanding that guides DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring)

Building a workflow orchestration system requires clear boundaries around what the system will and won't do. Think of this like planning a construction project - before breaking ground, architects must define exactly what type of building they're constructing, what features it will include, and what amenities are explicitly out of scope. Without these clear boundaries, scope creep inevitably leads to an over-engineered system that satisfies no one well.

This section establishes the functional requirements, performance expectations, and intentional limitations that will guide every architectural decision throughout the project. These goals serve as both a North Star for development priorities and a shield against feature creep that could derail the core mission.

### Functional Goals

The workflow orchestrator must deliver five core capabilities that form the foundation of any production-ready orchestration system. Each capability builds upon the others, creating a cohesive platform for managing complex data workflows.

**DAG Definition and Validation**

The system must provide a declarative way to define workflows as directed acyclic graphs using a human-readable format. This capability forms the foundation upon which all other features depend - without reliable DAG representation, scheduling and execution become impossible.

The DAG definition system must support task dependency declarations using upstream and downstream relationships, allowing workflow authors to express complex dependency chains naturally. Tasks should accept parameters that can be provided at trigger time, enabling workflow reuse across different datasets or environments. The system must validate these definitions rigorously, detecting circular dependencies before they can cause runtime deadlocks.

| Requirement | Description | Acceptance Criteria |
|-------------|-------------|-------------------|
| Task Definition | Declarative task specification with metadata | Tasks include ID, type, dependencies, retry config, timeout |
| Dependency Declaration | Express upstream/downstream relationships | Support multiple dependency types: success, failure, completion |
| Parameter Support | Runtime parameter injection | Task parameters templated at execution time |
| Cycle Detection | Validate DAG acyclicity | Reject DAGs with circular dependencies before registration |
| File-based Configuration | Load DAGs from configuration files | Support `YAML_CONFIG` format with schema validation |

> **Design Insight:** DAG validation at registration time rather than execution time prevents costly runtime failures. A workflow that fails to start due to circular dependencies wastes no compute resources, whereas discovering cycles mid-execution can leave partial work scattered across multiple systems.

**Scheduling and Triggering**

The orchestrator must reliably trigger workflow executions based on time-based schedules or manual requests. This scheduling system acts as the heartbeat of the entire platform, ensuring workflows run when needed without manual intervention.

Time-based scheduling must support standard `CRON_SCHEDULE` expressions with proper timezone handling, accounting for daylight saving time transitions and leap years. The system should track both scheduled and actual execution times, providing visibility into schedule drift and system performance.

| Schedule Type | Format | Example | Behavior |
|---------------|--------|---------|----------|
| Cron Expression | 5-field cron syntax | `0 2 * * *` (daily at 2 AM) | Recurring schedule with timezone support |
| Manual Trigger | API or UI initiated | Immediate execution | Creates DAG run with current timestamp |
| Backfill | Historical time range | 2024-01-01 to 2024-01-31 | Generates runs for missed intervals |
| One-time | Single future execution | 2024-06-15 14:30 | Non-recurring scheduled run |

The scheduler must handle catchup scenarios intelligently, providing configurable behavior for workflows that miss their scheduled time due to system downtime or maintenance windows. Some workflows need to run for every missed interval (data processing pipelines), while others should only run once when the system recovers (cleanup jobs).

**Dependency-Aware Task Execution**

Task execution must respect dependency relationships, ensuring upstream tasks complete successfully before downstream tasks begin. This dependency resolution forms the core orchestration logic that distinguishes workflow systems from simple job queues.

The execution engine must maintain task state throughout the workflow lifecycle, tracking transitions from queued to running to completed states. When upstream tasks fail, the system must intelligently determine which downstream tasks should be skipped, retried, or marked as upstream_failed.

| Task State | Description | Trigger Conditions | Valid Transitions |
|------------|-------------|-------------------|-------------------|
| Queued | Waiting for dependencies | All upstreams completed successfully | Running, Skipped |
| Running | Currently executing | Dependencies satisfied, executor available | Success, Failed |
| Success | Completed successfully | Task returned exit code 0 | None (terminal) |
| Failed | Execution failed | Task error, timeout, or crash | Queued (if retries remain) |
| Skipped | Bypassed due to upstream failure | Upstream task failed | None (terminal) |
| Retry | Waiting for retry attempt | Failed task with remaining retries | Queued |

The system must support configurable retry logic with exponential backoff, allowing transient failures to be handled automatically without manual intervention. Task timeouts must be enforced consistently, preventing runaway tasks from consuming resources indefinitely.

**Distributed Task Execution**

The orchestrator must distribute task execution across multiple worker nodes, enabling horizontal scaling and fault tolerance. This distributed architecture allows the system to handle workloads that exceed the capacity of any single machine while providing resilience against worker failures.

Worker management includes health monitoring through heartbeat mechanisms, automatic task reassignment when workers become unavailable, and load balancing to optimize resource utilization across the worker pool.

| Executor Type | Use Case | Scalability | Complexity |
|---------------|----------|-------------|------------|
| Local Sequential | Development and testing | Single machine only | Low |
| Local Parallel | Small production workloads | Multiple cores, single machine | Medium |
| Distributed Queue | Large production workloads | Multiple machines | High |
| Kubernetes | Container-based workloads | Auto-scaling clusters | High |

The system must handle worker failures gracefully, detecting when workers become unresponsive and reassigning their tasks to healthy workers. Task serialization must work across process and network boundaries, ensuring all necessary context reaches the executing worker.

**Basic Monitoring and Observability**

The orchestrator must provide visibility into workflow execution through both programmatic APIs and human-friendly interfaces. This monitoring capability enables operators to understand system behavior, diagnose issues, and optimize workflow performance.

The monitoring system must track workflow execution history, task-level logs, and system health metrics. Users should be able to view the current state of running workflows, inspect failed task output, and trigger manual interventions when necessary.

| Monitoring Component | Capability | Data Provided |
|---------------------|------------|---------------|
| DAG Graph Visualization | Show task dependencies and status | Visual representation with status colors |
| Execution History | Track past workflow runs | Start time, duration, success/failure, trigger reason |
| Task Logs | Capture task output | Stdout, stderr, and custom log messages |
| Manual Controls | User intervention | Trigger runs, clear task states, mark success/failure |
| Basic Metrics | System health | Queue depth, worker utilization, error rates |

> **Critical Requirement:** All monitoring data must be persisted to survive system restarts. Operators need historical context to understand patterns and diagnose recurring issues.

### Non-Functional Goals

Beyond core functionality, the workflow orchestrator must meet performance, reliability, and operational requirements that determine its suitability for production workloads.

**Performance Targets**

The system must handle realistic production workloads without becoming a bottleneck for data processing pipelines. Performance requirements should account for both steady-state operation and peak load scenarios.

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| DAG Registration | < 1 second per DAG | Time from file change to available for scheduling |
| Schedule Latency | < 30 seconds | Time between scheduled time and actual execution start |
| Task Throughput | 1000+ tasks/minute | Sustained task completion rate |
| Worker Scalability | 100+ concurrent workers | Maximum workers without coordinator bottleneck |
| Database Operations | < 100ms p95 latency | State updates and queries |
| Memory Usage | < 1GB coordinator process | Resident memory under normal load |

Performance must degrade gracefully under overload conditions rather than failing catastrophically. The system should provide backpressure mechanisms to prevent resource exhaustion when task submission rate exceeds processing capacity.

**Reliability Requirements**

The orchestrator must provide strong consistency guarantees for workflow state while remaining available during partial system failures. Reliability encompasses both correctness (workflows behave as defined) and fault tolerance (system continues operating despite component failures).

Task execution must be idempotent where possible, allowing safe retries without duplicating side effects. The system must persist all critical state changes before acknowledging completion, ensuring workflow progress survives system restarts.

| Reliability Aspect | Requirement | Implementation Strategy |
|-------------------|-------------|-------------------------|
| State Consistency | Workflow state always reflects reality | Database transactions for state changes |
| Execution Guarantees | Tasks run at-least-once | Persistent task queue with acknowledgment |
| Recovery Time | < 5 minutes after restart | Fast startup with state reconstruction |
| Data Durability | Zero state loss | Database persistence for all workflow metadata |
| Partial Failure Handling | Continue operation with reduced capacity | Graceful degradation when workers unavailable |

> **Design Principle:** The system should fail fast and fail safely. It's better to reject new work than to accept it and execute it incorrectly.

**Operational Simplicity**

The orchestrator must be straightforward to deploy, configure, and maintain in production environments. Operational complexity often determines whether a system succeeds or fails in practice, regardless of its technical capabilities.

Configuration should follow the principle of secure defaults, requiring minimal setup for basic operation while allowing customization for advanced use cases. The system should provide clear error messages and diagnostic information when problems occur.

| Operational Aspect | Requirement | Benefit |
|-------------------|-------------|---------|
| Deployment | Single binary with embedded UI | Reduces deployment complexity |
| Configuration | Environment variables + config file | Standard deployment patterns |
| Dependencies | PostgreSQL/MySQL + Redis only | Minimal infrastructure requirements |
| Logging | Structured JSON logs | Integration with log aggregation systems |
| Health Checks | HTTP endpoints for liveness/readiness | Kubernetes and load balancer integration |
| Metrics | Prometheus-compatible metrics | Standard monitoring integration |

### Explicit Non-Goals

Clearly defining what the orchestrator will NOT do is as important as defining what it will do. These non-goals prevent scope creep and keep the project focused on delivering core value effectively.

**Advanced Security Features**

The initial version will not include sophisticated security features like multi-tenancy, role-based access control, or audit logging. While security is important for production systems, these features add significant complexity that would distract from core orchestration functionality.

Authentication will be limited to basic mechanisms suitable for trusted environments. The system assumes deployment within secure network boundaries where all users have equivalent access rights.

| Security Feature | Status | Rationale |
|------------------|--------|-----------|
| Multi-tenancy | Not Included | Adds complexity to all data models and APIs |
| RBAC | Not Included | Requires user management and permission systems |
| Audit Logging | Not Included | Requires additional storage and compliance features |
| Encryption at Rest | Not Included | Database-level encryption is sufficient |
| OAuth/SAML | Not Included | Basic auth meets initial requirements |

**Visual DAG Editor**

The system will not include a drag-and-drop visual editor for creating workflows. While visual editors can be user-friendly, they introduce significant UI complexity and often produce less maintainable workflows than code-based definitions.

DAG definition will remain file-based using `YAML_CONFIG` format, which provides version control integration, code review processes, and programmatic generation capabilities that visual editors typically lack.

**Advanced Workflow Patterns**

Complex workflow patterns like dynamic task generation, conditional branching, and sub-workflow composition are explicitly excluded from the initial scope. These patterns, while powerful, introduce significant complexity in both implementation and user understanding.

| Pattern | Status | Alternative |
|---------|--------|-------------|
| Dynamic Task Creation | Not Included | Pre-define all possible tasks |
| Conditional Branches | Not Included | Use separate DAGs for different scenarios |
| Sub-workflows | Not Included | Flatten into single DAG or use task groups |
| Loop Constructs | Not Included | Unroll loops into explicit task sequences |
| Event-driven Triggers | Not Included | Use cron schedules or manual triggers |

**Enterprise Integration Features**

Advanced integration capabilities like data lineage tracking, workflow versioning, SLA monitoring, and integration with enterprise data catalogs are beyond the initial scope. These features, while valuable for large organizations, require deep integration with external systems and add significant complexity.

The focus remains on reliable workflow orchestration rather than comprehensive data platform integration.

**High Availability and Multi-Region Deployment**

The system will not include built-in support for active-active high availability or multi-region deployment patterns. While these capabilities are important for large-scale production systems, they introduce distributed systems complexity that would significantly expand the project scope.

Single-region deployment with database-level replication provides sufficient availability for the target use cases while keeping architectural complexity manageable.

> **Decision: Scope Limitation Strategy**
> - **Context**: Workflow orchestration systems can grow extremely complex with hundreds of potential features
> - **Options Considered**: 
>   1. Build comprehensive platform matching Airflow feature-for-feature
>   2. Focus on core orchestration with clear scope boundaries
>   3. Build minimal system and add features based on user feedback
> - **Decision**: Focus on core orchestration with clear scope boundaries
> - **Rationale**: Core orchestration done well is more valuable than comprehensive features done poorly. Clear boundaries prevent scope creep that often derails projects.
> - **Consequences**: Some users may need additional tools for advanced features, but core system will be more reliable and maintainable

### Success Criteria and Validation

The orchestrator will be considered successful when it can reliably execute real-world data workflows that demonstrate all core capabilities working together. Success is measured not just by feature completeness but by system reliability and operational simplicity.

**Functional Validation**

Each functional goal must pass specific validation tests that demonstrate real-world usage patterns rather than toy examples.

| Goal | Validation Test | Success Criteria |
|------|-----------------|------------------|
| DAG Definition | Load complex DAG with 20+ tasks and multiple dependency chains | No validation errors, correct dependency graph |
| Scheduling | Run hourly workflow for 48 hours with system restarts | All expected runs triggered within 30 seconds of schedule |
| Task Execution | Execute DAG with parallel branches and join points | Correct execution order, proper state transitions |
| Distributed Execution | Run 100 concurrent tasks across 5 workers with simulated failures | All tasks complete successfully with automatic retry/reassignment |
| Monitoring | View execution progress and debug failed task through UI | Complete visibility without external tools |

**Performance Validation**

Performance targets must be validated under realistic load conditions using representative workflows rather than synthetic benchmarks.

**Operational Validation**

The system must demonstrate operational simplicity through straightforward deployment and recovery procedures.

| Scenario | Test | Success Criteria |
|----------|------|------------------|
| Fresh Deployment | Deploy to clean environment with minimal configuration | System operational within 10 minutes |
| Restart Recovery | Restart coordinator during workflow execution | All running workflows resume correctly |
| Worker Failure | Terminate worker process during task execution | Tasks automatically reassigned and completed |
| Database Connectivity | Temporarily disconnect from database | Graceful degradation, automatic reconnection |

### Implementation Guidance

The goals and non-goals defined above translate into specific technology choices and architectural constraints that guide implementation decisions throughout the project.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Database | PostgreSQL with standard SQL | PostgreSQL with JSONB for flexible metadata |
| Message Queue | Redis pub/sub | Redis Streams or RabbitMQ |
| Configuration Format | YAML with Go structs | YAML with JSON Schema validation |
| HTTP Framework | Go standard library net/http | Gin or Echo framework |
| Frontend | Server-side rendered HTML | React SPA with WebSocket updates |
| Task Serialization | JSON encoding | Protocol Buffers or MessagePack |

**Goal-Driven Configuration Structure**

The configuration system must support all functional goals while maintaining operational simplicity. Here's the complete configuration structure that enables all target capabilities:

```go
// Config represents the complete system configuration
type Config struct {
    Server   ServerConfig   `yaml:"server"`
    Database DatabaseConfig `yaml:"database"`
    Redis    RedisConfig    `yaml:"redis"`
    DAGs     DAGConfig      `yaml:"dags"`
    Logging  LoggingConfig  `yaml:"logging"`
}

// ServerConfig defines HTTP server parameters
type ServerConfig struct {
    Host         string        `yaml:"host"`
    Port         int           `yaml:"port"`
    ReadTimeout  time.Duration `yaml:"read_timeout"`
    WriteTimeout time.Duration `yaml:"write_timeout"`
}

// DatabaseConfig defines database connection parameters
type DatabaseConfig struct {
    Driver   string `yaml:"driver"`
    Host     string `yaml:"host"`
    Port     int    `yaml:"port"`
    Database string `yaml:"database"`
    Username string `yaml:"username"`
    Password string `yaml:"password"`
    SSLMode  string `yaml:"ssl_mode"`
}

// RedisConfig defines Redis connection parameters
type RedisConfig struct {
    Host     string `yaml:"host"`
    Port     int    `yaml:"port"`
    Password string `yaml:"password"`
    DB       int    `yaml:"db"`
}

// DAGConfig defines DAG discovery and loading parameters
type DAGConfig struct {
    Directory      string        `yaml:"directory"`
    ScanInterval   time.Duration `yaml:"scan_interval"`
    FilePattern    string        `yaml:"file_pattern"`
    ReloadOnChange bool          `yaml:"reload_on_change"`
}

// LoggingConfig defines logging behavior
type LoggingConfig struct {
    Level  string `yaml:"level"`
    Format string `yaml:"format"`
}
```

**Goal Validation Functions**

Each functional goal must have corresponding validation functions that verify correct behavior:

```go
// ValidateDAG ensures DAG meets all functional requirements
func (d *DAG) ValidateDAG() error {
    // TODO 1: Check all task IDs are unique within the DAG
    // TODO 2: Validate all dependency references point to existing tasks
    // TODO 3: Run cycle detection using depth-first search
    // TODO 4: Ensure at least one task has no upstream dependencies (entry point)
    // TODO 5: Validate cron schedule format if specified
    // TODO 6: Check task parameter templates are valid
}

// GetTaskByID retrieves task by identifier for dependency resolution
func (d *DAG) GetTaskByID(taskID string) (*Task, bool) {
    // TODO: Return task and existence flag from internal task map
}

// GetUpstreamTasks returns all dependency tasks for execution ordering
func (d *DAG) GetUpstreamTasks(taskID string) ([]*Task, error) {
    // TODO 1: Look up task by ID
    // TODO 2: Resolve dependency task IDs to task objects
    // TODO 3: Return slice of upstream task pointers
}

// GetDownstreamTasks returns all dependent tasks for failure propagation
func (d *DAG) GetDownstreamTasks(taskID string) ([]*Task, error) {
    // TODO 1: Scan all tasks in DAG
    // TODO 2: Find tasks that list taskID in dependencies
    // TODO 3: Return slice of downstream task pointers
}
```

**Infrastructure Initialization**

The configuration and infrastructure initialization must support all non-functional goals:

```go
// Load reads configuration from file with validation
func Load(configPath string) (*Config, error) {
    // TODO 1: Read YAML file from configPath
    // TODO 2: Parse YAML into Config struct
    // TODO 3: Validate all required fields are present
    // TODO 4: Apply defaults for optional fields
    // TODO 5: Validate configuration constraints (positive timeouts, valid ports)
}

// NewDatabase creates database connection with reliability features
func NewDatabase(config DatabaseConfig) (*Database, error) {
    // TODO 1: Build connection string from config parameters
    // TODO 2: Open database connection with retry logic
    // TODO 3: Configure connection pool for performance targets
    // TODO 4: Test connection with simple query
    // TODO 5: Return wrapped connection with health check methods
}

// RunMigrations executes database schema setup for persistence goals
func RunMigrations(migrationsPath string) error {
    // TODO 1: Read migration files from directory
    // TODO 2: Check current schema version
    // TODO 3: Apply pending migrations in transaction
    // TODO 4: Update schema version on success
}

// NewRedisClient creates Redis connection for distributed execution goals
func NewRedisClient(config RedisConfig) (*RedisClient, error) {
    // TODO 1: Create Redis client with connection parameters
    // TODO 2: Test connection with ping command
    // TODO 3: Configure client for task queue reliability
    // TODO 4: Return wrapped client with health check methods
}
```

**Milestone Checkpoints**

After defining goals and non-goals, validate your understanding:

1. **Goal Clarity Test**: Can you explain each functional goal to a junior developer without referring to implementation details? If not, the goal needs more user-focused description.

2. **Scope Boundary Test**: When someone requests a feature not in the non-goals list, can you quickly determine whether it fits within the defined scope? If not, the boundaries need clarification.

3. **Success Criteria Test**: For each functional goal, can you describe what "working correctly" looks like from a user perspective? If not, add specific acceptance criteria.

4. **Configuration Completeness Test**: Does the configuration structure support all functional and non-functional goals? Missing configuration often indicates unclear requirements.

**Common Pitfalls in Goal Definition**

⚠️ **Pitfall: Vague Performance Requirements**
Many projects define performance goals like "fast" or "scalable" without specific metrics. This makes it impossible to know when requirements are met and leads to endless optimization cycles. Always specify measurable targets with clear testing methodology.

⚠️ **Pitfall: Scope Creep Through Dependencies**
Functional goals often imply additional features that weren't explicitly considered. For example, "distributed execution" might seem to require sophisticated load balancing, but that could be a non-goal. Regularly review implied requirements and decide whether they're in scope.

⚠️ **Pitfall: Ignoring Operational Requirements**
Technical goals (like DAG validation) are easier to define than operational ones (like deployment simplicity), leading to systems that work in development but fail in production. Give equal attention to both functional and operational requirements.

⚠️ **Pitfall: Non-Goals That Are Too Restrictive**
Excluding features that are natural extensions of core functionality can lead to awkward architectural choices. For example, excluding all security features might force design decisions that make security impossible to add later. Consider which non-goals are temporary vs permanent.


## High-Level Architecture

> **Milestone(s):** All milestones (architectural foundation for DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring)

### Component Overview

Think of the workflow orchestrator as a **manufacturing plant control system**. Just as a modern factory has separate control centers for planning production schedules, monitoring assembly lines, and managing worker assignments, our orchestrator divides responsibilities among specialized components. The scheduler acts as the production planner, determining when each manufacturing run should start based on schedules and resource availability. The DAG parser serves as the blueprint reader, interpreting workflow definitions like an engineer reading assembly instructions. The executor functions as the floor supervisor, coordinating workers and ensuring tasks execute in the correct order. Finally, the web UI provides the plant manager's dashboard, offering real-time visibility into operations and manual override capabilities.

This separation of concerns ensures that each component can be developed, tested, and scaled independently while maintaining clear interfaces and responsibilities. The architecture follows a **hub-and-spoke pattern** where the database serves as the central coordination point, with each component reading and updating shared state through well-defined data contracts.

![System Component Architecture](./diagrams/system-components.svg)

#### DAG Parser Component

The DAG parser serves as the **workflow definition interpreter**, responsible for discovering, parsing, and validating workflow files from the filesystem. This component continuously monitors a configured directory for workflow definition files, parsing them from the YAML configuration format into internal data structures. The parser's primary responsibility is ensuring that workflow definitions are syntactically correct, semantically valid, and free of circular dependencies before they enter the execution system.

The parser maintains an in-memory registry of all discovered DAGs, updating this registry when files are added, modified, or removed. It performs comprehensive validation including cycle detection using topological sorting algorithms, dependency resolution verification, and parameter validation. The component also handles file watching capabilities, automatically reloading workflows when their definition files change during development.

| Responsibility | Description | Interfaces |
|---|---|---|
| File Discovery | Scan configured directories for workflow files matching patterns | `DiscoverDAGFiles(directory string) ([]string, error)` |
| YAML Parsing | Convert workflow files into internal DAG structures | `ParseDAGFile(filepath string) (*DAG, error)` |
| Cycle Detection | Validate DAG acyclicity using topological sort | `ValidateDAG() error` |
| Registry Management | Maintain in-memory catalog of all valid workflows | `RegisterDAG(dag *DAG) error`, `GetDAG(id string) (*DAG, bool)` |
| File Watching | Monitor filesystem changes and reload workflows | `WatchForChanges() error` |

#### Scheduler Component

The scheduler operates as the **workflow execution coordinator**, determining when DAG runs should be triggered based on time-based schedules or manual requests. This component continuously evaluates cron expressions to identify workflows that should execute, creates new DAGRun instances with appropriate execution dates, and manages the backfill process for historical periods.

The scheduler maintains awareness of currently running workflows to prevent overlapping executions when configured, and handles timezone conversions to ensure consistent scheduling across distributed deployments. It also manages the catchup behavior, determining whether missed schedules should be automatically executed or skipped based on configuration.

| Responsibility | Description | Interfaces |
|---|---|---|
| Schedule Evaluation | Parse cron expressions and calculate next run times | `EvaluateSchedule(dag *DAG) (time.Time, error)` |
| DAGRun Creation | Instantiate new workflow executions with proper state | `CreateDAGRun(dagID string, executionDate time.Time) (*DAGRun, error)` |
| Manual Triggering | Handle user-initiated workflow executions | `TriggerDAG(dagID string, params map[string]string) error` |
| Backfill Management | Execute workflows for historical time periods | `BackfillDAG(dagID string, start, end time.Time) error` |
| Concurrency Control | Prevent overlapping executions when configured | `CheckConcurrencyLimits(dag *DAG) bool` |

#### Executor Component

The executor functions as the **task distribution engine**, responsible for managing the lifecycle of individual tasks within DAG runs. This component receives ready tasks from the scheduler, determines appropriate workers for execution, and tracks task state transitions from queued through completion. The executor abstracts the underlying execution infrastructure, supporting both local sequential execution for development and distributed execution across worker pools for production scenarios.

The executor handles dependency resolution, ensuring tasks only execute after their upstream dependencies complete successfully. It manages task queuing, timeout enforcement, retry logic, and result collection. The component also implements different execution strategies through pluggable executor backends, allowing the same workflow definitions to run in various deployment environments.

| Responsibility | Description | Interfaces |
|---|---|---|
| Dependency Resolution | Determine when tasks are ready for execution | `GetReadyTasks(dagRun *DAGRun) ([]*TaskInstance, error)` |
| Task Distribution | Assign tasks to available workers | `SubmitTask(task *TaskInstance) error` |
| State Management | Track task lifecycle and status transitions | `UpdateTaskState(taskID string, state TaskState) error` |
| Worker Coordination | Manage worker pools and health monitoring | `RegisterWorker(worker *Worker) error` |
| Result Collection | Gather task outputs and propagate to downstream tasks | `CollectTaskResult(taskID string) (*TaskResult, error)` |

#### Web UI Component

The web UI serves as the **operational dashboard**, providing human operators with visibility into workflow execution and manual control capabilities. This component renders interactive visualizations of DAG structures, displays real-time execution status, and allows users to trigger manual runs or clear failed tasks. The UI communicates with core components through a REST API layer, maintaining separation between presentation and business logic.

The web interface includes features for viewing task logs, monitoring execution history, and managing workflow configuration. It implements real-time updates through WebSocket connections, ensuring operators receive immediate notification of status changes without polling.

| Responsibility | Description | Interfaces |
|---|---|---|
| DAG Visualization | Render interactive workflow graphs | `RenderDAGGraph(dagID string) ([]byte, error)` |
| Execution Monitoring | Display real-time task and DAG run status | `GetDAGRunStatus(runID string) (*DAGRunStatus, error)` |
| Log Viewing | Stream and display task execution logs | `GetTaskLogs(taskID string) (io.Reader, error)` |
| Manual Operations | Handle user-triggered runs and state changes | `HandleManualTrigger(dagID string, params map[string]string) error` |
| Real-time Updates | Push status changes via WebSocket | `BroadcastStatusUpdate(update *StatusUpdate) error` |

#### Database Component

The database serves as the **system state repository**, persisting all workflow metadata, execution history, and configuration. This component stores DAG definitions, tracks DAGRun and TaskInstance states, and maintains audit logs of all system operations. The database design supports concurrent access from multiple components while ensuring data consistency through appropriate isolation levels and locking strategies.

The persistence layer implements connection pooling, query optimization, and migration management to support production deployments. It provides transactional guarantees for critical state transitions and supports both read-heavy monitoring queries and write-heavy execution updates.

| Responsibility | Description | Interfaces |
|---|---|---|
| State Persistence | Store workflow and execution metadata | `CreateDAGRun(*DAGRun) error`, `UpdateTaskState(*TaskInstance) error` |
| Query Interface | Retrieve workflow data for components | `GetActiveDAGRuns() ([]*DAGRun, error)` |
| Transaction Management | Ensure consistency during state updates | `WithTransaction(func(*sql.Tx) error) error` |
| Connection Pooling | Manage database connections efficiently | `NewDatabase(config DatabaseConfig) (*Database, error)` |
| Migration Support | Handle schema evolution | `RunMigrations(migrationsPath string) error` |

#### Message Queue Component

The message queue facilitates **asynchronous task distribution** in distributed execution environments, enabling the executor to send tasks to remote workers without blocking on task completion. This component handles task serialization, delivery guarantees, and worker acknowledgments. It supports both Redis-based simple queuing and more robust message queue systems like RabbitMQ for high-availability deployments.

The message queue implements retry policies for failed deliveries, dead letter queues for permanently failed tasks, and priority-based routing for different task types. It also provides visibility into queue depths and processing rates for operational monitoring.

| Responsibility | Description | Interfaces |
|---|---|---|
| Task Queuing | Distribute tasks to worker queues | `EnqueueTask(queue string, task *Task) error` |
| Message Delivery | Ensure reliable task delivery to workers | `DequeueTask(queue string) (*Task, error)` |
| Worker Communication | Handle worker registration and heartbeats | `RegisterWorker(workerID string, queues []string) error` |
| Queue Management | Monitor queue depths and processing rates | `GetQueueStats(queue string) (*QueueStats, error)` |
| Retry Handling | Manage failed task retries and dead letters | `RetryTask(task *Task, delay time.Duration) error` |

> **Key Architectural Insight**: The separation between executor (task distribution) and workers (task execution) enables horizontal scaling. The executor can run as a single instance managing state consistency, while workers can scale independently based on processing demand. This pattern prevents the complexity of distributed consensus from affecting the core orchestration logic.

### Recommended File Structure

The Go codebase organization follows **domain-driven design principles**, grouping related functionality into cohesive packages while maintaining clear dependency boundaries. The structure separates public interfaces from internal implementations, enabling independent development and testing of each component.

```
workflow-orchestrator/
├── cmd/                          # Application entry points
│   ├── scheduler/                # Scheduler service binary
│   │   └── main.go              # Scheduler startup and configuration
│   ├── worker/                   # Worker service binary  
│   │   └── main.go              # Worker startup and task execution loop
│   └── webui/                    # Web UI service binary
│       └── main.go              # HTTP server and API handlers
├── internal/                     # Private application packages
│   ├── config/                   # Configuration management
│   │   ├── config.go            # Config struct definitions and loading
│   │   └── config_test.go       # Configuration parsing tests
│   ├── dag/                      # DAG parsing and validation
│   │   ├── parser.go            # YAML parsing and file discovery
│   │   ├── validator.go         # Cycle detection and dependency validation
│   │   ├── registry.go          # In-memory DAG registry
│   │   └── models.go            # DAG and Task struct definitions
│   ├── scheduler/                # Scheduling engine
│   │   ├── scheduler.go         # Cron evaluation and DAGRun creation
│   │   ├── cron.go              # Cron expression parsing utilities
│   │   └── backfill.go          # Historical execution logic
│   ├── executor/                 # Task execution coordination
│   │   ├── executor.go          # Core execution interface
│   │   ├── local.go             # Local sequential executor implementation
│   │   ├── distributed.go       # Distributed message queue executor
│   │   └── worker.go            # Worker registration and monitoring
│   ├── operators/                # Task operation implementations
│   │   ├── operator.go          # Common operator interface
│   │   ├── python.go            # Python command execution
│   │   ├── bash.go              # Shell command execution
│   │   └── sql.go               # Database query execution
│   ├── storage/                  # Database persistence layer
│   │   ├── database.go          # Database connection and migrations
│   │   ├── dag_store.go         # DAG metadata persistence
│   │   ├── run_store.go         # DAGRun and TaskInstance storage
│   │   └── migrations/          # SQL schema migration files
│   ├── queue/                    # Message queue abstraction
│   │   ├── queue.go             # Queue interface definition
│   │   ├── redis.go             # Redis-based queue implementation
│   │   └── memory.go            # In-memory queue for testing
│   ├── webui/                    # Web interface components
│   │   ├── handlers/             # HTTP request handlers
│   │   │   ├── dag_handler.go   # DAG management endpoints
│   │   │   ├── run_handler.go   # Execution monitoring endpoints
│   │   │   └── api_handler.go   # REST API implementation
│   │   ├── templates/            # HTML templates
│   │   └── static/               # CSS, JavaScript, assets
│   └── common/                   # Shared utilities
│       ├── logger/               # Structured logging
│       ├── errors/               # Error handling utilities
│       └── metrics/              # Monitoring and metrics
├── pkg/                          # Public API packages
│   └── client/                   # Go client library for API
├── docs/                         # Documentation
├── scripts/                      # Build and deployment scripts
├── migrations/                   # Database schema migrations
├── examples/                     # Example DAG definitions
│   ├── simple_dag.yaml          # Basic workflow example
│   └── data_pipeline.yaml       # Complex pipeline example
├── docker/                       # Container definitions
│   ├── Dockerfile.scheduler     # Scheduler container
│   ├── Dockerfile.worker        # Worker container
│   └── docker-compose.yml       # Development environment
├── go.mod                        # Go module definition
├── go.sum                        # Dependency checksums
├── Makefile                      # Build automation
└── README.md                     # Project documentation
```

The package organization follows Go conventions with the `internal` directory preventing external imports of private implementation details. Each component has dedicated packages allowing independent development while the `common` package provides shared utilities. The `cmd` directory supports multiple deployment patterns with separate binaries for scheduler, worker, and web UI services.

> **Design Principle**: The file structure reflects the component boundaries defined in the architecture. Each major component (`dag`, `scheduler`, `executor`) has its own package with clear interfaces, enabling teams to work on different components simultaneously without merge conflicts.

### Technology Stack Decisions

The technology choices balance **operational simplicity** with **production readiness**, prioritizing proven technologies over cutting-edge solutions. Each decision considers deployment complexity, operational overhead, and the learning curve for developers implementing the system.

#### Database Technology Selection

> **Decision: PostgreSQL as Primary Database**
> - **Context**: Need persistent storage for workflow metadata, execution history, and system state with ACID guarantees and concurrent access support
> - **Options Considered**: PostgreSQL, MySQL, SQLite, MongoDB
> - **Decision**: PostgreSQL with optional SQLite for development
> - **Rationale**: PostgreSQL provides JSON column support for flexible metadata storage, excellent concurrency control, and robust transaction guarantees. Its advisory locks feature supports distributed coordination. SQLite offers zero-configuration development experience.
> - **Consequences**: Enables complex queries for monitoring dashboards, supports horizontal read scaling, but requires database administration expertise for production deployments

| Database Option | Pros | Cons | Use Case |
|---|---|---|---|
| PostgreSQL | ACID compliance, JSON support, advisory locks, mature ecosystem | Requires administration, setup complexity | Production deployments |
| SQLite | Zero configuration, embedded, perfect for development | Limited concurrency, no network access | Development and testing |
| MySQL | Wide adoption, good tooling | Limited JSON support, weaker consistency guarantees | Alternative production option |
| MongoDB | Schema flexibility, horizontal scaling | Eventually consistent, complex transactions | Not recommended for workflow state |

#### Message Queue Technology Selection

> **Decision: Redis with Fallback to In-Memory Queue**
> - **Context**: Need asynchronous task distribution for distributed execution with delivery guarantees and worker coordination
> - **Options Considered**: Redis, RabbitMQ, Apache Kafka, In-memory channels
> - **Decision**: Redis for distributed deployments, Go channels for local execution
> - **Rationale**: Redis provides simple pub/sub and list operations sufficient for task queuing, requires minimal operational overhead, and offers persistence options. Go channels enable development without external dependencies.
> - **Consequences**: Simplified deployment with Redis as single external dependency, but limited to single Redis instance without clustering complexity

| Queue Option | Pros | Cons | Best For |
|---|---|---|---|
| Redis | Simple operations, minimal setup, persistence options | Single point of failure, limited clustering | Small to medium deployments |
| RabbitMQ | Robust messaging, clustering, delivery guarantees | Complex administration, resource intensive | Enterprise deployments |
| Apache Kafka | High throughput, horizontal scaling, durability | Operational complexity, heavyweight for simple queuing | Event streaming use cases |
| Go Channels | Zero dependencies, type safety, built-in language support | Process-local only, no persistence | Development and local execution |

#### Web Framework Technology Selection

> **Decision: Go net/http with Gorilla Toolkit**
> - **Context**: Need HTTP API for workflow management and web UI for operational dashboard with real-time updates
> - **Options Considered**: Go net/http, Gin, Echo, gRPC + HTTP gateway
> - **Decision**: Standard library net/http enhanced with Gorilla mux for routing and WebSocket support
> - **Rationale**: Standard library provides stability and simplicity, Gorilla toolkit adds essential features without framework lock-in, WebSocket support enables real-time UI updates
> - **Consequences**: Minimal dependencies, maximum compatibility, but requires more boilerplate compared to full frameworks

| Framework Option | Pros | Cons | Decision Factor |
|---|---|---|---|
| net/http + Gorilla | Standard library, minimal dependencies, flexibility | More boilerplate, manual middleware setup | Chosen for simplicity |
| Gin | High performance, extensive middleware, JSON binding | Framework dependency, opinionated structure | Good alternative |
| Echo | Minimal, fast, good documentation | Smaller ecosystem than Gin | Suitable but less mature |
| gRPC + Gateway | Type safety, performance, streaming support | Complex setup, HTTP/2 requirements | Overkill for this use case |

#### Configuration Management Technology Selection

> **Decision: YAML Configuration with Environment Variable Override**
> - **Context**: Need flexible configuration for different deployment environments with secrets management and operational overrides
> - **Options Considered**: YAML files, JSON files, Environment variables only, HashiCorp Consul
> - **Decision**: YAML configuration files with environment variable override capability
> - **Rationale**: YAML provides human-readable configuration with comments, environment variables enable container-friendly overrides without file modifications, combination supports both development convenience and production security
> - **Consequences**: Easy development setup and production deployment flexibility, but requires careful handling of sensitive values

| Config Option | Pros | Cons | Use Case |
|---|---|---|---|
| YAML + ENV | Human readable, comments, container friendly | Manual merge logic, no validation | Primary choice |
| JSON only | Structured, validated, language agnostic | No comments, not human friendly | API communication |
| ENV variables only | Container native, secure | Flat namespace, difficult complex config | Container overrides only |
| External config service | Centralized, audit trail, dynamic updates | Additional dependency, complexity | Enterprise environments |

#### Logging and Monitoring Technology Selection

> **Decision: Structured Logging with Optional Metrics Export**
> - **Context**: Need observability for debugging workflow execution issues and monitoring system performance
> - **Options Considered**: Standard log package, Logrus, Zap, Prometheus metrics, OpenTelemetry
> - **Decision**: Structured JSON logging with configurable levels and optional Prometheus metrics endpoint
> - **Rationale**: Structured logging enables automated log analysis, JSON format integrates with log aggregation systems, Prometheus metrics provide operational visibility without vendor lock-in
> - **Consequences**: Excellent debugging capabilities and monitoring integration, but requires log aggregation infrastructure for production use

| Logging Option | Pros | Cons | Selection |
|---|---|---|---|
| JSON structured | Machine parseable, searchable, standardized | Not human readable in raw form | Chosen for production |
| Text with levels | Human readable, simple, familiar | Difficult to parse, limited structure | Development option |
| Binary formats | Compact, fast, typed | Vendor specific, tooling requirements | Not suitable |

The technology stack emphasizes **operational simplicity** while providing production-ready capabilities. Each component has fallback options enabling development without external dependencies while supporting distributed deployment when needed.

⚠️ **Pitfall: Over-Engineering Infrastructure**
A common mistake is selecting complex technologies like Kubernetes operators, service mesh, or distributed databases before understanding actual requirements. This orchestrator design prioritizes proven, simple technologies that can run effectively on a single server or scale to moderate distributed deployments. Complex infrastructure can be added later when specific scaling or availability requirements emerge, but starting simple enables faster development and easier debugging.

### Implementation Guidance

#### Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|---|---|---|
| Database | SQLite with file storage | PostgreSQL with connection pooling |
| Message Queue | Go channels with goroutines | Redis with pub/sub and lists |
| Web Framework | net/http with basic routing | Gorilla mux + WebSocket + middleware |
| Configuration | YAML files with gopkg.in/yaml.v3 | Viper with environment override |
| Logging | Standard log package | Structured logging with logrus/zap |
| Testing | Go testing package | Testify with mocks and assertions |
| Database Migrations | SQL files with custom runner | golang-migrate library |
| HTTP Client | net/http.Client | Retryable HTTP with backoff |

#### Recommended File Structure

```
workflow-orchestrator/
├── cmd/
│   ├── scheduler/
│   │   └── main.go              # Entry point: config loading + component startup
│   ├── worker/
│   │   └── main.go              # Worker process: connect to queue + execute tasks
│   └── webui/
│       └── main.go              # Web server: API handlers + static file serving
├── internal/
│   ├── config/
│   │   ├── config.go            # Config struct + Load function
│   │   └── config_test.go       # Configuration parsing tests
│   ├── dag/
│   │   ├── models.go            # DAG, Task, OperatorConfig structs
│   │   ├── parser.go            # YAML parsing + file discovery
│   │   ├── validator.go         # ValidateDAG with cycle detection
│   │   └── registry.go          # In-memory DAG storage
│   ├── storage/
│   │   ├── database.go          # Database + NewDatabase + RunMigrations
│   │   └── models.go            # Database entity structs
│   └── queue/
│       ├── queue.go             # Queue interface definition
│       ├── redis.go             # Redis implementation
│       └── memory.go            # In-memory implementation for testing
```

#### Infrastructure Starter Code

**Configuration Management (`internal/config/config.go`)**:
```go
package config

import (
    "fmt"
    "os"
    "time"
    "gopkg.in/yaml.v3"
)

// Config represents the complete application configuration
type Config struct {
    Server   ServerConfig   `yaml:"server"`
    Database DatabaseConfig `yaml:"database"`
    Redis    RedisConfig    `yaml:"redis"`
    DAGs     DAGConfig      `yaml:"dags"`
    Logging  LoggingConfig  `yaml:"logging"`
}

type ServerConfig struct {
    Host         string        `yaml:"host"`
    Port         int           `yaml:"port"`
    ReadTimeout  time.Duration `yaml:"read_timeout"`
    WriteTimeout time.Duration `yaml:"write_timeout"`
}

type DatabaseConfig struct {
    Driver   string `yaml:"driver"`
    Host     string `yaml:"host"`
    Port     int    `yaml:"port"`
    Database string `yaml:"database"`
    Username string `yaml:"username"`
    Password string `yaml:"password"`
    SSLMode  string `yaml:"ssl_mode"`
}

type RedisConfig struct {
    Host     string `yaml:"host"`
    Port     int    `yaml:"port"`
    Password string `yaml:"password"`
    DB       int    `yaml:"db"`
}

type DAGConfig struct {
    Directory      string        `yaml:"directory"`
    ScanInterval   time.Duration `yaml:"scan_interval"`
    FilePattern    string        `yaml:"file_pattern"`
    ReloadOnChange bool          `yaml:"reload_on_change"`
}

type LoggingConfig struct {
    Level  string `yaml:"level"`
    Format string `yaml:"format"`
}

// Load reads configuration from file with environment variable overrides
func Load(configPath string) (*Config, error) {
    // Read YAML file
    data, err := os.ReadFile(configPath)
    if err != nil {
        return nil, fmt.Errorf("reading config file: %w", err)
    }
    
    var config Config
    if err := yaml.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("parsing config YAML: %w", err)
    }
    
    // Apply environment variable overrides
    applyEnvOverrides(&config)
    
    return &config, nil
}

func applyEnvOverrides(config *Config) {
    if host := os.Getenv("SERVER_HOST"); host != "" {
        config.Server.Host = host
    }
    if port := os.Getenv("SERVER_PORT"); port != "" {
        // Parse port string to int (implementation details)
    }
    // Add other environment overrides as needed
}
```

**Database Connection (`internal/storage/database.go`)**:
```go
package storage

import (
    "database/sql"
    "fmt"
    "path/filepath"
    "time"
    
    _ "github.com/lib/pq"           // PostgreSQL driver
    _ "github.com/mattn/go-sqlite3" // SQLite driver
)

type Database struct {
    *sql.DB
}

func NewDatabase(config DatabaseConfig) (*Database, error) {
    var dsn string
    
    switch config.Driver {
    case "postgres":
        dsn = fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
            config.Host, config.Port, config.Username, config.Password, 
            config.Database, config.SSLMode)
    case "sqlite3":
        dsn = config.Database // File path for SQLite
    default:
        return nil, fmt.Errorf("unsupported database driver: %s", config.Driver)
    }
    
    db, err := sql.Open(config.Driver, dsn)
    if err != nil {
        return nil, fmt.Errorf("opening database connection: %w", err)
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)
    
    // Test connection
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("database ping failed: %w", err)
    }
    
    return &Database{db}, nil
}

func (d *Database) RunMigrations(migrationsPath string) error {
    // Read migration files from directory
    files, err := filepath.Glob(filepath.Join(migrationsPath, "*.sql"))
    if err != nil {
        return fmt.Errorf("reading migration files: %w", err)
    }
    
    // Execute each migration file in order
    for _, file := range files {
        if err := d.executeMigrationFile(file); err != nil {
            return fmt.Errorf("executing migration %s: %w", file, err)
        }
    }
    
    return nil
}

func (d *Database) executeMigrationFile(filepath string) error {
    // Implementation: read file content and execute SQL statements
    // Handle transaction boundaries and error recovery
    return nil // TODO: Implement migration execution
}
```

**Redis Queue Client (`internal/queue/redis.go`)**:
```go
package queue

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    "github.com/go-redis/redis/v8"
)

type RedisClient struct {
    *redis.Client
}

func NewRedisClient(config RedisConfig) (*RedisClient, error) {
    client := redis.NewClient(&redis.Options{
        Addr:     fmt.Sprintf("%s:%d", config.Host, config.Port),
        Password: config.Password,
        DB:       config.DB,
    })
    
    // Test connection
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := client.Ping(ctx).Err(); err != nil {
        return nil, fmt.Errorf("redis connection failed: %w", err)
    }
    
    return &RedisClient{client}, nil
}

func (r *RedisClient) EnqueueTask(queueName string, task interface{}) error {
    ctx := context.Background()
    
    // Serialize task to JSON
    data, err := json.Marshal(task)
    if err != nil {
        return fmt.Errorf("serializing task: %w", err)
    }
    
    // Push to Redis list (FIFO queue)
    return r.LPush(ctx, queueName, data).Err()
}

func (r *RedisClient) DequeueTask(queueName string, timeout time.Duration) ([]byte, error) {
    ctx := context.Background()
    
    // Blocking pop from Redis list
    result := r.BRPop(ctx, timeout, queueName)
    if result.Err() == redis.Nil {
        return nil, nil // No task available
    }
    if result.Err() != nil {
        return nil, result.Err()
    }
    
    // Return task data (second element after queue name)
    values := result.Val()
    if len(values) < 2 {
        return nil, fmt.Errorf("invalid Redis response format")
    }
    
    return []byte(values[1]), nil
}
```

#### Core Logic Skeleton Code

**DAG Definition Models (`internal/dag/models.go`)**:
```go
package dag

import (
    "time"
)

// DAG represents a complete workflow definition
type DAG struct {
    ID          string            `yaml:"id" json:"id"`
    Description string            `yaml:"description" json:"description"`
    Schedule    string            `yaml:"schedule" json:"schedule"` // Cron expression
    Tags        []string          `yaml:"tags" json:"tags"`
    Parameters  map[string]string `yaml:"parameters" json:"parameters"`
    Tasks       []Task            `yaml:"tasks" json:"tasks"`
    
    // Internal fields populated during parsing
    taskIndex map[string]*Task // Task lookup by ID
}

// Task represents an individual unit of work within a DAG
type Task struct {
    ID           string            `yaml:"id" json:"id"`
    Type         string            `yaml:"type" json:"type"` // python, bash, sql
    Operator     OperatorConfig    `yaml:"operator" json:"operator"`
    Dependencies []string          `yaml:"dependencies" json:"dependencies"`
    Retries      int               `yaml:"retries" json:"retries"`
    Timeout      time.Duration     `yaml:"timeout" json:"timeout"`
    Parameters   map[string]string `yaml:"parameters" json:"parameters"`
}

// OperatorConfig contains type-specific execution configuration
type OperatorConfig struct {
    PythonCommand string            `yaml:"python_command,omitempty" json:"python_command,omitempty"`
    BashCommand   string            `yaml:"bash_command,omitempty" json:"bash_command,omitempty"`
    SQLQuery      string            `yaml:"sql_query,omitempty" json:"sql_query,omitempty"`
    Environment   map[string]string `yaml:"environment,omitempty" json:"environment,omitempty"`
}

// ValidateDAG checks DAG structure and detects cycles
func (d *DAG) ValidateDAG() error {
    // TODO 1: Build task index map for O(1) lookup by task ID
    // TODO 2: Validate all task dependencies reference existing tasks
    // TODO 3: Detect cycles using topological sort or DFS
    // TODO 4: Check for orphaned tasks with no upstream dependencies
    // TODO 5: Validate operator configurations based on task type
    // TODO 6: Ensure task IDs are unique within the DAG
    return nil
}

// GetTaskByID retrieves a task by its identifier
func (d *DAG) GetTaskByID(taskID string) (*Task, bool) {
    // TODO 1: Check if taskIndex map is populated
    // TODO 2: Return task from index map with existence flag
    return nil, false
}

// GetUpstreamTasks returns all tasks this task depends on
func (d *DAG) GetUpstreamTasks(taskID string) ([]*Task, error) {
    // TODO 1: Find target task by ID
    // TODO 2: Iterate through task's Dependencies slice
    // TODO 3: Look up each dependency task by ID
    // TODO 4: Return slice of upstream task pointers
    return nil, nil
}

// GetDownstreamTasks returns all tasks that depend on this task
func (d *DAG) GetDownstreamTasks(taskID string) ([]*Task, error) {
    // TODO 1: Iterate through all tasks in DAG
    // TODO 2: Check if target task ID appears in each task's Dependencies
    // TODO 3: Collect tasks that have target as dependency
    // TODO 4: Return slice of downstream task pointers
    return nil, nil
}
```

#### Language-Specific Hints

- **YAML Parsing**: Use `gopkg.in/yaml.v3` for better error reporting and struct tag support compared to the standard library
- **Database Drivers**: Import PostgreSQL driver `github.com/lib/pq` and SQLite driver `github.com/mattn/go-sqlite3` as blank imports
- **Goroutine Management**: Use `sync.WaitGroup` to coordinate worker goroutines and `context.Context` for cancellation
- **JSON Handling**: Standard library `encoding/json` handles struct serialization for API responses and Redis message payloads
- **File Watching**: Consider `github.com/fsnotify/fsnotify` for filesystem change notifications in DAG file monitoring
- **HTTP Routing**: `github.com/gorilla/mux` provides path variables and method routing beyond standard library capabilities
- **Testing**: Use `testing.TB` interface for helper functions that work in both tests and benchmarks

#### Milestone Checkpoint

After implementing the high-level architecture:

1. **Configuration Loading Test**:
   ```bash
   # Create example config.yaml and test loading
   go run cmd/scheduler/main.go --config=examples/config.yaml --dry-run
   # Should output: "Configuration loaded successfully: Server listening on :8080"
   ```

2. **Database Connection Verification**:
   ```bash
   # Test database connectivity and migration
   go test ./internal/storage/... -v
   # Should show: "Database connection established" and "Migrations applied successfully"
   ```

3. **Component Integration Test**:
   ```bash
   # Run integration test that starts all components
   go test ./internal/... -tags=integration
   # Should demonstrate: DAG parsing → Database storage → Queue messaging
   ```

**Expected Behavior**: All major components should instantiate successfully with their dependencies. The system should demonstrate basic connectivity between database, message queue, and configuration loading without executing actual workflows.

**Troubleshooting Signs**:
- **"Database connection refused"**: Check PostgreSQL/SQLite configuration and ensure database server is running
- **"Redis connection timeout"**: Verify Redis server availability and network connectivity
- **"Config file not found"**: Ensure YAML configuration file exists and has correct permissions
- **"Port already in use"**: Check if web server port conflicts with existing services


## Data Model

> **Milestone(s):** DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring (foundational data structures used across all milestones)

The data model forms the foundational layer of our workflow orchestration system, defining how workflows, tasks, and execution state are represented, stored, and manipulated throughout the system lifecycle. Think of the data model as the **blueprint language** for our orchestration system - just as architectural blueprints define the structure of a building with precise specifications for rooms, connections, and materials, our data model defines the precise structure of workflows with their tasks, dependencies, and execution metadata.

![Data Model Relationships](./diagrams/data-model.svg)

This data model must serve multiple masters simultaneously: the DAG parser needs to transform configuration files into structured representations, the scheduler requires temporal metadata for triggering runs, the executor needs dependency information for task ordering, and the monitoring system demands detailed state tracking for observability. The challenge lies in designing entities that are rich enough to support all these use cases while remaining simple enough to reason about and maintain.

### Core Entities

The workflow orchestration system revolves around five primary entities that capture the complete lifecycle from workflow definition to task execution. These entities form a hierarchical relationship where abstract workflow definitions cascade down through concrete execution instances to individual task runs.

#### DAG Entity

The `DAG` entity represents the **template** or **blueprint** for a workflow - it defines what should happen but not when it happens. Think of it as a recipe that can be executed many times with different ingredients (parameters) and at different times (scheduled runs).

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for the DAG, used as primary key and referenced in scheduling |
| Description | string | Human-readable description explaining the workflow's business purpose |
| Schedule | string | Cron expression defining when this DAG should automatically trigger |
| Tags | []string | Classification labels for organizing and filtering DAGs in the UI |
| Parameters | map[string]string | Default parameter values that can be overridden at trigger time |
| Tasks | []Task | Complete list of tasks that comprise this workflow |

The `DAG` entity serves as the **immutable definition** that gets versioned and stored. When the scheduler needs to create a new run, it uses this template to instantiate a `DAGRun`. The parameters map allows for workflow reusability - the same DAG can process different datasets or target different environments by overriding parameter values at runtime.

#### Task Entity

The `Task` entity defines an individual unit of work within a DAG. Each task represents a single operation that can be executed independently once its dependencies are satisfied. Tasks are **stateless definitions** - they specify what to do but maintain no information about execution state.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier within the DAG scope, used for dependency references |
| Type | string | Operator type determining execution method (python, bash, sql) |
| Operator | OperatorConfig | Type-specific configuration for the task execution |
| Dependencies | []string | List of upstream task IDs that must complete before this task runs |
| Retries | int | Maximum number of retry attempts if the task fails |
| Timeout | time.Duration | Maximum execution time before the task is killed |
| Parameters | map[string]string | Task-specific parameters merged with DAG parameters at runtime |

The `Dependencies` field creates the **edges** in our directed acyclic graph, while the task itself represents a **node**. The separation between `Type` and `Operator` allows for extensible task types while maintaining a consistent interface - new operator types can be added without changing the core `Task` structure.

#### OperatorConfig Entity

The `OperatorConfig` entity contains the type-specific configuration needed to execute different kinds of tasks. This design follows the **union pattern** where different fields are populated based on the task type.

| Field | Type | Description |
|-------|------|-------------|
| PythonCommand | string | Python script path or inline code for python tasks |
| BashCommand | string | Shell command to execute for bash tasks |
| SQLQuery | string | SQL statement to run for sql tasks |
| Environment | map[string]string | Environment variables to set during task execution |

This approach allows us to add new operator types without breaking existing configurations. The `Environment` field is shared across all operator types, providing a consistent way to inject runtime configuration like database connection strings or API keys.

#### DAGRun Entity

The `DAGRun` entity represents a **specific execution instance** of a DAG - it captures when the workflow was triggered, with what parameters, and tracks the overall execution state. Think of it as the **work order** created from the DAG template.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for this specific DAG execution instance |
| DAGID | string | Foreign key reference to the DAG template being executed |
| ExecutionDate | time.Time | Logical date/time this run represents (not necessarily when it started) |
| StartDate | time.Time | Actual timestamp when execution began |
| EndDate | *time.Time | Actual timestamp when execution completed (null while running) |
| State | DAGRunState | Current execution state of the entire workflow |
| Parameters | map[string]string | Resolved parameter values for this run (DAG defaults + overrides) |
| ExternalTriggerID | *string | Optional identifier if this run was triggered by external event |

The distinction between `ExecutionDate` and `StartDate` is crucial for **data pipeline semantics**. The execution date represents the logical time period this workflow processes (e.g., "process data for 2024-01-15"), while the start date tracks when the actual computation began. This separation enables **backfilling** historical periods and maintains data lineage even when workflows run late.

#### TaskInstance Entity

The `TaskInstance` entity represents the **actual execution** of a specific task within a specific DAG run. This is where the rubber meets the road - task instances track the concrete work being performed, including attempts, logs, and results.

| Field | Type | Description |
|-------|------|-------------|
| ID | string | Unique identifier for this task execution instance |
| DAGRunID | string | Foreign key to the DAG run this task belongs to |
| TaskID | string | Reference to the task definition within the DAG |
| State | TaskInstanceState | Current execution state (queued, running, success, failed, etc.) |
| StartDate | *time.Time | When this task instance began executing |
| EndDate | *time.Time | When this task instance finished executing |
| Duration | *time.Duration | Total execution time (calculated from start/end dates) |
| TryNumber | int | Current attempt number (1 for first attempt, 2+ for retries) |
| MaxTries | int | Maximum retry attempts allowed (copied from task definition) |
| Hostname | *string | Worker node that executed this task |
| LogPath | *string | File path or URL to execution logs |
| ExitCode | *int | Process exit code for completed tasks |
| XComValues | map[string]interface{} | Cross-task communication data produced by this task |

The `TaskInstance` maintains the complete **audit trail** for task execution. The `TryNumber` field enables retry tracking - each retry attempt creates a new record with an incremented try number, preserving the history of all attempts. The `XComValues` field stores small data payloads that tasks can pass to their downstream dependencies.

### State Enumerations

State management is the **central nervous system** of workflow orchestration. States track the lifecycle of both DAG runs and individual task instances, providing the foundation for dependency resolution, failure handling, and monitoring. The state model must be precise enough to capture all meaningful execution phases while remaining simple enough to avoid race conditions and inconsistent transitions.

#### DAG Run States

DAG run states track the **aggregate status** of an entire workflow execution. The state transitions follow a strict progression from initial trigger through completion or failure.

| State | Description | Entry Condition | Exit Condition |
|-------|-------------|----------------|----------------|
| queued | DAG run created but not yet started | Scheduler creates run from trigger | Scheduler begins processing tasks |
| running | At least one task is actively executing | First task starts running | All tasks reach terminal state |
| success | All tasks completed successfully | Last task succeeds with no failures | N/A (terminal state) |
| failed | At least one task failed permanently | Any task exhausts retries and fails | Manual clear or retry trigger |
| up_for_retry | Temporary failure with retry attempts remaining | Task fails but has retries available | Retry timer expires and task retries |

The `up_for_retry` state deserves special attention - it represents a **suspended animation** where the DAG run is neither completely failed nor actively running. This state prevents the scheduler from marking the run as failed while retry timers are counting down, maintaining the distinction between temporary setbacks and permanent failures.

> **Key Insight**: DAG run states are **derived** from task instance states rather than independently managed. This ensures consistency - the DAG run state always accurately reflects the aggregate status of its constituent tasks.

#### Task Instance States

Task instance states capture the **fine-grained execution lifecycle** of individual tasks. These states must support parallel execution, retry logic, and dependency resolution while providing clear signals to both the scheduler and monitoring systems.

| Current State | Event | Next State | Action Taken |
|---------------|-------|------------|--------------|
| none | Task dependencies satisfied | queued | Task added to execution queue |
| queued | Executor picks up task | running | Worker process starts task execution |
| running | Task completes successfully | success | Downstream dependencies evaluated |
| running | Task fails with retries remaining | up_for_retry | Retry timer started |
| running | Task fails with no retries | failed | Downstream tasks marked as upstream_failed |
| running | Task exceeds timeout | failed | Worker process killed, cleanup performed |
| up_for_retry | Retry timer expires | queued | Task re-added to execution queue |
| success | Manual clear operation | none | Task reset for re-execution |
| failed | Manual clear operation | none | Task reset for re-execution |
| * | DAG run cancelled | skipped | Task execution bypassed |

The state machine includes several **derived states** that help with dependency resolution:

| Derived State | Condition | Purpose |
|---------------|-----------|---------|
| upstream_failed | Upstream dependency failed | Skip task execution while preserving failure signal |
| skipped | Task bypassed due to branching logic | Indicate intentional non-execution |
| removed | Task deleted from DAG definition | Handle DAG evolution and task removal |

The transition from `running` to either `up_for_retry` or `failed` depends on the task's retry configuration. This decision point is **critical** for workflow reliability - premature failure marking can stop entire workflows, while excessive retrying can mask systematic problems.

> **Decision: Task State Granularity**
> - **Context**: Need to balance detailed state tracking with system complexity
> - **Options Considered**: 
>   1. Simple 3-state model (pending, running, done)
>   2. Detailed 9-state model with retry and failure distinctions
>   3. Hierarchical states with sub-states
> - **Decision**: Detailed 9-state model with explicit retry states
> - **Rationale**: Workflow orchestration requires precise failure handling and retry logic. Simple states hide critical information needed for dependency resolution and monitoring. Hierarchical states add complexity without clear benefits.
> - **Consequences**: More complex state management code but clearer operational semantics and better debugging capabilities.

### Database Schema Design

The database schema translates our logical data model into physical storage structures that support the access patterns required by workflow orchestration. The schema must optimize for both **write-heavy workloads** (task state updates) and **read-heavy workloads** (dependency resolution and monitoring queries).

#### Table Structures

The core tables directly mirror our entity model with additional fields for database management and performance optimization.

**dags table:**
```sql
CREATE TABLE dags (
    id VARCHAR(255) PRIMARY KEY,
    description TEXT NOT NULL,
    schedule VARCHAR(100) NOT NULL,
    tags JSON,
    parameters JSON,
    definition JSON NOT NULL,  -- Complete DAG definition including tasks
    file_path VARCHAR(500),    -- Source file path for reloading
    file_hash VARCHAR(64),     -- Content hash for change detection
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

The `definition` field stores the **complete DAG structure** as JSON, enabling atomic DAG updates and version consistency. The `file_hash` field supports **change detection** - when DAG files are modified, the parser can quickly identify which DAGs need reloading by comparing file hashes.

**dag_runs table:**
```sql
CREATE TABLE dag_runs (
    id VARCHAR(255) PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP NULL,
    state VARCHAR(50) NOT NULL,
    parameters JSON,
    external_trigger_id VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (dag_id) REFERENCES dags(id) ON DELETE CASCADE,
    UNIQUE KEY unique_dag_execution (dag_id, execution_date)
);
```

The `unique_dag_execution` constraint prevents **duplicate runs** for the same DAG and execution date, which is essential for data pipeline consistency. The foreign key cascade ensures that deleting a DAG automatically cleans up all its runs.

**task_instances table:**
```sql
CREATE TABLE task_instances (
    id VARCHAR(255) PRIMARY KEY,
    dag_run_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    state VARCHAR(50) NOT NULL,
    start_date TIMESTAMP NULL,
    end_date TIMESTAMP NULL,
    duration INTEGER NULL,  -- Milliseconds
    try_number INTEGER NOT NULL DEFAULT 1,
    max_tries INTEGER NOT NULL,
    hostname VARCHAR(255) NULL,
    log_path VARCHAR(500) NULL,
    exit_code INTEGER NULL,
    xcom_values JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (dag_run_id) REFERENCES dag_runs(id) ON DELETE CASCADE,
    UNIQUE KEY unique_task_instance (dag_run_id, task_id, try_number)
);
```

The `unique_task_instance` constraint combines DAG run, task ID, and try number to ensure **retry tracking consistency**. Each retry attempt gets its own record, preserving the complete execution history for debugging and analysis.

#### Index Strategy

Database indexes are **critical** for workflow orchestration performance, especially as the number of DAG runs and task instances grows into the millions. The index strategy must support both operational queries (finding ready tasks) and analytical queries (monitoring dashboards).

**Primary Operational Indexes:**
| Index Name | Columns | Purpose |
|------------|---------|---------|
| idx_dag_runs_state | (state, dag_id) | Find running DAG runs for scheduling |
| idx_task_instances_state | (state, dag_run_id) | Find ready tasks for execution |
| idx_task_instances_hostname | (hostname, state) | Worker health monitoring |
| idx_dag_runs_execution_date | (dag_id, execution_date DESC) | Recent run history queries |

**Monitoring and Analytics Indexes:**
| Index Name | Columns | Purpose |
|------------|---------|---------|
| idx_task_instances_duration | (task_id, duration) | Performance trend analysis |
| idx_dag_runs_created_at | (created_at DESC) | Recent activity monitoring |
| idx_task_instances_try_number | (task_id, try_number) | Failure pattern analysis |

The **composite index ordering** is crucial for query performance. For example, `(state, dag_run_id)` supports queries that filter by state first (high selectivity) and then narrow down to specific DAG runs.

> **Decision: JSON vs Relational Storage for Parameters**
> - **Context**: Need to store flexible key-value parameters for DAGs and tasks
> - **Options Considered**:
>   1. JSON columns for flexible schema
>   2. Separate parameter tables with key-value rows
>   3. Serialized binary format
> - **Decision**: JSON columns with database JSON functions
> - **Rationale**: Modern databases provide excellent JSON support with indexing and query capabilities. Separate tables add complexity and join overhead. Binary formats sacrifice debugging capability.
> - **Consequences**: Requires database JSON support but provides flexibility and reasonable query performance. Parameters larger than a few KB may impact performance.

#### Data Retention and Archival

Workflow orchestration systems generate **massive amounts** of historical data over time. A production system might create thousands of task instances per day, leading to millions of records within months. The schema must support efficient data lifecycle management.

**Retention Strategy:**
| Data Type | Retention Period | Archival Method |
|-----------|------------------|-----------------|
| Active DAG runs | Indefinite | Keep in primary tables |
| Completed runs (success) | 90 days | Archive to separate tables |
| Failed runs | 365 days | Archive with extended retention |
| Task instance logs | 30 days | Move to object storage |
| XCom values | 7 days | Purge after downstream consumption |

The archival process uses **date-based partitioning** to enable efficient bulk operations:

```sql
-- Archive old successful runs
INSERT INTO dag_runs_archive 
SELECT * FROM dag_runs 
WHERE state = 'success' AND end_date < DATE_SUB(NOW(), INTERVAL 90 DAY);

DELETE FROM dag_runs 
WHERE state = 'success' AND end_date < DATE_SUB(NOW(), INTERVAL 90 DAY);
```

#### Connection Pooling and Performance

Database connection management is **critical** for workflow orchestration systems due to the high frequency of state updates during task execution. Each task state change triggers a database write, and with hundreds of concurrent tasks, connection exhaustion becomes a real concern.

**Connection Pool Configuration:**
| Parameter | Recommended Value | Rationale |
|-----------|------------------|-----------|
| Max Connections | 50-100 | Balance concurrency with database limits |
| Idle Timeout | 300 seconds | Prevent connection buildup during quiet periods |
| Connection Lifetime | 3600 seconds | Force periodic connection refresh |
| Prepared Statements | Enabled | Reduce parsing overhead for frequent queries |

The database layer implements **connection health checking** to detect and replace stale connections that might occur due to network issues or database maintenance.

⚠️ **Pitfall: Inadequate Transaction Isolation**
Many developers use the default database transaction isolation without considering workflow orchestration requirements. Using `READ UNCOMMITTED` can lead to race conditions where tasks appear ready before their dependencies actually complete. Use `READ COMMITTED` or higher to ensure consistent dependency resolution, accepting the small performance cost for correctness.

⚠️ **Pitfall: Missing Database Constraints**
Forgetting foreign key constraints allows orphaned records that can break dependency resolution. For example, task instances without valid DAG run parents will never be cleaned up and may cause memory leaks in the scheduler. Always define proper referential integrity constraints even if application logic "guarantees" consistency.

⚠️ **Pitfall: Inefficient State Transition Queries**
Writing separate UPDATE statements for each task state change creates unnecessary database round trips. Instead, batch state updates when possible and use database features like `UPDATE ... CASE` for conditional state transitions. This reduces lock contention and improves throughput under high concurrency.

### Implementation Guidance

The data model implementation requires careful attention to database interactions, concurrent access patterns, and performance optimization. The following guidance provides both foundational infrastructure and skeleton code for the core data access layer.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Database | PostgreSQL with pgx driver | PostgreSQL with connection pooling + read replicas |
| Migrations | golang-migrate with SQL files | Liquibase or custom migration framework |
| ORM vs Raw SQL | Raw SQL with squirrel query builder | GORM or sqlx for complex queries |
| JSON Handling | database/sql with JSON columns | JSONB with GIN indexes for complex queries |
| Connection Pool | pgxpool with default settings | Custom pool with circuit breaker patterns |

#### Recommended File Structure

```
internal/
├── models/
│   ├── dag.go              ← DAG and Task entity definitions
│   ├── dag_run.go          ← DAGRun entity and state enums
│   ├── task_instance.go    ← TaskInstance entity and state machine
│   └── states.go           ← State type definitions and constants
├── database/
│   ├── database.go         ← Database connection and configuration
│   ├── migrations/         ← SQL migration files
│   │   ├── 001_initial_schema.up.sql
│   │   ├── 001_initial_schema.down.sql
│   │   └── 002_add_indexes.up.sql
│   └── queries/            ← SQL query definitions
│       ├── dag_queries.sql
│       └── task_queries.sql
├── repository/
│   ├── dag_repository.go    ← DAG CRUD operations
│   ├── dag_run_repository.go ← DAG run CRUD operations
│   └── task_repository.go   ← Task instance CRUD operations
└── testdata/
    ├── sample_dags/         ← Test DAG definitions
    └── test_migrations/     ← Test database setup
```

#### Infrastructure Starter Code

**Database Connection Management (internal/database/database.go):**

```go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "time"

    "github.com/jackc/pgx/v5/pgxpool"
    _ "github.com/jackc/pgx/v5/stdlib"
    "github.com/golang-migrate/migrate/v4"
    "github.com/golang-migrate/migrate/v4/database/postgres"
    _ "github.com/golang-migrate/migrate/v4/source/file"
)

// Database wraps the connection pool with workflow-specific methods
type Database struct {
    pool *pgxpool.Pool
    db   *sql.DB // For migrations
}

// NewDatabase creates a new database connection with proper pooling
func NewDatabase(config DatabaseConfig) (*Database, error) {
    // Build connection string
    connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
        config.Host, config.Port, config.Username, config.Password, 
        config.Database, config.SSLMode)

    // Configure connection pool
    poolConfig, err := pgxpool.ParseConfig(connStr)
    if err != nil {
        return nil, fmt.Errorf("failed to parse connection config: %w", err)
    }

    // Set pool parameters optimized for workflow orchestration
    poolConfig.MaxConns = 50                    // Handle concurrent task updates
    poolConfig.MinConns = 5                     // Maintain baseline connections
    poolConfig.MaxConnLifetime = time.Hour      // Rotate connections periodically
    poolConfig.MaxConnIdleTime = 5 * time.Minute // Close idle connections
    
    // Create connection pool
    pool, err := pgxpool.NewWithConfig(context.Background(), poolConfig)
    if err != nil {
        return nil, fmt.Errorf("failed to create connection pool: %w", err)
    }

    // Test connection
    if err := pool.Ping(context.Background()); err != nil {
        pool.Close()
        return nil, fmt.Errorf("failed to ping database: %w", err)
    }

    // Create standard sql.DB for migrations
    db, err := sql.Open("pgx", connStr)
    if err != nil {
        pool.Close()
        return nil, fmt.Errorf("failed to create sql.DB: %w", err)
    }

    return &Database{pool: pool, db: db}, nil
}

// RunMigrations executes all pending database migrations
func (d *Database) RunMigrations(migrationsPath string) error {
    driver, err := postgres.WithInstance(d.db, &postgres.Config{})
    if err != nil {
        return fmt.Errorf("failed to create migration driver: %w", err)
    }

    m, err := migrate.NewWithDatabaseInstance(
        fmt.Sprintf("file://%s", migrationsPath),
        "postgres", 
        driver,
    )
    if err != nil {
        return fmt.Errorf("failed to create migration instance: %w", err)
    }

    if err := m.Up(); err != nil && err != migrate.ErrNoChange {
        return fmt.Errorf("failed to run migrations: %w", err)
    }

    return nil
}

// Close cleans up database connections
func (d *Database) Close() {
    if d.pool != nil {
        d.pool.Close()
    }
    if d.db != nil {
        d.db.Close()
    }
}

// Pool returns the underlying connection pool for repository use
func (d *Database) Pool() *pgxpool.Pool {
    return d.pool
}
```

**State Type Definitions (internal/models/states.go):**

```go
package models

// DAGRunState represents the overall state of a DAG execution
type DAGRunState string

const (
    DAGRunStateQueued     DAGRunState = "queued"
    DAGRunStateRunning    DAGRunState = "running" 
    DAGRunStateSuccess    DAGRunState = "success"
    DAGRunStateFailed     DAGRunState = "failed"
    DAGRunStateUpForRetry DAGRunState = "up_for_retry"
)

// IsTerminal returns true if the DAG run has reached a final state
func (s DAGRunState) IsTerminal() bool {
    return s == DAGRunStateSuccess || s == DAGRunStateFailed
}

// TaskInstanceState represents the state of an individual task execution
type TaskInstanceState string

const (
    TaskInstanceStateNone           TaskInstanceState = "none"
    TaskInstanceStateQueued         TaskInstanceState = "queued"
    TaskInstanceStateRunning        TaskInstanceState = "running"
    TaskInstanceStateSuccess        TaskInstanceState = "success"
    TaskInstanceStateFailed         TaskInstanceState = "failed"
    TaskInstanceStateUpForRetry     TaskInstanceState = "up_for_retry"
    TaskInstanceStateUpstreamFailed TaskInstanceState = "upstream_failed"
    TaskInstanceStateSkipped        TaskInstanceState = "skipped"
    TaskInstanceStateRemoved        TaskInstanceState = "removed"
)

// IsTerminal returns true if the task instance has reached a final state
func (s TaskInstanceState) IsTerminal() bool {
    switch s {
    case TaskInstanceStateSuccess, TaskInstanceStateFailed, 
         TaskInstanceStateUpstreamFailed, TaskInstanceStateSkipped, 
         TaskInstanceStateRemoved:
        return true
    default:
        return false
    }
}

// CanRetry returns true if the task instance can be retried from this state
func (s TaskInstanceState) CanRetry() bool {
    return s == TaskInstanceStateFailed || s == TaskInstanceStateUpForRetry
}
```

#### Core Logic Skeleton Code

**DAG Entity with Validation (internal/models/dag.go):**

```go
package models

import (
    "fmt"
    "time"
)

// DAG represents a workflow definition with tasks and dependencies
type DAG struct {
    ID          string            `json:"id" db:"id"`
    Description string            `json:"description" db:"description"`
    Schedule    string            `json:"schedule" db:"schedule"`
    Tags        []string          `json:"tags" db:"tags"`
    Parameters  map[string]string `json:"parameters" db:"parameters"`
    Tasks       []Task            `json:"tasks" db:"-"`
    
    // Metadata fields
    FilePath  string    `json:"file_path" db:"file_path"`
    FileHash  string    `json:"file_hash" db:"file_hash"`
    IsActive  bool      `json:"is_active" db:"is_active"`
    CreatedAt time.Time `json:"created_at" db:"created_at"`
    UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
    
    // Internal indexes for fast lookups
    taskIndex map[string]*Task      // Task ID -> Task pointer
    depGraph  map[string][]string   // Task ID -> dependency list
}

// ValidateDAG validates the DAG definition and checks for cycles
func (d *DAG) ValidateDAG() error {
    // TODO 1: Validate DAG ID is not empty and follows naming conventions
    // TODO 2: Validate all tasks have unique IDs within the DAG
    // TODO 3: Build task index map for fast lookups
    // TODO 4: Validate all task dependencies reference valid task IDs
    // TODO 5: Check for cycles using topological sort algorithm
    // TODO 6: Validate cron schedule expression is parseable
    // TODO 7: Ensure at least one task has no dependencies (entry point)
    // TODO 8: Validate task timeout values are positive
    // TODO 9: Check for orphaned tasks with no upstream or downstream connections
    // Hint: Use depth-first search with visited/visiting/visited states for cycle detection
    return fmt.Errorf("validation not implemented")
}

// GetTaskByID retrieves a task by its identifier
func (d *DAG) GetTaskByID(taskID string) (*Task, bool) {
    // TODO 1: Check if task index is built, build it if necessary
    // TODO 2: Look up task in the index map
    // TODO 3: Return task pointer and existence boolean
    return nil, false
}

// GetUpstreamTasks returns all tasks that must complete before the given task
func (d *DAG) GetUpstreamTasks(taskID string) ([]*Task, error) {
    // TODO 1: Find the target task and validate it exists
    // TODO 2: Get the dependency list from the task definition
    // TODO 3: Resolve each dependency ID to a Task pointer
    // TODO 4: Return error if any dependency references invalid task
    return nil, fmt.Errorf("upstream resolution not implemented")
}

// GetDownstreamTasks returns all tasks that depend on the given task
func (d *DAG) GetDownstreamTasks(taskID string) ([]*Task, error) {
    // TODO 1: Iterate through all tasks in the DAG
    // TODO 2: Check each task's Dependencies slice for the target task ID
    // TODO 3: Collect tasks that list the target as a dependency
    // TODO 4: Return the collected downstream tasks
    return nil, fmt.Errorf("downstream resolution not implemented")
}
```

**Task Repository with CRUD Operations (internal/repository/task_repository.go):**

```go
package repository

import (
    "context"
    "database/sql"
    "fmt"
    
    "github.com/jackc/pgx/v5/pgxpool"
    "your-project/internal/models"
)

// TaskInstanceRepository handles database operations for task instances
type TaskInstanceRepository struct {
    pool *pgxpool.Pool
}

// NewTaskInstanceRepository creates a new repository instance
func NewTaskInstanceRepository(pool *pgxpool.Pool) *TaskInstanceRepository {
    return &TaskInstanceRepository{pool: pool}
}

// CreateTaskInstance inserts a new task instance record
func (r *TaskInstanceRepository) CreateTaskInstance(ctx context.Context, ti *models.TaskInstance) error {
    // TODO 1: Prepare INSERT statement with all task instance fields
    // TODO 2: Execute query with parameter binding to prevent SQL injection
    // TODO 3: Handle unique constraint violations gracefully
    // TODO 4: Set created_at and updated_at timestamps
    // TODO 5: Return error with context if insertion fails
    return fmt.Errorf("create task instance not implemented")
}

// UpdateTaskInstanceState updates the state and related fields of a task instance
func (r *TaskInstanceRepository) UpdateTaskInstanceState(ctx context.Context, taskInstanceID string, 
    newState models.TaskInstanceState, hostname *string, exitCode *int) error {
    // TODO 1: Begin transaction to ensure atomic state update
    // TODO 2: Validate state transition is legal (optional but recommended)
    // TODO 3: Update state, hostname, exit_code, and updated_at fields
    // TODO 4: Set end_date if transitioning to terminal state
    // TODO 5: Calculate and store duration if start_date exists
    // TODO 6: Commit transaction or rollback on error
    return fmt.Errorf("update state not implemented")
}

// GetReadyTaskInstances returns task instances that are ready to execute
func (r *TaskInstanceRepository) GetReadyTaskInstances(ctx context.Context, limit int) ([]*models.TaskInstance, error) {
    // TODO 1: Query for task instances in 'queued' state
    // TODO 2: Join with dag_runs to ensure parent run is still active
    // TODO 3: Order by creation time for fair scheduling
    // TODO 4: Apply LIMIT to prevent overwhelming the executor
    // TODO 5: Scan results into TaskInstance structs
    return nil, fmt.Errorf("get ready tasks not implemented")
}
```

#### Language-Specific Hints

**Go Database Best Practices:**
- Use `pgxpool` for connection pooling with concurrent access
- Always pass `context.Context` to database operations for cancellation
- Use `database/sql.NullString` for optional string fields in database structs
- Implement `sql.Scanner` and `driver.Valuer` interfaces for custom types like enums
- Use prepared statements for frequently executed queries
- Handle `sql.ErrNoRows` separately from other database errors

**JSON Handling in PostgreSQL:**
- Use `JSONB` data type for better performance and indexing
- Create GIN indexes on JSONB columns for fast queries: `CREATE INDEX idx_parameters ON dags USING GIN (parameters);`
- Use PostgreSQL JSON functions: `JSON_EXTRACT_PATH_TEXT(parameters, 'key')` for queries
- Validate JSON structure in application code before database insertion

**Enum Type Implementation:**
- Define enums as string constants rather than iota for database compatibility
- Implement `String()` method for enum types to support logging and debugging
- Consider using PostgreSQL enum types for strict database-level validation
- Provide validation functions to check enum values before database operations

#### Milestone Checkpoints

**After implementing core entities:**
1. Run unit tests: `go test ./internal/models/... -v`
2. Expected: All entity validation functions should pass with valid input
3. Expected: Cycle detection should reject DAGs with circular dependencies
4. Manual verification: Create a sample DAG definition and ensure `ValidateDAG()` catches common errors

**After implementing database schema:**
1. Run migrations: `go run cmd/migrate/main.go up`
2. Expected: All tables created with proper indexes and constraints
3. Check schema: `psql -c "\d+ dags"` should show correct column types
4. Insert test data: Sample DAG and DAG run should insert without foreign key errors

**After implementing repositories:**
1. Run integration tests: `go test ./internal/repository/... -v -tags=integration`
2. Expected: CRUD operations should succeed with test database
3. Performance test: Insert 1000 task instances and verify query performance stays under 100ms
4. Concurrent access test: Multiple goroutines updating different task states should not deadlock

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Cycle detected" error on valid DAG | Incorrect cycle detection algorithm | Add debug logging to show visit order | Fix topological sort implementation |
| Task dependencies never resolve | Case sensitivity in task ID matching | Check exact task ID strings in database | Enforce consistent ID casing |
| Database deadlocks during state updates | Updates acquiring locks in different orders | Enable PostgreSQL deadlock logging | Order UPDATE operations by primary key |
| Memory leaks in long-running processes | Missing database connection closure | Monitor connection pool stats | Ensure proper context cancellation |
| JSON parameter queries fail | Invalid JSON stored in database | Validate JSON before insertion | Add JSON schema validation |


## DAG Parser and Validation

> **Milestone(s):** DAG Definition (foundational component for parsing and validating workflow definitions)

The DAG parser and validation component serves as the entry point for all workflow definitions in the orchestration system. Think of this component as a **skilled translator and quality inspector** in a publishing house. Just as a translator converts manuscripts from various languages into a standard format while an inspector checks for plot holes and inconsistencies, the DAG parser converts workflow definitions from human-readable formats into internal data structures while validating that the workflow logic is sound and executable.

This component handles three critical responsibilities: loading DAG definitions from configuration files, parsing task dependencies into an internal graph representation, and validating that the resulting graph is a valid directed acyclic graph with no circular dependencies. The quality of this component directly impacts the reliability of the entire orchestration system, as invalid workflows that slip through validation can cause runtime failures, infinite loops, or resource exhaustion.

![DAG Validation and Registration Flow](./diagrams/dag-validation-flow.svg)

### DAG Definition Format

The workflow orchestrator uses a YAML-based domain-specific language (DSL) for defining workflows. This format strikes a balance between human readability and machine parseability, allowing data engineers to express complex workflow dependencies without requiring programming knowledge. The YAML format leverages familiar declarative syntax while providing enough structure to support advanced features like parameterization, retry policies, and timeout configurations.

A complete DAG definition includes metadata about the workflow itself, scheduling information, default task parameters, and an array of task definitions with their dependencies. The parser expects specific top-level keys and validates the structure before attempting to construct the internal graph representation.

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the DAG, used in scheduling and monitoring |
| `description` | string | No | Human-readable description of the workflow purpose |
| `schedule` | string | No | Cron expression or trigger type (e.g., "@daily", "0 2 * * *") |
| `tags` | array[string] | No | Classification tags for grouping and filtering DAGs |
| `parameters` | map[string]string | No | Default parameters available to all tasks in the workflow |
| `default_task_config` | object | No | Default retry, timeout, and environment settings for tasks |
| `tasks` | array[Task] | Yes | List of task definitions that comprise the workflow |

The task definition format provides flexibility for different operator types while maintaining a consistent interface. Each task specifies its execution type, command or query, dependency relationships, and runtime configuration. The parser validates that all referenced dependencies exist and that the resulting graph forms a valid DAG.

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the task within the DAG |
| `type` | string | Yes | Operator type: "python", "bash", or "sql" |
| `operator` | OperatorConfig | Yes | Type-specific configuration for task execution |
| `dependencies` | array[string] | No | List of upstream task IDs that must complete successfully |
| `retries` | integer | No | Maximum retry attempts on failure (default: 0) |
| `timeout` | string | No | Maximum execution duration (e.g., "30m", "2h") |
| `parameters` | map[string]string | No | Task-specific parameters that override DAG defaults |

The operator configuration varies by task type but follows a consistent pattern. Python tasks specify the command to execute and optional virtual environment settings. Bash tasks define shell commands with working directory options. SQL tasks include the query text and database connection parameters.

| Operator Type | Configuration Fields | Example Values |
|---------------|---------------------|----------------|
| Python | `python_command`, `environment`, `working_directory` | `"python scripts/process_data.py"` |
| Bash | `bash_command`, `working_directory`, `shell` | `"./scripts/backup.sh"` |
| SQL | `sql_query`, `database_connection`, `parameters` | `"SELECT COUNT(*) FROM users"` |

Here's an example DAG definition that demonstrates the YAML format and common workflow patterns:

```yaml
id: "customer_analytics_pipeline"
description: "Daily customer analytics data processing workflow"
schedule: "0 2 * * *"  # Run at 2 AM daily
tags: ["analytics", "customer", "daily"]
parameters:
  data_date: "{{ execution_date }}"
  output_bucket: "analytics-results"

default_task_config:
  retries: 2
  timeout: "1h"
  environment:
    PYTHONPATH: "/opt/analytics"

tasks:
  - id: "extract_customer_data"
    type: "sql"
    operator:
      sql_query: |
        SELECT customer_id, purchase_amount, purchase_date
        FROM transactions
        WHERE DATE(created_at) = '{{ params.data_date }}'
      database_connection: "analytics_db"
    
  - id: "transform_data"
    type: "python"
    dependencies: ["extract_customer_data"]
    operator:
      python_command: "python transform_customer_data.py"
      environment:
        INPUT_TABLE: "raw_customer_data"
        OUTPUT_TABLE: "processed_customer_data"
    
  - id: "generate_report"
    type: "python"
    dependencies: ["transform_data"]
    operator:
      python_command: "python generate_analytics_report.py"
      working_directory: "/opt/analytics/reports"
    timeout: "30m"
    
  - id: "upload_results"
    type: "bash"
    dependencies: ["generate_report"]
    operator:
      bash_command: "aws s3 sync ./output/ s3://{{ params.output_bucket }}/"
```

The parser handles parameter templating using double-brace syntax similar to Jinja2 templates. Common template variables include `{{ execution_date }}` for the scheduled run date, `{{ params.parameter_name }}` for DAG or task parameters, and `{{ task_instance.try_number }}` for retry-specific logic. The templating engine resolves these variables at runtime based on the execution context.

> **Key Design Insight**: The YAML format prioritizes readability and version control compatibility over programmatic flexibility. While this limits dynamic workflow generation, it ensures that workflows are auditable, reviewable, and maintainable by teams without deep programming expertise.

The parser implements strict validation rules to catch common configuration errors early. Missing required fields generate clear error messages indicating the expected structure. Invalid cron expressions are validated using a cron parser library. Task dependency references are checked against the complete task list to prevent orphaned dependencies. Operator configurations are validated against type-specific schemas to ensure all required fields are present.

### Cycle Detection Algorithm

The cycle detection algorithm forms the core of DAG validation, ensuring that workflow definitions represent valid directed acyclic graphs without circular dependencies. The algorithm uses topological sorting with Kahn's algorithm, which provides both cycle detection and a valid execution order for the tasks.

Think of cycle detection as a **prerequisite checker for university courses**. Just as a computer science program cannot require that Operating Systems is a prerequisite for Data Structures while Data Structures is a prerequisite for Operating Systems, a workflow cannot have task A depending on task B while task B depends on task A. The algorithm systematically checks that every task can eventually run by following the dependency chain to completion.

The topological sort algorithm works by maintaining an in-degree count for each task (the number of upstream dependencies) and processing tasks in order of their dependency depth. Tasks with no dependencies can run immediately, and as tasks complete, their downstream dependencies have their in-degree decremented. If the algorithm processes all tasks successfully, the graph is acyclic. If tasks remain unprocessed after the algorithm completes, those tasks are part of a cycle.

| Algorithm Step | Action Taken | Data Structure Updated | Purpose |
|----------------|--------------|------------------------|---------|
| 1. Initialize | Calculate in-degree for each task | `inDegree map[string]int` | Count upstream dependencies |
| 2. Seed Queue | Add tasks with in-degree 0 to queue | `queue []string` | Identify tasks ready to run |
| 3. Process Task | Remove task from queue, add to result | `result []string` | Build topological order |
| 4. Update Dependencies | Decrement in-degree of downstream tasks | `inDegree map[string]int` | Mark dependency as satisfied |
| 5. Queue Ready Tasks | Add tasks with in-degree 0 to queue | `queue []string` | Find newly ready tasks |
| 6. Repeat | Continue until queue is empty | All structures | Process entire graph |
| 7. Validate | Check if all tasks were processed | `result []string` | Detect remaining cycles |

The algorithm maintains several data structures during execution. The adjacency list represents the graph structure with each task mapping to its downstream dependencies. The in-degree map tracks how many upstream dependencies each task has remaining. The processing queue holds tasks that are ready to run (in-degree of zero), and the result list accumulates the topological ordering.

Here's the detailed algorithm flow:

1. **Graph Construction Phase**: Parse the DAG definition and build an adjacency list representation where each task maps to its list of downstream dependents. Simultaneously calculate the in-degree for each task by counting how many upstream dependencies it declares.

2. **Initial Queue Population**: Identify all tasks with an in-degree of zero (no upstream dependencies) and add them to the processing queue. These tasks can run immediately when the workflow starts.

3. **Iterative Processing**: While the queue is not empty, remove a task from the front of the queue and add it to the result list. For each downstream task of the current task, decrement its in-degree count to reflect that one dependency has been satisfied.

4. **Dependency Resolution**: After decrementing in-degrees, check if any downstream tasks now have an in-degree of zero. Add these newly ready tasks to the processing queue for the next iteration.

5. **Cycle Detection**: After the queue becomes empty, compare the length of the result list with the total number of tasks in the DAG. If they match, the graph is acyclic and the result list contains a valid topological ordering. If the result list is shorter, the remaining tasks are part of one or more cycles.

6. **Cycle Reporting**: For tasks not included in the result list, perform a depth-first search to identify the specific cycles and report them in user-friendly error messages.

The algorithm handles several edge cases that commonly occur in workflow definitions. Self-referencing tasks (a task that depends on itself) are detected during the dependency parsing phase. Disconnected subgraphs are handled correctly as long as each subgraph is individually acyclic. Tasks with no dependencies and no dependents (isolated tasks) are processed normally and appear early in the topological ordering.

> **Performance Consideration**: The topological sort algorithm has O(V + E) time complexity where V is the number of tasks and E is the number of dependencies. This scales well for typical workflow sizes (hundreds of tasks) but may require optimization for very large workflows with thousands of interdependent tasks.

The algorithm provides detailed error reporting when cycles are detected. Instead of simply reporting "cycle detected," the validation logic identifies the specific tasks involved in each cycle and presents them as a chain of dependencies. For example: "Cycle detected: task_a → task_b → task_c → task_a". This level of detail helps workflow authors quickly identify and fix dependency issues.

![Sample DAG Dependency Graph](./diagrams/dag-dependency-graph.svg)

The cycle detection integrates with the broader validation pipeline to provide comprehensive feedback. Tasks with invalid operator configurations are flagged even if the dependency graph is valid. Missing task dependencies are reported with suggestions for similar task names (using string distance algorithms). The validator accumulates all errors before reporting them, allowing workflow authors to fix multiple issues simultaneously rather than discovering them one at a time.

### Architecture Decision: DAG Definition Language

The choice of workflow definition language significantly impacts both the developer experience and the system's capabilities. This decision affects how workflows are authored, versioned, reviewed, and maintained throughout their lifecycle.

> **Decision: YAML-Based Configuration DSL**
> - **Context**: Workflow definitions need to be human-readable, version-controllable, auditable, and accessible to non-programmers while supporting complex dependency graphs and parameterization.
> - **Options Considered**: YAML configuration files, Python-based DSL (like Airflow), Custom domain-specific language with parser
> - **Decision**: YAML-based configuration with template support
> - **Rationale**: YAML provides the optimal balance of readability, tooling support, and declarative clarity while avoiding the complexity and security risks of executable code in workflow definitions.
> - **Consequences**: Enables broader team adoption and easier code review but limits dynamic workflow generation and requires careful template design for parameterization.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| YAML Config | Human-readable, version control friendly, no security risks, familiar syntax, excellent tooling support | Limited dynamic capabilities, requires templating for parameterization, verbose for complex logic | ✅ Yes |
| Python DSL | Full programming language power, dynamic workflow generation, familiar to data engineers, rich ecosystem | Security risks with code execution, harder to review, requires Python knowledge, complex validation | ❌ No |
| Custom Language | Tailored to workflow needs, optimal syntax, compile-time validation, no security concerns | High development cost, tooling overhead, learning curve, maintenance burden | ❌ No |

The YAML-based approach provides several key advantages for workflow orchestration. The declarative nature makes workflows self-documenting and easy to understand at a glance. Version control systems handle YAML files naturally, providing clear diffs when workflows change. Code review processes can focus on business logic rather than programming syntax. The format is accessible to data analysts and business users who may not have programming backgrounds but understand the data processing requirements.

YAML's structured format enables comprehensive validation before workflow execution. The parser can check for required fields, validate data types, and ensure referential integrity of task dependencies. Static analysis tools can detect potential issues like unreachable tasks or missing parameters. The declarative nature also facilitates workflow visualization tools that can automatically generate dependency graphs from the YAML definitions.

The template system bridges the gap between static configuration and dynamic parameterization. Using double-brace syntax (`{{ variable_name }}`), workflows can reference execution context variables, DAG parameters, and cross-task communication values. The templating engine resolves these references at runtime while maintaining the static nature of the underlying workflow definition.

However, the YAML approach does impose limitations on dynamic workflow generation. Workflows cannot programmatically generate tasks based on runtime data or external API responses. Complex conditional logic requires careful modeling with task dependencies and trigger rules rather than imperative programming constructs. Loops must be unrolled into explicit task definitions, which can be verbose for workflows that process many similar datasets.

> **Design Trade-off**: The YAML format prioritizes workflow maintainability and team collaboration over programmatic flexibility. This aligns with the principle that workflows should be stable, predictable, and auditable rather than dynamically generated from business logic.

The decision to use YAML over Python-based DSLs (like Apache Airflow) eliminates entire classes of security vulnerabilities. YAML configurations cannot execute arbitrary code, access system resources, or import external libraries. This makes the workflow definitions safe to store in version control, share across teams, and review without security concerns. The static nature also enables better caching and optimization in the orchestration engine.

For teams that require more dynamic capabilities, the architecture supports hybrid approaches. The YAML workflow can invoke Python scripts or REST APIs that contain the dynamic logic, keeping the workflow structure static while allowing the tasks themselves to be flexible. This separation of concerns maintains the benefits of declarative workflow definition while providing escape hatches for complex scenarios.

### Common Pitfalls

DAG parsing and validation presents several common pitfalls that can lead to runtime failures, incorrect behavior, or poor developer experience. Understanding these pitfalls helps both implementers of the parser and users of the workflow orchestrator avoid common mistakes.

⚠️ **Pitfall: Incomplete Cycle Detection**

Many implementations only check for direct cycles (A depends on B, B depends on A) but miss longer cycles that span multiple tasks (A → B → C → D → A). This occurs when using simple dependency checking rather than proper graph algorithms. The incomplete detection allows invalid workflows into the system, causing deadlocks during execution where tasks wait indefinitely for dependencies that can never complete.

The fix requires implementing a complete topological sort algorithm that processes the entire dependency graph. Simple recursive checks or pairwise comparisons are insufficient for complex workflows with hundreds of interdependent tasks. The algorithm must track the processing state of every task and identify any remaining unprocessed tasks as participants in cycles.

⚠️ **Pitfall: Poor Error Message Quality**

Generic error messages like "invalid DAG" or "cycle detected" provide little actionable information for workflow authors. Users cannot efficiently fix their workflows without understanding which specific tasks or dependencies cause problems. This leads to frustration and time wasted on trial-and-error debugging.

Effective error reporting requires collecting and presenting specific context about validation failures. Cycle detection should identify the exact task sequence involved in each cycle. Missing dependency errors should specify which task references a non-existent upstream dependency. Invalid operator configurations should highlight which required fields are missing and provide examples of correct syntax.

⚠️ **Pitfall: Ignoring Template Validation**

Template expressions in YAML configurations (like `{{ execution_date }}`) often receive no validation during DAG parsing. Invalid template syntax or references to non-existent variables are only discovered at runtime, causing task failures after workflows have been scheduled and started. This late failure detection wastes compute resources and delays workflow completion.

The parser should validate template syntax during the initial DAG loading phase. While template variables cannot be resolved until runtime, the template expressions themselves can be checked for proper syntax, balanced braces, and valid function calls. Unknown template variables should generate warnings that help users identify potential typos or missing parameter definitions.

⚠️ **Pitfall: Inconsistent File Discovery**

DAG file discovery mechanisms that rely on simple filename patterns (like `*.yaml`) can inadvertently load backup files, temporary files, or incomplete workflow definitions. Users may have files like `workflow.yaml.backup` or `test_workflow.yaml.tmp` in their DAG directory that should not be processed as active workflows. Loading these files can create duplicate DAG IDs or invalid workflow definitions.

Robust file discovery requires both inclusion and exclusion patterns. The parser should actively ignore common backup file extensions (`.backup`, `.bak`, `.tmp`) and hidden files. Additionally, the parser should validate that each discovered file contains a complete workflow definition with required fields before attempting to register it in the system.

⚠️ **Pitfall: Missing Atomicity in DAG Updates**

When DAG definitions change, the parser may partially update the internal representation before encountering a validation error. This can leave the system in an inconsistent state where some tasks reflect the old definition while others reflect the new definition. Subsequent workflow executions may fail with confusing errors about missing tasks or invalid dependencies.

DAG updates must be atomic - either the entire new definition is valid and gets applied, or the update fails completely and the previous definition remains active. The parser should validate the complete new DAG definition in isolation before replacing any part of the existing registration. This requires careful transaction management and rollback capabilities in the DAG registry.

⚠️ **Pitfall: Inadequate Parameter Validation**

DAG and task parameters often lack proper validation beyond basic type checking. Parameters may have business-specific constraints (date ranges, enum values, numeric bounds) that are not enforced during parsing. Invalid parameter values are discovered only during task execution, causing workflow failures after resources have been allocated.

The parser should support parameter schemas that define validation rules for each parameter. Common validations include required vs optional parameters, data type constraints, regular expression patterns for strings, and range constraints for numeric values. Parameter validation should occur both during DAG registration and when workflows are manually triggered with custom parameter values.

⚠️ **Pitfall: Inefficient Re-parsing**

File-watching mechanisms that trigger full DAG re-parsing for any file change can create performance problems in directories with many workflow definitions. Changing a single character in one DAG file may trigger re-parsing of hundreds of unrelated workflows, causing unnecessary CPU usage and potential service disruption during the parsing phase.

Efficient DAG management requires tracking file modification timestamps and parsing only files that have actually changed. The parser should maintain a registry of last-parsed timestamps for each DAG file and skip unchanged files during directory scans. Additionally, parsing should be rate-limited to prevent resource exhaustion when many files change simultaneously.

### Implementation Guidance

The DAG parser and validation component requires careful attention to error handling, performance, and maintainability. The implementation should prioritize clear error messages and robust validation while maintaining good performance for large numbers of workflow definitions.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| YAML Parsing | `gopkg.in/yaml.v3` (standard Go YAML library) | `sigs.k8s.io/yaml` (Kubernetes YAML with JSON validation) |
| Template Engine | `text/template` (Go standard library) | `github.com/Masterminds/sprig` (enhanced template functions) |
| File Watching | `fsnotify/fsnotify` (cross-platform file events) | `radovskyb/watcher` (polling with filters) |
| Cron Parsing | `robfig/cron/v3` (standard cron expression library) | `gorhill/cronexpr` (advanced cron with seconds) |
| Graph Algorithms | Custom implementation (for learning) | `gonum/graph` (full graph algorithms library) |

#### Recommended File Structure

```
internal/parser/
  parser.go              ← main DAG parser with file discovery
  validator.go           ← cycle detection and validation logic
  template.go            ← parameter templating engine
  errors.go              ← structured error types and formatting
  parser_test.go         ← comprehensive parser tests
  validator_test.go      ← cycle detection test cases
  testdata/              ← sample DAG files for testing
    valid_dag.yaml       ← example valid workflow
    cyclic_dag.yaml      ← example with cycles for testing
    invalid_syntax.yaml  ← malformed YAML for error testing
```

#### Infrastructure Starter Code

Complete YAML parsing utilities with error handling and validation:

```go
// internal/parser/yaml.go
package parser

import (
    "fmt"
    "io/ioutil"
    "path/filepath"
    "strings"
    "time"
    
    "gopkg.in/yaml.v3"
)

// DAGDefinition represents the raw YAML structure before validation
type DAGDefinition struct {
    ID          string            `yaml:"id"`
    Description string            `yaml:"description"`
    Schedule    string            `yaml:"schedule"`
    Tags        []string          `yaml:"tags"`
    Parameters  map[string]string `yaml:"parameters"`
    Tasks       []TaskDefinition  `yaml:"tasks"`
}

// TaskDefinition represents a task in the YAML format
type TaskDefinition struct {
    ID           string                 `yaml:"id"`
    Type         string                 `yaml:"type"`
    Operator     map[string]interface{} `yaml:"operator"`
    Dependencies []string               `yaml:"dependencies"`
    Retries      int                    `yaml:"retries"`
    Timeout      string                 `yaml:"timeout"`
    Parameters   map[string]string      `yaml:"parameters"`
}

// FileDiscovery handles finding and loading DAG files
type FileDiscovery struct {
    directory   string
    filePattern string
}

// NewFileDiscovery creates a new file discovery instance
func NewFileDiscovery(directory, pattern string) *FileDiscovery {
    return &FileDiscovery{
        directory:   directory,
        filePattern: pattern,
    }
}

// DiscoverDAGFiles scans the configured directory for DAG definition files
func (fd *FileDiscovery) DiscoverDAGFiles() ([]string, error) {
    pattern := filepath.Join(fd.directory, fd.filePattern)
    matches, err := filepath.Glob(pattern)
    if err != nil {
        return nil, fmt.Errorf("failed to scan DAG directory %s: %w", fd.directory, err)
    }
    
    // Filter out backup and temporary files
    var validFiles []string
    for _, file := range matches {
        if fd.isValidDAGFile(file) {
            validFiles = append(validFiles, file)
        }
    }
    
    return validFiles, nil
}

// isValidDAGFile checks if a file should be processed as a DAG definition
func (fd *FileDiscovery) isValidDAGFile(filename string) bool {
    base := filepath.Base(filename)
    
    // Skip hidden files
    if strings.HasPrefix(base, ".") {
        return false
    }
    
    // Skip common backup and temporary file patterns
    excludePatterns := []string{".backup", ".bak", ".tmp", ".swp", "~"}
    for _, pattern := range excludePatterns {
        if strings.Contains(base, pattern) {
            return false
        }
    }
    
    return true
}

// LoadDAGDefinition reads and parses a YAML DAG file
func LoadDAGDefinition(filename string) (*DAGDefinition, error) {
    data, err := ioutil.ReadFile(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to read DAG file %s: %w", filename, err)
    }
    
    var dag DAGDefinition
    if err := yaml.Unmarshal(data, &dag); err != nil {
        return nil, fmt.Errorf("failed to parse YAML in %s: %w", filename, err)
    }
    
    return &dag, nil
}
```

Complete template engine for parameter resolution:

```go
// internal/parser/template.go
package parser

import (
    "bytes"
    "fmt"
    "strings"
    "text/template"
    "time"
)

// TemplateEngine handles parameter templating in DAG definitions
type TemplateEngine struct {
    funcs template.FuncMap
}

// NewTemplateEngine creates a template engine with built-in functions
func NewTemplateEngine() *TemplateEngine {
    funcs := template.FuncMap{
        "execution_date": func() string {
            return time.Now().Format("2006-01-02")
        },
        "now": func() string {
            return time.Now().Format(time.RFC3339)
        },
    }
    
    return &TemplateEngine{funcs: funcs}
}

// TemplateContext holds variables available during template rendering
type TemplateContext struct {
    ExecutionDate time.Time
    Parameters    map[string]string
    TaskInstance  map[string]interface{}
}

// ValidateTemplate checks template syntax without rendering
func (te *TemplateEngine) ValidateTemplate(text string) error {
    tmpl, err := template.New("validation").Funcs(te.funcs).Parse(text)
    if err != nil {
        return fmt.Errorf("template syntax error: %w", err)
    }
    
    // Check for balanced braces
    if strings.Count(text, "{{") != strings.Count(text, "}}") {
        return fmt.Errorf("unbalanced template braces in: %s", text)
    }
    
    _ = tmpl // Template parsed successfully
    return nil
}

// RenderTemplate processes template expressions with the given context
func (te *TemplateEngine) RenderTemplate(text string, ctx TemplateContext) (string, error) {
    tmpl, err := template.New("render").Funcs(te.funcs).Parse(text)
    if err != nil {
        return "", fmt.Errorf("template parse error: %w", err)
    }
    
    var buf bytes.Buffer
    if err := tmpl.Execute(&buf, ctx); err != nil {
        return "", fmt.Errorf("template execution error: %w", err)
    }
    
    return buf.String(), nil
}
```

#### Core Logic Skeleton Code

Main DAG parser with comprehensive validation pipeline:

```go
// internal/parser/parser.go
package parser

import (
    "fmt"
    "time"
    
    "github.com/workflow-orchestrator/internal/models"
)

// Parser handles loading and validating DAG definitions
type Parser struct {
    discovery *FileDiscovery
    validator *Validator
    templates *TemplateEngine
}

// NewParser creates a parser with all validation components
func NewParser(dagDirectory string) *Parser {
    return &Parser{
        discovery: NewFileDiscovery(dagDirectory, "*.yaml"),
        validator: NewValidator(),
        templates: NewTemplateEngine(),
    }
}

// LoadAndValidateDAG processes a DAG file through the complete pipeline
func (p *Parser) LoadAndValidateDAG(filename string) (*models.DAG, error) {
    // TODO 1: Load the raw DAG definition from YAML file
    // TODO 2: Validate basic structure and required fields
    // TODO 3: Validate template syntax in all string fields
    // TODO 4: Convert to internal DAG representation
    // TODO 5: Validate operator configurations by type
    // TODO 6: Perform cycle detection on dependency graph
    // TODO 7: Return validated DAG or accumulated errors
    // Hint: Use ParseDAGDefinition, ValidateStructure, ConvertToDAG, ValidateDAG
}

// ParseDAGDefinition loads and parses a single DAG file
func (p *Parser) ParseDAGDefinition(filename string) (*DAGDefinition, error) {
    // TODO 1: Use LoadDAGDefinition to read the YAML file
    // TODO 2: Check for required top-level fields (id, tasks)
    // TODO 3: Validate that DAG ID is a valid identifier
    // TODO 4: Return parsed definition or detailed error
    // Hint: Required fields are ID and Tasks array
}

// ValidateStructure performs basic validation before graph processing
func (p *Parser) ValidateStructure(def *DAGDefinition) error {
    // TODO 1: Check that DAG ID is not empty and follows naming rules
    // TODO 2: Validate that at least one task is defined
    // TODO 3: Check for duplicate task IDs within the DAG
    // TODO 4: Validate cron expression syntax if schedule is provided
    // TODO 5: Validate template syntax in description and parameter values
    // TODO 6: Return first validation error or nil if valid
    // Hint: Use map[string]bool to track duplicate task IDs
}

// ConvertToDAG transforms validated definition to internal representation
func (p *Parser) ConvertToDAG(def *DAGDefinition) (*models.DAG, error) {
    // TODO 1: Create new models.DAG with basic fields from definition
    // TODO 2: Convert each TaskDefinition to models.Task
    // TODO 3: Parse timeout strings to time.Duration
    // TODO 4: Validate and convert operator configurations
    // TODO 5: Build task lookup map for dependency resolution
    // TODO 6: Return converted DAG with all tasks properly linked
    // Hint: Use time.ParseDuration for timeout conversion
}

// DiscoverAndLoadAllDAGs scans directory and loads all valid DAGs
func (p *Parser) DiscoverAndLoadAllDAGs() (map[string]*models.DAG, []error) {
    // TODO 1: Use discovery.DiscoverDAGFiles to find all YAML files
    // TODO 2: Process each file with LoadAndValidateDAG
    // TODO 3: Collect successfully loaded DAGs in map by ID
    // TODO 4: Accumulate errors for files that failed to load
    // TODO 5: Check for duplicate DAG IDs across files
    // TODO 6: Return map of valid DAGs and slice of all errors
    // Hint: Continue processing after individual file failures
}
```

Cycle detection validator with detailed error reporting:

```go
// internal/parser/validator.go
package parser

import (
    "fmt"
    "strings"
    
    "github.com/workflow-orchestrator/internal/models"
)

// Validator handles DAG validation including cycle detection
type Validator struct {
    // No fields needed for stateless validation
}

// NewValidator creates a new DAG validator
func NewValidator() *Validator {
    return &Validator{}
}

// ValidateDAG performs comprehensive DAG validation including cycle detection
func (v *Validator) ValidateDAG(dag *models.DAG) error {
    // TODO 1: Call ValidateDependencyReferences to check task ID references
    // TODO 2: Call DetectCycles to ensure graph is acyclic
    // TODO 3: Call ValidateReachability to find orphaned tasks
    // TODO 4: Combine all validation errors into single error message
    // TODO 5: Return nil if all validations pass
    // Hint: Use multierror pattern to accumulate validation failures
}

// DetectCycles uses topological sort to find cycles in the dependency graph
func (v *Validator) DetectCycles(dag *models.DAG) error {
    // TODO 1: Build adjacency list mapping each task to its downstream dependents
    // TODO 2: Calculate in-degree for each task (number of upstream dependencies)
    // TODO 3: Initialize queue with tasks that have in-degree zero
    // TODO 4: Process tasks from queue, decrementing downstream in-degrees
    // TODO 5: Add tasks with zero in-degree to queue for next iteration
    // TODO 6: Check if all tasks were processed (result length == task count)
    // TODO 7: If unprocessed tasks remain, identify cycles with DFS
    // TODO 8: Return detailed error listing cycle paths
    // Hint: Use Kahn's algorithm for topological sort
}

// buildAdjacencyList creates graph representation for cycle detection
func (v *Validator) buildAdjacencyList(dag *models.DAG) (map[string][]string, map[string]int) {
    // TODO 1: Initialize adjacency list map and in-degree map
    // TODO 2: Add entry for each task in both maps
    // TODO 3: For each task, process its Dependencies slice
    // TODO 4: Add current task to dependency's adjacency list (downstream)
    // TODO 5: Increment current task's in-degree count
    // TODO 6: Return both maps for use in topological sort
    // Hint: Handle tasks with no dependencies (in-degree 0)
}

// findCyclesWithDFS identifies specific cycle paths for error reporting
func (v *Validator) findCyclesWithDFS(unprocessed []string, adjList map[string][]string) []string {
    // TODO 1: Initialize visited and recursion stack tracking maps
    // TODO 2: For each unprocessed task, start DFS if not visited
    // TODO 3: In DFS, mark task as visited and add to recursion stack
    // TODO 4: Recursively visit each downstream task
    // TODO 5: If downstream task is in recursion stack, cycle found
    // TODO 6: Build cycle path by backtracking through stack
    // TODO 7: Remove task from recursion stack before returning
    // TODO 8: Return slice of cycle descriptions for error message
    // Hint: Recursion stack detects back edges indicating cycles
}

// ValidateDependencyReferences ensures all dependency task IDs exist
func (v *Validator) ValidateDependencyReferences(dag *models.DAG) error {
    // TODO 1: Build set of all valid task IDs in the DAG
    // TODO 2: For each task, check each dependency ID
    // TODO 3: Report error if dependency ID not found in task set
    // TODO 4: Suggest similar task names for typos using string distance
    // TODO 5: Return error listing all invalid dependency references
    // Hint: Use map[string]bool for O(1) task ID lookup
}

// ValidateReachability finds tasks that cannot be reached from entry points
func (v *Validator) ValidateReachability(dag *models.DAG) error {
    // TODO 1: Identify entry point tasks (no upstream dependencies)
    // TODO 2: Perform breadth-first search from all entry points
    // TODO 3: Mark all reachable tasks during traversal
    // TODO 4: Check if any tasks were not reached
    // TODO 5: Report unreachable tasks as potential orphans
    // TODO 6: Return error if unreachable tasks found
    // Hint: Unreachable tasks may indicate missing dependencies
}
```

#### Language-Specific Hints

- Use `gopkg.in/yaml.v3` for YAML parsing with struct tags for field mapping
- Implement `fmt.Stringer` interface on error types for better error messages
- Use `filepath.Glob` for file pattern matching with proper error handling
- Leverage `text/template` for parameter templating with custom function maps
- Use buffered channels for concurrent DAG processing if needed
- Implement proper context cancellation for long-running validation operations

#### Milestone Checkpoint

After implementing the DAG parser and validation component:

**Test Command**: `go test ./internal/parser/... -v`

**Expected Output**:
```
=== RUN   TestParseValidDAG
--- PASS: TestParseValidDAG (0.01s)
=== RUN   TestDetectSimpleCycle
--- PASS: TestDetectSimpleCycle (0.00s)
=== RUN   TestDetectComplexCycle
--- PASS: TestDetectComplexCycle (0.00s)
=== RUN   TestValidateTemplates
--- PASS: TestValidateTemplates (0.00s)
PASS
```

**Manual Verification Steps**:

1. **Create test DAG file**: Place a valid YAML workflow in the configured DAGs directory
2. **Run parser**: Execute `./workflow-orchestrator parse-dags` (if CLI exists)
3. **Check output**: Should show "Successfully parsed 1 DAG(s)" with no errors
4. **Test cycle detection**: Add a circular dependency and verify error message shows the cycle path
5. **Test template validation**: Include invalid template syntax and confirm clear error reporting

**Signs Something Is Wrong**:

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "No DAGs found" despite files in directory | File discovery pattern mismatch | Check `filePattern` matches actual filenames | Update pattern or rename files |
| Parse succeeds but cycle not detected | Incomplete adjacency list building | Add debug logging to `buildAdjacencyList` | Fix dependency iteration logic |
| Templates not validated | Validation step skipped | Check `ValidateStructure` calls template validation | Add template validation to pipeline |
| Poor error messages | Generic error handling | Review error formatting in validator | Add specific error context and suggestions |


## Scheduler Component

> **Milestone(s):** Scheduler (core scheduling engine implementation), Task Execution (dependency tracking for ready tasks), Worker & Executor (coordination with task distribution)

The scheduler component serves as the orchestrator's central nervous system, responsible for triggering workflow executions based on time schedules, managing the execution queue, and coordinating with the executor to ensure tasks run when their dependencies are satisfied. This component bridges the gap between static workflow definitions and dynamic runtime execution, making critical decisions about when workflows should run and how to handle scheduling conflicts.

### Mental Model: The Project Manager

Think of the scheduler as an experienced project manager overseeing multiple complex projects simultaneously. Just as a project manager maintains a master schedule, tracks project dependencies, assigns tasks to team members when prerequisites are complete, and handles unexpected delays or resource conflicts, the workflow scheduler performs analogous functions in the orchestration system.

The project manager mental model helps understand the scheduler's core responsibilities. A project manager starts each day by reviewing the master schedule to identify which projects should begin today, checks whether prerequisite deliverables from other teams have been completed, and then assigns available team members to ready tasks. When team members complete their assignments, the project manager updates the project status and identifies newly ready tasks that can now begin. If a team member calls in sick or a task takes longer than expected, the project manager reassigns work and adjusts timelines accordingly.

The workflow scheduler operates using the same principles. It periodically scans the schedule configuration to identify DAGs that should trigger based on their cron expressions, validates that any external dependencies are satisfied, and creates new DAGRun instances representing workflow executions. As task instances complete and report their status back through the executor, the scheduler evaluates whether downstream tasks have become ready to run and queues them for execution. When worker failures occur or tasks timeout, the scheduler makes decisions about retries and failure propagation through the dependency graph.

This mental model emphasizes several key aspects of scheduler design. First, the scheduler must maintain authoritative state about what work is scheduled, in progress, and completed - just as a project manager maintains the definitive project status. Second, the scheduler operates on a polling cycle, regularly checking for schedule triggers and task completions rather than being purely reactive. Third, the scheduler must handle partial failures gracefully, making intelligent decisions about retries and dependency propagation when individual tasks fail. Finally, the scheduler must coordinate with other system components (the executor and workers) while maintaining clear boundaries of responsibility.

### Cron Expression Parsing

Cron expression parsing forms the foundation of time-based workflow triggering, converting human-readable schedule specifications into precise execution timestamps. The scheduler must accurately interpret cron syntax, calculate next execution times, and handle timezone complexities to ensure workflows run at the intended moments across different geographic regions.

The cron expression parser implements the standard five-field format: minute, hour, day of month, month, and day of week. Each field supports numeric values, ranges (1-5), lists (1,3,5), and step values (*/15 for every 15 minutes). The parser validates expression syntax during DAG registration to catch configuration errors before runtime rather than during execution attempts.

| Cron Field | Valid Values | Special Characters | Example |
|------------|--------------|-------------------|---------|
| Minute | 0-59 | * , - / | 0,15,30,45 |
| Hour | 0-23 | * , - / | 8-17 |
| Day of Month | 1-31 | * , - / L W | 1,15 |
| Month | 1-12 or JAN-DEC | * , - / | */3 |
| Day of Week | 0-6 or SUN-SAT | * , - / L # | 1-5 |

The next execution time calculation algorithm processes each field independently to find the earliest future timestamp that satisfies all constraints. The algorithm starts with the current time and iteratively adjusts each field from left to right (minute, hour, day, month, year) until finding a valid combination. When a field value exceeds its maximum (for example, minute 60), the algorithm resets that field to its minimum valid value and increments the next larger unit.

Timezone handling introduces significant complexity because workflow schedules should trigger at consistent local times regardless of daylight saving transitions or server location. The scheduler stores all execution dates in UTC internally but interprets cron expressions in the DAG's configured timezone. During daylight saving transitions, the scheduler applies special logic to handle ambiguous times (when clocks fall back) and missing times (when clocks spring forward).

```
Algorithm: Calculate Next Execution Time
1. Parse the cron expression into field constraints (minute, hour, day, month, day-of-week)
2. Start with current time in the DAG's configured timezone
3. For each time unit from smallest to largest:
   a. Find the next valid value for that field given the cron constraint
   b. If the current value already satisfies the constraint and no smaller unit was adjusted, keep it
   c. If the field value must increase, reset all smaller units to their minimum valid values
4. Handle day-of-month vs day-of-week conflicts (union semantics - either condition triggers)
5. Convert the result to UTC for storage and comparison
6. Validate the calculated time is actually in the future (handle edge cases)
```

The scheduler maintains a next execution time cache for each active DAG to avoid repeatedly recalculating the same cron expressions. When the current time exceeds a DAG's next execution time, the scheduler creates a new DAGRun and immediately calculates the subsequent execution time for the next cycle. This approach ensures minimal computational overhead during the scheduling loop while maintaining accurate timing.

> **Key Design Insight**: Cron parsing must be deterministic and timezone-aware because workflow schedules often need to align with business processes in specific geographic regions. A data pipeline that processes daily sales reports should trigger at 6 AM local time regardless of server location or daylight saving changes.

Special handling is required for edge cases that occur infrequently but can break workflows when they happen. February 29th schedules should skip non-leap years rather than failing permanently. Schedules that reference the 31st day of the month should automatically adjust to the last day in months with fewer days. Hour 2:30 AM during spring-forward daylight saving transitions (when 2:00 AM becomes 3:00 AM) should trigger at 3:30 AM instead of being skipped entirely.

| Edge Case | Cron Expression | Standard Behavior | Scheduler Behavior |
|-----------|----------------|-------------------|-------------------|
| Leap Year | 0 6 29 2 * | Skip on non-leap years | Skip gracefully, next trigger Mar 1 |
| Short Month | 0 6 31 * * | Fail on months < 31 days | Adjust to last day of month |
| DST Spring Forward | 0 2 * * * | Skip missing hour | Trigger at next valid hour (3 AM) |
| DST Fall Back | 0 2 * * * | Trigger twice | Trigger once using UTC tracking |

### DAG Run Creation Logic

DAG run creation transforms a scheduling trigger into an executable workflow instance, initializing all necessary state tracking and dependency relationships. This process involves creating the master DAGRun record, instantiating individual TaskInstance objects for each task in the DAG, and establishing the initial execution queue with tasks that have no upstream dependencies.

When the scheduler determines a DAG should execute (either from cron schedule or manual trigger), it begins the run creation process by generating a unique DAGRun identifier and setting the execution date. The execution date represents the logical time period this workflow run is processing, which may differ from the actual start time when backfilling historical periods or recovering from scheduler downtime.

The DAGRun creation algorithm follows a systematic approach to ensure consistency and proper initialization. First, it validates that no conflicting DAGRun already exists for the same DAG and execution date, preventing duplicate processing of the same logical time period. Second, it creates the master DAGRun record with initial state `queued` and persists it to the database within a transaction to establish the authoritative execution record.

```
Algorithm: Create DAG Run
1. Generate unique DAGRun ID using format: {DAG_ID}__{EXECUTION_DATE}__{UUID}
2. Validate no existing DAGRun exists for same DAG ID and execution date
3. Create DAGRun record with state = queued, start_date = now, parameters from trigger
4. For each task in the DAG definition:
   a. Create TaskInstance record with state = none, try_number = 1
   b. Calculate max_tries from task retry configuration + 1 for initial attempt
   c. Set task parameters by merging DAG defaults with run-specific overrides
5. Identify tasks with no upstream dependencies (entry point tasks)
6. Transition entry point TaskInstances to state = queued
7. Commit all records in a single database transaction
8. Return DAGRun ID for executor coordination
```

Task instance initialization requires careful attention to parameter resolution and state management. Each TaskInstance inherits base parameters from the task definition, overlaid with DAG-level parameters, and finally run-specific parameters passed during manual triggers. This layered parameter resolution allows workflows to be flexible while maintaining predictable defaults.

| Parameter Source | Priority | Example | Use Case |
|------------------|----------|---------|----------|
| Task Definition | Lowest | `retries: 3` | Task-specific defaults |
| DAG Definition | Medium | `environment: production` | Workflow-wide settings |
| DAG Run | Highest | `date_range: 2024-01-01` | Runtime overrides |

The scheduler must handle concurrent DAG run creation requests gracefully, particularly when backfilling multiple historical periods or when manual triggers overlap with scheduled triggers. Database constraints on the combination of DAG ID and execution date prevent duplicate runs, but the scheduler should detect these conflicts and respond appropriately rather than failing unexpectedly.

State initialization follows a careful sequence to ensure the DAGRun and its TaskInstances are in a consistent state before the executor begins processing. All TaskInstances start in the `none` state, indicating they have been created but not yet evaluated for execution readiness. Entry point tasks (those with no upstream dependencies) are immediately transitioned to `queued` state, signaling to the executor that they are ready to begin execution.

> **Critical Design Principle**: DAG run creation must be atomic - either all TaskInstances are created successfully or none are created. Partial creation would leave the workflow in an undefined state where some tasks exist and others don't, making dependency resolution impossible.

The scheduler tracks DAGRun metadata that supports monitoring and debugging capabilities. The `ExternalTriggerID` field stores identifiers from manual triggers, allowing the web UI to show which user or API call initiated a particular run. The `Parameters` field preserves the exact parameter values used during execution, enabling reproducible reruns and debugging of parameter-related issues.

Error handling during DAGRun creation must distinguish between transient failures (database connection issues) that should be retried and permanent failures (invalid DAG configuration) that indicate deeper problems. Transient failures should log warnings and attempt creation again on the next scheduler cycle. Permanent failures should mark the DAG as unhealthy and alert administrators to configuration problems.

### Architecture Decision: Scheduling Strategy

> **Decision: Polling vs Event-Driven Scheduling**
> - **Context**: The scheduler must decide when to check for new work (schedule triggers, task completions) and how to coordinate with other components. This fundamental choice affects system responsiveness, resource usage, and complexity.
> - **Options Considered**: Pure polling approach, pure event-driven approach, hybrid polling with event optimization
> - **Decision**: Hybrid approach with primary polling loop enhanced by event notifications
> - **Rationale**: Provides predictable resource usage and simple failure recovery while enabling real-time responsiveness for interactive operations
> - **Consequences**: Enables bounded scheduler overhead with configurable polling intervals while supporting immediate response to manual triggers and task completions

| Approach | Pros | Cons | Resource Usage |
|----------|------|------|----------------|
| Pure Polling | Simple, predictable, failure-resilient | Higher latency, constant CPU usage | High CPU, low memory |
| Pure Event-Driven | Low latency, efficient resource usage | Complex failure handling, cascading failures | Low CPU, medium memory |
| Hybrid | Balanced latency and reliability | Moderate complexity | Medium CPU, medium memory |

The pure polling approach operates on a fixed-interval loop where the scheduler periodically scans for schedule triggers and task state changes. This strategy offers excellent reliability because the scheduler naturally recovers from missed events by detecting them on the next polling cycle. Database failures, message queue outages, or network partitions cannot permanently block scheduling because the scheduler rediscovers pending work when connectivity is restored.

However, pure polling introduces inherent latency equal to the polling interval and consumes CPU resources continuously, even when no work is pending. For interactive operations like manual DAG triggers, users expect immediate response rather than waiting up to the polling interval for their request to be processed. Additionally, frequent database scans to check for ready tasks can create significant load on the database as the number of active DAGs and task instances grows.

The pure event-driven approach uses message queues or database triggers to notify the scheduler immediately when relevant changes occur. Task completions generate events that wake the scheduler to evaluate newly ready downstream tasks. Manual triggers send immediate messages that bypass the normal scheduling cycle. This strategy minimizes resource usage by activating the scheduler only when necessary and provides excellent responsiveness for interactive operations.

Event-driven scheduling introduces complexity in failure handling because missed events can leave tasks stranded indefinitely. If a task completion message is lost due to network issues, the downstream tasks will never be queued for execution. Message queue failures can block the entire scheduling process, and complex retry logic is required to handle transient failures without creating duplicate work or message loops.

The hybrid approach combines a primary polling loop with event-based optimizations to balance reliability and responsiveness. The scheduler maintains its regular polling cycle to ensure all work is eventually discovered and processed, providing the reliability guarantees of pure polling. Additionally, it subscribes to high-priority events like manual triggers and task completions to enable immediate processing without waiting for the next poll cycle.

```
Hybrid Scheduling Algorithm:
1. Initialize primary polling loop with configurable interval (default 30 seconds)
2. Initialize event subscription for manual triggers and task completions
3. On each polling cycle:
   a. Scan for schedule triggers based on cron expressions
   b. Scan for ready tasks based on dependency completion
   c. Update any stale task states (timeout detection, worker failures)
4. On event notification:
   a. Process manual triggers immediately
   b. Evaluate specific DAGRun for newly ready tasks
   c. Skip next polling cycle if comprehensive work was performed
```

Event optimization reduces average latency from half the polling interval to near-zero for interactive operations while maintaining the polling safety net. The scheduler can skip or delay polling cycles when recent event processing has already performed comprehensive work scanning, reducing unnecessary database load during active periods.

> **Performance Consideration**: The hybrid approach allows tuning the polling interval based on operational requirements. Production environments prioritizing reliability can use longer intervals (60+ seconds) relying primarily on events, while development environments can use shorter intervals (10-15 seconds) for faster feedback cycles.

Configuration parameters control the balance between polling and event-driven behavior, allowing operators to tune the scheduler for their specific reliability and performance requirements. The `polling_interval` setting determines the maximum time before any work is discovered and processed. The `event_processing_timeout` setting prevents event handlers from blocking the polling loop indefinitely if downstream systems become slow.

| Configuration | Default | Description | Tuning Guidance |
|---------------|---------|-------------|-----------------|
| `polling_interval` | 30s | Maximum discovery latency | Decrease for faster response, increase for lower load |
| `event_processing_timeout` | 10s | Max time processing single event | Set based on typical task evaluation time |
| `skip_poll_after_events` | true | Skip poll cycle after event processing | Enable for reduced database load |
| `max_parallel_evaluations` | 10 | Concurrent DAGRuns evaluated | Tune based on database capacity |

### Common Pitfalls

⚠️ **Pitfall: Timezone Confusion Leading to Missed Schedules**

Many implementations fail to properly handle timezone conversions, particularly during daylight saving transitions or when scheduler servers and workflow target systems operate in different timezones. The most common mistake is storing execution times in local timezone rather than UTC, which causes ambiguity during daylight saving transitions when the same local time occurs twice (fall back) or never occurs (spring forward).

Consider a DAG scheduled to run daily at 2:30 AM Eastern Time. During the spring-forward transition when clocks jump from 2:00 AM to 3:00 AM, the local time 2:30 AM does not exist. A scheduler that stores times in local timezone may either fail to trigger the workflow or trigger it at the wrong time. Similarly, during fall-back transitions when 2:30 AM occurs twice, the scheduler might trigger the workflow twice for the same logical execution date.

The correct approach stores all execution times in UTC internally while interpreting cron expressions in the DAG's configured timezone. During spring-forward transitions, the scheduler should trigger workflows at the next available time (3:30 AM in this example). During fall-back transitions, the scheduler should use UTC timestamps to ensure each workflow triggers exactly once per logical execution date.

```
Correct Timezone Handling:
1. Parse cron expression in DAG's configured timezone
2. Calculate next execution time in that timezone
3. Convert to UTC for internal storage and comparison
4. During DST transitions, adjust to next/previous valid time
5. Log timezone adjustments for debugging
```

⚠️ **Pitfall: Scheduler Drift and Clock Synchronization Issues**

Scheduler drift occurs when the system gradually accumulates timing errors, causing workflows to trigger increasingly late or early compared to their intended schedule. This often results from imprecise sleep intervals, processing delays, or clock synchronization problems across distributed systems. Left unchecked, scheduler drift can cause workflows to miss their intended execution windows entirely.

The root cause usually involves calculating sleep intervals based on relative delays rather than absolute timestamps. For example, sleeping for exactly 60 seconds between polling cycles will gradually drift later as each cycle's processing time accumulates. After 1000 cycles with 100ms processing time each, the scheduler will be running 100 seconds late.

Instead, the scheduler should calculate sleep duration based on absolute target timestamps. If the scheduler should poll every 30 seconds starting at midnight, the target times are 00:00:00, 00:00:30, 00:01:00, etc. The sleep duration for each cycle is the time remaining until the next target, minus any processing time already consumed.

```
Drift-Resistant Polling:
1. Calculate next target poll time = current_target + polling_interval
2. Execute polling logic (schedule checks, task evaluation)
3. Calculate sleep_duration = next_target - current_time
4. If sleep_duration > 0, sleep for that duration
5. If sleep_duration <= 0, log warning about scheduler overload
```

Clock synchronization becomes critical in distributed deployments where multiple scheduler instances or workers may coordinate timing. Network Time Protocol (NTP) should be configured on all nodes, and the scheduler should detect significant clock skew that could affect workflow timing.

⚠️ **Pitfall: Memory Leaks from Unbounded DAGRun Accumulation**

Schedulers often develop memory leaks by maintaining in-memory state for all active DAGRuns without implementing proper cleanup policies. As workflows complete, failed runs accumulate, and historical executions multiply, the scheduler's memory usage grows unbounded until the process crashes or becomes unresponsive.

This pitfall typically manifests when the scheduler caches DAGRun state for performance optimization but never removes completed runs from the cache. Additionally, event subscriptions or polling loops may retain references to completed TaskInstances, preventing garbage collection even after the database records are cleaned up.

The solution involves implementing aggressive cleanup policies and periodic memory auditing. Completed DAGRuns should be removed from in-memory caches immediately after their final state update. Failed runs should be retained only for a configured period before cleanup. Most importantly, the scheduler should periodically audit its memory usage and force cleanup if consumption exceeds acceptable thresholds.

```
Memory Management Strategy:
1. Remove completed DAGRuns from cache within 1 polling cycle
2. Retain failed runs for configured debugging period (default 24 hours)
3. Implement periodic cache size auditing (every 100 polling cycles)
4. Force garbage collection if memory usage exceeds threshold
5. Log cache sizes and cleanup statistics for monitoring
```

⚠️ **Pitfall: Race Conditions in Concurrent DAGRun Creation**

Concurrent DAGRun creation can create race conditions when multiple scheduler instances or threads attempt to create runs for the same DAG and execution date simultaneously. This commonly occurs during backfill operations, scheduler failover scenarios, or when manual triggers overlap with scheduled triggers.

The race condition typically manifests as duplicate DAGRun creation attempts, leading to database constraint violations or inconsistent task instance counts. More subtly, one scheduler instance might create the DAGRun record while another instance simultaneously creates TaskInstance records, resulting in orphaned tasks or missing dependencies.

Database constraints provide the ultimate protection by preventing duplicate DAGRuns, but the application logic should handle constraint violations gracefully rather than treating them as unexpected errors. The scheduler should use database transactions to ensure atomic DAGRun creation and implement retry logic with exponential backoff to handle concurrent access patterns.

```
Race-Safe DAGRun Creation:
1. Begin database transaction
2. Check for existing DAGRun with same DAG ID and execution date
3. If exists, rollback transaction and log info message
4. If not exists, create DAGRun and all TaskInstance records
5. Commit transaction or rollback on any error
6. On constraint violation, log info and continue (another instance succeeded)
```

⚠️ **Pitfall: Inadequate Error Handling for External Dependencies**

Many scheduler implementations fail to properly handle errors when external systems (databases, message queues, file systems) become temporarily unavailable. The scheduler may crash entirely, enter infinite retry loops, or silently skip work when these dependencies fail, leading to missed workflows and data processing gaps.

Database connection failures during DAGRun creation should be retried with exponential backoff rather than terminating the scheduler. Message queue unavailability should degrade gracefully to polling-only mode rather than blocking all scheduling activity. File system issues during DAG discovery should log errors and continue with previously loaded DAGs rather than stopping all workflow processing.

The key principle is graceful degradation - the scheduler should continue operating with reduced functionality rather than complete failure when external systems are unavailable. Each external dependency should have clearly defined failure modes, retry policies, and fallback behaviors that allow core scheduling to continue.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Cron Parsing | Manual parsing with regex + time.Parse | github.com/robfig/cron/v3 library |
| Database Operations | database/sql with prepared statements | GORM ORM with connection pooling |
| Background Processing | time.Ticker with goroutines | github.com/go-co-op/gocron scheduler |
| Configuration | encoding/json or gopkg.in/yaml.v3 | github.com/spf13/viper multi-format |
| Logging | log/slog standard library | github.com/sirupsen/logrus structured |
| Metrics | Manual counters with HTTP endpoint | github.com/prometheus/client_golang |

#### Recommended File Structure

```
internal/scheduler/
├── scheduler.go              ← main Scheduler struct and Run() method
├── scheduler_test.go         ← comprehensive scheduler behavior tests
├── cron/
│   ├── parser.go            ← cron expression parsing and validation
│   ├── parser_test.go       ← cron edge case and timezone tests
│   └── timezone.go          ← timezone handling and DST logic
├── dagrun/
│   ├── creator.go           ← DAGRun and TaskInstance creation logic
│   ├── creator_test.go      ← DAGRun creation and concurrency tests
│   └── states.go           ← state machine definitions and transitions
├── events/
│   ├── subscriber.go        ← event subscription for hybrid scheduling
│   ├── types.go            ← event type definitions and payloads
│   └── publisher.go        ← event publishing for manual triggers
└── config/
    ├── config.go           ← scheduler configuration structures
    └── defaults.go         ← default values and validation
```

#### Infrastructure Starter Code

**Cron Parser Implementation** (`internal/scheduler/cron/parser.go`):
```go
package cron

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// CronParser handles parsing and next execution calculation for cron expressions
type CronParser struct {
	timezone *time.Location
}

// NewCronParser creates a parser for the specified timezone
func NewCronParser(tz string) (*CronParser, error) {
	location, err := time.LoadLocation(tz)
	if err != nil {
		return nil, fmt.Errorf("invalid timezone %s: %w", tz, err)
	}
	return &CronParser{timezone: location}, nil
}

// CronField represents parsed values for a single cron field
type CronField struct {
	Values []int  // Specific values (e.g., 1,3,5)
	Step   int    // Step interval (e.g., */15)
	IsAll  bool   // True if field is *
	Min    int    // Minimum valid value for this field
	Max    int    // Maximum valid value for this field
}

// CronExpression represents a fully parsed cron schedule
type CronExpression struct {
	Minute    CronField
	Hour      CronField
	DayMonth  CronField
	Month     CronField
	DayWeek   CronField
	Raw       string
	Timezone  *time.Location
}

// Parse converts a cron string into a CronExpression
func (p *CronParser) Parse(cronStr string) (*CronExpression, error) {
	fields := strings.Fields(cronStr)
	if len(fields) != 5 {
		return nil, fmt.Errorf("cron expression must have exactly 5 fields, got %d", len(fields))
	}

	expr := &CronExpression{
		Raw:      cronStr,
		Timezone: p.timezone,
	}

	var err error
	expr.Minute, err = p.parseField(fields[0], 0, 59)
	if err != nil {
		return nil, fmt.Errorf("minute field: %w", err)
	}

	expr.Hour, err = p.parseField(fields[1], 0, 23)
	if err != nil {
		return nil, fmt.Errorf("hour field: %w", err)
	}

	expr.DayMonth, err = p.parseField(fields[2], 1, 31)
	if err != nil {
		return nil, fmt.Errorf("day of month field: %w", err)
	}

	expr.Month, err = p.parseField(fields[3], 1, 12)
	if err != nil {
		return nil, fmt.Errorf("month field: %w", err)
	}

	expr.DayWeek, err = p.parseField(fields[4], 0, 6)
	if err != nil {
		return nil, fmt.Errorf("day of week field: %w", err)
	}

	return expr, nil
}

// parseField handles individual cron field parsing with ranges, lists, and steps
func (p *CronParser) parseField(field string, min, max int) (CronField, error) {
	result := CronField{Min: min, Max: max}

	if field == "*" {
		result.IsAll = true
		return result, nil
	}

	// Handle step values (*/N)
	stepRegex := regexp.MustCompile(`^\*/(\d+)$`)
	if matches := stepRegex.FindStringSubmatch(field); matches != nil {
		step, err := strconv.Atoi(matches[1])
		if err != nil || step <= 0 || step > max-min+1 {
			return result, fmt.Errorf("invalid step value: %s", field)
		}
		result.Step = step
		result.IsAll = true
		return result, nil
	}

	// Handle comma-separated lists and ranges
	parts := strings.Split(field, ",")
	for _, part := range parts {
		values, err := p.parseRange(part, min, max)
		if err != nil {
			return result, fmt.Errorf("invalid range %s: %w", part, err)
		}
		result.Values = append(result.Values, values...)
	}

	return result, nil
}

// parseRange handles range notation like 1-5 or single values
func (p *CronParser) parseRange(rangeStr string, min, max int) ([]int, error) {
	if strings.Contains(rangeStr, "-") {
		parts := strings.Split(rangeStr, "-")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid range format: %s", rangeStr)
		}

		start, err := strconv.Atoi(parts[0])
		if err != nil || start < min || start > max {
			return nil, fmt.Errorf("invalid range start: %s", parts[0])
		}

		end, err := strconv.Atoi(parts[1])
		if err != nil || end < min || end > max || end < start {
			return nil, fmt.Errorf("invalid range end: %s", parts[1])
		}

		var values []int
		for i := start; i <= end; i++ {
			values = append(values, i)
		}
		return values, nil
	}

	// Single value
	value, err := strconv.Atoi(rangeStr)
	if err != nil || value < min || value > max {
		return nil, fmt.Errorf("invalid value: %s", rangeStr)
	}

	return []int{value}, nil
}

// NextExecution calculates the next valid execution time after the given time
func (expr *CronExpression) NextExecution(after time.Time) time.Time {
	// Convert to the cron's timezone for calculation
	t := after.In(expr.Timezone)
	
	// Start with the next minute to avoid immediate re-triggering
	t = t.Truncate(time.Minute).Add(time.Minute)
	
	// Find next valid time - limit iterations to prevent infinite loops
	for i := 0; i < 366*24*60; i++ { // Max 1 year of minutes
		if expr.matches(t) {
			return t.UTC() // Convert back to UTC for storage
		}
		t = t.Add(time.Minute)
	}
	
	// Should never reach here with valid cron expressions
	return time.Time{}
}

// matches checks if the given time satisfies the cron expression
func (expr *CronExpression) matches(t time.Time) bool {
	return expr.matchesField(expr.Minute, t.Minute()) &&
		expr.matchesField(expr.Hour, t.Hour()) &&
		expr.matchesField(expr.Month, int(t.Month())) &&
		(expr.matchesField(expr.DayMonth, t.Day()) || expr.matchesField(expr.DayWeek, int(t.Weekday())))
}

// matchesField checks if a time component matches a cron field
func (expr *CronExpression) matchesField(field CronField, value int) bool {
	if field.IsAll {
		if field.Step > 0 {
			return (value-field.Min)%field.Step == 0
		}
		return true
	}

	for _, v := range field.Values {
		if v == value {
			return true
		}
	}
	return false
}
```

**DAGRun Creation Implementation** (`internal/scheduler/dagrun/creator.go`):
```go
package dagrun

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
	"workflow-orchestrator/internal/models"
)

// Creator handles DAGRun and TaskInstance creation
type Creator struct {
	db     *sql.DB
	logger Logger
}

// Logger interface for dependency injection
type Logger interface {
	Info(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
}

// NewCreator initializes a new DAGRun creator
func NewCreator(db *sql.DB, logger Logger) *Creator {
	return &Creator{
		db:     db,
		logger: logger,
	}
}

// CreateDAGRun creates a new DAG execution with all task instances
func (c *Creator) CreateDAGRun(dag *models.DAG, executionDate time.Time, triggerType string, parameters map[string]string) (*models.DAGRun, error) {
	runID := fmt.Sprintf("%s__%s__%s", 
		dag.ID, 
		executionDate.Format("2006-01-02T15:04:05Z"), 
		uuid.New().String()[:8])

	// Begin transaction for atomic creation
	tx, err := c.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback() // Safe to call even after commit

	// Check for existing DAGRun
	var existingCount int
	err = tx.QueryRow(`
		SELECT COUNT(*) FROM dag_runs 
		WHERE dag_id = $1 AND execution_date = $2
	`, dag.ID, executionDate).Scan(&existingCount)
	
	if err != nil {
		return nil, fmt.Errorf("failed to check existing DAGRun: %w", err)
	}
	
	if existingCount > 0 {
		return nil, fmt.Errorf("DAGRun already exists for %s at %s", dag.ID, executionDate.Format(time.RFC3339))
	}

	// Create DAGRun record
	dagRun := &models.DAGRun{
		ID:            runID,
		DAGID:         dag.ID,
		ExecutionDate: executionDate,
		StartDate:     time.Now(),
		State:         models.DAGRunStateQueued,
		Parameters:    mergeParameters(dag.Parameters, parameters),
	}

	if triggerType == "manual" {
		externalID := fmt.Sprintf("manual_%s", uuid.New().String()[:8])
		dagRun.ExternalTriggerID = &externalID
	}

	_, err = tx.Exec(`
		INSERT INTO dag_runs (id, dag_id, execution_date, start_date, state, parameters, external_trigger_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
	`, dagRun.ID, dagRun.DAGID, dagRun.ExecutionDate, dagRun.StartDate, dagRun.State, 
		serializeParameters(dagRun.Parameters), dagRun.ExternalTriggerID)
	
	if err != nil {
		return nil, fmt.Errorf("failed to create DAGRun: %w", err)
	}

	// Create TaskInstance records for each task
	var readyTasks []string
	for _, task := range dag.Tasks {
		taskInstance := &models.TaskInstance{
			ID:        fmt.Sprintf("%s__%s", runID, task.ID),
			DAGRunID:  runID,
			TaskID:    task.ID,
			State:     models.TaskInstanceStateNone,
			TryNumber: 1,
			MaxTries:  task.Retries + 1, // Retries + initial attempt
		}

		_, err = tx.Exec(`
			INSERT INTO task_instances (id, dag_run_id, task_id, state, try_number, max_tries)
			VALUES ($1, $2, $3, $4, $5, $6)
		`, taskInstance.ID, taskInstance.DAGRunID, taskInstance.TaskID, 
			taskInstance.State, taskInstance.TryNumber, taskInstance.MaxTries)
		
		if err != nil {
			return nil, fmt.Errorf("failed to create TaskInstance %s: %w", task.ID, err)
		}

		// Track entry point tasks (no dependencies)
		if len(task.Dependencies) == 0 {
			readyTasks = append(readyTasks, task.ID)
		}
	}

	// Queue ready tasks (entry points with no dependencies)
	for _, taskID := range readyTasks {
		taskInstanceID := fmt.Sprintf("%s__%s", runID, taskID)
		_, err = tx.Exec(`
			UPDATE task_instances SET state = $1 WHERE id = $2
		`, models.TaskInstanceStateQueued, taskInstanceID)
		
		if err != nil {
			return nil, fmt.Errorf("failed to queue ready task %s: %w", taskID, err)
		}
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return nil, fmt.Errorf("failed to commit DAGRun creation: %w", err)
	}

	c.logger.Info("Created DAGRun", 
		"dag_id", dag.ID, 
		"run_id", runID, 
		"execution_date", executionDate.Format(time.RFC3339),
		"ready_tasks", len(readyTasks))

	return dagRun, nil
}

// mergeParameters combines DAG defaults with run-specific overrides
func mergeParameters(dagParams, runParams map[string]string) map[string]string {
	result := make(map[string]string)
	
	// Start with DAG defaults
	for k, v := range dagParams {
		result[k] = v
	}
	
	// Override with run-specific parameters
	for k, v := range runParams {
		result[k] = v
	}
	
	return result
}

// serializeParameters converts parameter map to JSON for database storage
func serializeParameters(params map[string]string) string {
	if len(params) == 0 {
		return "{}"
	}
	
	// Simple JSON serialization - use proper JSON library in production
	result := "{"
	first := true
	for k, v := range params {
		if !first {
			result += ","
		}
		result += fmt.Sprintf(`"%s":"%s"`, k, v)
		first = false
	}
	result += "}"
	return result
}
```

#### Core Logic Skeleton

**Main Scheduler Implementation** (`internal/scheduler/scheduler.go`):
```go
package scheduler

import (
	"context"
	"database/sql"
	"sync"
	"time"

	"workflow-orchestrator/internal/models"
	"workflow-orchestrator/internal/scheduler/cron"
	"workflow-orchestrator/internal/scheduler/dagrun"
)

// Scheduler coordinates workflow execution timing and DAGRun creation
type Scheduler struct {
	db             *sql.DB
	cronParser     *cron.CronParser
	dagrunCreator  *dagrun.Creator
	config         Config
	logger         Logger
	
	// State tracking
	activeDAGs     map[string]*models.DAG
	nextExecution  map[string]time.Time
	mu             sync.RWMutex
}

// Config defines scheduler configuration parameters
type Config struct {
	PollingInterval       time.Duration
	EventProcessingTimeout time.Duration
	SkipPollAfterEvents   bool
	MaxParallelEvaluations int
	Timezone              string
}

// NewScheduler creates a new scheduler instance
func NewScheduler(db *sql.DB, config Config, logger Logger) (*Scheduler, error) {
	cronParser, err := cron.NewCronParser(config.Timezone)
	if err != nil {
		return nil, fmt.Errorf("failed to create cron parser: %w", err)
	}

	return &Scheduler{
		db:             db,
		cronParser:     cronParser,
		dagrunCreator:  dagrun.NewCreator(db, logger),
		config:         config,
		logger:         logger,
		activeDAGs:     make(map[string]*models.DAG),
		nextExecution:  make(map[string]time.Time),
	}, nil
}

// Run starts the scheduler's main polling and event processing loop
func (s *Scheduler) Run(ctx context.Context) error {
	s.logger.Info("Starting scheduler", "polling_interval", s.config.PollingInterval)

	// TODO 1: Initialize ticker for polling interval
	// TODO 2: Load active DAGs from database and calculate next execution times
	// TODO 3: Start main event loop with select statement for ctx.Done(), ticker, and events
	// TODO 4: On each poll cycle, call s.processPendingSchedules() and s.evaluateReadyTasks()
	// TODO 5: On event notifications, call s.handleManualTrigger() or s.handleTaskCompletion()
	// TODO 6: Implement graceful shutdown when context is cancelled
	
	return nil
}

// processPendingSchedules checks for DAGs that should trigger based on their cron schedule
func (s *Scheduler) processPendingSchedules() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	now := time.Now()
	
	// TODO 1: Iterate through all active DAGs
	// TODO 2: For each DAG, check if current time >= next execution time
	// TODO 3: If ready to execute, call s.triggerDAGRun() with execution date
	// TODO 4: Calculate and store next execution time using cronParser.NextExecution()
	// TODO 5: Handle any errors by logging and continuing with other DAGs
	// TODO 6: Return count of triggered DAGRuns for monitoring
	
	return nil
}

// evaluateReadyTasks checks for task instances that can start based on dependency completion
func (s *Scheduler) evaluateReadyTasks() error {
	// TODO 1: Query database for task instances in 'none' state
	// TODO 2: For each task instance, check if all upstream dependencies are in 'success' state
	// TODO 3: If dependencies satisfied, transition task instance to 'queued' state
	// TODO 4: Handle dependency failures by transitioning to 'upstream_failed' state
	// TODO 5: Use batch updates for efficiency when queuing multiple tasks
	// TODO 6: Log task state transitions for debugging
	
	return nil
}

// triggerDAGRun creates a new DAG execution for the specified execution date
func (s *Scheduler) triggerDAGRun(dag *models.DAG, executionDate time.Time, triggerType string, parameters map[string]string) (*models.DAGRun, error) {
	// TODO 1: Validate that executionDate is not in the future for scheduled runs
	// TODO 2: Call dagrunCreator.CreateDAGRun() with merged parameters
	// TODO 3: Handle database constraint violations gracefully (duplicate runs)
	// TODO 4: Log successful DAGRun creation with run ID and execution date
	// TODO 5: Return created DAGRun for caller notification/monitoring
	
	return nil, nil
}

// HandleManualTrigger processes immediate DAG execution requests from API or UI
func (s *Scheduler) HandleManualTrigger(dagID string, executionDate *time.Time, parameters map[string]string) error {
	// TODO 1: Look up DAG definition from activeDAGs map
	// TODO 2: Use current time as execution date if not specified
	// TODO 3: Validate that manual triggers are allowed for this DAG
	// TODO 4: Call triggerDAGRun() with triggerType="manual"
	// TODO 5: Return immediate response to API caller without waiting for task execution
	
	return nil
}

// loadActiveDAGs refreshes the scheduler's DAG registry from database
func (s *Scheduler) loadActiveDAGs() error {
	// TODO 1: Query database for all DAGs with active=true
	// TODO 2: Parse cron expressions and validate syntax
	// TODO 3: Calculate initial next execution time for each DAG
	// TODO 4: Update activeDAGs and nextExecution maps atomically
	// TODO 5: Log count of loaded DAGs and any parsing errors
	// TODO 6: Handle database errors with exponential backoff retry
	
	return nil
}

// checkUpstreamDependencies determines if a task's dependencies are satisfied
func (s *Scheduler) checkUpstreamDependencies(taskInstance *models.TaskInstance) (bool, error) {
	// TODO 1: Get task definition to find upstream dependency list
	// TODO 2: Query task_instances table for upstream tasks in same DAGRun
	// TODO 3: Check that ALL upstream tasks are in 'success' state
	// TODO 4: Return false if any upstream task is 'failed' or 'upstream_failed'
	// TODO 5: Return false if any upstream task is still 'running' or 'queued'
	// TODO 6: Handle missing upstream tasks as configuration error
	
	return false, nil
}
```

#### Milestone Checkpoint

After implementing the scheduler component, verify the following behavior:

**Test Schedule Triggering:**
```bash
# Start the scheduler with debug logging
go run cmd/server/main.go --log-level=debug

# Create a test DAG with 1-minute cron schedule
echo 'id: test_dag
schedule: "* * * * *"
tasks:
  - id: hello_task
    type: python
    operator:
      python_command: "print(\"Hello from scheduled task\")"
' > dags/test_minute.yaml

# Verify DAGRuns are created every minute
curl http://localhost:8080/api/v1/dags/test_dag/runs | jq '.runs | length'
```

**Expected Output:**
- Scheduler logs show "Loaded 1 active DAGs"
- New DAGRun created every minute with increasing execution_date
- TaskInstances transition from 'none' → 'queued' → 'running' → 'success'
- Next execution time calculated correctly

**Test Manual Triggering:**
```bash
# Trigger DAG manually with parameters
curl -X POST http://localhost:8080/api/v1/dags/test_dag/trigger \
  -H "Content-Type: application/json" \
  -d '{"execution_date": "2024-01-01T12:00:00Z", "parameters": {"env": "test"}}'

# Verify immediate DAGRun creation
curl http://localhost:8080/api/v1/dags/test_dag/runs | jq '.runs[0]'
```

**Expected Behavior:**
- DAGRun created immediately without waiting for next poll cycle
- External trigger ID populated in DAGRun record
- Parameters merged correctly (DAG defaults + manual overrides)

**Signs of Problems:**
- "No DAGs loaded" → Check DAG file discovery and parsing
- "Constraint violation on DAGRun creation" → Multiple scheduler instances or race condition
- "Task stuck in 'none' state" → Dependency evaluation not running
- "Scheduler consuming 100% CPU" → Polling interval too short or infinite loop in evaluation

![DAG Execution Sequence](./diagrams/dag-execution-sequence.svg)


## Task Executor and Worker Management

> **Milestone(s):** Task Execution (dependency resolution and task execution logic), Worker & Executor (distributed task execution and worker management)

The task executor and worker management system forms the operational heart of our workflow orchestrator, responsible for coordinating the distributed execution of tasks across multiple worker nodes while maintaining strict dependency ordering and handling various failure scenarios. This component bridges the gap between the scheduler's high-level orchestration decisions and the actual execution of user-defined work, managing the complex choreography of task distribution, worker health monitoring, and result collection in a fault-tolerant manner.

### Mental Model: The Dispatch Center

Understanding the task executor requires thinking about it as a **dispatch center** coordinating field teams across a large organization. Imagine a construction company managing multiple job sites simultaneously. The dispatch center receives work orders (task instances) from project managers (the scheduler), determines which crews (workers) are available and capable of handling specific types of work, assigns tasks to appropriate teams, monitors progress through radio check-ins (heartbeats), and coordinates the handoff of materials and results between sequential work phases (dependency resolution).

Just as a dispatch center must handle radio failures, crew shortages, equipment breakdowns, and weather delays while keeping projects moving forward, our task executor must gracefully manage worker crashes, network partitions, resource contention, and task failures while ensuring that dependent work waits for upstream completion and that no critical work is lost or duplicated.

The dispatch center analogy helps clarify the key responsibilities: **task readiness evaluation** (ensuring all prerequisite work is complete), **resource allocation** (matching tasks to capable workers), **progress monitoring** (tracking task execution and worker health), and **coordination** (managing the flow of work and results between interdependent tasks).

![DAG Execution Sequence](./diagrams/dag-execution-sequence.svg)

### Dependency Resolution Algorithm

The dependency resolution algorithm serves as the core logic for determining when tasks become eligible for execution based on the completion status of their upstream dependencies. This algorithm operates on the principle that a task can only begin execution when all of its prerequisite tasks have reached a successful completion state, implementing a **topological execution order** that respects the directed acyclic graph structure defined in the DAG specification.

The dependency resolution process begins by maintaining an **adjacency list representation** of the task dependency graph, where each task maintains references to its upstream dependencies and downstream dependents. For each task instance in a DAG run, the system tracks the current execution state and uses this information to evaluate the readiness of downstream tasks whenever state transitions occur.

The algorithm operates through the following steps:

1. **State Change Detection**: When any task instance transitions to a terminal state (success, failed, skipped), the dependency resolver evaluates all immediate downstream tasks to determine if their execution prerequisites have been satisfied.

2. **Upstream Status Aggregation**: For each downstream task being evaluated, the resolver queries the current state of all upstream dependencies, collecting their execution states and any relevant metadata such as exit codes or XCom values.

3. **Trigger Rule Evaluation**: The system applies the configured trigger rule for the task (typically "all success" but may include "all done", "one success", or "none failed") to determine whether the combination of upstream states satisfies the execution prerequisites.

4. **Resource Availability Check**: If the trigger rule is satisfied, the resolver verifies that sufficient execution slots are available in the target worker pool and that any resource constraints (CPU, memory, concurrency limits) can be accommodated.

5. **Task Queuing**: Eligible tasks are added to the execution queue with appropriate priority ordering, typically based on DAG priority, task priority, and execution date to ensure fair scheduling across multiple concurrent DAG runs.

The dependency resolution algorithm must handle several complex scenarios that arise in real-world workflow execution:

**Partial Failure Propagation**: When an upstream task fails, the resolver must determine which downstream tasks should be marked as "upstream failed" versus which should still execute based on their trigger rules. This requires maintaining a clear distinction between task-level failures and dependency chain failures.

**Dynamic Dependency Updates**: In some cases, task dependencies may be determined or modified at runtime based on the results of upstream tasks. The resolver must support updating the dependency graph structure and re-evaluating task readiness when these dynamic changes occur.

**Cross-DAG Dependencies**: Advanced scenarios may require tasks in one DAG run to wait for completion of tasks in different DAG runs or external systems. The resolver architecture should accommodate these external dependency types while maintaining performance for the common case of intra-DAG dependencies.

The dependency resolution state tracking utilizes the following data structures:

| Component | Type | Description |
|-----------|------|-------------|
| DependencyGraph | map[string][]string | Maps each task ID to its list of upstream dependency task IDs |
| TaskStates | map[string]TaskInstanceState | Current execution state for each task instance in the DAG run |
| ReadyQueue | chan *TaskInstance | Buffered channel containing tasks eligible for immediate execution |
| WaitingTasks | map[string]*TaskInstance | Tasks waiting for upstream dependencies to complete |
| TriggerRules | map[string]TriggerRule | Execution trigger logic for each task (all_success, all_done, etc.) |

![Task Instance State Machine](./diagrams/task-state-machine.svg)

### Executor Implementation Types

The workflow orchestrator supports multiple executor implementations to accommodate different deployment scenarios and scalability requirements. Each executor type provides the same interface for task distribution and result collection while implementing different strategies for worker management and task execution isolation.

#### Local Sequential Executor

The **Local Sequential Executor** provides the simplest execution model, running all tasks sequentially within the same process as the scheduler. This executor type offers several advantages for development and small-scale deployments: zero configuration complexity, immediate task execution without network overhead, simplified debugging with all logs in a single process, and elimination of distributed system failure modes.

The local executor maintains a simple task queue and processes tasks one at a time using a dedicated goroutine. Task execution occurs through direct method calls to operator implementations, with task context and parameters passed as function arguments. The executor captures stdout and stderr output directly, stores results in the database, and immediately updates task instance states without requiring network communication.

However, the local sequential executor has significant limitations for production use. All tasks must execute on the same machine as the scheduler, creating a single point of failure and limiting scalability. Resource-intensive tasks can block other DAG processing, and there is no isolation between task executions, allowing one task to potentially affect others through shared process state.

| Aspect | Local Sequential Executor |
|--------|---------------------------|
| **Scalability** | Limited to single machine resources |
| **Isolation** | Shared process space, minimal isolation |
| **Fault Tolerance** | Single point of failure |
| **Setup Complexity** | Zero configuration required |
| **Network Dependencies** | None |
| **Use Cases** | Development, testing, small workflows |

#### Distributed Message Queue Executor

The **Distributed Message Queue Executor** implements a scalable architecture where task execution is distributed across multiple worker processes that may run on different machines. This executor uses a message queue system (Redis or RabbitMQ) to coordinate task distribution and result collection, providing horizontal scalability and fault isolation.

The distributed executor operates through a **producer-consumer pattern** where the scheduler publishes task execution messages to named queues, and worker processes consume these messages, execute the corresponding tasks, and publish results back through the message queue system. This architecture decouples task production from task execution, allowing the system to buffer work during peak periods and scale worker capacity independently of scheduler load.

Task distribution in the distributed executor follows a **pull model** where workers actively request new tasks from the queue system rather than having tasks pushed to them. This approach provides natural load balancing as busy workers consume tasks more slowly, and idle workers can immediately pick up available work. The message queue system handles worker failures gracefully by re-queueing tasks that are not acknowledged within a configurable timeout period.

The distributed executor implements **task serialization** to package task definitions, context, and parameters into messages that can be transmitted over the network. Task context includes the execution date, DAG run parameters, upstream task results (XCom values), and any template variables needed for task execution. Workers deserialize these messages, reconstruct the task execution environment, and invoke the appropriate operator implementation.

Result collection follows a similar pattern where workers publish task completion messages containing the execution status, output logs, return values, and any XCom data produced by the task. The scheduler subscribes to result channels and processes completion messages to update database state and trigger downstream dependency resolution.

| Component | Technology Options | Trade-offs |
|-----------|-------------------|------------|
| **Message Queue** | Redis Streams, RabbitMQ, Apache Pulsar | Redis: Simple setup, limited durability; RabbitMQ: Full AMQP features, higher complexity; Pulsar: Highest throughput, most complex |
| **Serialization** | JSON, Protocol Buffers, MessagePack | JSON: Human readable, larger size; Protobuf: Compact, requires schemas; MessagePack: Compact, schema-free |
| **Worker Discovery** | Static configuration, Service discovery, Load balancer | Static: Simple but inflexible; Service discovery: Dynamic but complex; Load balancer: Automatic but requires infrastructure |

The distributed executor addresses the scalability and fault tolerance limitations of the local executor but introduces new complexity around network communication, message durability, and distributed system failure modes. Network partitions can isolate workers from the message queue, requiring careful timeout and retry configuration. Message queue failures become a single point of failure unless high availability configurations are implemented.

### Worker Health Monitoring

Worker health monitoring provides the critical infrastructure for detecting worker failures and ensuring task execution reliability in distributed deployments. The monitoring system must distinguish between different failure scenarios—worker crashes, network partitions, resource exhaustion, and task hangs—and implement appropriate recovery strategies for each case.

#### Heartbeat Mechanism

The **heartbeat mechanism** forms the foundation of worker health monitoring, requiring each active worker to periodically report its operational status to the scheduler. Workers send heartbeat messages at regular intervals (typically every 30-60 seconds) containing their worker ID, current timestamp, available resource capacity, active task count, and recent task completion statistics.

The scheduler maintains a **worker registry** that tracks the last heartbeat time for each known worker along with metadata about worker capabilities and current task assignments. A dedicated monitoring goroutine periodically scans this registry to identify workers whose heartbeat timestamps exceed the configured failure detection threshold. Workers that fail to send heartbeats within the timeout window are marked as "presumed dead" and become candidates for task reassignment.

The heartbeat system must account for network variability and temporary connectivity issues that could cause false positive failure detection. Rather than immediately marking workers as failed after a single missed heartbeat, the system implements a **grace period** where workers can recover from temporary issues. Only after multiple consecutive missed heartbeats (typically 3-5 intervals) does the system initiate failure recovery procedures.

Heartbeat messages also carry **resource availability information** that enables intelligent task scheduling. Workers report their current CPU usage, memory consumption, available execution slots, and any resource constraints that might affect their ability to accept new tasks. The scheduler uses this information for load balancing and capacity planning, avoiding overloading workers that are approaching resource limits.

| Field | Type | Description |
|-------|------|-------------|
| WorkerID | string | Unique identifier for the worker process |
| Timestamp | time.Time | Current time when heartbeat was sent |
| AvailableSlots | int | Number of tasks this worker can accept |
| ActiveTasks | []string | Task instance IDs currently executing |
| ResourceUsage | ResourceMetrics | Current CPU, memory, and disk usage |
| LastTaskCompleted | *time.Time | Timestamp of most recent task completion |
| Version | string | Worker software version for compatibility checking |

#### Task Reassignment Strategies

When worker failures are detected, the system must implement **task reassignment strategies** to ensure that work is not lost and that DAG execution can continue. The reassignment process involves several phases: identifying affected tasks, determining reassignment eligibility, selecting target workers, and updating task state to reflect the reassignment.

**Task Identification** begins by querying the database for all task instances that were assigned to the failed worker and are currently in "running" state. These tasks are candidates for reassignment, but the system must determine whether each task should be retried immediately, delayed, or marked as failed based on the task's retry configuration and execution history.

**Reassignment Eligibility** evaluation considers multiple factors: the task's remaining retry attempts, the elapsed execution time (tasks that have been running for a long time may be close to completion), the availability of alternative workers capable of executing the task type, and any task-specific constraints that might prevent reassignment.

**Target Worker Selection** uses a load balancing algorithm that considers worker capacity, task type compatibility, and historical performance metrics. The scheduler maintains statistics on worker execution times for different task types and can use this information to route tasks to workers that have demonstrated good performance for similar workloads.

The reassignment process updates the `TaskInstance` record to reflect the new worker assignment, resets the start time, increments the try number, and clears any partial execution state that might be incompatible with the new worker. If the reassignment exhausts the task's configured retry attempts, the task is marked as failed, and the failure is propagated to downstream tasks through the dependency resolution system.

![Worker Health Monitoring](./diagrams/worker-health-monitoring.svg)

#### Graceful Worker Shutdown

**Graceful worker shutdown** handling ensures that planned worker maintenance or scaling operations do not result in task failures or data loss. When a worker receives a shutdown signal (SIGTERM), it should complete any currently executing tasks before terminating, rather than immediately stopping and potentially corrupting task results.

The graceful shutdown process follows these steps:

1. **Stop Accepting New Tasks**: The worker immediately stops polling the task queue for new work and sends a "draining" status in its heartbeat messages to inform the scheduler.

2. **Complete Active Tasks**: The worker continues executing any tasks that are already in progress, allowing them to complete normally and report their results.

3. **Timeout Protection**: If active tasks do not complete within a configured timeout period (typically 5-10 minutes), the worker forcibly terminates them and reports their status as "interrupted" to allow for retry on other workers.

4. **Final Status Report**: Before terminating, the worker sends a final heartbeat indicating its shutdown status and clears any resource reservations or queue subscriptions.

The scheduler recognizes "draining" workers and excludes them from new task assignments while continuing to monitor them for completion of their active tasks. This approach minimizes the impact of worker shutdowns on DAG execution while ensuring that maintenance operations can proceed reliably.

### Architecture Decision: Task Distribution Strategy

> **Decision: Pull-Based Task Distribution with Message Queues**
> - **Context**: We need to distribute tasks from the scheduler to multiple workers efficiently while handling worker failures, load balancing, and back-pressure scenarios. The task distribution mechanism affects system scalability, fault tolerance, and operational complexity.
> - **Options Considered**: Push-based direct messaging, Pull-based message queues, Hybrid approach with both push and pull
> - **Decision**: Implement pull-based task distribution using Redis Streams as the primary message queue with workers actively consuming tasks
> - **Rationale**: Pull-based distribution provides natural load balancing as busy workers consume tasks more slowly, simplifies failure handling since unacknowledged tasks can be automatically retried, eliminates the need for the scheduler to maintain real-time worker state, and allows workers to join/leave the pool dynamically without scheduler coordination
> - **Consequences**: Enables horizontal worker scaling with minimal configuration changes, introduces dependency on Redis infrastructure, requires careful tuning of queue acknowledgment timeouts, adds slight latency due to polling intervals, but provides better fault tolerance than direct push mechanisms

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| **Push-based Direct Messaging** | Lower latency, simpler message flow, no queue infrastructure | Requires scheduler to track worker state, difficult failure handling, no natural load balancing | ❌ |
| **Pull-based Message Queues** | Natural load balancing, simple failure handling, dynamic worker scaling, back-pressure handling | Requires queue infrastructure, polling overhead, slightly higher latency | ✅ |
| **Hybrid Push/Pull** | Can optimize for both scenarios, flexible routing | Significantly more complex, difficult to debug, multiple failure modes | ❌ |

The pull-based approach using Redis Streams provides several key advantages for our workflow orchestrator. **Natural load balancing** occurs automatically as workers that are processing resource-intensive tasks will consume new tasks more slowly, while idle workers immediately pick up available work. **Failure handling** is simplified because tasks that are not acknowledged within the timeout period are automatically returned to the queue for retry by other workers.

**Dynamic worker scaling** becomes straightforward as new workers can join the pool by simply connecting to the Redis instance and beginning to consume tasks, without requiring scheduler reconfiguration. Similarly, workers can be removed from the pool without coordination—their unfinished tasks will be picked up by remaining workers after the acknowledgment timeout expires.

The message queue also provides **back-pressure handling** where the scheduler can continue queuing tasks even when all workers are busy, allowing the system to buffer work during peak periods and process it when capacity becomes available. This prevents task loss during temporary worker capacity shortages.

**Task serialization** requirements are addressed through a standardized message format that includes the task definition, execution context, parameters, and any upstream results needed for execution:

| Field | Type | Description |
|-------|------|-------------|
| TaskInstanceID | string | Unique identifier for this task execution |
| TaskType | string | Operator type (python, bash, sql) for worker routing |
| OperatorConfig | json | Task-specific configuration and command details |
| ExecutionContext | json | DAG run ID, execution date, and template variables |
| UpstreamResults | map[string]interface{} | XCom values from completed upstream tasks |
| RetryNumber | int | Current attempt number for retry logic |
| Timeout | duration | Maximum execution time before task termination |

### Common Pitfalls

⚠️ **Pitfall: Inadequate Worker Failure Detection**
Many implementations use overly aggressive failure detection timeouts that cause false positives when workers are executing long-running tasks or experiencing temporary network issues. Setting heartbeat timeouts too low (under 60 seconds) can result in healthy workers being marked as failed and their tasks being unnecessarily retried. This creates wasted work and can lead to duplicate task execution if tasks have side effects. The fix is to implement a multi-tier timeout system where initial missed heartbeats trigger warnings but only sustained absence (3-5 missed heartbeats) triggers failure recovery, and to account for expected task execution times when setting timeout thresholds.

⚠️ **Pitfall: Race Conditions in Task State Updates**
Concurrent updates to task instance state can occur when workers report task completion at nearly the same time as failure detection logic marks them as failed due to missed heartbeats. This race condition can result in successful task results being overwritten by failure states, causing incorrect dependency resolution and downstream task blocking. The issue is particularly problematic for tasks that complete successfully but whose final status update is delayed by network latency. The solution requires implementing optimistic locking in the database layer using version numbers or timestamps, and ensuring that state transitions follow a strict precedence order where success/failure states cannot be overwritten by failure detection logic.

⚠️ **Pitfall: Resource Exhaustion During Parallel Execution**
Without proper resource limits and slot management, workers can accept more concurrent tasks than they have capacity to execute effectively, leading to memory exhaustion, CPU thrashing, and cascading task failures. This is especially problematic when tasks have variable resource requirements but the scheduler treats all tasks as equivalent for capacity planning. The fix involves implementing **resource slot management** where workers report their available CPU cores, memory capacity, and current utilization in heartbeat messages, and the scheduler considers these constraints when assigning tasks. Each task type should have resource requirements specified, and workers should reject tasks that would exceed their capacity limits.

⚠️ **Pitfall: Incomplete Task Cleanup After Worker Failure**
When workers fail unexpectedly, they may leave behind temporary files, database connections, external resource locks, or partially completed work that interferes with task retry attempts. Subsequent task executions may fail due to file conflicts, connection pool exhaustion, or external systems that believe the previous task attempt is still active. The solution requires implementing **idempotent task execution** patterns where tasks can safely run multiple times with the same inputs, and developing cleanup procedures that can detect and remove stale resources left by previous execution attempts. Task operators should use unique temporary directories and implement timeout-based resource cleanup.

⚠️ **Pitfall: Message Queue Durability and Ordering Issues**
Relying on message queue systems without understanding their durability and ordering guarantees can lead to task loss during queue system failures or incorrect task execution order when tasks have ordering dependencies beyond the DAG structure. Redis Streams, for example, can lose data if not properly configured with persistence, and most queue systems do not guarantee global ordering across multiple consumers. The fix requires configuring message queue persistence appropriately for the data durability requirements, implementing task acknowledgment patterns that prevent loss during failures, and avoiding reliance on message delivery order for correctness—all ordering should be enforced through explicit dependency declarations in the DAG definition.

### Implementation Guidance

The task executor and worker management system requires careful coordination between multiple components: the dependency resolver that determines task readiness, the executor that distributes tasks to workers, the worker health monitor that tracks worker status, and the result collector that processes task completion. This implementation guidance provides both infrastructure components that support the executor and skeleton code for the core logic that learners should implement.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Message Queue** | Redis with basic pub/sub | Redis Streams with consumer groups |
| **Task Serialization** | JSON encoding/decoding | Protocol Buffers with schema evolution |
| **Worker Communication** | HTTP polling with REST API | WebSocket or gRPC streaming |
| **Resource Monitoring** | Basic CPU/memory checks via /proc | Full metrics with Prometheus client |
| **Distributed Locking** | Redis-based locks with expiration | Consul or etcd distributed locks |
| **Log Streaming** | File-based logs with HTTP endpoints | Centralized logging with log aggregation |

#### Recommended File Structure

```
internal/
  executor/
    executor.go              ← Main executor interface and factory
    local/
      local_executor.go      ← Sequential executor implementation
      local_executor_test.go
    distributed/
      distributed_executor.go ← Message queue-based executor
      message_queue.go       ← Queue abstraction layer
      worker_pool.go         ← Worker registry and health monitoring
    dependency/
      resolver.go            ← Dependency resolution logic
      resolver_test.go
    worker/
      worker.go              ← Worker process implementation
      heartbeat.go           ← Heartbeat and health reporting
      task_runner.go         ← Task execution wrapper
  operators/
    base.go                  ← Common operator interface
    python_operator.go       ← Python task execution
    bash_operator.go         ← Shell command execution
    sql_operator.go          ← Database query execution
```

#### Infrastructure Code: Message Queue Abstraction

```go
package distributed

import (
    "context"
    "encoding/json"
    "fmt"
    "time"

    "github.com/go-redis/redis/v8"
)

// TaskMessage represents a serialized task ready for worker execution
type TaskMessage struct {
    TaskInstanceID   string                 `json:"task_instance_id"`
    TaskType         string                 `json:"task_type"`
    OperatorConfig   map[string]interface{} `json:"operator_config"`
    ExecutionContext map[string]interface{} `json:"execution_context"`
    UpstreamResults  map[string]interface{} `json:"upstream_results"`
    RetryNumber      int                    `json:"retry_number"`
    Timeout          time.Duration          `json:"timeout"`
}

// TaskResult represents the outcome of task execution
type TaskResult struct {
    TaskInstanceID string                 `json:"task_instance_id"`
    State          string                 `json:"state"` // success, failed, retry
    ExitCode       *int                   `json:"exit_code,omitempty"`
    LogOutput      string                 `json:"log_output"`
    XComValues     map[string]interface{} `json:"xcom_values,omitempty"`
    Duration       time.Duration          `json:"duration"`
    ErrorMessage   string                 `json:"error_message,omitempty"`
}

// MessageQueue provides task distribution and result collection
type MessageQueue struct {
    client      *redis.Client
    taskStream  string
    resultStream string
}

// NewMessageQueue creates a Redis-based message queue for task distribution
func NewMessageQueue(redisAddr, password string, db int) (*MessageQueue, error) {
    client := redis.NewClient(&redis.Options{
        Addr:     redisAddr,
        Password: password,
        DB:       db,
    })

    // Test connection
    ctx := context.Background()
    if err := client.Ping(ctx).Err(); err != nil {
        return nil, fmt.Errorf("failed to connect to Redis: %w", err)
    }

    return &MessageQueue{
        client:       client,
        taskStream:   "orchestrator:tasks",
        resultStream: "orchestrator:results",
    }, nil
}

// PublishTask sends a task to the worker queue
func (mq *MessageQueue) PublishTask(ctx context.Context, task *TaskMessage) error {
    data, err := json.Marshal(task)
    if err != nil {
        return fmt.Errorf("failed to marshal task: %w", err)
    }

    args := &redis.XAddArgs{
        Stream: mq.taskStream,
        Values: map[string]interface{}{
            "task_data": string(data),
            "timestamp": time.Now().Unix(),
        },
    }

    if err := mq.client.XAdd(ctx, args).Err(); err != nil {
        return fmt.Errorf("failed to publish task: %w", err)
    }

    return nil
}

// ConsumeTask reads the next available task from the queue
func (mq *MessageQueue) ConsumeTask(ctx context.Context, consumerGroup, consumerID string, timeout time.Duration) (*TaskMessage, string, error) {
    // Create consumer group if it doesn't exist
    mq.client.XGroupCreate(ctx, mq.taskStream, consumerGroup, "0")

    args := &redis.XReadGroupArgs{
        Group:    consumerGroup,
        Consumer: consumerID,
        Streams:  []string{mq.taskStream, ">"},
        Count:    1,
        Block:    timeout,
    }

    result := mq.client.XReadGroup(ctx, args)
    if result.Err() != nil {
        return nil, "", result.Err()
    }

    streams := result.Val()
    if len(streams) == 0 || len(streams[0].Messages) == 0 {
        return nil, "", fmt.Errorf("no messages available")
    }

    msg := streams[0].Messages[0]
    taskData, ok := msg.Values["task_data"].(string)
    if !ok {
        return nil, "", fmt.Errorf("invalid task data format")
    }

    var task TaskMessage
    if err := json.Unmarshal([]byte(taskData), &task); err != nil {
        return nil, "", fmt.Errorf("failed to unmarshal task: %w", err)
    }

    return &task, msg.ID, nil
}

// AckTask acknowledges successful task processing
func (mq *MessageQueue) AckTask(ctx context.Context, consumerGroup, messageID string) error {
    return mq.client.XAck(ctx, mq.taskStream, consumerGroup, messageID).Err()
}

// PublishResult sends task execution results back to the scheduler
func (mq *MessageQueue) PublishResult(ctx context.Context, result *TaskResult) error {
    data, err := json.Marshal(result)
    if err != nil {
        return fmt.Errorf("failed to marshal result: %w", err)
    }

    args := &redis.XAddArgs{
        Stream: mq.resultStream,
        Values: map[string]interface{}{
            "result_data": string(data),
            "timestamp":   time.Now().Unix(),
        },
    }

    return mq.client.XAdd(ctx, args).Err()
}

// ConsumeResults reads task results from workers
func (mq *MessageQueue) ConsumeResults(ctx context.Context, callback func(*TaskResult)) error {
    for {
        args := &redis.XReadArgs{
            Streams: []string{mq.resultStream, "$"},
            Count:   10,
            Block:   time.Second * 5,
        }

        result := mq.client.XRead(ctx, args)
        if result.Err() != nil {
            if result.Err() == redis.Nil {
                continue // Timeout, try again
            }
            return result.Err()
        }

        for _, stream := range result.Val() {
            for _, msg := range stream.Messages {
                resultData, ok := msg.Values["result_data"].(string)
                if !ok {
                    continue
                }

                var taskResult TaskResult
                if err := json.Unmarshal([]byte(resultData), &taskResult); err != nil {
                    continue // Skip malformed results
                }

                callback(&taskResult)
            }
        }
    }
}
```

#### Infrastructure Code: Worker Health Monitoring

```go
package distributed

import (
    "context"
    "sync"
    "time"

    "your-project/internal/models"
)

// WorkerStatus represents current worker health and capacity
type WorkerStatus struct {
    WorkerID        string            `json:"worker_id"`
    LastHeartbeat   time.Time         `json:"last_heartbeat"`
    AvailableSlots  int               `json:"available_slots"`
    ActiveTasks     []string          `json:"active_tasks"`
    ResourceUsage   ResourceMetrics   `json:"resource_usage"`
    Status          string            `json:"status"` // active, draining, failed
    Version         string            `json:"version"`
}

// ResourceMetrics tracks worker resource consumption
type ResourceMetrics struct {
    CPUPercent    float64 `json:"cpu_percent"`
    MemoryPercent float64 `json:"memory_percent"`
    DiskPercent   float64 `json:"disk_percent"`
}

// WorkerRegistry manages the pool of available workers
type WorkerRegistry struct {
    workers           map[string]*WorkerStatus
    mutex             sync.RWMutex
    heartbeatTimeout  time.Duration
    failureThreshold  int
    onWorkerFailed    func(string) // Callback for worker failure
}

// NewWorkerRegistry creates a worker health monitoring system
func NewWorkerRegistry(heartbeatTimeout time.Duration, failureThreshold int) *WorkerRegistry {
    return &WorkerRegistry{
        workers:          make(map[string]*WorkerStatus),
        heartbeatTimeout: heartbeatTimeout,
        failureThreshold: failureThreshold,
    }
}

// RegisterWorker adds a new worker to the registry
func (wr *WorkerRegistry) RegisterWorker(status *WorkerStatus) {
    wr.mutex.Lock()
    defer wr.mutex.Unlock()
    
    status.LastHeartbeat = time.Now()
    wr.workers[status.WorkerID] = status
}

// UpdateHeartbeat records a worker heartbeat
func (wr *WorkerRegistry) UpdateHeartbeat(workerID string, status *WorkerStatus) {
    wr.mutex.Lock()
    defer wr.mutex.Unlock()
    
    if worker, exists := wr.workers[workerID]; exists {
        status.LastHeartbeat = time.Now()
        wr.workers[workerID] = status
    }
}

// GetAvailableWorkers returns workers that can accept new tasks
func (wr *WorkerRegistry) GetAvailableWorkers(taskType string) []*WorkerStatus {
    wr.mutex.RLock()
    defer wr.mutex.RUnlock()
    
    var available []*WorkerStatus
    for _, worker := range wr.workers {
        if worker.Status == "active" && worker.AvailableSlots > 0 {
            available = append(available, worker)
        }
    }
    return available
}

// StartHealthMonitoring begins the worker failure detection loop
func (wr *WorkerRegistry) StartHealthMonitoring(ctx context.Context) {
    ticker := time.NewTicker(wr.heartbeatTimeout / 3) // Check 3x more frequently than timeout
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            wr.checkWorkerHealth()
        }
    }
}

// checkWorkerHealth identifies failed workers and triggers recovery
func (wr *WorkerRegistry) checkWorkerHealth() {
    wr.mutex.Lock()
    defer wr.mutex.Unlock()

    now := time.Now()
    for workerID, worker := range wr.workers {
        if worker.Status == "failed" {
            continue // Already marked as failed
        }

        timeSinceHeartbeat := now.Sub(worker.LastHeartbeat)
        if timeSinceHeartbeat > wr.heartbeatTimeout {
            worker.Status = "failed"
            if wr.onWorkerFailed != nil {
                go wr.onWorkerFailed(workerID) // Handle failure in background
            }
        }
    }
}
```

#### Core Logic Skeleton: Dependency Resolver

```go
package dependency

import (
    "context"
    "fmt"

    "your-project/internal/models"
)

// Resolver determines when tasks are ready for execution
type Resolver struct {
    db     *sql.DB
    logger Logger
}

// NewResolver creates a dependency resolution engine
func NewResolver(db *sql.DB, logger Logger) *Resolver {
    return &Resolver{
        db:     db,
        logger: logger,
    }
}

// EvaluateReadyTasks identifies task instances that can begin execution
// based on their upstream dependencies completion status
func (r *Resolver) EvaluateReadyTasks(ctx context.Context, dagRunID string) ([]*models.TaskInstance, error) {
    // TODO 1: Query database for all task instances in this DAG run that are in 'none' state
    // Use: SELECT * FROM task_instances WHERE dag_run_id = ? AND state = 'none'
    
    // TODO 2: For each task instance, get its upstream dependencies from the DAG definition
    // Use: GetUpstreamTasks(taskID) to get dependency list
    
    // TODO 3: Check the state of all upstream dependencies
    // Query: SELECT task_id, state FROM task_instances WHERE dag_run_id = ? AND task_id IN (...)
    
    // TODO 4: Apply trigger rule logic for each task
    // Common rules: all_success (all upstreams succeeded), all_done (all upstreams finished), none_failed
    
    // TODO 5: For tasks that pass trigger rule evaluation, verify resource availability
    // Check if executor has available worker slots for the task type
    
    // TODO 6: Return list of task instances that are ready for queuing
    // Update their state to 'queued' in database before returning
    
    // Hint: Use a transaction to ensure atomic state updates
    // Hint: Consider task priority when multiple tasks are ready
    
    return nil, fmt.Errorf("not implemented")
}

// CheckTriggerRule evaluates whether upstream task states satisfy the trigger condition
func (r *Resolver) CheckTriggerRule(upstreamStates map[string]models.TaskInstanceState, triggerRule string) bool {
    // TODO 1: Implement 'all_success' rule - all upstream tasks must be in 'success' state
    // TODO 2: Implement 'all_done' rule - all upstream tasks must be in terminal state (success/failed/skipped)
    // TODO 3: Implement 'none_failed' rule - no upstream tasks can be in 'failed' state
    // TODO 4: Implement 'one_success' rule - at least one upstream task must be 'success'
    // TODO 5: Add default case that returns false for unknown trigger rules
    
    return false
}

// PropagateUpstreamFailed marks tasks as upstream_failed when dependencies fail
func (r *Resolver) PropagateUpstreamFailed(ctx context.Context, failedTaskInstanceID string) error {
    // TODO 1: Get the DAG run ID and task ID for the failed task instance
    // TODO 2: Find all downstream tasks that depend on this failed task
    // Use: GetDownstreamTasks(taskID) from the DAG definition
    // TODO 3: For each downstream task, check if it should be marked as upstream_failed
    // Only mark if trigger rule is 'all_success' or similar strict rules
    // TODO 4: Update task instance states in database atomically
    // TODO 5: Recursively propagate upstream_failed to tasks that depend on the newly failed tasks
    
    return fmt.Errorf("not implemented")
}
```

#### Core Logic Skeleton: Distributed Executor

```go
package distributed

import (
    "context"
    "sync"
    
    "your-project/internal/models"
)

// DistributedExecutor manages task distribution across remote workers
type DistributedExecutor struct {
    messageQueue    *MessageQueue
    workerRegistry  *WorkerRegistry
    dependencyResolver *dependency.Resolver
    db              *sql.DB
    logger          Logger
    config          ExecutorConfig
    
    // Internal state
    taskQueue       chan *models.TaskInstance
    resultProcessor chan *TaskResult
    stopChan        chan struct{}
    wg              sync.WaitGroup
}

// ExecutorConfig contains distributed executor settings
type ExecutorConfig struct {
    MaxConcurrentTasks int           `json:"max_concurrent_tasks"`
    TaskTimeout        time.Duration `json:"task_timeout"`
    RetryDelay         time.Duration `json:"retry_delay"`
    WorkerPoolSize     int           `json:"worker_pool_size"`
}

// NewDistributedExecutor creates a distributed task execution engine
func NewDistributedExecutor(mq *MessageQueue, registry *WorkerRegistry, resolver *dependency.Resolver, db *sql.DB, config ExecutorConfig) *DistributedExecutor {
    return &DistributedExecutor{
        messageQueue:       mq,
        workerRegistry:     registry,
        dependencyResolver: resolver,
        db:                 db,
        config:            config,
        taskQueue:          make(chan *models.TaskInstance, 1000),
        resultProcessor:    make(chan *TaskResult, 1000),
        stopChan:           make(chan struct{}),
    }
}

// Start begins the distributed executor's task processing loops
func (de *DistributedExecutor) Start(ctx context.Context) error {
    // TODO 1: Start the dependency evaluation loop that finds ready tasks
    // Run evaluateReadyTasks() periodically (every 5-10 seconds)
    
    // TODO 2: Start the task distribution loop that sends tasks to workers
    // Process tasks from de.taskQueue and publish to message queue
    
    // TODO 3: Start the result collection loop that processes worker responses
    // Consume results from message queue and update database state
    
    // TODO 4: Start worker health monitoring
    // Call workerRegistry.StartHealthMonitoring()
    
    // TODO 5: Set up graceful shutdown handling
    // Listen for ctx.Done() and stop all loops cleanly
    
    // Hint: Use separate goroutines for each loop
    // Hint: Use de.wg.Add(1) and defer de.wg.Done() in each goroutine
    
    return fmt.Errorf("not implemented")
}

// SubmitTask queues a task instance for execution
func (de *DistributedExecutor) SubmitTask(ctx context.Context, taskInstance *models.TaskInstance) error {
    // TODO 1: Validate that task instance is in correct state for execution ('queued')
    // TODO 2: Add task to internal queue: de.taskQueue <- taskInstance
    // TODO 3: Update task instance state to 'queued' in database
    // TODO 4: Log task submission for debugging
    
    return fmt.Errorf("not implemented")
}

// processTaskQueue continuously distributes queued tasks to available workers
func (de *DistributedExecutor) processTaskQueue(ctx context.Context) {
    // TODO 1: Loop continuously until ctx.Done()
    // TODO 2: Read task instances from de.taskQueue channel
    // TODO 3: Find available workers for the task type using workerRegistry.GetAvailableWorkers()
    // TODO 4: Select best worker based on load balancing algorithm (round robin, least loaded, etc.)
    // TODO 5: Create TaskMessage from TaskInstance data
    // TODO 6: Publish task to message queue using messageQueue.PublishTask()
    // TODO 7: Update task instance state to 'running' and record assigned worker
    // TODO 8: Handle errors by requeuing task or marking as failed based on retry logic
    
    // Hint: Use select statement to handle both task queue and context cancellation
    // Hint: Implement exponential backoff for failed task submissions
}

// processResults handles task completion notifications from workers
func (de *DistributedExecutor) processResults(ctx context.Context) {
    // TODO 1: Start consuming results from message queue using messageQueue.ConsumeResults()
    // TODO 2: For each TaskResult, update the corresponding TaskInstance in database
    // TODO 3: Update state (success/failed), end_date, duration, exit_code, log_path
    // TODO 4: Store XCom values in separate xcom table for downstream task access
    // TODO 5: Trigger dependency evaluation for downstream tasks
    // TODO 6: Handle task failures by checking retry count and either retrying or marking final failure
    // TODO 7: Update worker registry to reflect completed task (decrement active task count)
    
    // Hint: Use database transactions for atomic state updates
    // Hint: Call dependencyResolver.EvaluateReadyTasks() after each successful completion
}

// handleWorkerFailure processes worker failure notifications
func (de *DistributedExecutor) handleWorkerFailure(workerID string) {
    // TODO 1: Query database for all task instances assigned to failed worker in 'running' state
    // TODO 2: For each affected task instance, determine if it should be retried or failed
    // TODO 3: Reset task state to 'queued' if retries remain, increment try_number
    // TODO 4: Clear worker assignment and reset start_date for retry attempts
    // TODO 5: Add retryable tasks back to de.taskQueue for redistribution
    // TODO 6: Mark non-retryable tasks as 'failed' and trigger upstream_failed propagation
    
    // Hint: Use task timeout information to decide if long-running tasks might have completed
    // Hint: Consider task idempotency when deciding retry eligibility
}
```

#### Milestone Checkpoints

**After implementing dependency resolution:**
```bash
# Test dependency resolution logic
go test ./internal/dependency/... -v

# Expected: All tests pass, showing correct trigger rule evaluation
# Check: Create test DAG with dependencies, verify tasks become ready in correct order
```

**After implementing distributed executor:**
```bash
# Start Redis for message queue
redis-server --port 6379

# Run executor tests
go test ./internal/executor/distributed/... -v

# Start a simple worker for integration testing
go run cmd/worker/main.go --redis-url localhost:6379

# Expected: Worker connects and receives heartbeat acknowledgment
# Check: Submit test task, verify it reaches worker and result returns to executor
```

**After implementing worker health monitoring:**
```bash
# Start multiple workers
go run cmd/worker/main.go --worker-id worker1 --redis-url localhost:6379 &
go run cmd/worker/main.go --worker-id worker2 --redis-url localhost:6379 &

# Kill one worker process and observe failure detection
kill %1

# Expected: Remaining worker picks up tasks from failed worker within heartbeat timeout
# Check: Task reassignment logs appear, no tasks are lost
```

#### Debugging Tips

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| **Tasks stuck in queued state** | No available workers or dependency resolution failure | Check worker registry for active workers, verify upstream task completion | Start workers, fix failed upstream dependencies |
| **Duplicate task execution** | Worker failure during result reporting | Check task acknowledgment in message queue logs | Implement idempotent task execution patterns |
| **Tasks timeout frequently** | Resource contention or incorrect timeout configuration | Monitor worker resource usage during task execution | Increase worker capacity or adjust timeout settings |
| **Memory leaks in executor** | Result channels not being drained properly | Check goroutine count and channel buffer sizes | Ensure result processing loop is running and handling backlog |
| **Worker registration failures** | Network connectivity or Redis configuration issues | Test Redis connection manually, check firewall settings | Fix network configuration, verify Redis accessibility |


## Task Operators and Execution Context

> **Milestone(s):** Task Execution (operator implementation and context passing), Worker & Executor (task execution environment and isolation)

### Mental Model: The Specialized Tools Workshop

Think of task operators as specialized tools in a well-organized workshop. Just as a carpenter's workshop contains different tools for different jobs—saws for cutting wood, drills for making holes, sanders for smoothing surfaces—a workflow orchestrator contains different operators for different types of computational work. Each tool has a standard interface (they all have handles and power switches), but their internal mechanisms are completely different.

The execution context is like the workbench that provides everything a tool needs to do its job: power supply, lighting, measuring instruments, and storage for materials. Every tool plugs into this standardized workbench, which ensures they have access to the same basic utilities while allowing them to perform their specialized functions. The workbench also provides communication channels so tools can pass their finished work to the next tool in the assembly line.

When a task fails, the workshop has a systematic retry protocol—check if it's a temporary issue like a power fluctuation, wait for conditions to improve, then try again with fresh materials. If a tool takes too long, the workshop supervisor steps in to prevent it from holding up the entire production line.

Task operators form the core execution layer of our workflow orchestrator, providing the actual computational logic that transforms inputs into outputs. Each operator type specializes in a particular execution environment—Python scripts for data processing, Bash commands for system operations, SQL queries for database interactions—while sharing a common interface that allows the orchestrator to treat them uniformly during scheduling and execution.

### Operator Interface Design

The operator interface design establishes a contract that all task types must implement, enabling the orchestrator to execute diverse workloads through a unified execution pipeline. This abstraction separates the concerns of task scheduling from task implementation, allowing operators to focus on their specific execution logic while the orchestrator handles dependency resolution, retry logic, and state management.

The core operator interface revolves around a standardized `Execute` method that receives an execution context and returns structured results. This design pattern mirrors the Command pattern from object-oriented design, where each operator encapsulates a specific action that can be invoked uniformly by the executor framework.

| Interface Method | Parameters | Returns | Description |
|------------------|------------|---------|-------------|
| `Execute` | `ctx ExecutionContext` | `*TaskResult`, `error` | Primary execution entry point that runs the operator's logic |
| `Validate` | `config OperatorConfig` | `error` | Validates operator configuration before execution |
| `GetRequiredResources` | `config OperatorConfig` | `ResourceRequirements` | Returns resource needs for scheduling decisions |
| `GetTimeout` | `config OperatorConfig` | `time.Duration` | Returns maximum execution time for this operator type |
| `SupportsRetry` | `config OperatorConfig` | `bool` | Indicates if this operator can be safely retried on failure |

The `ExecutionContext` structure provides operators with all the environmental information and utilities they need during execution. This context object serves as the primary communication channel between the orchestrator framework and the individual task implementations, ensuring that operators have access to workflow metadata, cross-task communication mechanisms, and logging infrastructure without tightly coupling them to the orchestrator's internal architecture.

| Context Field | Type | Description |
|---------------|------|-------------|
| `TaskInstance` | `*TaskInstance` | Complete task instance metadata including ID, retry count, and state |
| `DAGRun` | `*DAGRun` | Parent workflow execution context with run parameters and execution date |
| `Parameters` | `map[string]string` | Task-specific parameters merged with DAG-level defaults |
| `XCom` | `XComManager` | Interface for retrieving outputs from upstream tasks |
| `Logger` | `Logger` | Structured logger with task-specific metadata pre-populated |
| `Metrics` | `MetricsCollector` | Interface for recording task execution metrics and performance data |
| `Secrets` | `SecretManager` | Secure access to credentials and sensitive configuration values |
| `TempDir` | `string` | Isolated temporary directory for task-specific file operations |
| `WorkDir` | `string` | Working directory where task should execute (may be shared across tasks) |
| `Environment` | `map[string]string` | Environment variables specific to this task execution |

The operator configuration structure provides type-specific execution parameters while maintaining a consistent shape across all operator types. This design allows the orchestrator to serialize and deserialize operator configurations generically while ensuring each operator type receives the specific configuration it needs.

| Configuration Field | Type | Description |
|----------------------|------|-------------|
| `PythonCommand` | `string` | Python script content or file path for Python operators |
| `BashCommand` | `string` | Shell command string for Bash operators |
| `SQLQuery` | `string` | SQL statement for database operators |
| `Environment` | `map[string]string` | Operator-specific environment variable overrides |
| `WorkingDirectory` | `string` | Custom working directory for this operator |
| `Resources` | `ResourceRequirements` | CPU, memory, and disk requirements for scheduling |
| `Network` | `NetworkConfig` | Network access requirements and restrictions |

> **Decision: Unified Operator Interface vs Type-Specific Interfaces**
> - **Context**: Different task types have vastly different execution requirements—Python needs interpreter access, SQL needs database connections, Bash needs shell environment
> - **Options Considered**: 
>   1. Single interface with context-based dispatch
>   2. Type hierarchy with specialized interfaces per operator type
>   3. Plugin architecture with dynamic loading
> - **Decision**: Single unified interface with rich execution context
> - **Rationale**: Simplifies executor implementation, enables generic retry/timeout logic, supports future operator types without framework changes, maintains type safety through configuration validation
> - **Consequences**: Requires comprehensive context object, some operator types may not use all context fields, configuration validation becomes critical for type safety

The operator registry maintains a mapping from operator type strings to factory functions that create operator instances. This registry pattern enables dynamic operator loading and simplifies the addition of new operator types without modifying the core executor logic.

| Registry Method | Parameters | Returns | Description |
|-----------------|------------|---------|-------------|
| `RegisterOperator` | `typeName string`, `factory OperatorFactory` | `error` | Adds new operator type to registry |
| `CreateOperator` | `typeName string`, `config OperatorConfig` | `Operator`, `error` | Creates operator instance of specified type |
| `GetSupportedTypes` | - | `[]string` | Returns list of all registered operator type names |
| `ValidateConfig` | `typeName string`, `config OperatorConfig` | `error` | Validates configuration for specific operator type |

Each operator implementation follows a consistent internal structure that separates configuration parsing, resource setup, execution logic, and cleanup operations. This structure ensures that operators handle common concerns like error handling and resource management consistently while focusing their unique logic on the core execution steps.

The Python operator exemplifies this pattern by parsing the Python command from the configuration, setting up an isolated Python environment with the appropriate packages and environment variables, executing the script while capturing stdout and stderr, and cleaning up temporary files and processes. The operator validates that the Python command contains valid syntax before execution and provides detailed error messages that help users debug script failures.

Similarly, the Bash operator validates command syntax, sets up a restricted shell environment that prevents certain dangerous operations, executes the command with timeout enforcement, and captures both the exit code and output streams. The Bash operator includes additional security measures to prevent shell injection attacks when parameters are substituted into command strings.

The SQL operator connects to the configured database using connection pooling, validates query syntax, executes the query within a transaction context, and handles different result types (SELECT returns data, INSERT/UPDATE/DELETE return row counts). The SQL operator supports parameter binding to prevent SQL injection attacks and provides detailed error messages that include both database errors and query context.

### Cross-Task Communication (XCom)

Cross-task communication (XCom) provides a mechanism for tasks within a DAG run to share data with their downstream dependencies, enabling complex workflows where the output of one task influences the execution of subsequent tasks. The name "XCom" derives from "cross-communication," emphasizing its role as the primary data sharing mechanism between otherwise isolated task executions.

XCom addresses the fundamental challenge that tasks in a distributed workflow orchestrator typically run in separate processes, potentially on different machines, making direct memory sharing impossible. By providing a centralized data store with a simple key-value interface, XCom enables tasks to produce outputs that downstream tasks can consume, while maintaining the isolation and fault tolerance properties essential for distributed execution.

The XCom system stores small data payloads (typically under 1MB) that represent task outputs, intermediate results, or configuration parameters that downstream tasks need for their execution. XCom is explicitly designed for lightweight data sharing and should not be used for large datasets, which should instead be stored in external systems like databases, object storage, or shared filesystems, with XCom containing only references to these external resources.

| XCom Operation | Method Signature | Description |
|----------------|------------------|-------------|
| `Push` | `Push(key string, value interface{}) error` | Stores a value that downstream tasks can retrieve |
| `Pull` | `Pull(taskID, key string) (interface{}, error)` | Retrieves a value pushed by an upstream task |
| `PullAll` | `PullAll(taskID string) (map[string]interface{}, error)` | Retrieves all values pushed by a specific upstream task |
| `Clear` | `Clear(dagRunID, taskID string) error` | Removes all XCom values for a specific task instance |
| `List` | `List(dagRunID string) ([]XComEntry, error)` | Lists all XCom entries for a DAG run |

The XCom data model maintains a clear scope boundary where XCom values are isolated within individual DAG runs, preventing data leakage between different workflow executions. Each XCom entry includes metadata that tracks which task produced the value, when it was created, and its data type, enabling downstream tasks to make informed decisions about how to process the retrieved data.

| XCom Entry Field | Type | Description |
|------------------|------|-------------|
| `DAGRunID` | `string` | Unique identifier of the workflow execution |
| `TaskID` | `string` | Identifier of the task that produced this value |
| `Key` | `string` | User-defined key for retrieving this specific value |
| `Value` | `[]byte` | Serialized value data (JSON, protobuf, or custom format) |
| `Timestamp` | `time.Time` | When this value was stored |
| `DataType` | `string` | Type information for deserialization (json, string, int, etc.) |
| `Size` | `int64` | Size of the serialized value in bytes |

Task operators interact with XCom through the `XComManager` interface provided in their execution context, which abstracts the underlying storage implementation and provides type-safe serialization and deserialization of common data types. The XCom manager automatically handles the conversion between Go types and the serialized format stored in the backend, while providing explicit control over serialization when needed for custom data types.

```
Mental Model: The Relay Race Baton

Think of XCom as the baton in a relay race. Each runner (task) receives the baton from the previous runner, potentially adds their own information to it, and passes it to the next runner. The baton is small and lightweight—it contains essential information needed for coordination, but the real work happens during each runner's segment of the race.

Just as relay runners can't throw the baton to someone three positions ahead, XCom maintains clear handoff rules: tasks can only access XCom values from their direct upstream dependencies, ensuring data flow follows the DAG structure and preventing temporal paradoxes where a task tries to access data from a task that hasn't run yet.
```

The XCom implementation supports multiple backend storage systems to accommodate different deployment scenarios and performance requirements. The database backend provides ACID guarantees and supports complex queries for monitoring and debugging, while the Redis backend offers higher performance and built-in expiration for ephemeral data sharing.

| Backend Type | Pros | Cons | Use Cases |
|--------------|------|------|-----------|
| Database (PostgreSQL/MySQL) | ACID guarantees, complex queries, persistent storage, transaction support | Higher latency, requires database maintenance | Production deployments, audit requirements, complex XCom queries |
| Redis | High performance, built-in expiration, pub/sub support, clustering | No ACID guarantees, requires separate Redis maintenance | High-throughput workflows, temporary data sharing, distributed caching |
| In-Memory | Fastest access, no external dependencies, simple debugging | Lost on restart, memory usage growth, single-node only | Development, testing, lightweight deployments |

The XCom serialization system automatically handles common Go types including strings, integers, floating-point numbers, booleans, slices, and maps, while providing extension points for custom serialization logic. The serialization layer includes versioning support to handle schema evolution when XCom data structures change over time.

XCom access control ensures that tasks can only retrieve XCom values from their declared upstream dependencies, preventing tasks from accessing arbitrary data from unrelated task executions. This access control is enforced both at the API level and in the underlying storage queries, providing defense in depth against data access violations.

The XCom cleanup system automatically purges XCom data after DAG runs complete, with configurable retention periods for debugging and audit purposes. The cleanup system handles partial failures gracefully, ensuring that XCom storage doesn't grow indefinitely even when DAG runs fail to complete normally.

> **Decision: XCom Storage Backend**
> - **Context**: Need to store small data payloads between tasks with different consistency, performance, and operational requirements across deployment environments
> - **Options Considered**: 
>   1. Database-only storage with ACID guarantees
>   2. Redis-only storage for performance
>   3. Pluggable backend system supporting multiple storage types
> - **Decision**: Pluggable backend system with database and Redis implementations
> - **Rationale**: Different deployment scenarios have different requirements—development needs simplicity, production needs durability, high-throughput needs performance. Pluggable design supports all scenarios without forcing users into one solution.
> - **Consequences**: More complex implementation, requires configuration management for backend selection, enables optimal performance for each deployment scenario

### Task Retry and Timeout Handling

Task retry and timeout handling provides resilience against transient failures and runaway processes, ensuring that temporary issues don't cause permanent workflow failures while preventing individual tasks from consuming resources indefinitely. This system distinguishes between different types of failures and applies appropriate recovery strategies based on the failure characteristics and operator configuration.

The retry mechanism implements exponential backoff with jitter to prevent thundering herd problems when multiple failed tasks attempt to retry simultaneously. The backoff algorithm doubles the retry delay after each failure attempt, while adding random jitter to spread retry attempts across time and reduce system load during recovery periods.

| Retry Configuration | Type | Default | Description |
|---------------------|------|---------|-------------|
| `MaxRetries` | `int` | `3` | Maximum number of retry attempts before permanent failure |
| `RetryDelay` | `time.Duration` | `30s` | Base delay before first retry attempt |
| `BackoffMultiplier` | `float64` | `2.0` | Factor by which retry delay increases after each failure |
| `MaxRetryDelay` | `time.Duration` | `300s` | Maximum delay between retry attempts |
| `JitterRange` | `float64` | `0.1` | Random variation factor applied to retry delays (0.0-1.0) |
| `RetryExitCodes` | `[]int` | `[1, 2]` | Process exit codes that trigger retry (operator-specific) |
| `NoRetryExitCodes` | `[]int` | `[125, 126, 127]` | Exit codes that indicate permanent failure |

The retry decision logic evaluates multiple factors to determine whether a task failure should trigger a retry attempt or result in permanent failure. The system examines the exit code, error type, retry attempt count, and operator-specific retry policies to make intelligent retry decisions that maximize workflow reliability without wasting resources on hopeless retry attempts.

| Failure Type | Retry Decision | Rationale |
|--------------|----------------|-----------|
| Network timeout | Retry with backoff | Temporary connectivity issues often resolve |
| File not found | No retry | Missing files indicate configuration or dependency errors |
| Permission denied | No retry | Authorization issues require manual intervention |
| Out of memory | Retry once with delay | May indicate temporary resource contention |
| Database connection failed | Retry with backoff | Database may be temporarily overloaded |
| Invalid syntax error | No retry | Code errors require manual fixes |
| Process killed by signal | Retry based on signal type | SIGTERM (no retry) vs SIGKILL (retry once) |

The timeout enforcement system operates at multiple levels to provide comprehensive protection against runaway tasks. The task-level timeout enforces maximum execution time for individual task attempts, while the retry-level timeout limits the total time spent on all retry attempts combined, preventing tasks from consuming resources indefinitely through repeated failures.

Task timeout implementation uses operating system signals to terminate processes cleanly when possible, escalating to forceful termination if the process doesn't respond to the initial termination signal within a grace period. The timeout system captures partial output from terminated tasks to aid in debugging timeout issues.

| Timeout Type | Scope | Default Behavior | Configuration |
|--------------|-------|------------------|---------------|
| Task Timeout | Single execution attempt | Send SIGTERM, wait 30s, then SIGKILL | `timeout` field in task definition |
| Retry Timeout | All attempts combined | Fail task permanently | `retry_timeout` field in task definition |
| Operator Timeout | Per operator type | Use task timeout or operator default | Configured in operator registration |
| Global Timeout | System-wide maximum | 24 hours for any single task | Server configuration |

The retry state machine tracks task progression through multiple retry attempts while maintaining detailed audit information about each failure and retry decision. This state tracking enables the system to provide comprehensive retry status information in the web UI and supports debugging of retry logic issues.

| Retry State | Description | Next States | Trigger Events |
|-------------|-------------|-------------|---------------|
| `FirstAttempt` | Initial task execution | `Success`, `FailedWillRetry`, `FailedFinal` | Task completion or failure |
| `FailedWillRetry` | Failed, retry scheduled | `Retrying`, `FailedFinal` | Retry delay expires or max retries reached |
| `Retrying` | Currently executing retry | `Success`, `FailedWillRetry`, `FailedFinal` | Retry completion or failure |
| `Success` | Task completed successfully | Terminal state | N/A |
| `FailedFinal` | Permanent failure, no more retries | Terminal state | N/A |

The retry information storage system maintains detailed records of each retry attempt, including the specific error that triggered the retry, the retry delay applied, and the decision logic that determined whether to retry. This information proves invaluable for debugging workflow reliability issues and tuning retry parameters for specific task types.

| Retry Record Field | Type | Description |
|--------------------|------|-------------|
| `AttemptNumber` | `int` | Sequence number of this retry attempt |
| `StartTime` | `time.Time` | When this retry attempt began execution |
| `EndTime` | `time.Time` | When this retry attempt completed |
| `ExitCode` | `*int` | Process exit code if available |
| `ErrorMessage` | `string` | Error message from failed attempt |
| `RetryReason` | `string` | Why this failure triggered a retry |
| `NextRetryTime` | `*time.Time` | Scheduled time for next attempt |
| `BackoffDelay` | `time.Duration` | Delay applied before this retry |

Timeout handling includes special consideration for different operator types and their cleanup requirements. Python operators may need time to save intermediate results or release database connections, while Bash operators may need to clean up temporary files or terminate child processes. The timeout system provides operator-specific hooks for graceful shutdown procedures.

The retry and timeout system integrates with the XCom mechanism to handle partial results from failed task attempts. When a task fails after producing some XCom outputs, the system can be configured to either preserve these outputs for the next retry attempt or clear them to ensure retry attempts start with a clean state.

> The critical insight for retry and timeout design is that different failure modes require different recovery strategies. A network timeout may resolve within seconds, while a configuration error requires human intervention. The system must be sophisticated enough to distinguish between these scenarios while remaining simple enough for operators to understand and configure correctly.

### Architecture Decision: Task Isolation Strategy

Task isolation strategy determines how tasks are separated from each other during execution to prevent interference, ensure security, and enable resource management. The isolation level affects performance, security, operational complexity, and the types of workloads that can be safely executed within the orchestrator.

> **Decision: Task Isolation Strategy**
> - **Context**: Tasks may contain untrusted code, require different runtime environments, compete for system resources, or create conflicting file system changes. Need to balance security, performance, and operational complexity.
> - **Options Considered**: 
>   1. Thread-based execution within shared processes
>   2. Process isolation with separate OS processes per task
>   3. Container isolation using Docker or similar container runtimes
> - **Decision**: Process isolation as default with optional container isolation for enhanced security
> - **Rationale**: Process isolation provides strong security boundaries without container orchestration complexity, supports different runtime environments, enables resource limits via OS mechanisms, while container isolation remains available for high-security environments.
> - **Consequences**: Higher memory overhead than threads, simpler deployment than containers, strong security boundaries, enables per-task resource monitoring, requires process management logic in executor.

| Isolation Strategy | Security | Performance | Complexity | Resource Overhead | Use Cases |
|--------------------|----------|-------------|------------|-------------------|-----------|
| Thread-based | Low - shared memory space | High - minimal overhead | Low - simple implementation | Minimal - shared process | Trusted code, high throughput |
| Process-based | High - OS process boundaries | Medium - process creation cost | Medium - process management | Moderate - per-process memory | Mixed trust, different runtimes |
| Container-based | Highest - full isolation | Low - container startup cost | High - container orchestration | High - container images | Untrusted code, complex dependencies |

The process isolation implementation creates a separate operating system process for each task execution, providing memory isolation, file system separation through working directories, and independent environment variables. Process isolation leverages operating system security mechanisms to prevent tasks from interfering with each other or accessing sensitive system resources.

Process creation involves setting up an isolated execution environment that includes a dedicated working directory, filtered environment variables, resource limits configured through cgroups (on Linux) or job objects (on Windows), and standard input/output redirection for log capture. The process isolation system ensures that task processes cannot access files outside their designated working directory unless explicitly configured.

The process management system tracks all child processes created during task execution and ensures proper cleanup when tasks complete or are terminated. This includes handling process trees where tasks spawn additional child processes, ensuring that all related processes are terminated when a task timeout occurs or the workflow is cancelled.

Resource management within process isolation uses operating system mechanisms to enforce CPU, memory, and I/O limits for task processes. On Linux systems, the implementation uses cgroups to create resource containers that limit task resource consumption and provide accurate resource usage reporting. On other platforms, the system uses available OS-specific mechanisms or falls back to process monitoring with alerting when resource limits are exceeded.

Environment isolation ensures that each task process receives only the environment variables it needs for execution, preventing tasks from accessing sensitive system environment variables or configuration intended for other tasks. The environment setup process merges task-specific environment variables with safe system variables while filtering out sensitive values like database passwords or API keys.

File system isolation creates a dedicated working directory for each task execution and configures the task process to use this directory as its current working directory. The isolation system can optionally provide additional file system restrictions through chroot jails or similar mechanisms on supported platforms, further limiting task access to the broader file system.

The optional container isolation mode provides enhanced security for high-risk environments by executing tasks within Docker containers or similar container runtimes. Container isolation offers the strongest security boundaries by providing kernel-level isolation, completely separate file systems, and network isolation capabilities.

Container execution involves building or selecting appropriate container images that contain the required runtime environment for each task type, managing container lifecycle through the container runtime API, collecting logs and results from completed containers, and cleaning up container resources after task completion. The container isolation system integrates with existing container orchestration platforms when available.

Container resource management leverages container runtime features to enforce precise CPU and memory limits, provides detailed resource usage metrics through container monitoring APIs, and supports advanced features like GPU allocation for specialized workloads. Container isolation also enables more sophisticated networking configurations, including network policies that restrict task network access.

The hybrid isolation model allows different tasks within the same DAG to use different isolation strategies based on their security requirements and performance characteristics. Trusted data processing tasks might use process isolation for better performance, while tasks that execute user-provided code use container isolation for enhanced security.

Task communication across isolation boundaries requires careful design to maintain security while enabling necessary data sharing. The XCom system works consistently across all isolation modes by using the centralized data store, while direct inter-task communication is prevented to maintain security boundaries.

Process monitoring and health checks operate differently across isolation strategies, with container-based isolation providing richer health check mechanisms through container runtime APIs, while process-based isolation relies on operating system process monitoring and custom health check implementations.

⚠️ **Pitfall: Incomplete Process Cleanup**
Task processes may spawn child processes that outlive the main task process, leading to resource leaks and potential security issues. The process isolation system must track and terminate entire process trees, not just the primary task process. Implement process group management and recursive process termination to ensure complete cleanup.

⚠️ **Pitfall: Resource Limit Bypass**
Tasks may attempt to bypass resource limits by spawning multiple processes or using memory-mapped files. Implement comprehensive resource accounting that tracks all resources consumed by the task process tree, not just the primary process. Use cgroups or similar mechanisms that provide hierarchical resource management.

⚠️ **Pitfall: Environment Variable Leakage**
Sensitive configuration values may leak into task environments through inherited environment variables or shared configuration files. Implement explicit environment variable filtering that only passes through explicitly allowed variables, and ensure that task working directories don't contain sensitive configuration files.

### Common Pitfalls

⚠️ **Pitfall: XCom Size Limits Not Enforced**
Developers often try to pass large datasets through XCom, causing memory issues and database bloat. XCom is designed for small metadata, not bulk data transfer. Implement strict size limits (typically 1MB) and provide clear error messages directing users to use external storage systems for large datasets. Store references to external data in XCom rather than the data itself.

⚠️ **Pitfall: Retry Logic on Non-Idempotent Operations**
Retrying tasks that perform non-idempotent operations (like inserting database records without proper conflict handling) leads to duplicate data or inconsistent state. Design tasks to be idempotent by using upsert operations, checking for existing work before proceeding, or implementing proper transaction handling. Document retry behavior clearly for each operator type.

⚠️ **Pitfall: Infinite Retry Loops**
Misconfigured retry logic may retry tasks indefinitely when facing permanent failures, wasting resources and preventing workflow completion. Implement reasonable default retry limits, distinguish between transient and permanent failure types, and provide circuit breaker functionality that stops retrying after detecting permanent failure patterns.

⚠️ **Pitfall: Timeout Too Short for Normal Execution**
Setting task timeouts too aggressively causes tasks to be killed during normal execution, especially for data processing tasks with variable execution times. Provide operator-specific timeout defaults based on typical execution patterns, support dynamic timeout calculation based on data size or other parameters, and log timeout events with enough detail to help users tune timeout values.

⚠️ **Pitfall: Resource Isolation Bypass**
Tasks may bypass isolation boundaries through shared file systems, network connections, or system resources, leading to interference between tasks. Implement comprehensive isolation that covers file system access, network connections, shared memory, and system resources. Use container isolation for untrusted code that requires stronger security boundaries.

⚠️ **Pitfall: XCom Data Type Confusion**
Different programming languages and serialization formats may handle data types differently, causing tasks to receive data in unexpected formats from XCom. Implement explicit data type handling with clear serialization formats, provide type information in XCom metadata, and include data validation in task operator implementations.

⚠️ **Pitfall: Operator Configuration Validation Missing**
Invalid operator configurations may not be detected until task execution time, causing workflows to fail late in the execution process. Implement comprehensive configuration validation during DAG parsing that checks for required fields, validates command syntax where possible, and verifies that referenced resources exist and are accessible.

### Implementation Guidance

The task operator and execution context implementation requires careful coordination between the operator interface, execution environment, and retry handling systems. This guidance provides complete implementation patterns for building a robust task execution system that handles diverse operator types while maintaining consistency and reliability.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|------------------|
| Process Management | `os/exec` with `syscall` for resource limits | cgroups via `/sys/fs/cgroup` or libcgroup |
| Container Runtime | Docker CLI via `os/exec` | Docker API client or containerd API |
| Serialization | `encoding/json` for XCom data | Protocol Buffers or MessagePack |
| Timeout Handling | `context.Context` with `time.After` | OS signals with `syscall` package |
| Resource Monitoring | `/proc` filesystem parsing | cAdvisor integration or Prometheus metrics |

**Recommended File Structure:**
```
internal/
  executor/
    operators/
      operator.go           ← Base operator interface and factory
      python_operator.go    ← Python script execution
      bash_operator.go      ← Shell command execution  
      sql_operator.go       ← Database query execution
      operator_test.go      ← Operator unit tests
    context/
      execution_context.go  ← Task execution context implementation
      xcom_manager.go       ← Cross-task communication manager
      context_test.go       ← Context and XCom tests
    isolation/
      process_isolator.go   ← Process-based task isolation
      container_isolator.go ← Container-based task isolation
      isolator_test.go      ← Isolation strategy tests
    retry/
      retry_handler.go      ← Retry logic and backoff implementation
      timeout_manager.go    ← Task timeout enforcement
      retry_test.go         ← Retry and timeout tests
```

**Core Operator Interface (Complete Implementation):**

```go
// Package operators provides the task execution interface and implementations
// for different operator types in the workflow orchestrator.
package operators

import (
    "context"
    "time"
)

// Operator defines the interface that all task types must implement.
// This interface enables uniform task execution across diverse operator types.
type Operator interface {
    // Execute runs the operator's core logic with the provided execution context.
    // Returns task results and any execution error encountered.
    Execute(ctx ExecutionContext) (*TaskResult, error)
    
    // Validate checks operator configuration for correctness before execution.
    // Should catch configuration errors early to prevent runtime failures.
    Validate(config OperatorConfig) error
    
    // GetRequiredResources returns resource requirements for scheduling decisions.
    // Helps executor determine worker placement and resource allocation.
    GetRequiredResources(config OperatorConfig) ResourceRequirements
    
    // GetTimeout returns maximum execution time for this operator instance.
    // Provides operator-specific timeout defaults that can override global settings.
    GetTimeout(config OperatorConfig) time.Duration
    
    // SupportsRetry indicates if this operator can be safely retried on failure.
    // Some operators with side effects may not be suitable for automatic retry.
    SupportsRetry(config OperatorConfig) bool
}

// ExecutionContext provides operators with access to orchestrator services
// and environmental information needed during task execution.
type ExecutionContext struct {
    Context       context.Context     // Cancellation context for graceful shutdown
    TaskInstance  *TaskInstance      // Complete task metadata and state
    DAGRun        *DAGRun           // Parent workflow execution context  
    Parameters    map[string]string  // Resolved task parameters
    XCom          XComManager       // Cross-task communication interface
    Logger        Logger            // Task-specific structured logger
    Metrics       MetricsCollector  // Performance and execution metrics
    Secrets       SecretManager     // Secure credential access
    TempDir       string            // Isolated temporary directory
    WorkDir       string            // Task working directory
    Environment   map[string]string // Task environment variables
}

// OperatorFactory creates operator instances of a specific type.
// Used by the operator registry for dynamic operator instantiation.
type OperatorFactory func(config OperatorConfig) (Operator, error)

// OperatorRegistry manages available operator types and creates instances.
type OperatorRegistry struct {
    factories map[string]OperatorFactory
    mu        sync.RWMutex
}

// NewOperatorRegistry creates a registry with built-in operator types.
func NewOperatorRegistry() *OperatorRegistry {
    registry := &OperatorRegistry{
        factories: make(map[string]OperatorFactory),
    }
    
    // Register built-in operator types
    registry.RegisterOperator("python", NewPythonOperator)
    registry.RegisterOperator("bash", NewBashOperator)  
    registry.RegisterOperator("sql", NewSQLOperator)
    
    return registry
}

// RegisterOperator adds a new operator type to the registry.
func (r *OperatorRegistry) RegisterOperator(typeName string, factory OperatorFactory) error {
    // TODO 1: Validate typeName is not empty and contains only valid characters
    // TODO 2: Acquire write lock to ensure thread-safe registration
    // TODO 3: Check if typeName is already registered, return error if duplicate
    // TODO 4: Store factory function in internal map
    // TODO 5: Release write lock and return success
}

// CreateOperator instantiates an operator of the specified type.
func (r *OperatorRegistry) CreateOperator(typeName string, config OperatorConfig) (Operator, error) {
    // TODO 1: Acquire read lock for thread-safe access to factory map
    // TODO 2: Look up factory function by typeName
    // TODO 3: Return error if typeName not found in registry
    // TODO 4: Call factory function with provided config
    // TODO 5: Validate created operator using its Validate method
    // TODO 6: Return operator instance or any creation/validation errors
}
```

**XCom Manager Implementation (Complete):**

```go
// Package xcom provides cross-task communication functionality for sharing
// small data payloads between tasks in a workflow execution.
package xcom

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "time"
)

// XComManager provides the interface for cross-task data sharing.
type XComManager interface {
    Push(key string, value interface{}) error
    Pull(taskID, key string) (interface{}, error)
    PullAll(taskID string) (map[string]interface{}, error)
    Clear(dagRunID, taskID string) error
    List(dagRunID string) ([]XComEntry, error)
}

// XComEntry represents a single cross-task data sharing record.
type XComEntry struct {
    DAGRunID  string    `json:"dag_run_id"`
    TaskID    string    `json:"task_id"`
    Key       string    `json:"key"`
    Value     []byte    `json:"value"`
    Timestamp time.Time `json:"timestamp"`
    DataType  string    `json:"data_type"`
    Size      int64     `json:"size"`
}

// DatabaseXComManager implements XComManager using database storage.
type DatabaseXComManager struct {
    db        *sql.DB
    dagRunID  string
    taskID    string
    logger    Logger
    maxSize   int64  // Maximum size for XCom values
}

// NewDatabaseXComManager creates a database-backed XCom manager.
func NewDatabaseXComManager(db *sql.DB, dagRunID, taskID string, logger Logger) *DatabaseXComManager {
    return &DatabaseXComManager{
        db:       db,
        dagRunID: dagRunID,
        taskID:   taskID,
        logger:   logger,
        maxSize:  1024 * 1024, // 1MB default limit
    }
}

// Push stores a value that downstream tasks can retrieve.
func (x *DatabaseXComManager) Push(key string, value interface{}) error {
    // TODO 1: Serialize value to JSON bytes using json.Marshal
    // TODO 2: Check serialized size against maxSize limit
    // TODO 3: Determine data type string from value type
    // TODO 4: Begin database transaction for atomic insert
    // TODO 5: Execute INSERT or UPSERT query with all XCom fields
    // TODO 6: Commit transaction and log successful push operation
    // TODO 7: Return any serialization, size, or database errors
}

// Pull retrieves a value pushed by an upstream task.
func (x *DatabaseXComManager) Pull(taskID, key string) (interface{}, error) {
    // TODO 1: Query database for XCom entry matching dagRunID, taskID, key
    // TODO 2: Return error if no matching entry found
    // TODO 3: Deserialize value bytes based on stored data type
    // TODO 4: Return deserialized value or any query/deserialization errors
}
```

**Retry Handler Implementation (Complete):**

```go
// Package retry provides task retry logic with exponential backoff
// and configurable failure handling policies.
package retry

import (
    "context"
    "fmt" 
    "math"
    "math/rand"
    "time"
)

// RetryConfig defines retry behavior for task execution.
type RetryConfig struct {
    MaxRetries        int           `yaml:"max_retries"`
    RetryDelay        time.Duration `yaml:"retry_delay"`
    BackoffMultiplier float64       `yaml:"backoff_multiplier"`
    MaxRetryDelay     time.Duration `yaml:"max_retry_delay"`
    JitterRange       float64       `yaml:"jitter_range"`
    RetryExitCodes    []int         `yaml:"retry_exit_codes"`
    NoRetryExitCodes  []int         `yaml:"no_retry_exit_codes"`
}

// RetryHandler manages task retry logic and state tracking.
type RetryHandler struct {
    config   RetryConfig
    logger   Logger
    attempts []RetryAttempt
}

// RetryAttempt records details about a single retry attempt.
type RetryAttempt struct {
    AttemptNumber int           `json:"attempt_number"`
    StartTime     time.Time     `json:"start_time"`
    EndTime       time.Time     `json:"end_time"`
    ExitCode      *int          `json:"exit_code,omitempty"`
    ErrorMessage  string        `json:"error_message"`
    RetryReason   string        `json:"retry_reason"`
    NextRetryTime *time.Time    `json:"next_retry_time,omitempty"`
    BackoffDelay  time.Duration `json:"backoff_delay"`
}

// NewRetryHandler creates a retry handler with the specified configuration.
func NewRetryHandler(config RetryConfig, logger Logger) *RetryHandler {
    return &RetryHandler{
        config:   config,
        logger:   logger,
        attempts: make([]RetryAttempt, 0),
    }
}

// ShouldRetry determines if a failed task should be retried.
func (r *RetryHandler) ShouldRetry(exitCode *int, err error, attemptNumber int) (bool, string) {
    // TODO 1: Check if attempt number exceeds MaxRetries configuration
    // TODO 2: If exitCode provided, check against NoRetryExitCodes list
    // TODO 3: If exitCode provided, check if it's in RetryExitCodes list  
    // TODO 4: For error-based failures, classify error type for retry decision
    // TODO 5: Return retry decision and human-readable reason string
}

// CalculateBackoffDelay computes the delay before the next retry attempt.
func (r *RetryHandler) CalculateBackoffDelay(attemptNumber int) time.Duration {
    // TODO 1: Calculate base delay using exponential backoff formula
    // TODO 2: Apply BackoffMultiplier raised to attemptNumber power
    // TODO 3: Cap delay at MaxRetryDelay to prevent excessive waits
    // TODO 4: Add random jitter based on JitterRange configuration
    // TODO 5: Return final computed delay duration
}

// RecordAttempt logs details about a task execution attempt.
func (r *RetryHandler) RecordAttempt(attemptNumber int, startTime, endTime time.Time, 
                                   exitCode *int, err error) {
    // TODO 1: Create RetryAttempt struct with provided execution details
    // TODO 2: Determine retry reason based on exit code and error
    // TODO 3: If retry will be attempted, calculate next retry time
    // TODO 4: Append attempt record to internal attempts slice
    // TODO 5: Log attempt details for monitoring and debugging
}
```

**Milestone Checkpoints:**

After implementing the operator interface and basic operators:
```bash
# Test operator registration and creation
go test ./internal/executor/operators/... -v

# Expected: All operator types register successfully
# Expected: Configuration validation catches common errors
# Expected: Operators execute simple tasks and return results
```

After implementing XCom functionality:
```bash  
# Test cross-task communication
go test ./internal/executor/context/... -v -run TestXCom

# Expected: Tasks can push and pull data successfully
# Expected: Size limits are enforced properly
# Expected: DAG run isolation prevents data leakage
```

After implementing retry and timeout handling:
```bash
# Test retry logic with simulated failures
go test ./internal/executor/retry/... -v -run TestRetryLogic

# Expected: Transient failures trigger appropriate retry attempts
# Expected: Permanent failures are not retried
# Expected: Exponential backoff delays increase properly
# Expected: Tasks are terminated cleanly when timeouts occur
```

**Language-Specific Implementation Hints:**

- Use `context.Context` throughout for cancellation and timeout handling
- Implement `os/exec.Cmd` with `SysProcAttr` for process isolation on Unix systems
- Use `encoding/json` for XCom serialization with custom type handling for complex data
- Implement proper signal handling with `os/signal` for graceful task termination
- Use `sync.RWMutex` for thread-safe access to operator registry and retry state
- Leverage `time.Timer` for retry backoff delays and timeout enforcement
- Use `database/sql` with prepared statements for XCom database operations
- Implement comprehensive error handling with wrapped errors using `fmt.Errorf`


## Web UI and API Layer

> **Milestone(s):** Web UI & Monitoring (complete dashboard and API implementation for workflow monitoring and management)

The web UI and API layer serves as the primary interface between users and the workflow orchestrator, providing both programmatic access through REST endpoints and visual monitoring through a web dashboard. This component transforms the complex internal state of workflows, tasks, and executions into accessible information that operations teams and developers need for monitoring, debugging, and controlling their data pipelines.

### Mental Model: The Mission Control Center

Think of the web UI and API layer as a **mission control center** for space missions. Just as mission control provides real-time telemetry from spacecraft, multiple monitoring screens showing system status, and control panels for sending commands, our web interface provides:

- **Real-time dashboards** showing the health and progress of all running workflows
- **Historical mission logs** (task execution logs) that help diagnose what went wrong during failed runs
- **Control interfaces** for manually triggering workflows or stopping runaway executions
- **Communication channels** (APIs) that allow other systems to check status and send commands

The mission control analogy helps us understand that this layer must be reliable, provide clear information under pressure, and offer both high-level overview and detailed drill-down capabilities. When a critical data pipeline fails at 3 AM, the on-call engineer needs to quickly understand what happened and take corrective action.

![API and UI Architecture](./diagrams/api-architecture.svg)

### REST API Design

The REST API provides programmatic access to all workflow orchestrator functionality, following RESTful principles with resource-based URLs, standard HTTP methods, and consistent response formats. The API serves as both the backend for the web dashboard and the integration point for external systems that need to trigger workflows or query execution status.

The API design follows a hierarchical resource structure that mirrors the data model relationships. At the top level, we have DAGs as the primary resource, with DAGRuns nested underneath, and TaskInstances nested within DAGRuns. This structure provides intuitive URL patterns and clear resource ownership boundaries.

**Core Resource Endpoints:**

| Resource | Method | Endpoint | Description |
|----------|---------|----------|-------------|
| DAGs | GET | `/api/v1/dags` | List all registered DAGs with summary information |
| DAGs | GET | `/api/v1/dags/{dag_id}` | Get detailed DAG definition and current status |
| DAGs | POST | `/api/v1/dags/{dag_id}/dagRuns` | Trigger a new DAG run with optional parameters |
| DAGs | GET | `/api/v1/dags/{dag_id}/dagRuns` | List DAG runs for a specific DAG with pagination |
| DAG Runs | GET | `/api/v1/dagRuns/{dagrun_id}` | Get DAG run details including task instance status |
| DAG Runs | PATCH | `/api/v1/dagRuns/{dagrun_id}` | Update DAG run state (pause, resume, clear) |
| DAG Runs | DELETE | `/api/v1/dagRuns/{dagrun_id}` | Cancel a running DAG run |
| Task Instances | GET | `/api/v1/dagRuns/{dagrun_id}/taskInstances` | List all task instances in a DAG run |
| Task Instances | GET | `/api/v1/taskInstances/{task_instance_id}` | Get specific task instance details |
| Task Instances | GET | `/api/v1/taskInstances/{task_instance_id}/logs` | Stream or retrieve task execution logs |
| Task Instances | POST | `/api/v1/taskInstances/{task_instance_id}/clear` | Clear task instance state for retry |
| XCom | GET | `/api/v1/dagRuns/{dagrun_id}/xcom` | List XCom values for the DAG run |
| XCom | GET | `/api/v1/dagRuns/{dagrun_id}/xcom/{task_id}` | Get XCom values from specific task |

The API implements comprehensive error handling with standardized error response formats. All errors return consistent JSON structures that include error codes, human-readable messages, and contextual details for debugging. This consistency helps client applications handle errors gracefully and provides clear information for troubleshooting.

**Standard Error Response Format:**

| Field | Type | Description |
|-------|------|-------------|
| error_code | string | Machine-readable error identifier (e.g., "DAG_NOT_FOUND") |
| message | string | Human-readable error description |
| details | object | Additional context specific to the error type |
| request_id | string | Unique identifier for tracing this request |
| timestamp | string | ISO 8601 timestamp when error occurred |

Authentication and authorization are handled through configurable middleware that supports multiple strategies including API keys, JWT tokens, and OAuth 2.0. The API enforces fine-grained permissions that control access to specific DAGs and operations, ensuring that users can only interact with workflows they're authorized to manage.

**API Request/Response Examples:**

The DAG listing endpoint returns paginated results with summary information that helps users quickly identify the workflows they need. Each DAG entry includes current status, last run information, and schedule details:

```json
{
  "dags": [
    {
      "dag_id": "daily_etl_pipeline",
      "description": "Extract customer data and update analytics tables",
      "schedule_interval": "0 2 * * *",
      "is_active": true,
      "last_run": {
        "execution_date": "2024-01-15T02:00:00Z",
        "state": "success",
        "duration": "00:12:34"
      },
      "next_run": "2024-01-16T02:00:00Z",
      "tags": ["etl", "customer_data"]
    }
  ],
  "total_count": 25,
  "page": 1,
  "page_size": 20
}
```

The DAG run triggering endpoint accepts parameters that can override default values and control execution behavior:

```json
{
  "execution_date": "2024-01-15T10:00:00Z",
  "parameters": {
    "source_table": "customers_staging",
    "target_date": "2024-01-15"
  },
  "replace_existing": false
}
```

> **Design Insight**: The API design prioritizes discoverability and consistency over brevity. Verbose but clear endpoint names and comprehensive response objects help developers understand the system without constantly referring to documentation.

**Pagination and Filtering:**

All list endpoints implement cursor-based pagination to handle large datasets efficiently. The pagination strategy uses opaque tokens rather than offset/limit to avoid consistency issues when the underlying data changes during pagination:

| Parameter | Type | Description |
|-----------|------|-------------|
| limit | integer | Maximum number of items to return (default 20, max 100) |
| cursor | string | Opaque token for retrieving the next page |
| filter | string | Query string for filtering results |
| sort | string | Sort order specification |

The filtering system supports common query patterns for workflow monitoring, allowing users to find DAG runs by state, execution date range, or tag labels. This enables integration with monitoring systems that need to query for failed workflows or long-running executions.

### Dashboard Components

The web dashboard provides a comprehensive visual interface for monitoring and managing workflow orchestration, designed with operations teams in mind who need to quickly assess system health and take corrective actions. The dashboard follows a progressive disclosure design pattern, starting with high-level overviews and allowing drill-down into detailed information as needed.

**DAG List View:**

The primary dashboard screen displays all registered DAGs in a tabular format optimized for scanning and quick status assessment. Each row provides essential information that helps users identify workflows needing attention:

| Column | Information Displayed |
|--------|----------------------|
| DAG Name | Clickable identifier with description tooltip |
| Schedule | Cron expression with next execution time |
| Status | Visual indicator (green/yellow/red) of last run outcome |
| Last Run | Execution date, duration, and completion status |
| Actions | Quick buttons for manual trigger, view graph, pause/unpause |
| Run Stats | Success/failure counts for recent executions |

The status indicators use color coding and iconography that provide immediate visual feedback about DAG health. Green indicates successful recent runs, yellow shows workflows with some failed tasks but overall success, and red highlights DAGs with recent failures requiring attention.

**DAG Graph Visualization:**

The graph visualization renders the DAG structure as an interactive directed graph using a force-directed layout algorithm that positions tasks to minimize edge crossings and highlight the workflow structure. Tasks are represented as nodes with different shapes and colors indicating their type and current status:

| Task State | Visual Representation |
|------------|----------------------|
| Success | Green circle with checkmark icon |
| Running | Blue circle with spinner animation |
| Failed | Red circle with X icon |
| Queued | Gray circle with clock icon |
| Skipped | Yellow circle with skip icon |
| Upstream Failed | Orange circle with warning icon |

The graph supports interactive operations including zooming, panning, and clicking on task nodes to view detailed information in a sidebar panel. Dependency edges are drawn as arrows that show the flow direction, with different line styles indicating different types of dependencies (success, failure, or completion-based triggers).

**Task Instance Detail Panel:**

When users click on a task node in the graph or select from the task list, a detailed panel appears showing comprehensive task execution information:

| Section | Content |
|---------|---------|
| Basic Info | Task ID, operator type, start/end times, duration |
| Status | Current state, retry count, exit code |
| Configuration | Task parameters, timeout settings, retry policy |
| Dependencies | Upstream and downstream task relationships |
| Execution Context | Worker assignment, resource usage, environment |
| XCom Data | Cross-task communication values passed to/from this task |

The detail panel includes action buttons for common operations like clearing task state, viewing full logs, or manually marking tasks as success/failure for recovery scenarios.

**Run History and Timeline:**

The run history view provides both tabular and timeline visualizations of DAG execution history. The timeline view uses a Gantt chart format that shows overlapping runs and identifies bottlenecks in the execution schedule:

| Timeline Element | Visual Representation |
|------------------|----------------------|
| DAG Run | Horizontal bar spanning from start to end time |
| Task Execution | Nested bars within DAG run showing individual tasks |
| Retry Attempts | Dotted segments indicating retry behavior |
| Queue Time | Light gray segments showing wait time before execution |
| Overlap Detection | Visual warnings when runs overlap unexpectedly |

The timeline supports zooming from yearly overview down to minute-level detail, helping users identify patterns in execution times and spot scheduling conflicts or resource contention issues.

**Task Log Viewer:**

The integrated log viewer provides real-time streaming of task execution logs with syntax highlighting, filtering, and search capabilities. The viewer automatically follows log output for running tasks and maintains connection state to handle network interruptions gracefully:

| Feature | Implementation |
|---------|----------------|
| Real-time Streaming | WebSocket connection with automatic reconnection |
| Syntax Highlighting | Language detection for Python, SQL, and shell output |
| Log Level Filtering | Toggle visibility of debug, info, warning, error levels |
| Search and Navigation | Full-text search with regex support and result highlighting |
| Download | Export logs as text files for external analysis |
| Context Lines | Show surrounding lines when searching for specific content |

The log viewer handles large log files efficiently by implementing virtual scrolling and lazy loading of log content, ensuring responsive performance even for tasks that generate gigabytes of output.

**System Health Dashboard:**

A dedicated health dashboard provides operational metrics about the workflow orchestrator itself, helping administrators monitor system performance and capacity:

| Metric Category | Displayed Information |
|----------------|----------------------|
| Worker Status | Active workers, available slots, resource utilization |
| Queue Depth | Pending tasks by priority and estimated wait time |
| Execution Statistics | Tasks per minute, average duration, failure rates |
| Resource Usage | Database connections, memory usage, disk space |
| Schedule Drift | Difference between scheduled and actual execution times |
| Error Rates | Failed tasks, worker crashes, API error percentages |

The health dashboard includes configurable alerts and threshold monitoring that can trigger notifications when system metrics exceed acceptable ranges.

⚠️ **Pitfall: Overwhelming Information Density**
Junior developers often create dashboards that try to show everything at once, leading to cluttered interfaces that are difficult to scan quickly. The key is progressive disclosure - show the minimum information needed for decision-making at each level, with clear paths to drill down for more detail. Use visual hierarchy, whitespace, and color strategically to guide attention to the most important information.

### Real-Time Status Updates

The real-time status update system provides live monitoring capabilities that keep the web interface synchronized with the current state of workflow execution. This system must handle high-frequency updates efficiently while maintaining consistency across multiple browser sessions and providing graceful degradation when network connectivity is intermittent.

The real-time update architecture uses **Server-Sent Events (SSE)** as the primary transport mechanism because it provides a simpler programming model than WebSockets for one-way data flow while maintaining excellent browser support and automatic reconnection handling. SSE connections establish persistent HTTP connections that allow the server to push updates as they occur.

**Event Stream Architecture:**

The event streaming system is built around a publish-subscribe pattern where different components of the workflow orchestrator publish state change events to a central event bus, and the web interface subscribes to relevant event streams based on user context:

| Event Source | Event Types | Content |
|--------------|-------------|---------|
| Scheduler | DAG run created, scheduled, paused | DAG ID, execution date, state change |
| Executor | Task queued, started, completed | Task instance ID, state, worker assignment |
| Worker Registry | Worker connected, disconnected, heartbeat | Worker ID, status, available slots |
| XCom Manager | XCom value created, updated | DAG run ID, task ID, key, data size |
| DAG Parser | DAG updated, validation failed | DAG ID, change summary, error details |

The event bus implements topic-based routing that allows clients to subscribe only to events relevant to their current view. For example, when viewing a specific DAG's execution graph, the client subscribes to events for that DAG ID rather than receiving all system events.

**Client-Side Event Processing:**

The web interface implements an event processing system that applies incoming state changes to the current view incrementally, avoiding full page refreshes that would disrupt user interaction:

| Update Type | Processing Strategy |
|-------------|-------------------|
| Task State Change | Update node color and icon in graph visualization |
| Log Line Addition | Append to log viewer if currently viewing that task |
| DAG Run Progress | Update progress bars and completion percentages |
| Worker Status | Refresh worker health indicators and capacity meters |
| Queue Depth Changes | Update pending task counts and estimated wait times |

The client maintains a local state model that mirrors the server state for the currently viewed resources. Incoming events are applied as patches to this local state, and React components re-render only the specific elements that changed.

**Connection Management:**

The SSE connection management system handles network failures, browser tab switching, and connection limits gracefully. When connections are lost, the client automatically attempts reconnection with exponential backoff and can request state synchronization to catch up on missed events:

| Connection State | Behavior |
|----------------|----------|
| Connected | Normal event processing and UI updates |
| Reconnecting | Display connection status indicator, buffer updates |
| Disconnected | Show warning banner, allow manual refresh |
| Background Tab | Reduce update frequency to conserve resources |
| Page Visibility | Resume full updates when tab becomes active |

The connection management includes heartbeat monitoring that detects stale connections and proactively reconnects before users notice service degradation.

**Event Filtering and Throttling:**

To prevent overwhelming clients with high-frequency events during periods of intense activity, the event system implements filtering and throttling mechanisms:

| Throttling Strategy | Use Case |
|--------------------|----------|
| Event Coalescing | Multiple rapid state changes for same task instance |
| Rate Limiting | Maximum events per second per client connection |
| Priority Queueing | Critical events (failures) bypass throttling |
| Batch Updates | Group related events into single update messages |
| Client Filtering | Skip events for resources not currently visible |

The throttling system ensures that critical events like task failures are always delivered immediately while less urgent updates like progress increments may be batched or delayed.

> **Architecture Decision: Server-Sent Events vs WebSockets**
> - **Context**: Need bidirectional real-time communication between web interface and workflow orchestrator for status updates and control commands
> - **Options Considered**: 
>   1. Server-Sent Events (SSE) for updates + HTTP POST for commands
>   2. WebSockets for full bidirectional communication
>   3. Polling-based updates with HTTP requests
> - **Decision**: SSE for status updates, HTTP POST for commands
> - **Rationale**: SSE provides automatic reconnection, better browser support, simpler debugging, and built-in event formatting. Commands are infrequent enough that HTTP POST overhead is negligible. WebSockets add complexity for multiplexing and connection management without significant benefit for our use case.
> - **Consequences**: Slightly higher latency for commands due to separate HTTP requests, but much simpler client-side code and better reliability under network issues.

**Event Payload Format:**

Real-time events use a standardized JSON format that includes metadata for routing and processing:

| Field | Type | Purpose |
|-------|------|---------|
| event_type | string | Event classification for client-side routing |
| resource_type | string | Type of resource affected (dag, task_instance, worker) |
| resource_id | string | Specific identifier for the affected resource |
| timestamp | string | ISO 8601 timestamp when event occurred |
| data | object | Event-specific payload with state changes |
| correlation_id | string | Links related events in a sequence |

The standardized format enables generic client-side event processing logic that can handle new event types without code changes, supporting system extensibility.

⚠️ **Pitfall: Memory Leaks in Event Subscriptions**
A common mistake is creating event listeners or SSE connections without properly cleaning them up when components unmount or views change. This leads to memory leaks and duplicate event processing. Always implement cleanup functions that close connections and remove listeners, especially in single-page applications where components mount and unmount frequently.

### Architecture Decision: Frontend Technology

The choice of frontend technology stack significantly impacts development velocity, user experience, and long-term maintainability of the web interface. This decision must balance the complexity of real-time workflow monitoring with the need for responsive, interactive data visualization.

> **Decision: Hybrid Server-Side Rendering with Client-Side Interactivity**
> - **Context**: Need to support both quick page loads for operational monitoring and rich interactivity for workflow management, while maintaining simplicity for a small development team
> - **Options Considered**:
>   1. Pure server-side rendering (SSR) with minimal JavaScript
>   2. Single Page Application (SPA) with client-side routing
>   3. Hybrid approach with SSR for initial load and SPA for interactions
> - **Decision**: Hybrid approach using Go templates for initial rendering and React components for interactive elements
> - **Rationale**: Server-side rendering provides fast initial page loads critical for operational dashboards, while client-side components enable rich interactivity for graph visualization and real-time updates without full page reloads. This approach minimizes JavaScript bundle size while supporting complex UI requirements.
> - **Consequences**: Slightly more complex build process and component architecture, but better performance characteristics and easier debugging than pure SPA approach.

**Technology Stack Comparison:**

| Approach | Pros | Cons | Complexity |
|----------|------|------|------------|
| Pure SSR | Fast page loads, simple deployment, good SEO | Limited interactivity, full page refreshes | Low |
| Pure SPA | Rich interactivity, smooth UX, offline capability | Slow initial load, complex state management | High |
| Hybrid | Best of both approaches, progressive enhancement | More complex architecture, build tooling | Medium |

The hybrid approach uses server-side rendering for the initial page shell and static content, then enhances specific components with client-side JavaScript for interactivity. This strategy provides excellent perceived performance because users see content immediately while JavaScript loads and initializes interactive features in the background.

**Component Architecture:**

The frontend architecture divides functionality between server-rendered templates and client-side React components based on interactivity requirements:

| Component Type | Rendering Strategy | Use Cases |
|----------------|-------------------|-----------|
| Navigation | Server-side template | Static menus, breadcrumbs, authentication status |
| DAG List | Server-side with client enhancement | Initial table render, then real-time updates |
| DAG Graph | Client-side React | Interactive visualization requiring state management |
| Log Viewer | Client-side React | Real-time streaming, search, and filtering |
| Forms | Progressive enhancement | Basic HTML forms enhanced with validation |

The server-side templates use Go's `html/template` package with a component-like structure that promotes reusability and maintainability. Client-side components use React with functional components and hooks for state management, keeping the JavaScript bundle size minimal.

**State Management Strategy:**

Rather than using complex state management libraries like Redux, the hybrid architecture uses a simpler approach that leverages the server as the source of truth:

| State Type | Management Strategy |
|------------|-------------------|
| Initial Data | Embedded in HTML via server-side rendering |
| Real-time Updates | Local component state updated via SSE events |
| User Interactions | Optimistic updates with server reconciliation |
| Navigation State | Browser history API with server-side routing |
| Authentication | HTTP-only cookies managed server-side |

This approach reduces client-side complexity while providing responsive user interactions through optimistic updates that assume success and reconcile with server state when operations complete.

**Build and Deployment Pipeline:**

The hybrid approach requires coordinated building of both server-side templates and client-side JavaScript:

| Build Stage | Process | Output |
|-------------|---------|--------|
| Asset Compilation | Compile TypeScript/SCSS, optimize images | Static assets with content hashes |
| Bundle Generation | Webpack bundling with code splitting | JavaScript chunks for each major component |
| Template Processing | Go template compilation with asset references | HTML templates with correct asset URLs |
| Binary Building | Go build with embedded static assets | Single binary containing server and UI |

The build process embeds all static assets into the Go binary using `embed` directives, creating a single deployable artifact that doesn't require separate static file serving infrastructure.

⚠️ **Pitfall: Over-Engineering the Frontend**
Many developers jump directly to complex SPA frameworks and state management libraries because they're familiar from other projects. For a workflow orchestrator, the primary users are operations teams who prioritize fast loading and reliability over smooth animations. Start with server-side rendering and add client-side interactivity only where it provides clear user experience benefits.

**Progressive Enhancement Strategy:**

The interface is designed to work at multiple levels of JavaScript availability:

| JavaScript State | User Experience |
|------------------|----------------|
| Disabled | Basic functionality with form submissions and page reloads |
| Loading | Server-rendered content visible, loading indicators for enhanced features |
| Partial | Core interactivity available, some advanced features still loading |
| Full | Complete feature set with real-time updates and rich interactions |

This progressive enhancement ensures that the interface remains usable even when JavaScript fails to load or execute, which is critical for operational tooling that must work reliably during incidents.

The hybrid approach strikes an optimal balance between performance, complexity, and user experience for workflow orchestration interfaces, providing the responsiveness needed for operational monitoring while maintaining the simplicity required for reliable incident response.

### Implementation Guidance

The web UI and API layer implementation brings together HTTP serving, real-time communication, and frontend rendering in a cohesive package that provides both programmatic and visual access to the workflow orchestrator.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| HTTP Framework | `net/http` with `gorilla/mux` | `gin-gonic/gin` with middleware |
| Template Engine | `html/template` (stdlib) | `html/template` with custom helpers |
| Real-time Updates | Server-Sent Events | WebSocket with `gorilla/websocket` |
| Frontend Build | Manual asset serving | Webpack with Go embed |
| API Documentation | Hand-written docs | OpenAPI/Swagger generation |
| Authentication | Basic HTTP auth | JWT with middleware |

**Recommended File Structure:**
```
internal/web/
  server/
    server.go              ← HTTP server setup and middleware
    routes.go              ← Route definitions and handlers
    auth.go                ← Authentication middleware
  api/
    handlers/
      dags.go              ← DAG resource endpoints
      dagruns.go           ← DAG run endpoints  
      tasks.go             ← Task instance endpoints
      health.go            ← System health endpoints
    middleware/
      cors.go              ← Cross-origin request handling
      logging.go           ← Request/response logging
      ratelimit.go         ← API rate limiting
  ui/
    templates/
      layout.html          ← Base page template
      dag_list.html        ← DAG listing page
      dag_detail.html      ← DAG execution graph
      task_logs.html       ← Log viewer page
    static/
      js/
        dag_graph.js       ← Interactive DAG visualization
        log_viewer.js      ← Real-time log streaming
        sse_client.js      ← Server-sent events handling
      css/
        styles.css         ← Dashboard styling
  events/
    stream.go              ← Server-sent events implementation
    hub.go                 ← Event routing and client management
cmd/web/
  main.go                  ← Web server entry point
```

**Infrastructure Starter Code:**

Complete HTTP server setup with middleware stack:

```go
package server

import (
    "context"
    "database/sql"
    "html/template"
    "net/http"
    "time"
    
    "github.com/gorilla/mux"
)

// Server encapsulates the HTTP server and its dependencies
type Server struct {
    router    *mux.Router
    db        *sql.DB
    templates *template.Template
    eventHub  *events.Hub
    config    ServerConfig
    logger    Logger
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
    Host         string        `yaml:"host"`
    Port         int           `yaml:"port"`
    ReadTimeout  time.Duration `yaml:"read_timeout"`
    WriteTimeout time.Duration `yaml:"write_timeout"`
    CORS         CORSConfig    `yaml:"cors"`
}

type CORSConfig struct {
    AllowedOrigins []string `yaml:"allowed_origins"`
    AllowedMethods []string `yaml:"allowed_methods"`
    AllowedHeaders []string `yaml:"allowed_headers"`
}

// NewServer creates a new HTTP server with all middleware configured
func NewServer(db *sql.DB, eventHub *events.Hub, config ServerConfig, logger Logger) *Server {
    s := &Server{
        router:   mux.NewRouter(),
        db:       db,
        eventHub: eventHub,
        config:   config,
        logger:   logger,
    }
    
    // Load HTML templates with custom functions
    s.loadTemplates()
    
    // Setup middleware stack
    s.setupMiddleware()
    
    // Register all routes
    s.setupRoutes()
    
    return s
}

// Start begins serving HTTP requests
func (s *Server) Start(ctx context.Context) error {
    server := &http.Server{
        Addr:         fmt.Sprintf("%s:%d", s.config.Host, s.config.Port),
        Handler:      s.router,
        ReadTimeout:  s.config.ReadTimeout,
        WriteTimeout: s.config.WriteTimeout,
    }
    
    // Start server in goroutine
    go func() {
        if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            s.logger.Fatalf("HTTP server error: %v", err)
        }
    }()
    
    s.logger.Infof("HTTP server started on %s", server.Addr)
    
    // Wait for context cancellation
    <-ctx.Done()
    
    // Graceful shutdown
    shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    return server.Shutdown(shutdownCtx)
}

// setupMiddleware configures the middleware stack
func (s *Server) setupMiddleware() {
    // Request logging
    s.router.Use(s.loggingMiddleware)
    
    // CORS handling
    s.router.Use(s.corsMiddleware)
    
    // Request timeout
    s.router.Use(s.timeoutMiddleware(30 * time.Second))
    
    // Rate limiting
    s.router.Use(s.rateLimitMiddleware)
    
    // Authentication (applied selectively in routes)
    // Applied per-route rather than globally
}

// setupRoutes defines all HTTP endpoints
func (s *Server) setupRoutes() {
    // API routes
    api := s.router.PathPrefix("/api/v1").Subrouter()
    api.Use(s.authMiddleware) // Require auth for API endpoints
    
    // DAG management endpoints
    api.HandleFunc("/dags", s.listDAGs).Methods("GET")
    api.HandleFunc("/dags/{dag_id}", s.getDAG).Methods("GET")
    api.HandleFunc("/dags/{dag_id}/dagRuns", s.createDAGRun).Methods("POST")
    api.HandleFunc("/dags/{dag_id}/dagRuns", s.listDAGRuns).Methods("GET")
    
    // DAG run endpoints
    api.HandleFunc("/dagRuns/{dagrun_id}", s.getDAGRun).Methods("GET")
    api.HandleFunc("/dagRuns/{dagrun_id}", s.updateDAGRun).Methods("PATCH")
    api.HandleFunc("/dagRuns/{dagrun_id}/taskInstances", s.listTaskInstances).Methods("GET")
    
    // Task instance endpoints
    api.HandleFunc("/taskInstances/{task_instance_id}", s.getTaskInstance).Methods("GET")
    api.HandleFunc("/taskInstances/{task_instance_id}/logs", s.getTaskLogs).Methods("GET")
    api.HandleFunc("/taskInstances/{task_instance_id}/clear", s.clearTaskInstance).Methods("POST")
    
    // Real-time events
    api.HandleFunc("/events", s.handleSSEConnection).Methods("GET")
    
    // Health check
    api.HandleFunc("/health", s.healthCheck).Methods("GET")
    
    // UI routes (no auth required for read-only views)
    ui := s.router.PathPrefix("").Subrouter()
    ui.HandleFunc("/", s.renderDAGList).Methods("GET")
    ui.HandleFunc("/dags/{dag_id}", s.renderDAGDetail).Methods("GET")
    ui.HandleFunc("/dags/{dag_id}/runs/{dagrun_id}", s.renderDAGRun).Methods("GET")
    ui.HandleFunc("/tasks/{task_instance_id}/logs", s.renderTaskLogs).Methods("GET")
    
    // Static assets
    ui.PathPrefix("/static/").Handler(http.StripPrefix("/static/", 
        http.FileServer(http.FS(staticAssets))))
}
```

Event streaming infrastructure for real-time updates:

```go
package events

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
    "time"
)

// Event represents a state change in the workflow orchestrator
type Event struct {
    Type          string                 `json:"event_type"`
    ResourceType  string                 `json:"resource_type"`
    ResourceID    string                 `json:"resource_id"`
    Timestamp     time.Time              `json:"timestamp"`
    Data          map[string]interface{} `json:"data"`
    CorrelationID string                 `json:"correlation_id,omitempty"`
}

// Client represents a connected SSE client
type Client struct {
    ID       string
    Channel  chan Event
    Filters  map[string]string // Resource type/ID filters
    LastSeen time.Time
    writer   http.ResponseWriter
    flusher  http.Flusher
}

// Hub manages event distribution to connected clients
type Hub struct {
    clients    map[string]*Client
    register   chan *Client
    unregister chan *Client
    broadcast  chan Event
    mu         sync.RWMutex
    logger     Logger
}

// NewHub creates a new event hub
func NewHub(logger Logger) *Hub {
    return &Hub{
        clients:    make(map[string]*Client),
        register:   make(chan *Client),
        unregister: make(chan *Client),
        broadcast:  make(chan Event, 1000), // Buffer for burst events
        logger:     logger,
    }
}

// Start begins the hub's event processing loop
func (h *Hub) Start(ctx context.Context) {
    // Cleanup ticker for stale connections
    cleanupTicker := time.NewTicker(30 * time.Second)
    defer cleanupTicker.Stop()
    
    for {
        select {
        case client := <-h.register:
            h.mu.Lock()
            h.clients[client.ID] = client
            h.mu.Unlock()
            h.logger.Debugf("Client %s connected, total: %d", client.ID, len(h.clients))
            
        case client := <-h.unregister:
            h.mu.Lock()
            if _, ok := h.clients[client.ID]; ok {
                delete(h.clients, client.ID)
                close(client.Channel)
            }
            h.mu.Unlock()
            h.logger.Debugf("Client %s disconnected, total: %d", client.ID, len(h.clients))
            
        case event := <-h.broadcast:
            h.distributeEvent(event)
            
        case <-cleanupTicker.C:
            h.cleanupStaleClients()
            
        case <-ctx.Done():
            h.shutdown()
            return
        }
    }
}

// PublishEvent sends an event to all interested clients
func (h *Hub) PublishEvent(event Event) {
    select {
    case h.broadcast <- event:
    default:
        h.logger.Warnf("Event broadcast channel full, dropping event: %+v", event)
    }
}

// RegisterClient adds a new SSE client connection
func (h *Hub) RegisterClient(w http.ResponseWriter, r *http.Request, filters map[string]string) *Client {
    flusher, ok := w.(http.Flusher)
    if !ok {
        return nil
    }
    
    client := &Client{
        ID:       generateClientID(),
        Channel:  make(chan Event, 100),
        Filters:  filters,
        LastSeen: time.Now(),
        writer:   w,
        flusher:  flusher,
    }
    
    h.register <- client
    return client
}
```

**Core Logic Skeleton Code:**

API handlers that bridge HTTP requests to orchestrator operations:

```go
// listDAGs returns all registered DAGs with summary information
func (s *Server) listDAGs(w http.ResponseWriter, r *http.Request) {
    // TODO 1: Parse query parameters for pagination (limit, cursor) and filtering
    // TODO 2: Query database for DAG list with applied filters and pagination
    // TODO 3: For each DAG, fetch last run status and next scheduled execution
    // TODO 4: Build response object with DAGs array, pagination metadata
    // TODO 5: Set appropriate cache headers for DAG list data
    // TODO 6: Return JSON response with proper status code
    // Hint: Use prepared statements for efficient database queries
}

// getDAG returns detailed information about a specific DAG
func (s *Server) getDAG(w http.ResponseWriter, r *http.Request) {
    // TODO 1: Extract dag_id from URL path parameters using mux.Vars()
    // TODO 2: Validate dag_id format and check if DAG exists in database
    // TODO 3: Fetch DAG definition, configuration, and current status
    // TODO 4: Get recent run history and execution statistics
    // TODO 5: Build comprehensive DAG response object
    // TODO 6: Return JSON response or 404 if DAG not found
    // Hint: Return 404 for missing DAGs, 400 for invalid DAG IDs
}

// createDAGRun triggers a new execution of the specified DAG
func (s *Server) createDAGRun(w http.ResponseWriter, r *http.Request) {
    // TODO 1: Extract dag_id from URL and parse request body JSON
    // TODO 2: Validate execution_date format and parameter values
    // TODO 3: Check if DAG run already exists for this execution date
    // TODO 4: Call scheduler's HandleManualTrigger method
    // TODO 5: Return created DAG run details with 201 status
    // TODO 6: Handle conflicts (409) for duplicate executions
    // Hint: Use replace_existing parameter to control duplicate behavior
}

// handleSSEConnection establishes server-sent events connection
func (s *Server) handleSSEConnection(w http.ResponseWriter, r *http.Request) {
    // TODO 1: Set SSE response headers (text/event-stream, no-cache, CORS)
    // TODO 2: Parse query parameters for event filtering (dag_id, resource_type)
    // TODO 3: Register client with event hub using parsed filters
    // TODO 4: Start goroutine to send events to client connection
    // TODO 5: Handle client disconnection and cleanup resources
    // TODO 6: Implement heartbeat to detect stale connections
    // Hint: Use context cancellation to handle client disconnects gracefully
}
```

Frontend JavaScript for real-time graph updates:

```javascript
// DAG graph visualization with real-time updates
class DAGGraph {
    constructor(containerId, dagData) {
        // TODO 1: Initialize D3.js force simulation with nodes and links
        // TODO 2: Set up SVG container with zoom and pan capabilities
        // TODO 3: Create node and edge visual elements with initial styling
        // TODO 4: Connect to SSE endpoint with DAG-specific filters
        // TODO 5: Implement event handlers for task state changes
        // TODO 6: Add click handlers for task detail panels
        // Hint: Use D3's force simulation for automatic layout positioning
    }
    
    updateTaskState(taskId, newState) {
        // TODO 1: Find task node in graph data structure
        // TODO 2: Update node styling based on new state (color, icon)
        // TODO 3: Trigger smooth CSS transition for visual feedback
        // TODO 4: Update any connected edges if needed
        // Hint: Use CSS classes for state styling rather than inline styles
    }
    
    handleRealtimeEvent(event) {
        // TODO 1: Parse event data and extract relevant information
        // TODO 2: Determine which graph elements need updating
        // TODO 3: Apply updates with appropriate animations
        // TODO 4: Update task detail panel if currently selected
        // Hint: Batch multiple rapid updates to avoid UI thrashing
    }
}

// Server-Sent Events client with reconnection logic
class SSEClient {
    constructor(endpoint, filters = {}) {
        // TODO 1: Construct SSE URL with filter query parameters
        // TODO 2: Initialize EventSource with proper configuration
        // TODO 3: Set up event listeners for different event types
        // TODO 4: Implement automatic reconnection with exponential backoff
        // TODO 5: Add connection state indicators for user feedback
        // TODO 6: Handle browser tab visibility changes
        // Hint: Store reconnection state to avoid duplicate connections
    }
    
    connect() {
        // TODO 1: Create new EventSource connection to server
        // TODO 2: Set up onopen, onmessage, onerror handlers
        // TODO 3: Parse incoming JSON event data
        // TODO 4: Dispatch events to registered callbacks
        // TODO 5: Handle connection errors and initiate reconnection
        // Hint: Use exponential backoff for reconnection attempts
    }
    
    reconnect() {
        // TODO 1: Close existing connection if still open
        // TODO 2: Calculate backoff delay based on attempt count
        // TODO 3: Show reconnection status to user
        // TODO 4: Attempt new connection after delay
        // TODO 5: Reset backoff on successful connection
        // Hint: Cap maximum backoff delay and provide manual retry option
    }
}
```

**Language-Specific Hints:**

- Use `html/template.FuncMap` to add custom template functions for formatting dates, durations, and states
- Implement middleware as `func(http.Handler) http.Handler` for composable request processing
- Use `context.WithTimeout` for API requests to prevent hanging connections
- Leverage `embed` directive to include static assets in the binary: `//go:embed static`
- Use `gorilla/websocket.Upgrader` with proper origin checking for WebSocket connections
- Implement graceful shutdown with `sync.WaitGroup` to ensure connections close cleanly

**Milestone Checkpoint:**

After implementing the web UI and API layer:

1. **Start the web server**: `go run cmd/web/main.go --config config.yaml`
2. **Verify API endpoints**: `curl http://localhost:8080/api/v1/health` should return system status
3. **Check DAG listing**: Navigate to `http://localhost:8080/` should show registered DAGs
4. **Test real-time updates**: Open developer tools Network tab, look for EventSource connection
5. **Trigger a DAG run**: Use the manual trigger button, verify task states update in real-time
6. **View task logs**: Click on a running task, verify log streaming works

**Signs of Issues:**
- **Server won't start**: Check port conflicts, database connectivity, template loading errors
- **SSE not connecting**: Verify CORS headers, check browser network tab for connection errors  
- **Graph not rendering**: Check JavaScript console for D3.js errors, verify DAG data format
- **Logs not streaming**: Confirm WebSocket/SSE connection, check file permissions on log files

The web UI and API layer provides the essential interface for monitoring and managing workflow orchestration, combining efficient server-side rendering with rich client-side interactivity for an optimal user experience.


## Component Interactions and Data Flow

> **Milestone(s):** DAG Definition (DAG registration and validation flow), Scheduler (scheduling triggers and DAG run creation), Task Execution (task execution coordination), Worker & Executor (distributed task execution and worker communication), Web UI & Monitoring (real-time status updates and API interactions)

The component interactions and data flow represent the circulatory system of the workflow orchestrator. Just as the human circulatory system coordinates between the heart, lungs, brain, and organs to sustain life, our orchestrator must coordinate between the scheduler, executor, workers, database, and message queue to execute workflows reliably. Understanding these interactions is crucial because the majority of production issues in workflow orchestration stem from communication failures, race conditions, or inconsistent state between components rather than individual component bugs.

This section provides three critical flow patterns that form the backbone of the orchestrator: DAG registration (how workflows enter the system), task execution (how work gets distributed and completed), and failure recovery (how the system maintains reliability when components fail). Each flow involves multiple components working in concert, with careful attention to state consistency, error handling, and performance optimization.

### Mental Model: The Orchestra Performance

Think of the workflow orchestrator as a symphony orchestra performing a complex piece. The **DAG registration flow** is like musicians arriving, tuning their instruments, and the conductor reviewing the sheet music to ensure everyone understands their parts. The **task execution flow** is the actual performance, where the conductor (scheduler) cues different sections (workers) at precisely the right moments based on the musical score (DAG dependencies). The **failure recovery flow** is like handling when a musician misses their entrance or an instrument breaks mid-performance—the conductor must adapt, reassign parts, and keep the music flowing without the audience noticing the disruption.

Each musician (worker) must know not only their own part but also when to listen for cues from other sections (task dependencies). The conductor (scheduler) must maintain awareness of the entire performance while making real-time decisions about tempo, dynamics, and recovery from mistakes. The sheet music (DAG definition) provides the blueprint, but the actual performance requires constant coordination and communication between all participants.

### DAG Registration Flow

The DAG registration flow represents how workflow definitions transition from static files on disk to active, schedulable workflows in the orchestrator. This process involves file discovery, parsing, validation, template rendering, and atomic database registration. The flow must handle concurrent file modifications, validation errors, and database consistency while providing clear feedback about registration status.

![System Component Architecture](./diagrams/system-components.svg)

The registration process begins with the **FileDiscovery** component continuously scanning the configured DAG directory for new, modified, or deleted YAML files. This scanning happens at regular intervals defined by the `DAGConfig.ScanInterval` setting, typically every 30-60 seconds to balance responsiveness with system load. The discovery component maintains file modification timestamps and checksums to detect changes efficiently without re-parsing unchanged files.

When a DAG file is detected as new or modified, the **Parser** component takes over the processing. The parser first loads the raw YAML content and performs initial syntax validation to ensure the file contains valid YAML structure. Next, it deserializes the content into a `DAGDefinition` struct, mapping YAML fields to strongly-typed Go structures with proper validation of required fields and data types.

The template rendering phase processes any parameterized expressions within the DAG definition using the **TemplateEngine**. This includes schedule expressions, task parameters, and operator configurations that may contain variables like `{{ params.environment }}` or `{{ execution_date }}`. The template engine validates syntax without fully rendering dynamic content, ensuring templates will work correctly when provided with actual execution context.

Following successful parsing and template validation, the **Validator** component performs comprehensive DAG structure validation. This multi-step process includes dependency reference checking (ensuring all task dependencies refer to existing tasks), cycle detection using topological sort, reachability analysis (confirming all tasks can be reached from DAG entry points), and resource requirement validation.

The cycle detection algorithm implements a depth-first search with a three-color marking system: white (unvisited), gray (currently being processed), and black (completely processed). When the algorithm encounters a gray node during traversal, it has found a back edge indicating a cycle. The validator reports the specific tasks involved in any detected cycles, providing actionable error messages for DAG authors.

Once validation completes successfully, the DAG registration enters the atomic database transaction phase. The system creates a new `DAG` record with a unique identifier, stores all `Task` definitions with their dependency relationships, and indexes the DAG for efficient querying. If any database operation fails, the entire transaction rolls back, ensuring the system never contains partially registered DAGs.

| Registration Phase | Component | Input | Output | Error Handling |
|-------------------|-----------|--------|---------|----------------|
| File Discovery | FileDiscovery | Directory path, file pattern | List of changed files | Log scan errors, continue with available files |
| YAML Parsing | Parser | Raw file content | DAGDefinition struct | Return parse errors with line numbers |
| Template Validation | TemplateEngine | Template expressions | Validated templates | Report syntax errors with context |
| Structure Validation | Validator | DAGDefinition | Validated DAG | Return validation errors with task details |
| Database Registration | Creator | Validated DAG | Database records | Rollback transaction on any failure |
| Index Updates | Scheduler | Registered DAG | Updated schedule index | Log indexing errors, retry registration |

The registration flow handles several concurrent scenarios gracefully. When multiple DAG files change simultaneously, the parser processes them in parallel using a worker pool, with each worker handling one file at a time to prevent resource exhaustion. File deletion triggers a cleanup process that removes the DAG from active schedules, marks existing runs as orphaned, and archives historical data according to retention policies.

**Hot reloading** functionality allows the scheduler to pick up DAG changes without requiring a full system restart. When a DAG is successfully re-registered, the scheduler compares the new version with the existing version to identify changes in schedule, dependencies, or task configurations. Schedule changes trigger immediate recalculation of next execution times, while dependency changes affect how future DAG runs will execute without impacting currently running instances.

The registration flow maintains **audit trails** for all DAG changes, recording who modified which DAG when, what changes were made, and whether registration succeeded or failed. This audit information proves essential for debugging production issues, especially when DAG modifications inadvertently break existing workflows or introduce performance problems.

#### Registration State Machine

The DAG registration process follows a well-defined state machine that tracks each DAG file through its registration lifecycle:

| Current State | Event | Next State | Actions Taken |
|--------------|-------|------------|---------------|
| Unknown | File Discovered | Parsing | Begin YAML parsing, create temp record |
| Parsing | Parse Success | Validating | Execute structure validation checks |
| Parsing | Parse Failure | Failed | Log errors, notify monitoring, cleanup temp data |
| Validating | Validation Success | Registering | Begin database transaction |
| Validating | Validation Failure | Failed | Log validation errors with details |
| Registering | Database Success | Active | Commit transaction, update scheduler index |
| Registering | Database Failure | Failed | Rollback transaction, log database errors |
| Active | File Modified | Parsing | Create new version, keep current active |
| Active | File Deleted | Archived | Remove from schedules, archive historical data |
| Failed | File Modified | Parsing | Retry registration with new content |

**Common Registration Pitfalls:**

⚠️ **Pitfall: Incomplete Cycle Detection**
Many implementations only check for direct circular dependencies (A depends on B, B depends on A) but miss complex cycles involving multiple tasks (A → B → C → D → A). The topological sort algorithm naturally detects all cycles, but developers sometimes implement simpler checks that miss these cases. Always use depth-first search with proper marking to detect all possible cycles.

⚠️ **Pitfall: Missing Template Validation**
Skipping template syntax validation during registration means errors only surface at execution time, potentially failing workflows hours or days after DAG deployment. Always validate template syntax during registration, even if you can't render the full content without execution context. Use dummy context values to test template rendering during validation.

⚠️ **Pitfall: Non-Atomic Registration**
Partially successful registration can leave the system in an inconsistent state where some tasks exist in the database but others don't, or where the DAG exists but its schedule isn't active. Always wrap the entire registration process in a database transaction and rollback completely on any failure.

### Task Execution Flow

The task execution flow orchestrates the complete lifecycle of individual task execution within a DAG run, from initial scheduling trigger through final result persistence. This flow involves careful coordination between the scheduler (determining readiness), executor (task distribution), workers (actual execution), and database (state persistence), with multiple feedback loops ensuring consistency and reliability.

![DAG Execution Sequence](./diagrams/dag-execution-sequence.svg)

The execution flow begins when the **Scheduler** identifies a DAG that should run based on its cron schedule or manual trigger. The scheduler creates a new `DAGRun` record with state `queued` and generates corresponding `TaskInstance` records for every task in the DAG, each initially in state `none`. This atomic creation ensures that either a complete DAG run exists with all its task instances or nothing is created if any step fails.

Following DAG run creation, the scheduler enters the **dependency evaluation phase**, continuously scanning for task instances that have become ready to execute. The `DependencyResolver` component examines each task's upstream dependencies, checking that all required predecessor tasks have completed successfully (or according to their specific trigger rules). Tasks with no upstream dependencies become immediately eligible for execution.

When the dependency resolver identifies ready tasks, it transitions their state from `none` to `queued` and submits them to the **DistributedExecutor**. The executor maintains an internal queue of pending tasks and uses the **WorkerRegistry** to identify available workers with appropriate capacity and capabilities for each task type. The executor considers factors like worker resource availability, task resource requirements, and worker specialization (some workers may only handle certain operator types).

Task distribution occurs through the **MessageQueue** component, which publishes `TaskMessage` records to Redis streams consumed by worker processes. Each task message contains complete execution context: task configuration, operator parameters, upstream results (via XCom), retry information, and timeout settings. The message queue provides delivery guarantees and consumer group functionality, ensuring tasks are distributed exactly once even if multiple executor processes are running.

Worker processes continuously poll the message queue for available tasks using consumer groups to prevent duplicate task assignment. When a worker receives a task, it acknowledges the message, transitions the task instance state to `running`, and begins execution through the appropriate **Operator** implementation. The operator creates an **ExecutionContext** with all necessary services (logging, XCom, metrics collection) and executes the task's core logic.

During task execution, workers periodically update the task instance with progress information and maintain heartbeat signals with the **WorkerRegistry**. This heartbeat mechanism allows the system to detect worker failures and implement task timeout enforcement. If a worker fails to report within the configured heartbeat interval, the executor marks it as unhealthy and reassigns its active tasks to other available workers.

Task completion involves the worker publishing a **TaskResult** message back through the message queue, containing the final execution state (`success`, `failed`, `up_for_retry`), exit code, log output, XCom values, and execution duration. The executor consumes these results and updates the corresponding task instance records atomically, ensuring state consistency even under concurrent operations.

| Execution Phase | Component | Primary Actions | State Changes | Error Recovery |
|----------------|-----------|-----------------|---------------|----------------|
| Schedule Trigger | Scheduler | Create DAGRun and TaskInstances | DAGRun: queued, Tasks: none | Rollback creation on failure |
| Dependency Check | DependencyResolver | Evaluate upstream completion | Tasks: none → queued | Continue with available tasks |
| Task Distribution | DistributedExecutor | Publish to message queue | Tasks: queued → running | Retry message publishing |
| Task Execution | Worker | Execute operator logic | Tasks: running → success/failed | Handle timeout and resource limits |
| Result Collection | DistributedExecutor | Update database records | Tasks: final states | Retry result processing |
| Downstream Trigger | DependencyResolver | Evaluate new ready tasks | New tasks: none → queued | Handle upstream failures |

The execution flow implements sophisticated **trigger rule evaluation** beyond simple "all upstream tasks successful" dependencies. Tasks can specify trigger rules like `all_success` (default), `all_failed` (run only when all upstream tasks fail), `all_done` (run when all upstream tasks complete regardless of state), `one_success` (run when at least one upstream task succeeds), or `none_failed` (run unless any upstream task failed). These trigger rules enable complex workflow patterns like cleanup tasks, error handling paths, and conditional execution branches.

**XCom (cross-communication)** functionality allows tasks to pass small data payloads to downstream tasks. When a task completes successfully, any values it pushed to XCom become available to downstream tasks through the `XComManager`. The XCom system handles serialization, size limits, and cleanup automatically. Downstream tasks access upstream results using keys like `task_instance.xcom_pull(task_ids='upstream_task', key='result_data')`.

The execution flow handles **partial failures** gracefully through upstream failure propagation. When a task fails, the dependency resolver identifies all downstream tasks that cannot proceed and marks them as `upstream_failed`, preventing unnecessary execution attempts while maintaining clear visibility into why tasks were skipped. This propagation respects trigger rules—tasks with `all_failed` trigger rules will still execute when their upstream dependencies fail.

**Task retry logic** integrates seamlessly into the execution flow. When a task fails but has remaining retry attempts, the executor waits for the configured retry delay (with exponential backoff), then re-queues the task for another execution attempt. The retry counter increments with each attempt, and the system maintains detailed logs of all retry attempts for debugging purposes.

#### Task State Transition Flow

The task execution flow manages complex state transitions that must handle concurrent operations, failure scenarios, and retry logic:

| Current State | Trigger Event | Next State | Required Conditions | Side Effects |
|--------------|---------------|------------|-------------------|--------------|
| none | Dependencies met | queued | All upstream tasks in final states matching trigger rule | Add to executor queue |
| queued | Worker available | running | Worker has capacity and task matches capabilities | Send TaskMessage to worker |
| running | Task completes successfully | success | Exit code 0 and no exceptions | Update duration, store XCom, trigger downstream |
| running | Task fails with retries left | up_for_retry | Exit code non-zero and try_number < max_retries | Schedule retry with backoff delay |
| running | Task fails with no retries | failed | Exit code non-zero and try_number >= max_retries | Mark downstream as upstream_failed |
| running | Worker timeout | up_for_retry or failed | No heartbeat within timeout period | Reassign to new worker or fail |
| up_for_retry | Retry delay expired | queued | Current time > retry_time | Increment try_number, reset timer |
| success | Manual clear | none | User administrative action | Clear XCom, reset counters |

**Performance Optimization Strategies:**

The execution flow incorporates several performance optimizations critical for handling large-scale workflows. **Task batching** groups multiple small tasks for distribution in single message queue operations, reducing network overhead. **Dependency caching** maintains in-memory indexes of task relationships to avoid repeated database queries during ready task evaluation. **State change buffering** batches database updates for multiple task state transitions, reducing database load during high-throughput periods.

**Worker affinity** allows tasks to specify preferences for workers with particular capabilities, such as GPU access, specific software installations, or network connectivity to external systems. The executor considers these preferences during task assignment while maintaining load balancing across the worker pool.

⚠️ **Pitfall: Race Conditions in State Updates**
Concurrent state updates can lead to lost updates or inconsistent task states. For example, if a worker reports task completion while the scheduler is marking the task as timed out, the final state depends on which update arrives last. Always use database transactions with proper isolation levels and consider implementing optimistic locking with version fields on task instances.

⚠️ **Pitfall: Memory Leaks in Long-Running Executors**
The executor maintains various in-memory caches and queues that can grow unbounded if not properly managed. Task queues can accumulate failed tasks, worker registries can retain dead workers, and XCom caches can consume excessive memory. Implement regular cleanup routines with configurable retention policies and memory usage monitoring.

⚠️ **Pitfall: Inadequate Timeout Handling**
Task timeouts must be enforced at multiple levels: worker-side execution timeout, message queue message TTL, and scheduler-side stuck task detection. Relying on only one timeout mechanism can lead to zombie tasks that consume resources indefinitely. Implement layered timeout enforcement with escalating actions from gentle termination to forceful process killing.

### Failure Recovery Flow

The failure recovery flow represents the orchestrator's immune system, detecting various failure modes and implementing appropriate recovery strategies to maintain workflow execution continuity. This flow must handle worker crashes, network partitions, database connectivity issues, scheduler failures, and corrupt state scenarios while providing clear visibility into recovery actions and their effectiveness.

![Worker Health Monitoring](./diagrams/worker-health-monitoring.svg)

Failure detection operates through multiple parallel monitoring systems, each designed to catch specific failure patterns. The **WorkerRegistry** implements heartbeat-based failure detection, requiring workers to report their status every 30 seconds (configurable via `heartbeatTimeout`). When a worker fails to report within the timeout window, the registry marks it as `unhealthy` and triggers task reassignment procedures. This heartbeat system includes worker resource utilization, active task counts, and version information to support rolling updates and capacity planning.

The **MessageQueue** component provides another failure detection layer through message acknowledgment timeouts. When a worker receives a task message but fails to acknowledge completion within the configured timeout (typically 5-10 minutes longer than the task's maximum execution time), the message queue automatically makes the task available for reassignment. This mechanism handles cases where workers crash during task execution without properly reporting failure.

**Database connectivity monitoring** ensures the scheduler and executor can detect database outages and implement appropriate fallback behaviors. The system maintains a connection pool with health checks and implements exponential backoff retry logic for database operations. During database unavailability, the system can continue operating with cached state for a limited time, though new DAG runs cannot be created and task state updates will be queued for later persistence.

Task reassignment logic handles the complex process of moving failed or orphaned tasks to healthy workers. When the system detects a worker failure, the **TaskReassigner** component identifies all tasks currently assigned to that worker and evaluates their states. Tasks in `running` state get moved to `up_for_retry` if they have remaining retry attempts, or to `failed` if they've exhausted their retries. The reassigner respects task resource requirements and worker capabilities when selecting new workers for reassigned tasks.

The failure recovery flow implements **circuit breaker patterns** for external dependencies to prevent cascading failures. When a task repeatedly fails due to external service unavailability (detected through specific error patterns or exit codes), the system can temporarily disable similar tasks to prevent resource waste and allow the external service time to recover. Circuit breakers have configurable failure thresholds, timeout periods, and recovery testing strategies.

| Failure Type | Detection Method | Recovery Strategy | Recovery Time | Prevention |
|-------------|-----------------|-------------------|---------------|------------|
| Worker Crash | Missing heartbeat > 60s | Reassign running tasks to healthy workers | 2-5 minutes | Resource monitoring, health checks |
| Network Partition | Message queue timeout | Queue tasks until connectivity restored | Variable | Redundant network paths |
| Database Outage | Connection pool failures | Cache operations, retry with backoff | 1-10 minutes | Database clustering, read replicas |
| Scheduler Failure | Process monitoring | Start standby scheduler instance | 30-60 seconds | Leader election, health checks |
| Corrupt Task State | State validation checks | Reset task to last known good state | Immediate | Regular state consistency checks |
| Memory Exhaustion | Resource monitoring | Restart component, rebalance load | 2-3 minutes | Memory limits, garbage collection tuning |

**Scheduler failover** handles the critical scenario where the main scheduler process crashes or becomes unresponsive. The system implements a leader election mechanism using database locks or distributed consensus (Redis-based locking). Multiple scheduler instances can run simultaneously, but only the elected leader actively creates DAG runs and evaluates task dependencies. When the leader fails, standby instances detect the failure within 30-60 seconds and elect a new leader through the consensus protocol.

**State consistency recovery** addresses scenarios where components have inconsistent views of task or DAG run states due to partial failures, race conditions, or message delivery issues. The system implements periodic consistency checks that compare task states across different components (database, message queue, worker registry) and reconcile discrepancies. These checks run every 5-10 minutes and can automatically fix common inconsistencies or alert operators to manual intervention requirements.

The recovery flow maintains **detailed failure forensics** to support root cause analysis and system improvement. Each failure event generates structured logs containing failure type, affected components, recovery actions taken, recovery success/failure, and performance impact. This information feeds into alerting systems and helps operators identify systemic issues that require architectural improvements.

**Graceful degradation** strategies allow the orchestrator to continue operating with reduced functionality during partial system failures. For example, if the web UI becomes unavailable, workflows continue executing normally with status updates queued for display when the UI recovers. If XCom storage fails, tasks can still execute but cross-task communication will be limited to return codes and log-based information passing.

#### Recovery Decision Matrix

The failure recovery system uses a decision matrix to determine appropriate recovery actions based on failure type, severity, and system state:

| Component Failed | Impact Severity | Automatic Recovery | Manual Intervention | Escalation Time |
|-----------------|----------------|-------------------|-------------------|----------------|
| Single Worker | Low | Reassign tasks, restart worker | None | N/A |
| Worker Pool (>50%) | High | Scale up emergency workers | Review resource allocation | 15 minutes |
| Message Queue | Critical | Failover to backup queue | Check queue cluster health | 5 minutes |
| Database Primary | Critical | Failover to replica | Restore primary from backup | 10 minutes |
| Scheduler Leader | Medium | Elect new leader | Check scheduler logs | 2 minutes |
| Web UI | Low | Restart UI process | None | N/A |
| File Storage | Medium | Use cached DAGs | Restore file system | 20 minutes |

**Recovery Testing and Validation:**

The failure recovery flow includes automated testing mechanisms to validate recovery procedures without impacting production workflows. **Chaos engineering** features allow operators to inject controlled failures (worker crashes, network delays, database connection drops) to test recovery behavior. These tests run during maintenance windows and generate detailed reports on recovery effectiveness and timing.

**Recovery metrics** track the system's resilience over time, including mean time to detection (MTTD), mean time to recovery (MTTR), recovery success rates, and false positive rates for failure detection. These metrics help operators tune failure detection thresholds and improve recovery procedures based on actual system behavior patterns.

⚠️ **Pitfall: Split-Brain Scenarios in Leader Election**
Network partitions can cause multiple scheduler instances to believe they are the leader, resulting in duplicate DAG runs and conflicting task assignments. Always use strongly consistent storage (like database locks with proper isolation levels) for leader election rather than eventually consistent systems. Implement lease-based leadership with automatic expiration to handle leader process crashes.

⚠️ **Pitfall: Inadequate Resource Isolation During Recovery**
Recovery operations can consume significant system resources (CPU for reassignment logic, memory for state reconciliation, network for message queue operations), potentially impacting normal workflow execution. Always implement resource limits and prioritization for recovery operations, ensuring they don't starve regular workflow execution of necessary resources.

⚠️ **Pitfall: Infinite Recovery Loops**
Poorly configured recovery logic can create infinite loops where recovery actions trigger additional failures. For example, aggressive task reassignment during worker instability can overload remaining workers, causing cascading failures. Implement exponential backoff, maximum retry limits, and circuit breakers in recovery logic to prevent runaway recovery attempts.

### Implementation Guidance

The component interactions and data flow implementation requires careful coordination between multiple concurrent processes, robust error handling, and comprehensive observability. The following guidance provides complete infrastructure components and detailed implementation patterns for the three critical flows.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Inter-Process Communication | HTTP REST + JSON (net/http) | gRPC with Protocol Buffers (google.golang.org/grpc) |
| Message Queue | Redis Streams (go-redis/redis/v8) | Apache Kafka (segmentio/kafka-go) |
| State Management | PostgreSQL with GORM | PostgreSQL with database/sql + migrations |
| Health Monitoring | HTTP endpoints + polling | Prometheus metrics + alerting |
| Distributed Locking | Redis SETNX commands | etcd distributed locks |
| Log Aggregation | Structured logging to files | ELK stack or similar |

#### Recommended File Structure

```
internal/
├── flows/
│   ├── registration/
│   │   ├── flow.go              ← DAG registration orchestrator
│   │   ├── discovery.go         ← File system monitoring
│   │   ├── validator.go         ← DAG validation logic
│   │   └── registration_test.go
│   ├── execution/
│   │   ├── flow.go              ← Task execution orchestrator  
│   │   ├── dependency.go        ← Dependency resolution
│   │   ├── distributor.go       ← Task distribution logic
│   │   └── execution_test.go
│   └── recovery/
│       ├── flow.go              ← Failure detection and recovery
│       ├── monitor.go           ← Health monitoring
│       ├── reassign.go          ← Task reassignment logic
│       └── recovery_test.go
├── messaging/
│   ├── queue.go                 ← Message queue abstraction
│   ├── redis_queue.go           ← Redis implementation
│   └── messages.go              ← Message type definitions
├── registry/
│   ├── worker_registry.go       ← Worker health tracking
│   └── heartbeat.go             ← Heartbeat processing
└── coordination/
    ├── leader_election.go       ← Scheduler leader election
    └── state_sync.go            ← State consistency checking
```

#### Infrastructure Components

Complete message queue implementation using Redis Streams:

```go
package messaging

import (
    "context"
    "encoding/json"
    "fmt"
    "time"

    "github.com/go-redis/redis/v8"
    "go.uber.org/zap"
)

// MessageQueue provides reliable task distribution using Redis Streams
type MessageQueue struct {
    client       *redis.Client
    taskStream   string
    resultStream string
    logger       *zap.Logger
}

// TaskMessage represents a task sent to workers
type TaskMessage struct {
    TaskInstanceID   string                 `json:"task_instance_id"`
    TaskType         string                 `json:"task_type"`
    OperatorConfig   map[string]interface{} `json:"operator_config"`
    ExecutionContext map[string]interface{} `json:"execution_context"`
    UpstreamResults  map[string]interface{} `json:"upstream_results"`
    RetryNumber      int                    `json:"retry_number"`
    Timeout          time.Duration          `json:"timeout"`
}

// TaskResult represents a completed task result
type TaskResult struct {
    TaskInstanceID string                 `json:"task_instance_id"`
    State          string                 `json:"state"`
    ExitCode       *int                   `json:"exit_code,omitempty"`
    LogOutput      string                 `json:"log_output"`
    XComValues     map[string]interface{} `json:"xcom_values"`
    Duration       time.Duration          `json:"duration"`
    ErrorMessage   string                 `json:"error_message,omitempty"`
}

func NewMessageQueue(redisAddr, password string, db int, logger *zap.Logger) (*MessageQueue, error) {
    client := redis.NewClient(&redis.Options{
        Addr:     redisAddr,
        Password: password,
        DB:       db,
    })

    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    if err := client.Ping(ctx).Err(); err != nil {
        return nil, fmt.Errorf("redis connection failed: %w", err)
    }

    return &MessageQueue{
        client:       client,
        taskStream:   "orchestrator:tasks",
        resultStream: "orchestrator:results",
        logger:       logger,
    }, nil
}

func (mq *MessageQueue) PublishTask(ctx context.Context, task *TaskMessage) error {
    data, err := json.Marshal(task)
    if err != nil {
        return fmt.Errorf("failed to marshal task: %w", err)
    }

    args := &redis.XAddArgs{
        Stream: mq.taskStream,
        Values: map[string]interface{}{
            "task_id": task.TaskInstanceID,
            "data":    string(data),
        },
    }

    _, err = mq.client.XAdd(ctx, args).Result()
    if err != nil {
        return fmt.Errorf("failed to publish task: %w", err)
    }

    mq.logger.Debug("Task published", 
        zap.String("task_id", task.TaskInstanceID),
        zap.String("type", task.TaskType))
    
    return nil
}

func (mq *MessageQueue) ConsumeTask(ctx context.Context, consumerGroup, consumerID string, timeout time.Duration) (*TaskMessage, string, error) {
    // Create consumer group if it doesn't exist
    mq.client.XGroupCreateMkStream(ctx, mq.taskStream, consumerGroup, "0")

    streams, err := mq.client.XReadGroup(ctx, &redis.XReadGroupArgs{
        Group:    consumerGroup,
        Consumer: consumerID,
        Streams:  []string{mq.taskStream, ">"},
        Count:    1,
        Block:    timeout,
    }).Result()

    if err != nil {
        if err == redis.Nil {
            return nil, "", nil // No messages available
        }
        return nil, "", fmt.Errorf("failed to read from stream: %w", err)
    }

    if len(streams) == 0 || len(streams[0].Messages) == 0 {
        return nil, "", nil
    }

    msg := streams[0].Messages[0]
    data, ok := msg.Values["data"].(string)
    if !ok {
        return nil, "", fmt.Errorf("invalid message format")
    }

    var task TaskMessage
    if err := json.Unmarshal([]byte(data), &task); err != nil {
        return nil, "", fmt.Errorf("failed to unmarshal task: %w", err)
    }

    return &task, msg.ID, nil
}
```

Complete worker registry with health monitoring:

```go
package registry

import (
    "context"
    "sync"
    "time"

    "go.uber.org/zap"
)

// WorkerStatus represents the current state of a worker
type WorkerStatus struct {
    WorkerID        string            `json:"worker_id"`
    LastHeartbeat   time.Time         `json:"last_heartbeat"`
    AvailableSlots  int               `json:"available_slots"`
    ActiveTasks     []string          `json:"active_tasks"`
    ResourceUsage   ResourceMetrics   `json:"resource_usage"`
    Status          string            `json:"status"` // healthy, unhealthy, draining
    Version         string            `json:"version"`
}

type ResourceMetrics struct {
    CPUPercent    float64 `json:"cpu_percent"`
    MemoryPercent float64 `json:"memory_percent"`
    DiskPercent   float64 `json:"disk_percent"`
}

// WorkerRegistry tracks worker health and availability
type WorkerRegistry struct {
    workers           map[string]*WorkerStatus
    mutex             sync.RWMutex
    heartbeatTimeout  time.Duration
    failureThreshold  int
    onWorkerFailed    func(string)
    logger            *zap.Logger
}

func NewWorkerRegistry(heartbeatTimeout time.Duration, failureThreshold int, logger *zap.Logger) *WorkerRegistry {
    return &WorkerRegistry{
        workers:          make(map[string]*WorkerStatus),
        heartbeatTimeout: heartbeatTimeout,
        failureThreshold: failureThreshold,
        logger:           logger,
    }
}

func (wr *WorkerRegistry) RegisterWorker(status *WorkerStatus) {
    wr.mutex.Lock()
    defer wr.mutex.Unlock()

    status.LastHeartbeat = time.Now()
    wr.workers[status.WorkerID] = status
    
    wr.logger.Info("Worker registered",
        zap.String("worker_id", status.WorkerID),
        zap.Int("available_slots", status.AvailableSlots))
}

func (wr *WorkerRegistry) UpdateHeartbeat(workerID string, status *WorkerStatus) {
    wr.mutex.Lock()
    defer wr.mutex.Unlock()

    status.LastHeartbeat = time.Now()
    wr.workers[workerID] = status
}

func (wr *WorkerRegistry) GetAvailableWorkers(taskType string) []*WorkerStatus {
    wr.mutex.RLock()
    defer wr.mutex.RUnlock()

    var available []*WorkerStatus
    for _, worker := range wr.workers {
        if worker.Status == "healthy" && worker.AvailableSlots > 0 {
            available = append(available, worker)
        }
    }
    return available
}

func (wr *WorkerRegistry) StartHealthMonitoring(ctx context.Context) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            wr.checkWorkerHealth()
        }
    }
}

func (wr *WorkerRegistry) checkWorkerHealth() {
    wr.mutex.Lock()
    defer wr.mutex.Unlock()

    now := time.Now()
    for workerID, worker := range wr.workers {
        if now.Sub(worker.LastHeartbeat) > wr.heartbeatTimeout {
            if worker.Status == "healthy" {
                worker.Status = "unhealthy"
                wr.logger.Warn("Worker marked unhealthy",
                    zap.String("worker_id", workerID),
                    zap.Duration("last_heartbeat", now.Sub(worker.LastHeartbeat)))
                
                if wr.onWorkerFailed != nil {
                    go wr.onWorkerFailed(workerID)
                }
            }
        }
    }
}
```

#### Core Flow Orchestration Skeletons

DAG Registration Flow implementation skeleton:

```go
package registration

import (
    "context"
    "database/sql"
    "time"

    "your-project/internal/models"
    "go.uber.org/zap"
)

// RegistrationFlow orchestrates the complete DAG registration process
type RegistrationFlow struct {
    discovery  *FileDiscovery
    parser     *Parser
    validator  *Validator
    creator    *Creator
    db         *sql.DB
    logger     *zap.Logger
}

// ProcessDAGFile handles the complete registration flow for a single DAG file
func (rf *RegistrationFlow) ProcessDAGFile(ctx context.Context, filepath string) error {
    // TODO 1: Load DAG definition from file using parser.LoadDAGDefinition()
    // TODO 2: Validate DAG structure using validator.ValidateDAG()
    // TODO 3: Check for cycles using validator.DetectCycles()
    // TODO 4: Validate all dependency references using validator.ValidateDependencyReferences()
    // TODO 5: Begin database transaction for atomic registration
    // TODO 6: Create DAG record using creator.CreateDAG()
    // TODO 7: Create all task records using creator.CreateTasks()
    // TODO 8: Update scheduler index with new DAG
    // TODO 9: Commit transaction on success, rollback on any failure
    // TODO 10: Log registration result with timing metrics
    
    // Hint: Wrap database operations in a transaction to ensure atomicity
    // Hint: Use defer for cleanup and rollback logic
    // Hint: Record detailed timing metrics for performance monitoring
    return nil
}

// StartFileWatcher begins continuous monitoring of the DAG directory
func (rf *RegistrationFlow) StartFileWatcher(ctx context.Context) error {
    // TODO 1: Create file system watcher for DAG directory
    // TODO 2: Set up polling loop with configurable interval
    // TODO 3: Detect file changes (create, modify, delete) using checksums
    // TODO 4: Process changed files using ProcessDAGFile()
    // TODO 5: Handle file deletion by removing DAGs from database
    // TODO 6: Implement rate limiting to prevent excessive processing
    // TODO 7: Log all file system events with appropriate detail
    
    // Hint: Use time.Ticker for periodic scanning
    // Hint: Maintain a map of file paths to modification times
    // Hint: Process multiple changed files concurrently with worker pool
    return nil
}
```

Task Execution Flow implementation skeleton:

```go
package execution

import (
    "context"
    "database/sql"

    "your-project/internal/messaging"
    "your-project/internal/registry"
    "go.uber.org/zap"
)

// ExecutionFlow coordinates task execution across distributed workers
type ExecutionFlow struct {
    messageQueue      *messaging.MessageQueue
    workerRegistry    *registry.WorkerRegistry
    dependencyResolver *DependencyResolver
    db                *sql.DB
    logger            *zap.Logger
}

// ProcessDAGRun manages the complete execution of a single DAG run
func (ef *ExecutionFlow) ProcessDAGRun(ctx context.Context, dagRunID string) error {
    // TODO 1: Load DAG run and all task instances from database
    // TODO 2: Identify initially ready tasks using dependency resolver
    // TODO 3: Transition ready tasks from 'none' to 'queued' state
    // TODO 4: Submit queued tasks to message queue for worker processing
    // TODO 5: Start result processing loop to handle task completions
    // TODO 6: Continue until all tasks reach final states
    // TODO 7: Update DAG run state based on task completion results
    // TODO 8: Handle partial failures and upstream failure propagation
    
    // Hint: Use goroutines for concurrent task submission and result processing
    // Hint: Implement proper cancellation handling with context
    // Hint: Track task state changes for debugging and monitoring
    return nil
}

// EvaluateReadyTasks identifies tasks that can execute based on dependencies
func (ef *ExecutionFlow) EvaluateReadyTasks(ctx context.Context, dagRunID string) ([]*models.TaskInstance, error) {
    // TODO 1: Query all task instances for the DAG run
    // TODO 2: For each task in 'none' state, check upstream dependencies
    // TODO 3: Evaluate trigger rule against upstream task states
    // TODO 4: Include tasks with trigger rule 'all_success' only if all upstream succeeded
    // TODO 5: Include tasks with trigger rule 'all_failed' only if all upstream failed
    // TODO 6: Handle other trigger rules (all_done, one_success, none_failed)
    // TODO 7: Return list of tasks ready for execution
    
    // Hint: Use map[string]TaskInstanceState for efficient upstream state lookup
    // Hint: Tasks with no upstream dependencies are always ready initially
    return nil, nil
}

// SubmitTask sends a task to available workers via message queue
func (ef *ExecutionFlow) SubmitTask(ctx context.Context, taskInstance *models.TaskInstance) error {
    // TODO 1: Find available workers using worker registry
    // TODO 2: Select optimal worker based on task requirements and worker capacity  
    // TODO 3: Create TaskMessage with execution context and upstream results
    // TODO 4: Publish task message to message queue
    // TODO 5: Update task instance state to 'running'
    // TODO 6: Record task assignment in database for monitoring
    
    // Hint: Include XCom values from upstream tasks in the task message
    // Hint: Set appropriate timeout based on task configuration
    // Hint: Handle worker selection failure gracefully
    return nil
}
```

#### Milestone Checkpoints

After implementing DAG Registration Flow:
- Run: `go test ./internal/flows/registration/... -v`
- Expected: All tests pass, including cycle detection and validation tests  
- Manual test: Place a valid DAG file in the configured directory, verify it appears in database within scan interval
- Manual test: Place an invalid DAG file (with cycles), verify registration fails with descriptive error
- Check logs: Should show file discovery events, parsing results, and registration status

After implementing Task Execution Flow:
- Run: `go test ./internal/flows/execution/... -v`
- Expected: Dependency resolution tests pass, task distribution tests pass
- Manual test: Trigger a simple DAG manually, verify tasks execute in correct dependency order
- Manual test: Trigger a DAG with parallel tasks, verify they execute concurrently
- Check Redis: Verify task messages appear in streams and are consumed by workers

After implementing Failure Recovery Flow:
- Run: `go test ./internal/flows/recovery/... -v`  
- Expected: Worker failure detection and task reassignment tests pass
- Manual test: Kill a worker process during task execution, verify tasks get reassigned
- Manual test: Simulate database connectivity issues, verify graceful degradation
- Check metrics: Monitor MTTR (mean time to recovery) and false positive rates

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Tasks stuck in 'queued' state | No available workers or message queue issues | Check worker registry health, verify Redis connectivity | Scale up workers or fix message queue |
| DAG registration fails silently | File discovery not detecting changes | Check file permissions and scan interval settings | Fix directory permissions, verify file watcher |
| Tasks execute out of dependency order | Dependency resolution logic error | Enable debug logging for dependency evaluation | Fix trigger rule evaluation logic |
| Worker heartbeats missing | Network issues or worker overload | Check worker logs and network connectivity | Tune heartbeat intervals, investigate worker performance |
| Memory usage growing continuously | Missing cleanup in caches or queues | Profile memory usage and identify growing structures | Add cleanup routines with retention policies |
| Duplicate task executions | Race conditions in state transitions | Check database isolation levels and locking | Add optimistic locking or stronger isolation |


## Error Handling and Edge Cases

> **Milestone(s):** DAG Definition (validation error handling), Scheduler (scheduling failure recovery), Task Execution (task failure and retry handling), Worker & Executor (worker failure detection and recovery), Web UI & Monitoring (system health monitoring and error reporting)

A workflow orchestrator operates in a fundamentally unreliable distributed environment where failures are not exceptions but expected occurrences. Workers crash unexpectedly, network connections drop during critical communications, databases become temporarily unavailable, and tasks fail for countless reasons ranging from data corruption to resource exhaustion. The system's resilience depends entirely on its ability to detect, isolate, recover from, and learn from these failures while maintaining consistency and preventing cascade effects.

Think of error handling in workflow orchestration like managing a large construction project during unpredictable weather. When a thunderstorm hits, you don't abandon the entire project—you protect completed work, safely pause active operations, assess what can continue indoors, and have contingency plans ready. Similarly, our orchestrator must gracefully degrade functionality during partial failures while protecting completed work and maintaining system integrity.

### Failure Mode Analysis

Understanding failure modes requires systematic analysis of every component interaction and external dependency. Each failure mode has distinct symptoms, detection mechanisms, and recovery strategies that must be carefully designed to prevent worse outcomes than the original failure.

**Database Failures** represent one of the most critical failure categories since the database serves as the single source of truth for all workflow state. Database connection failures manifest as connection timeouts, authentication errors, or network unreachability. During these failures, the scheduler cannot update DAG run states, task instances remain in stale states, and workers cannot report completion status. The system must implement connection pooling with health checks, automatic retry with exponential backoff, and circuit breakers to prevent connection pool exhaustion.

Database transaction failures occur during concurrent modifications to workflow state, such as multiple schedulers attempting to create the same DAG run or workers simultaneously updating task instance status. These manifest as deadlocks, constraint violations, or serialization errors. Recovery requires implementing optimistic locking with version fields, proper transaction isolation levels, and idempotent operations that can safely retry.

Database unavailability during critical operations like DAG run creation or task completion requires careful state management. The system must implement write-ahead logging for critical state changes, allowing recovery after database restoration. Operations must be designed as compensating transactions that can roll back partial work when the database becomes unavailable mid-operation.

| Failure Mode | Symptoms | Detection Method | Recovery Strategy |
|--------------|----------|------------------|-------------------|
| Connection Pool Exhaustion | New connections timeout or fail | Connection pool metrics, failed connection attempts | Circuit breaker activation, connection cleanup, pool size increase |
| Transaction Deadlock | Database returns deadlock error codes | Database error monitoring, transaction timeout | Exponential backoff retry, transaction ordering rules |
| Database Unavailability | All queries fail with connection errors | Health check failures, connection timeout patterns | Circuit breaker, read-only mode, cached state serving |
| Constraint Violations | Unique constraint or foreign key errors | Database error codes, constraint violation logs | Idempotent retry logic, state reconciliation |
| Replication Lag | Stale reads after writes | Read-after-write consistency checks | Read from primary, eventual consistency handling |

**Worker Node Failures** encompass the most complex failure scenarios because they involve both detection challenges and recovery coordination. Worker process crashes result in orphaned tasks that remain in running state indefinitely unless detected and reassigned. The heartbeat mechanism must distinguish between temporary network issues and actual worker failures to avoid unnecessary task reassignment thrashing.

Worker resource exhaustion manifests as tasks timing out, extremely slow execution, or worker processes becoming unresponsive while still sending heartbeats. This requires monitoring worker resource metrics alongside heartbeat status to detect degraded workers before complete failure.

Network partitions between workers and the scheduler create split-brain scenarios where workers continue executing tasks while being marked as failed by the scheduler. This requires careful coordination to prevent duplicate task execution when connectivity restores.

| Worker Failure Type | Detection Signals | Time to Detection | Recovery Actions |
|---------------------|-------------------|-------------------|------------------|
| Process Crash | Missing heartbeat, TCP connection reset | 30-60 seconds | Mark tasks as failed, reassign to healthy workers |
| Resource Exhaustion | High CPU/memory in heartbeat, slow task completion | 2-5 minutes | Drain worker gracefully, prevent new task assignment |
| Network Partition | Heartbeat timeout, task result timeout | 1-2 minutes | Mark worker as lost, reassign tasks after partition timeout |
| Disk Full | Task execution failures, log write errors | Immediate on task failure | Emergency cleanup, task reassignment |
| Container/Pod Eviction | Kubernetes eviction events, sudden disconnection | 10-30 seconds | Graceful task completion if possible, otherwise reassignment |

**Scheduler Component Failures** create system-wide impact since the scheduler coordinates all workflow execution. Scheduler crashes during DAG run creation can leave partially created DAG runs with missing task instances. The recovery process must detect incomplete DAG runs and either complete their creation or mark them as failed.

Multiple scheduler instances running simultaneously (split-brain) can create duplicate DAG runs or conflicting task assignments. This requires implementing leader election with lease management to ensure only one active scheduler at any time.

Scheduler database connection issues prevent processing scheduled workflows, causing schedule drift where workflows trigger later than intended. The system must track missed schedules and implement catch-up logic with configurable limits to prevent overwhelming the system after extended outages.

| Scheduler Failure Scenario | Impact | Detection Method | Recovery Strategy |
|-----------------------------|--------|------------------|-------------------|
| Crash During DAG Run Creation | Incomplete DAG runs with missing task instances | Startup scan for incomplete DAG runs | Complete creation or mark as failed |
| Split-Brain (Multiple Active) | Duplicate DAG runs, conflicting assignments | Leader election heartbeat monitoring | Force re-election, cleanup duplicate runs |
| Schedule Processing Lag | Workflows trigger late, schedule drift accumulation | Schedule drift metrics, missed trigger alerts | Catch-up processing with backfill limits |
| Task Queue Overflow | New tasks not queued, executor backpressure | Queue depth metrics, task submission failures | Queue size limits, priority-based dropping |
| Dependency Resolution Errors | Tasks stuck in queued state indefinitely | Task state monitoring, dependency timeout | Force dependency re-evaluation, manual intervention |

**Message Queue Failures** disrupt the communication channel between schedulers and workers, preventing task distribution and result collection. Queue unavailability stops new task assignments while allowing currently running tasks to complete. The system must implement local queuing with persistence to buffer tasks during queue outages.

Message delivery failures can cause tasks to be delivered multiple times to different workers, requiring idempotent task execution and result deduplication. Consumer group failures prevent workers from receiving new tasks, creating uneven load distribution that requires automatic rebalancing.

Queue message corruption or deserialization errors prevent tasks from executing properly. The system must implement message validation, error queues for failed messages, and monitoring for message processing rates to detect corruption patterns.

| Queue Failure Type | Immediate Effect | Secondary Effect | Mitigation Strategy |
|---------------------|------------------|------------------|---------------------|
| Complete Queue Unavailability | Task distribution stops | Worker starvation, schedule delays | Local task queuing, circuit breaker activation |
| Message Delivery Duplicates | Multiple workers receive same task | Duplicate execution, resource waste | Idempotent execution, task instance locking |
| Consumer Group Imbalance | Uneven task distribution | Some workers idle, others overloaded | Consumer rebalancing, partition reassignment |
| Message Corruption | Task deserialization failures | Tasks stuck in queue | Message validation, dead letter queues |
| Queue Partition Failures | Subset of tasks not distributed | Partial system degradation | Partition redundancy, cross-partition routing |

**External Service Dependencies** include databases, file systems, APIs, and other services that tasks interact with during execution. These failures require careful isolation to prevent cascading throughout the workflow system. API rate limiting or temporary unavailability should trigger task retries with exponential backoff rather than immediate failure.

File system issues during log writing or temporary file creation can cause tasks to fail even when their core logic succeeds. The system must implement fallback logging mechanisms and temporary storage alternatives to maintain task execution capabilities.

Authentication service outages prevent task execution that requires credentials, but should not impact tasks that operate on public data. The system must implement credential caching and graceful degradation for authentication-dependent operations.

![Worker Health Monitoring](./diagrams/worker-health-monitoring.svg)

### Circuit Breaker Patterns

Circuit breakers provide crucial protection against cascading failures by detecting unhealthy dependencies and temporarily isolating them while allowing the system to continue operating in a degraded mode. Think of circuit breakers like the electrical safety devices in your home—when they detect dangerous conditions, they immediately cut power to prevent fires, and they can be reset once the problem is resolved.

The circuit breaker pattern transitions through three distinct states: **Closed** (normal operation), **Open** (dependency blocked), and **Half-Open** (testing recovery). Each state has specific behaviors and transition conditions that must be carefully tuned to balance system protection with availability.

**Closed State Operations** represent normal system behavior where all requests pass through to the protected dependency. The circuit breaker monitors success and failure rates, response times, and timeout patterns to detect when the dependency becomes unhealthy. Success rate tracking uses a sliding window approach to avoid being overly sensitive to brief failure spikes while still detecting sustained problems.

Response time monitoring identifies dependencies that become slow even if they don't explicitly fail, as slow dependencies can cause resource exhaustion through connection pool depletion and timeout accumulation. The circuit breaker must distinguish between acceptable latency variations and problematic performance degradation.

Failure rate calculation requires careful definition of what constitutes a failure versus a legitimate business error. Database connection timeouts represent infrastructure failures that should trip the circuit breaker, while SQL constraint violations represent application logic issues that should not affect the circuit breaker state.

**Open State Behavior** immediately rejects all requests to the protected dependency without attempting to connect, returning cached responses, default values, or error responses depending on the operation type. This prevents resource exhaustion and allows the dependency time to recover without additional load.

During the open state, the circuit breaker starts a recovery timer that determines when to transition to half-open state for testing. The recovery timeout should be long enough to allow the dependency to fully recover but short enough to minimize system degradation. Exponential backoff for the recovery timeout prevents rapid cycling between states during intermittent failures.

The open state must provide meaningful fallback behavior rather than simply failing all operations. For database operations, this might mean serving cached data or operating in read-only mode. For external API calls, this might mean returning default values or skipping optional processing steps.

**Half-Open State Testing** allows a limited number of requests through to test whether the dependency has recovered. The number of test requests must be small enough to avoid overwhelming a recovering system but large enough to provide statistical confidence in the recovery assessment.

Test request selection should prioritize simple, low-impact operations that accurately reflect system health without causing additional stress. Database health checks should use simple SELECT statements rather than complex queries. API health checks should use lightweight status endpoints rather than full business operations.

Recovery assessment requires defining success criteria for returning to closed state versus failure criteria for returning to open state. A single failure during half-open testing might indicate incomplete recovery, while multiple consecutive successes might indicate full recovery. The specific thresholds depend on the dependency's typical failure patterns and recovery characteristics.

| Circuit Breaker Component | Configuration Parameter | Closed State Value | Open State Value | Half-Open State Value |
|----------------------------|-------------------------|--------------------|--------------------|----------------------|
| Database Connection Pool | Failure Rate Threshold | 50% failures in 10 requests | N/A (no requests sent) | 20% failures in 5 test requests |
| Database Connection Pool | Response Time Threshold | 5 second timeout | N/A (immediate failure) | 2 second timeout for tests |
| Database Connection Pool | Recovery Timeout | N/A | 30 seconds initial, exponential backoff | 10 second test window |
| Message Queue Operations | Failure Rate Threshold | 30% failures in 20 requests | N/A (use local queuing) | 10% failures in 10 test requests |
| Message Queue Operations | Recovery Timeout | N/A | 60 seconds initial | 15 second test window |
| External API Dependencies | Failure Rate Threshold | 40% failures in 15 requests | N/A (skip optional features) | 15% failures in 8 test requests |
| External API Dependencies | Rate Limit Handling | Normal request rate | Zero requests | 50% normal rate for tests |

**Implementation Strategies** for circuit breakers require careful integration with existing error handling and retry mechanisms. Circuit breakers should operate at a higher level than individual request retries, making decisions based on aggregate patterns rather than single request outcomes.

Thread safety becomes critical in distributed executors where multiple workers might simultaneously evaluate circuit breaker states. The circuit breaker state must be shared across worker processes, either through a centralized service or distributed consensus mechanism.

Monitoring and alerting for circuit breaker state changes provides crucial operational visibility. Open circuit breakers indicate system degradation that requires investigation, while frequent state transitions might indicate configuration issues or unstable dependencies.

**Database Circuit Breaker Implementation** focuses on protecting against connection pool exhaustion and database unavailability. The circuit breaker monitors both connection acquisition failures and query execution failures, treating them as separate failure categories with different thresholds.

Connection pool monitoring tracks failed connection attempts separately from successful connections that encounter query errors. Connection acquisition failures indicate infrastructure problems, while query failures might indicate application logic issues that shouldn't necessarily trip the circuit breaker.

During open state, database operations must gracefully degrade to read-only mode using cached data or alternative data sources. Critical operations like task state updates might need to be queued for later processing when the database becomes available again.

> **Decision: Database Circuit Breaker Granularity**
> - **Context**: Database failures can be partial (specific tables locked) or complete (database unavailable), and different operations have different criticality levels
> - **Options Considered**: Single system-wide circuit breaker, per-table circuit breakers, per-operation-type circuit breakers
> - **Decision**: Per-operation-type circuit breakers (read, write, admin) with shared failure detection
> - **Rationale**: Allows graceful degradation where reads might work during write failures, while avoiding the complexity of per-table tracking
> - **Consequences**: More complex configuration but better availability during partial database failures

**Message Queue Circuit Breaker Implementation** protects against queue unavailability and message processing failures. The circuit breaker distinguishes between queue connectivity issues and message deserialization problems, applying different failure thresholds for each category.

Queue connectivity monitoring tracks connection establishment failures and heartbeat timeouts separately from message publishing and consumption errors. Connectivity failures indicate infrastructure problems requiring circuit breaker activation, while message processing errors might indicate data corruption requiring different handling.

During open state, the system must implement local task queuing to buffer tasks until queue connectivity restores. The local queue must have size limits and overflow handling to prevent memory exhaustion during extended outages.

**External Service Circuit Breaker Implementation** protects against API rate limiting, service unavailability, and authentication failures. Each external service requires separate circuit breaker configuration based on its reliability characteristics and the criticality of its operations.

Rate limiting detection requires parsing service response headers and error messages to distinguish between temporary rate limits and permanent service failures. Rate limits should transition the circuit breaker to a throttled state rather than fully open state, allowing reduced request rates rather than complete blocking.

Authentication failures need special handling since they might indicate credential expiration rather than service unavailability. The circuit breaker should integrate with credential refresh mechanisms to avoid unnecessary service blocking during routine credential rotation.

### Data Consistency Guarantees

Data consistency in a distributed workflow orchestrator requires careful balance between performance, availability, and correctness. The system must ensure that workflow state remains coherent across multiple components while handling concurrent modifications, partial failures, and recovery scenarios.

Think of data consistency like maintaining accurate financial records across multiple bank branches. Each branch needs to see consistent account balances, transfers must be atomic (money can't disappear or be duplicated), and the system must remain available even when some branches are temporarily offline. The workflow orchestrator faces similar challenges with task states, dependencies, and execution history.

**ACID Properties Implementation** ensures that critical workflow operations maintain database consistency even during failures. **Atomicity** requires that DAG run creation either completes entirely (DAG run record plus all task instances) or fails completely with no partial state. This prevents scenarios where a DAG run exists but has missing task instances that would never execute.

Task instance state transitions must be atomic to prevent race conditions where multiple workers attempt to claim the same task or update conflicting states simultaneously. The database transaction must include both the state change and any related metadata updates (start time, worker assignment, retry count) to maintain consistency.

Cross-task communication (XCom) operations require atomicity between task completion and data storage to prevent scenarios where a task completes successfully but its output data is lost due to storage failures. The task result and XCom data must be stored in the same transaction as the task completion status.

**Consistency** enforcement prevents the system from entering invalid states that violate business rules or dependency constraints. DAG run state must remain consistent with its constituent task instance states—a DAG run cannot be marked as successful while containing failed task instances without explicit configuration allowing this behavior.

Dependency relationships must be enforced at the database level through foreign key constraints to prevent orphaned task instances or references to non-existent upstream tasks. However, these constraints must be carefully designed to allow batch operations like DAG deletion without creating constraint violation cascades.

Task instance retry counts must remain consistent with actual execution attempts to prevent infinite retry loops or premature retry exhaustion. The retry counter increment and execution attempt creation must occur atomically to maintain accurate retry tracking.

**Isolation** levels determine how concurrent operations interact with each other and what consistency trade-offs are acceptable. Read Committed isolation prevents dirty reads where one transaction sees uncommitted changes from another, which is crucial for dependency resolution where tasks must see accurate upstream completion states.

Serializable isolation provides the strongest consistency guarantees but can significantly impact performance due to increased lock contention and serialization conflicts. This level may be necessary for critical operations like scheduler leader election or DAG run state reconciliation.

Snapshot isolation offers a middle ground where each transaction sees a consistent snapshot of the database at transaction start time, preventing non-repeatable reads while allowing better concurrency than serializable isolation.

**Durability** ensures that committed changes survive system failures and restarts. Database durability depends on proper fsync configuration and transaction log management, but application-level durability requires careful error handling and retry logic.

Critical state changes like task completions must be durably stored before acknowledging success to external systems. This prevents scenarios where a task reports successful completion but the database loses the state due to a crash before the transaction commits.

Write-ahead logging for critical operations provides additional durability guarantees beyond database transactions. The system can implement application-level WAL for operations that span multiple systems or require custom recovery logic.

| Operation Type | Isolation Level | Atomicity Requirements | Durability Requirements |
|----------------|----------------|------------------------|------------------------|
| DAG Run Creation | Read Committed | DAG run + all task instances in single transaction | Must survive scheduler restart |
| Task State Transition | Read Committed | State + metadata + worker assignment atomically | Must survive worker crash |
| Dependency Resolution | Read Committed | Read upstream states consistently | No durability requirements (can recompute) |
| XCom Operations | Read Committed | Task completion + data storage atomic | Must survive database restart |
| Scheduler Leader Election | Serializable | Leader lease + timestamp atomic | Must survive network partition |
| DAG Registration | Read Committed | DAG definition + task definitions atomic | Must survive parser restart |

**Concurrent Modification Handling** addresses scenarios where multiple components attempt to modify the same workflow entities simultaneously. Optimistic locking using version fields allows detecting concurrent modifications without holding database locks for extended periods.

Task instance claiming by workers requires careful coordination to prevent multiple workers from executing the same task. The claiming operation must atomically check current state, verify worker eligibility, and update the assignment in a single transaction with appropriate isolation level.

DAG run state updates from multiple sources (scheduler, executor, manual operations) require conflict resolution policies. The system must define precedence rules for conflicting updates and provide mechanisms for detecting and resolving inconsistent states.

Batch operations like DAG deletion or bulk task retry must handle partial failures gracefully, either by implementing compensation transactions or by providing idempotent retry mechanisms that can safely rerun after interruption.

**Eventual Consistency Strategies** provide better availability and performance for operations that don't require immediate consistency. Worker heartbeat updates and task progress metrics can use eventual consistency since slight delays don't affect correctness of workflow execution decisions.

Status aggregation from task instances to DAG runs can use eventual consistency with periodic reconciliation to ensure that temporary inconsistencies don't persist indefinitely. The reconciliation process must be idempotent and handle cases where multiple reconciliation attempts occur simultaneously.

Log aggregation and metrics collection can use eventual consistency since these systems support operational monitoring rather than core workflow execution logic. However, the system must provide mechanisms to detect and alert on consistency lag that exceeds acceptable thresholds.

**State Reconciliation Mechanisms** detect and repair inconsistencies that arise from partial failures, network partitions, or software bugs. Periodic consistency checks compare related entities to identify inconsistent states requiring correction.

DAG run state reconciliation compares the DAG run state with the aggregate states of its constituent task instances, identifying cases where the DAG run state doesn't accurately reflect task completion status. The reconciliation process must handle edge cases like skipped tasks, conditional branches, and partial success scenarios.

Worker assignment reconciliation detects orphaned tasks where workers have failed but tasks remain in assigned state indefinitely. The reconciliation process must distinguish between temporarily unresponsive workers and actual failures to avoid unnecessarily reassigning tasks.

Dependency state reconciliation identifies tasks that should be ready for execution based on upstream completion but remain in queued state due to notification failures or processing errors. This safety net ensures that workflows continue progressing even when the primary dependency notification mechanisms fail.

> **Decision: Consistency vs. Availability Trade-offs**
> - **Context**: Distributed workflow execution requires balancing data consistency with system availability during network partitions and component failures
> - **Options Considered**: Strong consistency with lower availability, eventual consistency with reconciliation, hybrid approach with operation-specific consistency levels
> - **Decision**: Hybrid approach using strong consistency for critical workflow state and eventual consistency for monitoring data
> - **Rationale**: Critical workflow decisions require accurate state information, while monitoring and metrics can tolerate brief inconsistencies
> - **Consequences**: More complex implementation but better balance of correctness and availability

**Distributed Transaction Coordination** handles operations that span multiple systems or databases, ensuring atomicity across system boundaries. While full two-phase commit is often too expensive for workflow operations, saga patterns provide a more practical approach for coordinating multi-step operations.

DAG run creation across multiple databases (metadata database, metrics database, log storage) can use saga patterns where each step includes compensation logic for rollback scenarios. The saga coordinator tracks progress and executes compensation transactions when failures occur.

Task execution that involves multiple external systems (database updates, file operations, API calls) requires coordination to ensure either all operations succeed or all are properly rolled back. The task execution context must provide transaction management capabilities that integrate with the overall workflow consistency model.

Cross-system messaging for task distribution and result collection requires at-least-once delivery guarantees with idempotent processing to ensure that messages are not lost during system failures while avoiding duplicate processing when messages are delivered multiple times.

### Common Pitfalls

⚠️ **Pitfall: Insufficient Circuit Breaker Granularity**
Many implementations use a single circuit breaker for an entire database or service, which can cause unnecessary service degradation. When the circuit breaker trips due to failures in one operation type (like complex analytical queries), it blocks all operations including simple health checks and critical metadata updates. This creates cascading failures where healthy operations become unavailable due to problems in unrelated functionality. The solution is implementing per-operation-type circuit breakers that can isolate specific failure modes while allowing healthy operations to continue.

⚠️ **Pitfall: Race Conditions in Task State Transitions**
Concurrent task state updates without proper locking can create inconsistent states where a task appears to be both running and completed, or where retry counts become inaccurate. This commonly occurs when workers report task completion while timeout handlers simultaneously mark tasks as failed. The database transaction isolation level must be carefully chosen, and task state updates must include version checks or row-level locking to prevent concurrent modifications from creating invalid states.

⚠️ **Pitfall: Incomplete Failure Detection in Worker Health Monitoring**
Relying solely on heartbeat timeouts for worker failure detection misses scenarios where workers become unresponsive due to resource exhaustion but still send heartbeats. A worker with 100% CPU utilization might continue sending heartbeats while being unable to process new tasks, creating a situation where tasks are assigned but never execute. Comprehensive health monitoring must include resource utilization metrics, task completion rates, and response time monitoring in addition to basic connectivity checks.

⚠️ **Pitfall: Ignoring Database Transaction Timeouts During Recovery**
Long-running recovery operations like DAG run state reconciliation can hold database locks for extended periods, blocking normal operations and potentially causing deadlocks. Recovery processes must be designed with appropriate transaction boundaries, using short-lived transactions with explicit checkpointing rather than single large transactions. The recovery process should be interruptible and resumable to avoid blocking critical operations during system recovery.

⚠️ **Pitfall: Insufficient Error Context Propagation**
Generic error handling that loses context about the specific operation, timing, and system state makes debugging distributed failures extremely difficult. When a task fails with "database connection error," the system needs to preserve information about which database operation was attempted, what the system load was at the time, whether other similar operations were succeeding, and what recovery attempts were made. Structured logging with correlation IDs and comprehensive error context enables effective debugging and system improvement.

⚠️ **Pitfall: Cascading Timeout Failures**
Setting uniform timeouts across all system components without considering the cumulative effect can create cascading failures where upstream timeouts are shorter than the sum of downstream processing times. If the scheduler has a 30-second timeout for task submission, but worker task claiming takes 15 seconds and task initialization takes 20 seconds, the scheduler will consistently timeout before workers can complete their processing. Timeout values must be coordinated across the entire system with appropriate buffers for processing delays and retry attempts.

### Implementation Guidance

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Circuit Breaker Library | Custom implementation with basic state machine | netflix/hystrix-go or similar production library |
| Database Connection Pooling | database/sql with manual health checks | pgxpool or similar with built-in health monitoring |
| Distributed Locking | Redis-based locks with expiration | etcd or Consul with leader election |
| Metrics and Monitoring | Prometheus client with basic counters | OpenTelemetry with distributed tracing |
| Error Tracking | Structured logging with error classification | Sentry or similar error aggregation service |
| Health Monitoring | HTTP health check endpoints | Kubernetes probes with custom health logic |

**Recommended File Structure:**
```
internal/
  errors/
    circuit_breaker.go           ← Circuit breaker implementation
    circuit_breaker_test.go      ← Circuit breaker tests
    failure_detector.go          ← Worker and service failure detection
    recovery_manager.go          ← System recovery coordination
    consistency_checker.go       ← Data consistency validation
  health/
    checker.go                   ← Health check aggregation
    database_health.go           ← Database-specific health checks
    worker_health.go             ← Worker health monitoring
    metrics.go                   ← Health metrics collection
  models/
    errors.go                    ← Error type definitions
    health_status.go             ← Health status types
```

**Circuit Breaker Infrastructure (Complete Implementation):**
```go
package errors

import (
    "context"
    "errors"
    "sync"
    "time"
)

// CircuitBreakerState represents the current state of a circuit breaker
type CircuitBreakerState int

const (
    CircuitClosed CircuitBreakerState = iota
    CircuitOpen
    CircuitHalfOpen
)

// CircuitBreakerConfig defines circuit breaker behavior
type CircuitBreakerConfig struct {
    FailureThreshold   int           // Number of failures to trigger open state
    SuccessThreshold   int           // Number of successes to return to closed state
    Timeout           time.Duration  // How long to wait before attempting recovery
    MaxRequests       int           // Maximum requests in half-open state
    WindowSize        int           // Size of the failure tracking window
}

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
    config    CircuitBreakerConfig
    state     CircuitBreakerState
    failures  int
    successes int
    requests  int
    nextAttempt time.Time
    mu        sync.RWMutex
}

// NewCircuitBreaker creates a new circuit breaker with the given configuration
func NewCircuitBreaker(config CircuitBreakerConfig) *CircuitBreaker {
    return &CircuitBreaker{
        config: config,
        state:  CircuitClosed,
    }
}

// Execute runs the given function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() error) error {
    if !cb.allowRequest() {
        return errors.New("circuit breaker is open")
    }

    err := fn()
    cb.recordResult(err == nil)
    return err
}

func (cb *CircuitBreaker) allowRequest() bool {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case CircuitClosed:
        return true
    case CircuitOpen:
        if time.Now().After(cb.nextAttempt) {
            cb.state = CircuitHalfOpen
            cb.requests = 0
            cb.successes = 0
            return true
        }
        return false
    case CircuitHalfOpen:
        return cb.requests < cb.config.MaxRequests
    }
    return false
}

func (cb *CircuitBreaker) recordResult(success bool) {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case CircuitClosed:
        if success {
            cb.failures = 0
        } else {
            cb.failures++
            if cb.failures >= cb.config.FailureThreshold {
                cb.state = CircuitOpen
                cb.nextAttempt = time.Now().Add(cb.config.Timeout)
            }
        }
    case CircuitHalfOpen:
        cb.requests++
        if success {
            cb.successes++
            if cb.successes >= cb.config.SuccessThreshold {
                cb.state = CircuitClosed
                cb.failures = 0
            }
        } else {
            cb.state = CircuitOpen
            cb.nextAttempt = time.Now().Add(cb.config.Timeout)
        }
    }
}
```

**Worker Health Monitoring Infrastructure (Complete Implementation):**
```go
package health

import (
    "context"
    "sync"
    "time"
)

// WorkerHealthStatus represents the health state of a worker
type WorkerHealthStatus struct {
    WorkerID        string
    LastHeartbeat   time.Time
    ResourceUsage   ResourceMetrics
    ActiveTasks     []string
    Status          string
    ConsecutiveFailures int
}

// ResourceMetrics contains resource utilization information
type ResourceMetrics struct {
    CPUPercent    float64
    MemoryPercent float64
    DiskPercent   float64
    TaskSlots     int
    UsedSlots     int
}

// HealthChecker monitors worker health and detects failures
type HealthChecker struct {
    workers           map[string]*WorkerHealthStatus
    heartbeatTimeout  time.Duration
    failureThreshold  int
    checkInterval     time.Duration
    onWorkerFailed    func(workerID string)
    mu                sync.RWMutex
}

// NewHealthChecker creates a new worker health monitoring system
func NewHealthChecker(heartbeatTimeout time.Duration, failureThreshold int, 
                     onWorkerFailed func(string)) *HealthChecker {
    return &HealthChecker{
        workers:          make(map[string]*WorkerHealthStatus),
        heartbeatTimeout: heartbeatTimeout,
        failureThreshold: failureThreshold,
        checkInterval:    heartbeatTimeout / 3, // Check 3x more frequently than timeout
        onWorkerFailed:   onWorkerFailed,
    }
}

// RecordHeartbeat updates the health status for a worker
func (hc *HealthChecker) RecordHeartbeat(workerID string, metrics ResourceMetrics, 
                                        activeTasks []string) {
    hc.mu.Lock()
    defer hc.mu.Unlock()

    worker, exists := hc.workers[workerID]
    if !exists {
        worker = &WorkerHealthStatus{
            WorkerID: workerID,
            Status:   "healthy",
        }
        hc.workers[workerID] = worker
    }

    worker.LastHeartbeat = time.Now()
    worker.ResourceUsage = metrics
    worker.ActiveTasks = activeTasks
    worker.ConsecutiveFailures = 0

    // Evaluate resource-based health
    if metrics.CPUPercent > 95.0 || metrics.MemoryPercent > 95.0 {
        worker.Status = "degraded"
    } else {
        worker.Status = "healthy"
    }
}

// StartHealthMonitoring begins the health check loop
func (hc *HealthChecker) StartHealthMonitoring(ctx context.Context) {
    ticker := time.NewTicker(hc.checkInterval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            hc.checkWorkerHealth()
        }
    }
}
```

**Core Error Handling Logic (Skeleton for Implementation):**
```go
// FailureDetector identifies and categorizes system failures
type FailureDetector struct {
    db              *sql.DB
    circuitBreakers map[string]*CircuitBreaker
    healthChecker   *HealthChecker
    logger          Logger
}

// DetectAndHandleFailure processes a failure event and determines recovery actions
func (fd *FailureDetector) DetectAndHandleFailure(ctx context.Context, 
                                                  failure FailureEvent) error {
    // TODO 1: Classify the failure type (database, network, worker, application)
    // TODO 2: Check if this failure pattern triggers circuit breaker activation
    // TODO 3: Determine the scope of impact (single task, DAG run, system-wide)
    // TODO 4: Select appropriate recovery strategy based on failure type and scope
    // TODO 5: Execute recovery actions while monitoring for additional failures
    // TODO 6: Record failure metrics and update system health status
    // TODO 7: Send alerts if failure severity exceeds notification thresholds
    // Hint: Use context timeout to prevent recovery operations from blocking too long
}

// ReconcileSystemState checks for and repairs data inconsistencies
func (fd *FailureDetector) ReconcileSystemState(ctx context.Context) error {
    // TODO 1: Identify DAG runs with inconsistent state vs. task instance states
    // TODO 2: Find task instances assigned to failed workers that need reassignment
    // TODO 3: Detect tasks stuck in queued state despite satisfied dependencies
    // TODO 4: Verify XCom data consistency between task completions and stored values
    // TODO 5: Check for orphaned resources (temp files, connections) requiring cleanup
    // TODO 6: Repair identified inconsistencies using appropriate transaction boundaries
    // TODO 7: Log reconciliation actions for audit and debugging purposes
    // Hint: Process reconciliation in batches to avoid long-running transactions
}

// EvaluateCircuitBreakerStatus determines if circuit breakers should change state
func (fd *FailureDetector) EvaluateCircuitBreakerStatus(service string, 
                                                      operationType string) error {
    // TODO 1: Retrieve failure metrics for the specific service and operation type
    // TODO 2: Calculate failure rate over the configured time window
    // TODO 3: Compare failure rate against circuit breaker thresholds
    // TODO 4: Update circuit breaker state if thresholds are exceeded
    // TODO 5: Configure fallback behavior appropriate for the operation type
    // TODO 6: Schedule recovery testing if circuit breaker moves to half-open state
    // Hint: Different operation types may need different failure rate thresholds
}
```

**Milestone Checkpoints:**

After implementing failure detection and circuit breakers:
- Run `go test ./internal/errors/...` - all circuit breaker state transitions should work correctly
- Start the system and kill a worker process - verify task reassignment occurs within 60 seconds
- Disconnect the database temporarily - verify circuit breaker trips and system continues with degraded functionality
- Monitor logs during recovery - should see clear failure classification and recovery action descriptions

After implementing consistency checking:
- Create a DAG run and manually update some task states directly in database - reconciliation should detect and fix inconsistencies
- Simulate worker crash during task execution - verify orphaned tasks are detected and reassigned
- Check system behavior during network partitions - should maintain internal consistency even with external connectivity issues

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Circuit breaker never trips despite failures | Threshold too high or failure classification incorrect | Check failure rate calculations and error type categorization | Lower thresholds or fix failure detection logic |
| System becomes unresponsive during recovery | Recovery operations holding locks too long | Monitor database lock wait times and transaction durations | Implement recovery checkpointing with shorter transactions |
| Inconsistent task states persist after reconciliation | Reconciliation logic incomplete or concurrent modifications | Trace reconciliation process and check for race conditions | Add proper locking and idempotency to reconciliation |
| Workers marked as failed but still processing tasks | Health check criteria too sensitive or network issues | Review heartbeat patterns and resource utilization trends | Adjust failure thresholds and implement graceful degradation |
| Database deadlocks during error recovery | Recovery operations conflicting with normal processing | Analyze deadlock graphs and transaction ordering | Implement consistent lock ordering and retry logic |


## Testing Strategy and Milestone Checkpoints

> **Milestone(s):** All milestones (comprehensive testing strategy covering DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring)

Building a robust workflow orchestration system requires a multi-layered testing strategy that validates both individual component behavior and complex system-wide interactions. Think of testing as building a safety net beneath a trapeze artist - each layer of the net catches different types of failures, from simple unit-level bugs to intricate race conditions and distributed system failures. The orchestrator must handle everything from malformed DAG definitions to worker crashes during task execution, making comprehensive testing essential for production reliability.

Testing a workflow orchestrator presents unique challenges compared to typical web applications. The system must coordinate time-based scheduling, manage complex dependency graphs, handle distributed task execution, and recover from various failure modes gracefully. Each milestone introduces new complexity layers that require specialized testing approaches, from validating cycle detection algorithms to ensuring proper task retry behavior across worker failures.

### Mental Model: The Quality Assurance Assembly Line

Imagine testing as a quality assurance assembly line in a manufacturing plant. Raw materials (code changes) enter at one end and pass through multiple inspection stations. The first station performs basic component checks - ensuring individual parts meet specifications. The second station tests sub-assemblies - verifying that multiple components work together correctly. The third station conducts end-to-end system tests - running complete products through real-world scenarios. Finally, the acceptance testing station validates that the finished product meets customer requirements under production-like conditions.

Each testing layer serves a distinct purpose and catches different categories of defects. Unit tests catch logic errors and edge cases in individual functions. Integration tests reveal interface mismatches and data flow issues between components. System tests uncover timing problems, resource contention, and failure propagation issues. Acceptance tests validate that business requirements are met and the system behaves correctly from an end-user perspective.

### Unit Testing Strategy

Unit testing forms the foundation of the orchestrator's quality assurance strategy, focusing on validating individual components in isolation. Each component requires specialized testing approaches based on its responsibilities and the complexity of its internal logic.

#### DAG Validation and Parsing Tests

The DAG parser and validator require extensive unit testing due to the complexity of cycle detection algorithms and the variety of malformed inputs they must handle. Testing focuses on validating the parsing logic, dependency resolution, and cycle detection under various graph structures.

**Core DAG Parser Test Categories:**

| Test Category | Focus Area | Key Scenarios | Expected Behavior |
|---------------|------------|---------------|-------------------|
| File Discovery | Directory scanning and pattern matching | Valid YAML files, non-YAML files, nested directories, empty directories | Only YAML files matching pattern are discovered |
| YAML Parsing | Syntax validation and structure mapping | Valid YAML, malformed syntax, missing required fields, extra fields | Parse valid files, reject invalid syntax with clear errors |
| Dependency Resolution | Task reference validation | Valid task IDs, missing task references, self-references | Build correct dependency graph, reject invalid references |
| Cycle Detection | Graph topology validation | Acyclic graphs, simple cycles, complex cycles, disconnected components | Accept valid DAGs, reject any graph containing cycles |
| Template Processing | Parameter substitution | Valid templates, missing variables, type mismatches, nested references | Render templates correctly, fail gracefully on errors |

The cycle detection algorithm requires particularly thorough testing since subtle bugs can allow invalid DAGs into the system, causing runtime deadlocks. Test cases should cover various graph topologies including linear chains, diamond patterns, complex multi-path structures, and edge cases like single-node cycles.

```markdown
**Cycle Detection Test Matrix:**

| Graph Structure | Node Count | Cycle Present | Expected Result | Validation Method |
|-----------------|------------|---------------|----------------|-------------------|
| Linear chain | 5 | No | Valid | Topological sort succeeds |
| Diamond pattern | 4 | No | Valid | Multiple valid orderings exist |
| Simple cycle | 3 | Yes | Invalid | Topological sort fails |
| Complex multi-cycle | 10 | Yes | Invalid | Any cycle causes rejection |
| Self-reference | 1 | Yes | Invalid | Node depends on itself |
| Disconnected components | 8 | No | Valid | Each component validated separately |
```

Template processing tests must validate both successful rendering and proper error handling for malformed templates. The template engine should gracefully handle missing variables, type mismatches, and circular template references without exposing internal implementation details to users.

#### Scheduling Logic Tests

The scheduler component contains complex time-based logic that requires careful testing of cron expression parsing, timezone handling, and execution date calculation. Testing must cover edge cases like daylight saving time transitions, leap years, and invalid date combinations.

**Scheduler Test Categories:**

| Component | Test Focus | Critical Scenarios | Validation Points |
|-----------|------------|-------------------|-------------------|
| Cron Parser | Expression validation and parsing | Valid expressions, invalid syntax, edge cases | Correct field extraction, clear error messages |
| Next Execution Calculator | Time arithmetic and timezone handling | Standard schedules, DST transitions, leap years | Accurate next execution times |
| DAG Run Creator | Atomic creation with task instances | Successful creation, database failures, concurrent requests | All-or-nothing semantics, proper cleanup |
| Schedule Processor | Polling loop and trigger detection | Multiple DAGs, missed schedules, catchup behavior | Correct trigger timing, no missed executions |
| Manual Trigger Handler | Immediate execution requests | Valid requests, parameter validation, conflicts | Proper validation, conflict resolution |

Timezone testing requires special attention since incorrect timezone handling can cause workflows to run at wrong times or skip executions entirely. Test cases should cover various timezone scenarios including UTC, fixed offsets, and zones with daylight saving time transitions.

The DAG run creation process must be thoroughly tested for atomicity since partial creation can leave the system in an inconsistent state. Tests should verify that either all task instances are created successfully or none are created, with proper cleanup of any partially created state.

#### Task Execution and Dependency Resolution Tests

Task execution logic involves complex dependency resolution, state transitions, and failure propagation that require comprehensive unit testing. Tests must validate correct behavior under normal conditions and various failure scenarios.

**Task Execution Test Matrix:**

| Test Area | Component | Scenarios | Expected Behavior |
|-----------|-----------|-----------|-------------------|
| Dependency Resolution | `Resolver.EvaluateReadyTasks` | All upstream success, partial failures, mixed states | Only ready tasks identified |
| State Transitions | Task instance state machine | Valid transitions, invalid transitions, concurrent updates | Proper state progression, rejected invalid transitions |
| Trigger Rule Evaluation | `CheckTriggerRule` | All success, all failed, all done, none failed | Correct evaluation per trigger rule |
| Upstream Failure Propagation | `PropagateUpstreamFailed` | Single failures, cascade failures, recovery scenarios | Proper propagation, no infinite loops |
| XCom Management | `DatabaseXComManager` | Push/pull operations, size limits, cleanup | Correct data flow, resource management |

Dependency resolution testing must cover various upstream completion patterns since tasks can have complex trigger rules beyond simple "all upstream success." Test cases should validate behavior for trigger rules like "all_done," "none_failed," and custom conditions.

State transition testing requires validation of both successful transitions and rejection of invalid state changes. The state machine must prevent impossible transitions like moving directly from "queued" to "success" without passing through "running" state.

#### Operator Implementation Tests

Each operator type requires specific testing approaches based on its execution model and external dependencies. Testing must validate both successful execution and proper error handling for various failure conditions.

**Operator Test Categories:**

| Operator Type | Test Focus | Mock Requirements | Validation Points |
|---------------|------------|-------------------|-------------------|
| PythonOperator | Script execution and output capture | Mock subprocess execution | Correct command generation, output handling |
| BashOperator | Shell command execution | Mock shell environment | Proper escaping, environment variables |
| SQLOperator | Database query execution | Mock database connection | Query validation, result handling |
| HTTPOperator | API request handling | Mock HTTP client | Request formatting, response processing |

Testing operators requires careful mocking of external dependencies to ensure tests are fast, reliable, and independent of external systems. Mock implementations should simulate both successful operations and various failure conditions like network timeouts, authentication failures, and resource exhaustion.

The execution context provided to operators must be thoroughly tested to ensure all required services and data are available during task execution. Context testing should validate parameter passing, XCom availability, logging configuration, and resource cleanup.

### Integration Testing

Integration testing validates that multiple components work correctly together, focusing on component interfaces, data flow, and distributed system interactions. Unlike unit tests that isolate components, integration tests exercise real component interactions with minimal mocking.

#### Component Interface Testing

Integration tests must validate that components communicate correctly through their defined interfaces, handling both successful interactions and various error conditions gracefully.

**Component Integration Test Matrix:**

| Component A | Component B | Interface | Test Scenarios | Validation Points |
|-------------|-------------|-----------|----------------|-------------------|
| Scheduler | DAG Parser | DAG loading | File changes, parsing errors, validation failures | Proper error propagation, state consistency |
| Scheduler | Executor | Task submission | Ready tasks, overloaded executor, worker failures | Correct queuing, backpressure handling |
| Executor | Workers | Task distribution | Available workers, worker failures, task results | Proper load balancing, failure detection |
| Web UI | Scheduler | Manual triggers | Valid requests, invalid DAG IDs, parameter errors | Correct validation, user feedback |
| Workers | Database | State updates | Successful completion, failures, timeouts | Atomic updates, consistency guarantees |

Database integration testing requires special attention since the database serves as the central coordination point for all components. Tests must validate that concurrent database operations maintain consistency and that transaction boundaries are properly respected.

Message queue integration testing must verify that task distribution and result collection work correctly under various load conditions and failure scenarios. Tests should validate message ordering, delivery guarantees, and proper handling of queue failures.

#### End-to-End Workflow Execution

End-to-end integration tests execute complete workflows from schedule trigger through task completion, validating the entire system behavior under realistic conditions.

**End-to-End Test Scenarios:**

| Scenario | DAG Structure | Complexity | Validation Points | Duration |
|----------|---------------|------------|-------------------|----------|
| Simple Linear | 3 tasks in sequence | Low | Correct execution order, state progression | 30 seconds |
| Diamond Dependencies | 4 tasks with join | Medium | Parallel execution, dependency coordination | 45 seconds |
| Complex Multi-Path | 10+ tasks with multiple branches | High | Resource management, concurrent execution | 2 minutes |
| Failure Recovery | Include intentional task failures | High | Retry behavior, failure propagation | 1 minute |
| Parameter Passing | Tasks with XCom dependencies | Medium | Data flow between tasks, parameter templating | 45 seconds |

Each end-to-end test should validate multiple aspects of system behavior including timing, resource usage, state consistency, and proper cleanup. Tests must run against a complete system instance with all components running and communicating through their production interfaces.

Long-running integration tests help identify issues like memory leaks, resource exhaustion, and gradual performance degradation that may not appear in short unit tests. These tests should run overnight or during CI pipeline execution to catch regression issues.

#### Database and Message Queue Integration

Integration tests must validate that the orchestrator works correctly with real database and message queue implementations, testing connection handling, failure recovery, and performance characteristics.

**Infrastructure Integration Tests:**

| Component | Test Type | Scenarios | Validation Criteria |
|-----------|-----------|-----------|-------------------|
| Database | Connection Management | Pool exhaustion, connection timeouts, network failures | Proper connection recovery, no leaked connections |
| Database | Transaction Handling | Concurrent updates, deadlock scenarios, long transactions | ACID compliance, deadlock resolution |
| Message Queue | Message Delivery | High throughput, queue full conditions, consumer failures | Message durability, delivery guarantees |
| Message Queue | Failure Recovery | Broker restarts, network partitions, consumer crashes | Automatic reconnection, message replay |

Database integration tests should use real database instances rather than in-memory alternatives to catch issues specific to production database behavior. Tests should validate that database migrations work correctly and that the schema supports expected query patterns efficiently.

Message queue integration tests must verify that task distribution remains reliable under various failure conditions. Tests should simulate broker failures, network partitions, and consumer crashes to validate that the system recovers gracefully without losing tasks or delivering duplicates.

### Milestone Verification Steps

Each development milestone requires specific verification steps to ensure that implemented functionality works correctly and integrates properly with existing components. These checkpoints provide concrete validation criteria for completion.

#### Milestone 1: DAG Definition Verification

After implementing DAG definition and validation functionality, the following verification steps confirm that the component works correctly:

**Verification Checklist:**

| Verification Step | Command/Action | Expected Result | Failure Indicators |
|------------------|----------------|-----------------|-------------------|
| Valid DAG Parsing | Place valid DAG file in configured directory | DAG appears in system, no errors logged | Parse errors, missing tasks, incorrect dependencies |
| Cycle Detection | Submit DAG with circular dependency | Validation fails with clear error message | Cycle not detected, unclear error, system crash |
| File Discovery | Add multiple DAG files with various names | Only matching files processed | Wrong files processed, file pattern ignored |
| Template Rendering | DAG with parameterized task configs | Templates rendered correctly at runtime | Template errors, missing variables, malformed output |
| Dependency Validation | DAG with invalid task references | Clear error identifying missing tasks | Vague errors, successful parsing of invalid DAG |

**Manual Testing Commands:**

```bash
# Test basic DAG loading
echo "Creating test DAG file..."
cat > dags/test_dag.yaml << EOF
id: test_linear
description: Simple test DAG
schedule: "0 12 * * *"
tasks:
  - id: task1
    type: bash
    operator:
      bash_command: "echo 'Task 1 complete'"
  - id: task2
    type: bash
    operator:
      bash_command: "echo 'Task 2 complete'"
    dependencies: [task1]
EOF

# Verify DAG discovery and parsing
curl http://localhost:8080/api/v1/dags/test_linear
# Expected: DAG details with 2 tasks, task2 depends on task1

# Test cycle detection
cat > dags/cycle_dag.yaml << EOF
id: test_cycle
tasks:
  - id: task1
    dependencies: [task2]
  - id: task2
    dependencies: [task1]
EOF

# Verify cycle rejection
curl http://localhost:8080/api/v1/dags/test_cycle
# Expected: 404 or validation error about circular dependency
```

**Automated Test Validation:**

The DAG definition milestone requires passing comprehensive unit tests that validate all parsing, validation, and discovery functionality:

```bash
# Run DAG parser tests
go test ./internal/parser/... -v
# Expected output: All tests pass, coverage > 90%

# Run cycle detection tests specifically
go test ./internal/parser/ -run TestCycleDetection -v
# Expected: Various cycle scenarios tested and handled correctly

# Run integration tests for file discovery
go test ./internal/parser/ -run TestFileDiscovery -v
# Expected: File watching, pattern matching, and reload behavior validated
```

#### Milestone 2: Scheduler Verification

Scheduler implementation verification focuses on cron parsing, schedule evaluation, and DAG run creation functionality:

**Scheduler Verification Matrix:**

| Test Category | Verification Action | Success Criteria | Common Issues |
|---------------|-------------------|------------------|---------------|
| Cron Parsing | Submit various cron expressions | Valid expressions parsed, invalid rejected | Timezone handling, edge cases missed |
| Schedule Triggering | Wait for scheduled execution | DAG run created at correct time | Clock drift, missed schedules |
| Manual Triggers | Trigger DAG via API | Immediate DAG run creation | Parameter validation, conflict handling |
| Catchup Behavior | Start scheduler with past schedule | Historical runs created if configured | Too many runs, infinite loops |
| Concurrent Scheduling | Multiple DAGs with same schedule | Both trigger correctly | Race conditions, deadlocks |

**Time-Based Verification:**

```bash
# Test immediate manual trigger
curl -X POST http://localhost:8080/api/v1/dags/test_linear/trigger \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"env": "test"}}'
# Expected: DAG run created immediately with "queued" state

# Verify DAG run creation
curl http://localhost:8080/api/v1/dags/test_linear/runs
# Expected: List showing triggered run with execution date and parameters

# Test cron schedule (use * * * * * for every minute during testing)
# Update test DAG schedule and wait for automatic trigger
# Expected: New DAG run appears within polling interval
```

**Scheduling Edge Cases:**

```bash
# Test timezone handling
curl -X POST http://localhost:8080/api/v1/dags \
  -H "Content-Type: application/json" \
  -d '{
    "id": "timezone_test",
    "schedule": "0 9 * * MON",
    "timezone": "America/New_York",
    "tasks": [{"id": "test", "type": "bash", "operator": {"bash_command": "date"}}]
  }'
# Expected: Schedule calculated correctly for EST/EDT timezone

# Test leap year handling
# Submit DAG with schedule "0 0 29 2 *" (Feb 29)
# Expected: Only triggers on leap years
```

#### Milestone 3: Task Execution Verification

Task execution verification validates dependency resolution, state transitions, and proper task completion handling:

**Task Execution Verification Steps:**

| Verification Area | Test Action | Expected Behavior | Validation Method |
|------------------|-------------|-------------------|-------------------|
| Dependency Resolution | Trigger DAG with dependencies | Tasks execute in correct order | Monitor task start times |
| Parallel Execution | DAG with independent tasks | Tasks run concurrently | Multiple running states |
| Failure Handling | Task with exit code 1 | Task marked failed, downstream skipped | State transitions logged |
| Retry Mechanism | Failing task with retry config | Multiple attempts with backoff | Attempt counter increments |
| XCom Passing | Tasks with data dependencies | Data flows correctly | XCom values retrievable |

**State Transition Monitoring:**

```bash
# Monitor task execution progress
watch -n 2 'curl -s http://localhost:8080/api/v1/dags/test_linear/runs/latest/tasks | jq ".[] | {id, state, start_date}"'
# Expected: States progress from "queued" -> "running" -> "success"

# Test failure propagation
curl -X POST http://localhost:8080/api/v1/dags/failure_test/trigger
# Monitor dependent tasks transition to "upstream_failed"

# Verify retry behavior
# Create task that fails twice then succeeds
# Expected: Multiple task instances with incrementing try_number
```

**XCom Data Flow Testing:**

```bash
# Test XCom push/pull
curl http://localhost:8080/api/v1/dags/xcom_test/runs/latest/tasks/producer/xcom
# Expected: XCom values pushed by producer task

curl http://localhost:8080/api/v1/dags/xcom_test/runs/latest/tasks/consumer/logs
# Expected: Consumer task log shows pulled XCom values
```

#### Milestone 4: Worker & Executor Verification

Worker and executor verification focuses on distributed task execution, worker health monitoring, and failure recovery:

**Distributed Execution Verification:**

| Component | Verification Test | Success Indicators | Monitoring Points |
|-----------|------------------|-------------------|-------------------|
| Worker Registration | Start multiple workers | All workers appear in registry | Heartbeat timestamps updated |
| Task Distribution | High task volume | Tasks distributed across workers | Load balancing observed |
| Worker Failure | Kill worker process | Tasks reassigned, worker removed | Failure detection latency |
| Result Collection | Mixed success/failure tasks | All results collected correctly | No orphaned tasks |
| Resource Management | Resource-intensive tasks | Proper slot management | No oversubscription |

**Worker Health Monitoring:**

```bash
# Check worker registry
curl http://localhost:8080/api/v1/workers
# Expected: List of active workers with heartbeat times and resource usage

# Simulate worker failure
pkill -f worker-process
# Monitor worker removal from registry and task reassignment

# Verify task distribution
curl http://localhost:8080/api/v1/workers/stats
# Expected: Task counts distributed across available workers
```

**Message Queue Validation:**

```bash
# Monitor task queue depth
redis-cli XLEN orchestrator:tasks
# Expected: Queue depth increases with task submission, decreases with worker processing

# Check task assignment
redis-cli XRANGE orchestrator:tasks - +
# Expected: Tasks in queue have proper format and timeout information
```

#### Milestone 5: Web UI & Monitoring Verification

Web UI verification ensures that the dashboard correctly displays workflow state and provides functional management controls:

**UI Functionality Verification:**

| Feature | Test Action | Expected Result | Browser Verification |
|---------|-------------|-----------------|---------------------|
| DAG List | Navigate to main page | All DAGs displayed with status | Tables populated, no errors |
| DAG Graph | Click DAG name | Visual dependency graph | Nodes connected correctly |
| Task Logs | Click running task | Log output displayed | Real-time updates |
| Manual Trigger | Click trigger button | New run starts | State changes reflected |
| Real-time Updates | Keep browser open during execution | Status updates automatically | No page refresh needed |

**API Endpoint Validation:**

```bash
# Test all API endpoints
curl http://localhost:8080/api/v1/dags
curl http://localhost:8080/api/v1/dags/test_linear
curl http://localhost:8080/api/v1/dags/test_linear/runs
curl http://localhost:8080/api/v1/dags/test_linear/runs/latest/tasks
# Expected: All endpoints return valid JSON responses

# Test Server-Sent Events
curl -H "Accept: text/event-stream" http://localhost:8080/api/v1/events
# Expected: Stream of events as system state changes
```

**Dashboard Visual Verification:**

The web UI requires manual verification of visual elements and interactive functionality:

1. **DAG Graph Rendering**: Verify that task nodes are positioned correctly and dependency edges are clearly visible
2. **State Color Coding**: Confirm that different task states use distinct, intuitive colors
3. **Real-time Updates**: Ensure that task status changes appear automatically without page refresh
4. **Responsive Design**: Test dashboard on different screen sizes and devices
5. **Error Handling**: Verify graceful handling of network errors and invalid inputs

> **Critical Integration Point**: The milestone verification steps should be run in sequence, as later milestones depend on earlier functionality. A failure in an earlier milestone's verification may indicate foundational issues that will cascade to subsequent milestones.

### Common Testing Pitfalls

⚠️ **Pitfall: Insufficient Test Data Variety**
Many developers test only with simple, well-formed inputs and miss edge cases that occur in production. DAG definitions can have complex dependency structures, unusual parameter combinations, and edge-case timing scenarios that simple test cases don't cover.

Testing with only linear DAGs misses issues with parallel execution, complex join patterns, and resource contention. Test data should include DAGs with various topologies: diamonds, wide parallel sections, deep linear chains, and mixed patterns. Parameter testing should cover edge cases like empty strings, very long values, special characters, and type mismatches.

⚠️ **Pitfall: Ignoring Timing and Race Conditions**
Workflow orchestration involves significant timing dependencies and concurrent operations that can lead to race conditions. Unit tests that don't account for timing may miss critical bugs that only appear under load or specific timing conditions.

Scheduler tests must account for clock skew, system load variations, and the non-deterministic nature of cron trigger timing. Task execution tests should validate behavior when multiple tasks start simultaneously or when worker responses arrive out of order. Use proper synchronization primitives and deterministic timing in tests rather than relying on sleep statements.

⚠️ **Pitfall: Mocking External Dependencies Incorrectly**
Over-mocking can hide integration issues, while under-mocking makes tests slow and unreliable. The key is mocking at the right boundary - close enough to the component under test to isolate it, but not so close that you mock away important behavior.

Mock database calls for unit tests but use real databases for integration tests. Mock HTTP clients for operator tests but use real message queues for executor tests. Ensure that mocks accurately simulate both successful operations and realistic failure modes including timeouts, connection errors, and partial failures.

⚠️ **Pitfall: Inadequate Failure Scenario Testing**
Many tests focus on the "happy path" and don't adequately test failure conditions. In a distributed system like a workflow orchestrator, failures are common and the system's response to failures is often more complex than its response to success.

Test various failure modes: worker crashes during task execution, database connection losses during state updates, message queue failures during task distribution, and network partitions between components. Validate that the system detects failures quickly, recovers gracefully, and maintains data consistency throughout failure and recovery cycles.

⚠️ **Pitfall: Not Testing Resource Limits and Scalability**
Development testing often occurs with small data sets and minimal concurrency, missing issues that appear when the system approaches resource limits. Production workflows can have hundreds of tasks, dozens of concurrent DAG runs, and sustained high throughput.

Include performance tests that validate behavior under resource pressure: maximum concurrent tasks, large DAG definitions, high task throughput, and extended runtime periods. Monitor memory usage, database connection pools, and message queue capacity during testing to identify resource leaks and bottlenecks before production deployment.

### Implementation Guidance

The testing strategy requires a comprehensive test infrastructure that supports both isolated unit testing and integrated system validation. This guidance provides the foundation for implementing robust testing across all orchestrator components.

#### Technology Recommendations

| Testing Layer | Simple Option | Advanced Option | Use Case |
|---------------|---------------|-----------------|----------|
| Unit Testing Framework | `testing` package (Go standard) | `testify` with assertions and mocks | Basic Go testing vs rich assertion library |
| Integration Testing | `httptest` + real DB | `testcontainers` + Docker | Simple HTTP testing vs containerized dependencies |
| Mocking | Manual interfaces | `gomock` or `counterfeiter` | Simple mocks vs generated mock objects |
| Database Testing | SQLite in-memory | PostgreSQL with `testcontainers` | Fast lightweight tests vs production-like testing |
| Message Queue Testing | Channel-based fake | Redis with `testcontainers` | Simple synchronous testing vs real async behavior |
| Load Testing | Basic Go routines | `k6` or `hey` | Simple concurrent requests vs sophisticated load patterns |

#### Recommended Test Structure

```
internal/
├── parser/
│   ├── parser.go
│   ├── parser_test.go              ← unit tests for parsing logic
│   ├── validator_test.go           ← cycle detection and validation tests
│   └── testdata/
│       ├── valid_dags/             ← test DAG files for parsing
│       └── invalid_dags/           ← malformed DAGs for error testing
├── scheduler/
│   ├── scheduler_test.go           ← cron parsing and schedule evaluation
│   ├── creator_test.go             ← DAG run creation logic
│   └── integration_test.go         ← scheduler + database integration
├── executor/
│   ├── executor_test.go            ← task distribution and result handling
│   ├── dependency_test.go          ← dependency resolution algorithms
│   └── worker_test.go              ← worker health monitoring
├── operators/
│   ├── python_test.go              ← Python operator execution
│   ├── bash_test.go                ← Bash operator execution
│   └── mock_executor.go            ← shared test utilities
└── testutils/                      ← shared testing infrastructure
    ├── database.go                 ← test database setup/teardown
    ├── fixtures.go                 ← test data generation
    └── assertions.go               ← custom test assertions
```

#### Test Infrastructure Starter Code

**Database Test Utilities** (`internal/testutils/database.go`):

```go
package testutils

import (
    "database/sql"
    "fmt"
    "testing"
    
    _ "github.com/lib/pq"
    "github.com/ory/dockertest/v3"
)

type TestDatabase struct {
    DB       *sql.DB
    Pool     *dockertest.Pool
    Resource *dockertest.Resource
}

// SetupTestDatabase creates a PostgreSQL container for integration testing
func SetupTestDatabase(t *testing.T) *TestDatabase {
    pool, err := dockertest.NewPool("")
    if err != nil {
        t.Fatalf("Could not connect to docker: %s", err)
    }

    resource, err := pool.Run("postgres", "13", []string{
        "POSTGRES_PASSWORD=test",
        "POSTGRES_DB=orchestrator_test",
        "POSTGRES_USER=test",
    })
    if err != nil {
        t.Fatalf("Could not start postgres container: %s", err)
    }

    var db *sql.DB
    if err := pool.Retry(func() error {
        var err error
        dsn := fmt.Sprintf("postgres://test:test@localhost:%s/orchestrator_test?sslmode=disable",
            resource.GetPort("5432/tcp"))
        db, err = sql.Open("postgres", dsn)
        if err != nil {
            return err
        }
        return db.Ping()
    }); err != nil {
        t.Fatalf("Could not connect to postgres: %s", err)
    }

    // Run migrations
    if err := runTestMigrations(db); err != nil {
        t.Fatalf("Could not run migrations: %s", err)
    }

    return &TestDatabase{
        DB:       db,
        Pool:     pool,
        Resource: resource,
    }
}

func (td *TestDatabase) Cleanup() {
    td.DB.Close()
    td.Pool.Purge(td.Resource)
}

func runTestMigrations(db *sql.DB) error {
    // TODO: Execute SQL migration files for test schema
    // Read migration files from migrations/ directory
    // Execute in order to create test database schema
    return nil
}
```

**Test Fixtures and Data Generation** (`internal/testutils/fixtures.go`):

```go
package testutils

import (
    "time"
    
    "github.com/workflow-orchestrator/internal/models"
)

// CreateTestDAG generates a DAG for testing with configurable structure
func CreateTestDAG(id string, taskCount int, dependencies map[string][]string) *models.DAG {
    dag := &models.DAG{
        ID:          id,
        Description: fmt.Sprintf("Test DAG with %d tasks", taskCount),
        Schedule:    "0 12 * * *",
        Tags:        []string{"test"},
        Parameters:  map[string]string{"env": "test"},
        Tasks:       make([]models.Task, 0, taskCount),
    }

    for i := 0; i < taskCount; i++ {
        taskID := fmt.Sprintf("task_%d", i+1)
        task := models.Task{
            ID:   taskID,
            Type: "bash",
            Operator: models.OperatorConfig{
                BashCommand: fmt.Sprintf("echo 'Executing %s'", taskID),
                Environment: map[string]string{"TASK_ID": taskID},
            },
            Dependencies: dependencies[taskID],
            Retries:      1,
            Timeout:      time.Minute * 5,
            Parameters:   map[string]string{},
        }
        dag.Tasks = append(dag.Tasks, task)
    }

    return dag
}

// CreateLinearDAG creates a simple linear dependency chain
func CreateLinearDAG(taskCount int) *models.DAG {
    dependencies := make(map[string][]string)
    for i := 2; i <= taskCount; i++ {
        dependencies[fmt.Sprintf("task_%d", i)] = []string{fmt.Sprintf("task_%d", i-1)}
    }
    return CreateTestDAG("linear_dag", taskCount, dependencies)
}

// CreateDiamondDAG creates a diamond-shaped dependency pattern
func CreateDiamondDAG() *models.DAG {
    dependencies := map[string][]string{
        "task_2": {"task_1"},
        "task_3": {"task_1"},
        "task_4": {"task_2", "task_3"},
    }
    return CreateTestDAG("diamond_dag", 4, dependencies)
}

// CreateComplexDAG creates a complex multi-path DAG for stress testing
func CreateComplexDAG() *models.DAG {
    dependencies := map[string][]string{
        "task_2":  {"task_1"},
        "task_3":  {"task_1"},
        "task_4":  {"task_1"},
        "task_5":  {"task_2", "task_3"},
        "task_6":  {"task_3", "task_4"},
        "task_7":  {"task_4"},
        "task_8":  {"task_5", "task_6"},
        "task_9":  {"task_6", "task_7"},
        "task_10": {"task_8", "task_9"},
    }
    return CreateTestDAG("complex_dag", 10, dependencies)
}
```

#### Core Component Test Skeletons

**DAG Parser Test Framework** (`internal/parser/parser_test.go`):

```go
package parser

import (
    "testing"
    "path/filepath"
    
    "github.com/workflow-orchestrator/internal/testutils"
)

func TestParser_LoadDAGDefinition(t *testing.T) {
    tests := []struct {
        name     string
        filename string
        wantErr  bool
        validate func(*testing.T, *models.DAGDefinition)
    }{
        {
            name:     "valid linear DAG",
            filename: "valid_linear.yaml",
            wantErr:  false,
            validate: func(t *testing.T, dag *models.DAGDefinition) {
                // TODO: Validate DAG has correct ID, 3 tasks, proper dependencies
                // TODO: Check task types and operator configurations
                // TODO: Verify schedule and parameter definitions
            },
        },
        {
            name:     "DAG with cycle",
            filename: "circular_dependency.yaml", 
            wantErr:  true,
            validate: nil,
        },
        // TODO: Add more test cases for various DAG structures and error conditions
    }

    parser := NewParser(/* TODO: provide test configuration */)
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // TODO: Load DAG definition from test file
            // TODO: Validate parsing result matches expectations
            // TODO: If validation function provided, call it to verify DAG structure
        })
    }
}

func TestValidator_DetectCycles(t *testing.T) {
    // TODO: Test cycle detection with various graph structures
    // TODO: Verify topological sort works correctly
    // TODO: Ensure clear error messages for cycle detection failures
}
```

**Scheduler Component Tests** (`internal/scheduler/scheduler_test.go`):

```go
package scheduler

import (
    "context"
    "testing"
    "time"
    
    "github.com/workflow-orchestrator/internal/testutils"
)

func TestScheduler_ProcessPendingSchedules(t *testing.T) {
    db := testutils.SetupTestDatabase(t)
    defer db.Cleanup()
    
    scheduler := NewScheduler(db.DB, /* TODO: other dependencies */)
    
    // TODO: Create test DAG with specific schedule
    // TODO: Set system time to trigger condition
    // TODO: Call processPendingSchedules()
    // TODO: Verify DAG run was created with correct execution date
    // TODO: Validate all task instances were created atomically
}

func TestCronParser_NextExecution(t *testing.T) {
    tests := []struct {
        name     string
        cronExpr string
        from     time.Time
        expected time.Time
    }{
        {
            name:     "daily at noon",
            cronExpr: "0 12 * * *",
            from:     time.Date(2023, 1, 15, 10, 0, 0, 0, time.UTC),
            expected: time.Date(2023, 1, 15, 12, 0, 0, 0, time.UTC),
        },
        // TODO: Add timezone tests, DST transition tests, leap year tests
    }
    
    parser := NewCronParser(time.UTC)
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // TODO: Parse cron expression
            // TODO: Calculate next execution time
            // TODO: Verify result matches expected time
        })
    }
}
```

**Task Execution Tests** (`internal/executor/dependency_test.go`):

```go
package executor

import (
    "context"
    "testing"
    
    "github.com/workflow-orchestrator/internal/models"
    "github.com/workflow-orchestrator/internal/testutils"
)

func TestResolver_EvaluateReadyTasks(t *testing.T) {
    db := testutils.SetupTestDatabase(t)
    defer db.Cleanup()
    
    resolver := NewResolver(db.DB, /* logger */)
    
    // TODO: Create DAG run with specific task states
    // TODO: Set some tasks to "success", others to "queued"
    // TODO: Call EvaluateReadyTasks()
    // TODO: Verify only tasks with satisfied dependencies are returned
    // TODO: Test various trigger rule scenarios (all_success, all_done, etc.)
}

func TestRetryHandler_ShouldRetry(t *testing.T) {
    config := RetryConfig{
        MaxRetries:       3,
        RetryDelay:       time.Second * 10,
        BackoffMultiplier: 2.0,
        RetryExitCodes:   []int{1, 2},
        NoRetryExitCodes: []int{3},
    }
    
    handler := NewRetryHandler(config, /* logger */)
    
    tests := []struct {
        name          string
        exitCode      *int
        attemptNumber int
        shouldRetry   bool
        expectedReason string
    }{
        {
            name:           "retriable exit code within limit",
            exitCode:       intPtr(1),
            attemptNumber:  1,
            shouldRetry:    true,
            expectedReason: "retriable_exit_code",
        },
        // TODO: Add test cases for max retries exceeded, non-retriable exit codes
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // TODO: Call ShouldRetry with test parameters
            // TODO: Verify retry decision and reasoning
            // TODO: Check backoff delay calculation
        })
    }
}

func intPtr(i int) *int { return &i }
```

#### Milestone Verification Scripts

**Automated Milestone Checkpoints** (`scripts/verify-milestone.sh`):

```bash
#!/bin/bash
set -e

MILESTONE=$1
if [ -z "$MILESTONE" ]; then
    echo "Usage: $0 <milestone-number>"
    exit 1
fi

case $MILESTONE in
    1)
        echo "=== Verifying Milestone 1: DAG Definition ==="
        echo "Running DAG parser tests..."
        go test ./internal/parser/... -v
        
        echo "Testing cycle detection..."
        go test ./internal/parser/ -run TestCycleDetection -v
        
        echo "Verifying file discovery..."
        go test ./internal/parser/ -run TestFileDiscovery -v
        
        echo "Manual DAG loading test..."
        # TODO: Start server and test DAG loading via API
        ;;
    2)
        echo "=== Verifying Milestone 2: Scheduler ==="
        echo "Running scheduler tests..."
        go test ./internal/scheduler/... -v
        
        echo "Testing cron parsing..."
        go test ./internal/scheduler/ -run TestCronParser -v
        
        echo "Manual trigger test..."
        # TODO: Test manual DAG triggering via API
        ;;
    3)
        echo "=== Verifying Milestone 3: Task Execution ==="
        echo "Running task execution tests..."
        go test ./internal/executor/... -v
        
        echo "Testing dependency resolution..."
        go test ./internal/executor/ -run TestDependencyResolution -v
        
        echo "End-to-end execution test..."
        # TODO: Trigger DAG and verify complete execution
        ;;
    4)
        echo "=== Verifying Milestone 4: Worker & Executor ==="
        echo "Running distributed execution tests..."
        go test ./internal/executor/ -run TestDistributed -v
        
        echo "Testing worker health monitoring..."
        go test ./internal/executor/ -run TestWorkerHealth -v
        
        echo "Load testing with multiple workers..."
        # TODO: Start multiple workers and test task distribution
        ;;
    5)
        echo "=== Verifying Milestone 5: Web UI & Monitoring ==="
        echo "Running API tests..."
        go test ./internal/server/... -v
        
        echo "Testing SSE connections..."
        go test ./internal/events/ -run TestSSE -v
        
        echo "UI functionality test..."
        # TODO: Automated browser testing or manual checklist
        ;;
    *)
        echo "Unknown milestone: $MILESTONE"
        exit 1
        ;;
esac

echo "Milestone $MILESTONE verification complete!"
```

#### Performance and Load Testing

**Benchmark Test Template** (`internal/scheduler/scheduler_bench_test.go`):

```go
package scheduler

import (
    "context"
    "testing"
    
    "github.com/workflow-orchestrator/internal/testutils"
)

func BenchmarkScheduler_ProcessPendingSchedules(b *testing.B) {
    db := testutils.SetupTestDatabase(&testing.T{})
    defer db.Cleanup()
    
    scheduler := NewScheduler(db.DB, /* dependencies */)
    
    // TODO: Create multiple DAGs with various schedules
    // TODO: Set up realistic database state
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        // TODO: Call processPendingSchedules and measure performance
        // TODO: Reset state between iterations if necessary
    }
}

func BenchmarkExecutor_TaskDistribution(b *testing.B) {
    // TODO: Benchmark task distribution across multiple workers
    // TODO: Measure throughput and latency under various load levels
    // TODO: Test with different task types and execution times
}
```

This comprehensive testing strategy provides the foundation for building a reliable workflow orchestration system. Each milestone builds upon previous testing infrastructure, ensuring that new functionality integrates correctly with existing components while maintaining system reliability and performance characteristics.


## Debugging Guide

> **Milestone(s):** All milestones (comprehensive debugging support for DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring)

Building a distributed workflow orchestrator involves many moving parts that can fail in complex and often subtle ways. This debugging guide provides systematic approaches for diagnosing and resolving the most common issues encountered during development and operation of the system.

### Mental Model: The Detective Investigation

Think of debugging a workflow orchestrator like conducting a detective investigation at a crime scene. Each component leaves traces of its behavior in logs, database state, and system metrics. When something goes wrong, you need to gather evidence from multiple sources, establish a timeline of events, identify the root cause, and apply the appropriate fix. Just as a detective follows clues methodically rather than guessing randomly, effective debugging requires a structured approach to narrow down the problem space systematically.

The key insight is that workflow orchestration failures often manifest as cascading effects - a small issue in one component (like a malformed DAG definition) can cause failures throughout the system (scheduler crashes, tasks stuck in queued state, workers becoming unresponsive). By understanding the dependency relationships between components, you can often trace backwards from symptoms to find the original cause.

### DAG Parsing and Validation Issues

DAG parsing and validation failures represent some of the most common issues in workflow orchestration, often occurring during development when creating new workflows or modifying existing ones. These failures can be particularly frustrating because they prevent workflows from even being registered in the system.

#### Cycle Detection Failures

Cycle detection is critical for ensuring that workflows form valid directed acyclic graphs. When the `ValidateDAG()` method fails during cycle detection, it typically indicates that the dependency declarations in the DAG definition create circular dependencies.

The most common symptom is receiving an error message during DAG file processing that indicates "cycle detected in DAG dependencies" or similar. This error occurs when the topological sort algorithm implemented in `DetectCycles()` encounters a situation where it cannot find any tasks with zero in-degree (no upstream dependencies remaining).

| Symptom | Root Cause | Diagnostic Steps | Resolution |
|---------|------------|------------------|------------|
| "Cycle detected in task dependencies" error during DAG registration | Circular dependency chain in task definitions | 1. Print all task dependencies using `GetUpstreamTasks()` for each task<br>2. Manually trace dependency paths to find loops<br>3. Use graph visualization tools to render the dependency structure | Remove or restructure dependencies to eliminate cycles |
| Tasks appear multiple times during topological sort | Self-referencing task with itself in dependencies list | Check each task's Dependencies field for its own TaskID | Remove self-references from dependency declarations |
| Infinite loop during cycle detection algorithm | Incorrect implementation of topological sort with queue never emptying | Add debug logging to show queue state and in-degree counts at each iteration | Fix topological sort implementation to properly decrement in-degree counters |

#### Missing Dependency References

Another frequent issue occurs when task dependency declarations reference tasks that don't exist in the DAG definition. The `ValidateDependencyReferences()` method should catch these during validation, but implementation bugs can allow invalid references to slip through.

| Error Pattern | Likely Cause | Investigation Method | Fix |
|---------------|--------------|---------------------|-----|
| Task instances created but never become ready for execution | Dependency references non-existent task ID | Query `TaskInstance` table for tasks stuck in `none` or `queued` state indefinitely | Correct task ID in dependency declaration or add missing task definition |
| "Task not found" errors during dependency resolution | Typo in task ID or case sensitivity mismatch | Compare dependency strings with actual task IDs using case-sensitive comparison | Standardize task naming convention and validate references |
| Tasks become ready immediately despite having declared dependencies | Validation skipped non-existent dependencies instead of failing | Add comprehensive logging to `ValidateDependencyReferences()` method | Implement strict validation that rejects DAGs with invalid references |

#### Malformed DAG Definition Files

YAML parsing errors and schema validation failures often result from syntax errors, incorrect field types, or missing required fields in DAG definition files. The `LoadDAGDefinition()` method should provide clear error messages, but cryptic YAML parsing errors can be difficult to interpret.

> **Critical Insight**: YAML is notoriously sensitive to indentation and type interpretation. A string that looks like a number ("01:30") might be parsed as a time value, and incorrect indentation can create unexpected nesting structures.

Common malformed definition patterns include:

| YAML Issue | Manifestation | Detection Method | Correction |
|------------|---------------|------------------|------------|
| Incorrect indentation causing nested structure | Fields appearing under wrong parent in parsed DAGDefinition | Print parsed structure and compare with intended hierarchy | Fix indentation using consistent spaces (never tabs) |
| String values interpreted as other types | Numeric task IDs becoming integers instead of strings | Type assertion errors during task creation | Quote all string values explicitly in YAML |
| Missing required fields | Null pointer exceptions when accessing DAG properties | Validate all required fields are present after parsing | Add comprehensive schema validation |
| Invalid cron expressions in schedule field | Parsing succeeds but schedule never triggers | Test cron expression with `CronParser.Parse()` method | Use valid five-field cron syntax |

#### Template Validation Errors

When DAG definitions include template expressions for parameterization, template validation can fail due to invalid syntax, undefined variables, or type mismatches. The `ValidateTemplate()` method should catch these during DAG registration.

Template-related errors often manifest as:

1. **Syntax Errors**: Malformed template expressions with unmatched braces or invalid function calls
2. **Undefined Variables**: References to parameters or context variables that aren't available
3. **Type Mismatches**: Template expressions that expect string values but receive numbers or booleans
4. **Circular References**: Template variables that reference themselves directly or indirectly

⚠️ **Pitfall: Late Template Evaluation**
Templates are often validated syntactically during DAG registration but only evaluated at execution time. This means template variable errors might not surface until a DAG run actually starts, making them harder to catch during development.

To avoid this pitfall, implement comprehensive template validation that simulates execution-time evaluation with mock data covering all expected parameter types and edge cases.

### Scheduling Problems

Scheduling issues in workflow orchestrators can be particularly insidious because they often involve time-based logic, timezone handling, and complex interactions between the scheduler polling loop and database state. These problems frequently manifest as workflows that don't trigger when expected or trigger at incorrect times.

#### Missed Schedules

Missed schedules represent one of the most critical operational issues, as they can result in data processing gaps and SLA violations. The scheduler's main responsibility is to detect when DAGs should trigger based on their cron expressions and create corresponding `DAGRun` instances.

The most common root causes for missed schedules include:

| Cause Category | Specific Issues | Diagnostic Approach | Resolution Strategy |
|----------------|----------------|-------------------|-------------------|
| Scheduler Performance | Processing loop takes longer than schedule interval | Monitor scheduler loop duration vs polling interval | Optimize database queries, increase polling frequency, or implement parallel processing |
| Database Contention | Lock contention during DAG run creation | Check database logs for lock timeouts and query blocking | Add database connection pooling, optimize indexes, implement read replicas |
| Timezone Handling | Incorrect timezone conversion in cron evaluation | Compare expected vs actual execution dates in different timezones | Standardize on UTC for internal processing, convert only for display |
| Clock Skew | System clock drift between scheduler and database | Compare system time with database server time | Implement NTP synchronization and clock skew monitoring |

#### Cron Expression Parsing Issues

Cron expression parsing involves complex logic for interpreting the five-field format and calculating next execution times. The `CronParser.Parse()` method must handle edge cases like leap years, month boundaries, and daylight saving time transitions.

Common cron parsing problems include:

1. **Invalid Field Values**: Cron expressions with out-of-range values (e.g., hour 25, minute 60)
2. **Special Character Misinterpretation**: Incorrect handling of wildcards (*), ranges (-), lists (,), and step values (/)
3. **Timezone Transitions**: Incorrect behavior during daylight saving time changes
4. **February 29th Edge Case**: Schedules that specify day 29-31 during February

> **Design Principle**: Always validate cron expressions during DAG registration rather than at execution time. This prevents invalid schedules from being stored and provides immediate feedback to workflow developers.

#### Scheduler Performance Problems

As the number of registered DAGs grows, scheduler performance can degrade due to inefficient database queries, excessive polling, or inadequate indexing. Performance problems typically manifest as increasing latency between scheduled time and actual execution.

Key performance bottlenecks include:

| Bottleneck Area | Symptoms | Measurement Method | Optimization Approach |
|-----------------|----------|-------------------|----------------------|
| DAG Discovery | Long delays between file changes and DAG updates | Monitor `ProcessDAGFile()` execution time | Implement file watching instead of polling, cache parsed DAGs |
| Schedule Evaluation | High CPU usage during cron calculations | Profile `NextExecution()` method calls | Cache next execution times, batch schedule evaluations |
| Database Queries | Slow response times during DAG run creation | Analyze query execution plans | Add appropriate indexes, optimize WHERE clauses |
| Lock Contention | Timeouts during concurrent DAG run creation | Monitor database lock wait events | Implement row-level locking, reduce transaction scope |

⚠️ **Pitfall: Scheduler Drift**
Over time, processing delays can cause the scheduler to fall behind, resulting in a backlog of overdue DAG runs. Without proper catchup handling, this can lead to either overwhelming the system with simultaneous executions or permanent schedule gaps.

To prevent scheduler drift, implement configurable catchup behavior that limits the number of backfill runs created simultaneously and provides options for skipping overdue executions entirely.

#### Timezone and Daylight Saving Time Issues

Timezone handling represents one of the most complex aspects of scheduling, particularly when dealing with daylight saving time transitions. The scheduler must correctly interpret local time schedules while maintaining consistent internal representation in UTC.

Critical timezone considerations include:

1. **Spring Forward Transitions**: Schedules that fall within the "lost hour" during spring transitions
2. **Fall Back Transitions**: Schedules that occur twice during fall transitions
3. **Timezone Database Updates**: Changes to timezone definitions that affect future calculations
4. **Cross-Timezone Dependencies**: DAGs with tasks scheduled in different timezones

### Task Execution Failures

Task execution failures encompass a wide range of issues from simple task implementation bugs to complex distributed system problems involving worker communication, dependency resolution, and resource management.

#### Worker Communication Issues

Distributed task execution relies on reliable communication between the scheduler, executor, and worker processes through message queues. Communication failures can result in tasks appearing stuck in the `queued` state or workers becoming unresponsive.

| Communication Problem | Observable Symptoms | Diagnostic Commands | Resolution Steps |
|----------------------|-------------------|-------------------|------------------|
| Message queue connectivity | Tasks queued but never picked up by workers | Check Redis/RabbitMQ connection status and queue lengths | Verify network connectivity, restart message queue service |
| Worker heartbeat failures | Workers showing as offline despite running processes | Monitor `WorkerRegistry.UpdateHeartbeat()` calls | Check network latency, adjust heartbeat timeout values |
| Task result publishing | Tasks complete on workers but remain in `running` state | Examine worker logs for result publishing errors | Verify message queue write permissions, check serialization errors |
| Consumer group coordination | Multiple workers processing same task | Inspect message queue consumer group membership | Reset consumer group state, ensure unique consumer IDs |

The `MessageQueue.ConsumeTask()` and `MessageQueue.PublishResult()` methods are critical for worker communication. Failures in these methods often indicate underlying infrastructure issues rather than application logic bugs.

#### Task Timeout Handling

Task timeout enforcement prevents runaway tasks from consuming resources indefinitely, but incorrect timeout handling can lead to tasks being killed prematurely or never being terminated despite exceeding their limits.

Common timeout-related issues include:

1. **Premature Timeouts**: Tasks killed before completion due to overly aggressive timeout values
2. **Timeout Not Enforced**: Tasks running indefinitely despite configured timeout limits
3. **Grace Period Handling**: Improper cleanup when tasks are forcibly terminated
4. **Timeout Inheritance**: Child processes continuing to run after parent task termination

> **Implementation Detail**: Timeout enforcement requires careful coordination between the executor and worker processes. The executor must track task start times and send termination signals, while workers must implement graceful shutdown procedures that clean up resources properly.

#### Dependency Resolution Problems

The dependency resolution system determines when tasks are ready for execution based on upstream task completion. Bugs in this system can cause tasks to never become ready despite all dependencies being satisfied, or tasks to start prematurely before dependencies complete.

The `DependencyResolver.EvaluateReadyTasks()` method implements the core logic for dependency checking. Common issues include:

| Dependency Issue | Root Cause | Debugging Approach | Fix Strategy |
|------------------|------------|-------------------|--------------|
| Tasks stuck in queued state indefinitely | Upstream task marked as success but dependency check fails | Query TaskInstance states for all upstream dependencies | Fix state transition logic or dependency reference resolution |
| Tasks start before all dependencies complete | Race condition between task completion and dependency evaluation | Add logging to show exact timing of state changes | Implement proper synchronization in dependency checking |
| Circular dependency deadlock | Runtime circular dependencies not caught during validation | Use dependency graph visualization to identify cycles | Add runtime cycle detection or improve validation |
| Trigger rule evaluation errors | Complex trigger rules not properly implemented | Test trigger rule logic with various upstream state combinations | Simplify trigger rules or fix evaluation logic |

#### XCom Communication Failures

Cross-task communication through XCom allows tasks to pass data between each other, but XCom failures can break workflows that depend on data sharing. The `XComManager` interface provides methods for storing and retrieving task outputs.

XCom-related problems typically involve:

1. **Serialization Errors**: Data types that cannot be properly serialized/deserialized
2. **Size Limitations**: Attempting to store data larger than configured limits
3. **Key Collisions**: Multiple tasks writing to the same XCom key
4. **Cleanup Issues**: XCom data not properly removed after DAG run completion

### Debugging Tools and Techniques

Effective debugging of a workflow orchestrator requires a comprehensive toolkit of logging strategies, monitoring points, diagnostic commands, and troubleshooting procedures. The distributed nature of the system makes traditional debugging approaches insufficient.

#### Logging Strategies

Comprehensive logging forms the foundation of effective debugging for distributed systems. The orchestrator should implement structured logging with consistent formats across all components, making it possible to correlate events across the scheduler, executor, and multiple worker processes.

Essential logging categories include:

| Log Category | Purpose | Key Information | Format Considerations |
|--------------|---------|----------------|----------------------|
| DAG Lifecycle | Track DAG registration, validation, and scheduling | DAG ID, operation type, timestamps, error details | Include correlation IDs for multi-step operations |
| Task Execution | Monitor individual task instances from queue to completion | Task instance ID, state transitions, duration, exit codes | Structure for easy filtering by DAG run or task type |
| Worker Health | Track worker registration, heartbeats, and failures | Worker ID, resource usage, active tasks, error conditions | Enable real-time monitoring and alerting |
| System Performance | Monitor resource usage and performance metrics | Database query times, queue depths, memory usage | Format for time-series analysis and trending |

> **Logging Anti-Pattern**: Avoid logging sensitive information such as task parameters that might contain credentials, personal data, or proprietary information. Implement log sanitization that redacts sensitive fields while preserving debugging utility.

The `LoggingConfig` structure should support dynamic log level changes without service restart, enabling operators to increase verbosity when investigating issues without disrupting running workflows.

#### Monitoring and Observability

Beyond basic logging, comprehensive monitoring provides real-time visibility into system health and performance. Effective monitoring strategies combine metrics collection, alerting, and visualization to enable proactive issue detection.

Key monitoring dimensions include:

1. **Component Health**: Service availability, response times, error rates
2. **Resource Utilization**: CPU, memory, disk usage, network throughput
3. **Business Metrics**: DAG success rates, task completion times, schedule adherence
4. **Queue Depths**: Task queues, result queues, worker availability

Monitoring should integrate with the `Event` system to provide real-time updates through the `Hub` component, enabling both automated alerting and live dashboard updates.

#### Diagnostic Commands and Queries

Database queries and command-line tools provide essential capabilities for investigating system state and diagnosing issues. These diagnostic tools should be scriptable and provide both summary and detailed views of system status.

Essential diagnostic queries include:

| Query Purpose | SQL Example Pattern | Use Case |
|---------------|-------------------|----------|
| Task State Summary | `SELECT state, COUNT(*) FROM task_instances WHERE dag_run_id = ? GROUP BY state` | Quickly assess DAG run progress |
| Long-Running Tasks | `SELECT task_id, start_date, EXTRACT(EPOCH FROM NOW() - start_date) as duration FROM task_instances WHERE state = 'running' ORDER BY start_date` | Identify potentially stuck tasks |
| Worker Status Overview | `SELECT worker_id, last_heartbeat, available_slots FROM worker_status WHERE last_heartbeat > NOW() - INTERVAL '5 minutes'` | Check worker availability |
| Failed Task Analysis | `SELECT task_id, error_message, try_number FROM task_instances WHERE state = 'failed' AND dag_run_id = ?` | Debug task failures |

#### Circuit Breaker and Health Check Integration

The `CircuitBreaker` and `HealthChecker` components provide essential diagnostic information about system resilience and failure patterns. Debugging often requires understanding not just what failed, but why the failure detection and recovery mechanisms responded as they did.

Circuit breaker diagnostics should include:

1. **State History**: Timeline of circuit breaker state changes (closed, open, half-open)
2. **Failure Patterns**: Types of failures that triggered circuit opening
3. **Recovery Attempts**: Success/failure rates during half-open state testing
4. **Impact Analysis**: How circuit breaker actions affected system behavior

Health check diagnostics provide insights into worker and service availability:

| Health Check Type | Diagnostic Information | Troubleshooting Value |
|------------------|----------------------|----------------------|
| Worker Heartbeat | Last heartbeat timestamp, consecutive failures, resource metrics | Identify network issues, resource exhaustion, or worker crashes |
| Database Connectivity | Connection pool status, query response times, error rates | Diagnose database performance or availability issues |
| Message Queue Health | Queue depths, consumer lag, connection status | Identify message processing bottlenecks |
| External Service Dependencies | Response times, error rates, circuit breaker states | Understand impact of external service issues |

⚠️ **Pitfall: Diagnostic Tool Performance Impact**
Intensive diagnostic queries and monitoring can themselves impact system performance. Implement diagnostic tools with appropriate rate limiting, query optimization, and resource isolation to prevent debugging activities from exacerbating production issues.

#### Troubleshooting Workflows

Systematic troubleshooting workflows help operators diagnose issues efficiently by following structured procedures rather than ad-hoc investigation. These workflows should be documented and practiced regularly.

**Standard Troubleshooting Procedure:**

1. **Symptom Identification**: Clearly define what is not working as expected
2. **Impact Assessment**: Determine scope and severity of the issue
3. **Initial Data Gathering**: Collect logs, metrics, and system status information
4. **Hypothesis Formation**: Develop theories about root causes based on available evidence
5. **Testing and Validation**: Use diagnostic tools to test hypotheses systematically
6. **Resolution Implementation**: Apply fixes with proper change control and monitoring
7. **Post-Resolution Verification**: Confirm resolution and implement preventive measures

**Component-Specific Investigation Paths:**

For DAG parsing issues:
1. Check file discovery logs for scanning errors
2. Validate YAML syntax using external parser
3. Verify all referenced tasks exist in definition
4. Run cycle detection algorithm manually
5. Test template expressions with sample data

For scheduling problems:
1. Verify system time synchronization
2. Test cron expression parsing independently
3. Check database for scheduling conflicts
4. Monitor scheduler loop performance
5. Validate timezone configuration

For task execution failures:
1. Check worker availability and resource usage
2. Verify message queue connectivity and processing
3. Examine task logs for implementation errors
4. Test dependency resolution logic
5. Validate timeout and retry configurations

### Implementation Guidance

The debugging capabilities of a workflow orchestrator are crucial for operational success. This implementation guidance provides practical tools and techniques for building comprehensive debugging support into the system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Structured Logging | `logrus` with JSON formatter | `zap` with structured fields and sampling |
| Metrics Collection | Prometheus client with basic counters | OpenTelemetry with distributed tracing |
| Health Monitoring | HTTP health check endpoints | Kubernetes readiness/liveness probes |
| Diagnostic Queries | Raw SQL with parameter binding | Query builder with type safety |
| Log Aggregation | File-based logging with rotation | ELK stack or centralized logging service |

#### Debugging Infrastructure Starter Code

Complete logging infrastructure that provides structured, correlatable logs across all components:

```go
package logging

import (
    "context"
    "fmt"
    "github.com/sirupsen/logrus"
    "os"
    "runtime"
    "time"
)

// Logger provides structured logging with correlation support
type Logger struct {
    logger *logrus.Logger
    fields logrus.Fields
}

// NewLogger creates a logger with the specified configuration
func NewLogger(config LoggingConfig) *Logger {
    logger := logrus.New()
    
    // Configure output format
    if config.Format == "json" {
        logger.SetFormatter(&logrus.JSONFormatter{
            TimestampFormat: time.RFC3339,
            FieldMap: logrus.FieldMap{
                logrus.FieldKeyTime:  "timestamp",
                logrus.FieldKeyLevel: "level",
                logrus.FieldKeyMsg:   "message",
            },
        })
    } else {
        logger.SetFormatter(&logrus.TextFormatter{
            TimestampFormat: time.RFC3339,
            FullTimestamp:   true,
        })
    }
    
    // Set log level
    level, err := logrus.ParseLevel(config.Level)
    if err != nil {
        level = logrus.InfoLevel
    }
    logger.SetLevel(level)
    
    logger.SetOutput(os.Stdout)
    
    return &Logger{
        logger: logger,
        fields: make(logrus.Fields),
    }
}

// WithField adds a field to logger context
func (l *Logger) WithField(key string, value interface{}) *Logger {
    newFields := make(logrus.Fields)
    for k, v := range l.fields {
        newFields[k] = v
    }
    newFields[key] = value
    
    return &Logger{
        logger: l.logger,
        fields: newFields,
    }
}

// WithError adds error information to logger context
func (l *Logger) WithError(err error) *Logger {
    return l.WithField("error", err.Error()).
        WithField("error_type", fmt.Sprintf("%T", err))
}

// Debug logs debug level message
func (l *Logger) Debug(msg string) {
    l.logger.WithFields(l.fields).Debug(msg)
}

// Info logs info level message
func (l *Logger) Info(msg string) {
    l.logger.WithFields(l.fields).Info(msg)
}

// Error logs error level message
func (l *Logger) Error(msg string) {
    // Add caller information for errors
    _, file, line, ok := runtime.Caller(1)
    if ok {
        l.logger.WithFields(l.fields).
            WithField("caller", fmt.Sprintf("%s:%d", file, line)).
            Error(msg)
    } else {
        l.logger.WithFields(l.fields).Error(msg)
    }
}

// Fatal logs fatal level message and exits
func (l *Logger) Fatal(msg string) {
    _, file, line, ok := runtime.Caller(1)
    if ok {
        l.logger.WithFields(l.fields).
            WithField("caller", fmt.Sprintf("%s:%d", file, line)).
            Fatal(msg)
    } else {
        l.logger.WithFields(l.fields).Fatal(msg)
    }
}
```

Diagnostic query utilities for common troubleshooting scenarios:

```go
package diagnostics

import (
    "database/sql"
    "fmt"
    "time"
)

// DiagnosticQueries provides pre-built queries for common debugging scenarios
type DiagnosticQueries struct {
    db     *sql.DB
    logger Logger
}

// NewDiagnosticQueries creates diagnostic query utilities
func NewDiagnosticQueries(db *sql.DB, logger Logger) *DiagnosticQueries {
    return &DiagnosticQueries{
        db:     db,
        logger: logger,
    }
}

// TaskStateSummary returns count of tasks in each state for a DAG run
func (d *DiagnosticQueries) TaskStateSummary(dagRunID string) (map[string]int, error) {
    query := `
        SELECT state, COUNT(*) as count 
        FROM task_instances 
        WHERE dag_run_id = $1 
        GROUP BY state
    `
    
    rows, err := d.db.Query(query, dagRunID)
    if err != nil {
        return nil, fmt.Errorf("failed to query task states: %w", err)
    }
    defer rows.Close()
    
    results := make(map[string]int)
    for rows.Next() {
        var state string
        var count int
        if err := rows.Scan(&state, &count); err != nil {
            return nil, fmt.Errorf("failed to scan task state row: %w", err)
        }
        results[state] = count
    }
    
    return results, nil
}

// LongRunningTasks identifies tasks that have been running longer than threshold
func (d *DiagnosticQueries) LongRunningTasks(threshold time.Duration) ([]LongRunningTask, error) {
    query := `
        SELECT 
            ti.id,
            ti.task_id,
            ti.dag_run_id,
            ti.start_date,
            EXTRACT(EPOCH FROM NOW() - ti.start_date) as duration_seconds
        FROM task_instances ti
        WHERE ti.state = 'running'
        AND ti.start_date < NOW() - INTERVAL '%d seconds'
        ORDER BY ti.start_date ASC
    `
    
    formattedQuery := fmt.Sprintf(query, int(threshold.Seconds()))
    rows, err := d.db.Query(formattedQuery)
    if err != nil {
        return nil, fmt.Errorf("failed to query long running tasks: %w", err)
    }
    defer rows.Close()
    
    var tasks []LongRunningTask
    for rows.Next() {
        var task LongRunningTask
        if err := rows.Scan(
            &task.InstanceID,
            &task.TaskID,
            &task.DAGRunID,
            &task.StartDate,
            &task.DurationSeconds,
        ); err != nil {
            return nil, fmt.Errorf("failed to scan long running task: %w", err)
        }
        tasks = append(tasks, task)
    }
    
    return tasks, nil
}

type LongRunningTask struct {
    InstanceID      string
    TaskID          string
    DAGRunID        string
    StartDate       time.Time
    DurationSeconds float64
}
```

#### Core Debugging Logic Skeleton

Diagnostic command implementation that operators can use for troubleshooting:

```go
// DiagnoseDAGRun performs comprehensive analysis of a DAG run's status
func (d *DiagnosticQueries) DiagnoseDAGRun(ctx context.Context, dagRunID string) (*DAGRunDiagnosis, error) {
    // TODO 1: Query basic DAG run information (state, start time, execution date)
    // TODO 2: Get task state summary using TaskStateSummary method
    // TODO 3: Identify any failed tasks and retrieve error messages
    // TODO 4: Check for tasks stuck in queued state beyond normal timeouts
    // TODO 5: Analyze dependency chains for any blocked task sequences
    // TODO 6: Verify worker availability for any queued tasks
    // TODO 7: Check XCom usage patterns for any communication issues
    // TODO 8: Generate recommendations based on identified issues
    // Hint: Use context cancellation to avoid long-running diagnostic queries
    return nil, nil
}

// AnalyzeSchedulingIssues examines DAG scheduling patterns and identifies problems
func (d *DiagnosticQueries) AnalyzeSchedulingIssues(ctx context.Context, dagID string, timeWindow time.Duration) (*SchedulingAnalysis, error) {
    // TODO 1: Query recent DAG runs within time window
    // TODO 2: Calculate expected vs actual execution times based on schedule
    // TODO 3: Identify any missed schedule windows
    // TODO 4: Analyze execution duration trends
    // TODO 5: Check for scheduling conflicts or resource contention
    // TODO 6: Examine catchup behavior and backfill patterns
    // TODO 7: Generate scheduling health score and recommendations
    // Hint: Handle timezone conversions carefully when comparing scheduled vs actual times
    return nil, nil
}

// ValidateWorkerHealth checks worker status and identifies communication issues
func (d *DiagnosticQueries) ValidateWorkerHealth(ctx context.Context) (*WorkerHealthReport, error) {
    // TODO 1: Query all registered workers from worker registry
    // TODO 2: Check heartbeat timestamps against timeout threshold
    // TODO 3: Analyze resource usage trends and capacity
    // TODO 4: Identify workers with consistently high failure rates
    // TODO 5: Test message queue connectivity for each worker
    // TODO 6: Verify task distribution patterns for load balancing
    // TODO 7: Generate worker health summary with specific recommendations
    // Hint: Use worker registry methods rather than direct database queries
    return nil, nil
}
```

#### Milestone Checkpoints

**After implementing basic logging:**
- Run: `go test ./internal/logging/...`
- Verify: Log messages appear with proper structure and correlation IDs
- Test: Generate logs from multiple components and verify consistent formatting

**After implementing diagnostic queries:**
- Run: `go run cmd/diagnostic/main.go --dag-run-id=<test-run-id>`
- Verify: Query returns task state summary and identifies any issues
- Test: Create DAG run with failed tasks and verify diagnostic output

**After implementing health monitoring:**
- Run: `curl http://localhost:8080/health` and verify response
- Verify: Worker heartbeats update regularly in database
- Test: Stop worker process and verify failure detection

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| No log output visible | Logger not properly initialized | Check LoggingConfig and output destination | Initialize logger in main() and pass to all components |
| Logs missing correlation information | Missing context propagation | Verify Logger.WithField() usage across components | Add consistent field names for DAG ID, run ID, task ID |
| High memory usage during debugging | Excessive log buffering | Monitor memory usage during high-volume logging | Implement log sampling or reduce log level in production |
| Diagnostic queries timeout | Complex queries on large datasets | Use EXPLAIN ANALYZE to check query performance | Add appropriate indexes and implement query timeouts |
| Health checks show false positives | Overly sensitive failure thresholds | Adjust heartbeat timeout and failure threshold values | Implement graduated failure detection with multiple severity levels |


## Future Extensions

> **Milestone(s):** Future development phases beyond the core workflow orchestrator (extending capabilities after completing DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring)

Building a workflow orchestrator is an iterative process where the core functionality serves as a foundation for increasingly sophisticated features. Once the fundamental DAG-based scheduling, execution, and monitoring capabilities are established, several natural extensions emerge to address real-world enterprise requirements and operational complexity. These extensions transform the orchestrator from a basic task scheduler into a comprehensive workflow management platform.

### Mental Model: The Growing Factory

Think of the workflow orchestrator as a factory that starts with a basic assembly line but gradually adds specialized departments. Initially, you have workers following fixed blueprints (DAGs) on a scheduled basis. As the factory grows, you add:

- **Sensor stations** that detect when raw materials arrive and automatically start production
- **Template workshops** that create standardized blueprints for common products
- **Performance optimization centers** that streamline operations and reduce waste

Each extension builds upon the existing infrastructure while adding new capabilities that make the entire operation more responsive, efficient, and scalable.

### Extension Categories and Evolution Path

The future extensions fall into three primary categories that address different aspects of workflow management maturity. **Event-driven capabilities** transform the orchestrator from purely time-based scheduling to reactive processing. **Workflow templating** addresses code reuse and standardization challenges. **Performance optimizations** tackle scale and efficiency concerns as deployment size grows.

| Extension Category | Primary Purpose | Complexity Level | Dependencies |
|-------------------|----------------|------------------|--------------|
| Sensor-Based Triggering | Event-driven workflow initiation | Medium | Core scheduler, file system monitoring |
| Workflow Templating | Code reuse and standardization | Medium | DAG parser, parameter system |
| Performance Optimizations | Scale and efficiency improvements | High | Database, caching layer, metrics |

The recommended implementation order follows dependency relationships and value delivery. Sensor-based triggering provides immediate operational value by eliminating polling and manual intervention. Workflow templating addresses developer productivity and maintenance overhead. Performance optimizations become critical as the system scales beyond proof-of-concept usage.

## Sensor-Based Triggering

> **Decision: Event-Driven Workflow Triggering**
> - **Context**: The core orchestrator relies on time-based cron scheduling, requiring workflows to poll for external changes or run on fixed intervals regardless of actual data availability
> - **Options Considered**: Continue with cron-only scheduling, add simple file watching, implement comprehensive sensor framework
> - **Decision**: Implement a comprehensive sensor framework supporting multiple trigger types
> - **Rationale**: Event-driven triggering eliminates unnecessary resource consumption, reduces latency between data availability and processing, and provides more responsive workflows
> - **Consequences**: Adds complexity to the scheduler component but enables real-time data processing and more efficient resource utilization

Sensor-based triggering transforms the orchestrator from a reactive system that runs on predetermined schedules to a responsive system that reacts immediately to external events. This capability is essential for real-time data processing, ETL pipelines that depend on file arrivals, and workflows triggered by external system state changes.

### Sensor Architecture and Types

The sensor framework extends the existing scheduler with event detection capabilities while maintaining compatibility with cron-based schedules. Each sensor type implements a common interface but monitors different event sources and maintains appropriate state for reliable event detection.

| Sensor Type | Event Source | Use Cases | State Requirements |
|-------------|--------------|-----------|-------------------|
| FileSensor | File system changes | Data ingestion from file drops | Last modification time, file size |
| DatabaseSensor | Database table changes | Trigger on new records or updates | Row counts, timestamp columns |
| HTTPSensor | External API endpoints | React to service availability or data changes | Response content, headers, status codes |
| S3Sensor | Cloud storage bucket changes | Process uploaded files or objects | Object metadata, ETags |
| TimeDeltaSensor | Time-based conditions with dependencies | Trigger after upstream DAG completion plus delay | Upstream completion timestamps |

### Sensor Implementation Pattern

Each sensor follows a standardized lifecycle that integrates with the existing scheduler polling loop. The sensor manager coordinates multiple sensors while ensuring each operates independently and can recover from failures gracefully.

The sensor execution flow follows these numbered steps:

1. **Initialization Phase**: Sensor loads its configuration and establishes baseline state by querying the monitored resource
2. **Polling Loop**: Sensor periodically checks the monitored resource for changes according to its configured polling interval
3. **State Comparison**: Current state is compared against stored baseline to detect meaningful changes based on sensor-specific criteria
4. **Event Generation**: When a trigger condition is met, sensor creates a trigger event with metadata about the detected change
5. **DAG Triggering**: Event is passed to the scheduler which creates a new DAGRun with sensor metadata available as parameters
6. **State Persistence**: Sensor updates its stored state to prevent duplicate triggers from the same event
7. **Error Handling**: Failed sensor checks are logged and retried with exponential backoff to handle transient failures

### File System Sensor Implementation

File sensors monitor directories for file arrivals, modifications, or deletions and trigger workflows when specified conditions are met. This sensor type is particularly valuable for batch processing workflows that depend on external file drops.

| Configuration Field | Type | Description | Example Value |
|---------------------|------|-------------|---------------|
| DirectoryPath | string | Monitored directory path | `/data/incoming/` |
| FilePattern | string | Glob pattern for matching files | `*.csv` |
| TriggerCondition | string | Event type that triggers workflow | `file_created`, `file_modified`, `file_deleted` |
| MinFileSize | int64 | Minimum file size to avoid triggering on partial uploads | `1024` |
| StabilizationTime | time.Duration | Wait time to ensure file is completely written | `30s` |
| RecursiveWatch | bool | Monitor subdirectories | `true` |

The file sensor maintains state about previously seen files to detect new arrivals and modifications reliably. File metadata including size, modification time, and checksum are tracked to distinguish between meaningful changes and spurious file system events.

```go
type FileSensorState struct {
    LastScanTime    time.Time                 `json:"last_scan_time"`
    KnownFiles      map[string]FileMetadata   `json:"known_files"`
    ProcessedEvents map[string]time.Time      `json:"processed_events"`
}

type FileMetadata struct {
    Size         int64     `json:"size"`
    ModTime      time.Time `json:"mod_time"`
    Checksum     string    `json:"checksum"`
    ProcessedAt  time.Time `json:"processed_at"`
}
```

### Database Sensor Implementation

Database sensors monitor table changes by polling for row count increases, timestamp column updates, or specific query result changes. These sensors enable workflows that process newly arrived data in staging tables or react to business logic state changes.

| Configuration Field | Type | Description | Example Value |
|---------------------|------|-------------|---------------|
| ConnectionString | string | Database connection parameters | `postgres://user:pass@host/db` |
| Query | string | SQL query to monitor for changes | `SELECT COUNT(*) FROM orders WHERE created_at > $1` |
| TimestampColumn | string | Column for tracking new records | `created_at` |
| PollInterval | time.Duration | Frequency of database polling | `5m` |
| BatchSize | int | Maximum records to process per trigger | `1000` |
| Timeout | time.Duration | Query execution timeout | `30s` |

The database sensor tracks the last processed timestamp or row count to detect new data arrivals. Query results are compared against stored baseline values, with changes triggering workflow execution with relevant row identifiers or counts passed as parameters.

### HTTP Endpoint Sensor Implementation

HTTP sensors monitor external APIs or webhook endpoints for availability, content changes, or specific response patterns. These sensors enable workflows that depend on external service state or data availability exposed through REST APIs.

| Configuration Field | Type | Description | Example Value |
|---------------------|------|-------------|---------------|
| URL | string | Monitored endpoint URL | `https://api.example.com/status` |
| Method | string | HTTP method for requests | `GET` |
| Headers | map[string]string | Request headers | `{"Authorization": "Bearer token"}` |
| ExpectedStatusCode | int | Status code indicating success | `200` |
| ResponseContentPattern | string | Regex pattern for response content matching | `"status": "ready"` |
| TimeoutDuration | time.Duration | Request timeout | `10s` |
| RetryAttempts | int | Failed request retry count | `3` |

HTTP sensors maintain response content hashes or specific field values to detect meaningful changes while ignoring irrelevant response variations like timestamps or request IDs.

### Sensor State Management and Persistence

Sensor state persistence ensures reliable event detection across system restarts and prevents duplicate workflow triggers from the same event. State is stored in the main database alongside DAG and task execution metadata.

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `sensor_instances` | Sensor configuration and status | `sensor_id`, `dag_id`, `config`, `state`, `last_check` |
| `sensor_events` | Historical trigger events | `event_id`, `sensor_id`, `event_type`, `metadata`, `triggered_at` |
| `sensor_state` | Current sensor state data | `sensor_id`, `state_data`, `updated_at` |

State serialization uses JSON format for flexibility while maintaining queryability for debugging and monitoring. Large state objects are compressed to minimize database storage overhead.

The sensor state update process follows these steps:

1. **Atomic State Read**: Current sensor state is read within a database transaction to ensure consistency
2. **Event Detection**: New state is compared against stored state using sensor-specific comparison logic
3. **Event Recording**: Detected events are recorded in the events table with full metadata for auditing
4. **DAG Triggering**: Scheduler creates DAGRun instances for each triggered workflow with event metadata
5. **State Update**: New state is written atomically, ensuring consistent state even if subsequent operations fail
6. **Cleanup**: Old events and state snapshots are cleaned up based on configurable retention policies

### Sensor Integration with Scheduler

Sensors integrate with the existing scheduler component through a plugin-like architecture that maintains separation of concerns while enabling unified workflow triggering. The scheduler polls sensors according to their configured intervals while continuing to process cron-based schedules.

The sensor polling integration modifies the scheduler's main loop to include sensor evaluation:

1. **Cron Schedule Evaluation**: Existing logic checks for DAGs that should trigger based on their cron expressions
2. **Sensor Poll Cycle**: Each registered sensor is checked if its poll interval has elapsed since the last check
3. **Parallel Sensor Execution**: Multiple sensors execute concurrently to avoid blocking the scheduler main loop
4. **Event Collection**: Sensor events are collected and batched for efficient processing
5. **DAGRun Creation**: Events trigger DAGRun creation using the same logic as cron-triggered executions
6. **Sensor State Persistence**: Updated sensor states are persisted atomically after successful DAGRun creation
7. **Error Handling**: Sensor failures are logged but don't affect cron scheduling or other sensor operations

### Common Pitfalls in Sensor Implementation

⚠️ **Pitfall: Duplicate Event Processing**
Sensors that don't maintain proper state can trigger the same workflow multiple times for a single event. This occurs when sensor state isn't updated atomically with DAGRun creation, causing restarts to re-detect the same event. The fix involves wrapping sensor state updates and DAGRun creation in a single database transaction, ensuring either both succeed or both fail.

⚠️ **Pitfall: Sensor State Corruption**
File system sensors that don't handle concurrent file operations can maintain inconsistent state about which files have been processed. This happens when files are modified while the sensor is scanning, leading to missed events or duplicate processing. The solution involves using file locking or checksums to verify file stability before processing.

⚠️ **Pitfall: Resource Exhaustion from Aggressive Polling**
Database sensors with very short poll intervals can overwhelm the monitored database with constant queries, especially when monitoring large tables. This degrades both sensor and database performance. The fix involves implementing exponential backoff when no changes are detected and using database-specific change detection mechanisms like triggers or change streams where available.

## Workflow Templating

> **Decision: Template-Based Workflow Generation**
> - **Context**: Teams frequently create similar workflows with slight variations, leading to code duplication and maintenance overhead as DAG definitions proliferate
> - **Options Considered**: Continue with manual DAG creation, simple parameter substitution, comprehensive template system with inheritance
> - **Decision**: Implement comprehensive template system with inheritance and composition capabilities
> - **Rationale**: Template systems reduce development time, ensure consistency across similar workflows, and enable centralized best practice enforcement
> - **Consequences**: Adds complexity to DAG parsing but dramatically improves developer productivity and workflow maintainability

Workflow templating addresses the code reuse and standardization challenges that emerge as organizations develop numerous similar workflows. Rather than duplicating DAG definitions with minor variations, templates enable parameterized workflow patterns that can be instantiated with specific configurations.

### Template Architecture and Hierarchy

The template system extends the existing DAG parser with a preprocessing layer that expands templates into concrete DAG definitions before normal validation and registration. Templates support inheritance, composition, and parameterization to handle complex workflow patterns efficiently.

| Template Type | Purpose | Complexity | Use Cases |
|--------------|---------|------------|-----------|
| Base Templates | Common workflow patterns | Low | ETL skeletons, ML pipeline frameworks |
| Parameterized Templates | Configurable workflows | Medium | Environment-specific deployments, multi-tenant processing |
| Composed Templates | Complex workflows from reusable components | High | Enterprise data pipelines, microservice orchestration |
| Template Libraries | Shareable template collections | High | Cross-team standardization, best practice enforcement |

### Template Definition Format

Templates use an extended YAML format that includes template-specific directives for parameterization, inheritance, and composition. The template engine processes these directives before passing the resulting DAG definition to the standard parser.

```yaml
# Template definition with inheritance and parameters
template:
  name: "etl_pipeline"
  version: "1.2.0"
  extends: "base_pipeline"
  parameters:
    source_table: 
      type: "string"
      required: true
      description: "Source table name for extraction"
    target_bucket:
      type: "string" 
      required: true
      description: "S3 bucket for processed data"
    batch_size:
      type: "integer"
      default: 1000
      description: "Records per processing batch"
      
dag:
  id: "{{ .template.name }}_{{ .source_table }}"
  description: "ETL pipeline for {{ .source_table }} → {{ .target_bucket }}"
  schedule: "{{ .schedule | default '0 2 * * *' }}"
  
  tasks:
    extract:
      type: "sql"
      operator:
        query: "SELECT * FROM {{ .source_table }} WHERE updated_at > '{{ .execution_date }}'"
        connection: "{{ .database_connection }}"
      parameters:
        batch_size: "{{ .batch_size }}"
        
    transform:
      type: "python"
      dependencies: ["extract"]
      operator:
        python_command: |
          import pandas as pd
          from processors import clean_data
          
          df = pd.read_json('{{ upstream.extract.output_path }}')
          cleaned_df = clean_data(df, batch_size={{ .batch_size }})
          cleaned_df.to_json('{{ .temp_dir }}/transformed.json')
```

### Template Parameter System

Template parameters provide type-safe configuration with validation, default values, and documentation. The parameter system prevents common template instantiation errors while enabling flexible workflow customization.

| Parameter Type | Validation Rules | Example Usage |
|---------------|------------------|---------------|
| string | Length limits, regex patterns | Database connection strings, file paths |
| integer | Range constraints, positive values | Batch sizes, timeout values, retry counts |
| boolean | True/false validation | Feature flags, debug modes |
| array | Element type validation, length limits | Tag lists, email recipients |
| object | Schema validation, required fields | Complex configuration objects |
| template_reference | Template existence validation | Composed workflow references |

Template parameter validation occurs during template expansion to provide early error detection with clear feedback about configuration issues.

The parameter resolution process follows these steps:

1. **Parameter Discovery**: Template engine scans template definition for parameter declarations and usage
2. **Type Validation**: Instance parameters are validated against declared types and constraints  
3. **Default Application**: Missing parameters are filled with declared default values where available
4. **Required Validation**: All required parameters must be provided or have defaults
5. **Cross-Reference Resolution**: Parameters referencing other parameters or template values are resolved
6. **Template Expansion**: Validated parameters are substituted into template expressions throughout the definition
7. **Output Generation**: Fully expanded DAG definition is generated for standard parser processing

### Template Inheritance and Composition

Template inheritance enables hierarchical workflow patterns where specialized templates extend base templates with additional tasks or modified configuration. Composition allows complex workflows to be assembled from reusable template components.

```yaml
# Base template for all ML pipelines
template:
  name: "ml_base"
  parameters:
    model_type:
      type: "string"
      required: true
    dataset_path:
      type: "string" 
      required: true
      
dag:
  tags: ["ml", "{{ .model_type }}"]
  
  tasks:
    prepare_data:
      type: "python"
      operator:
        python_command: "prepare_dataset('{{ .dataset_path }}')"
        
    validate_data:
      type: "python" 
      dependencies: ["prepare_data"]
      operator:
        python_command: "validate_dataset('{{ upstream.prepare_data.output_path }}')"

---
# Specialized template extending ML base
template:
  name: "ml_classification"
  extends: "ml_base"
  parameters:
    num_classes:
      type: "integer"
      required: true
      minimum: 2
    algorithms:
      type: "array"
      default: ["random_forest", "svm"]
      
dag:
  # Inherits base tasks and adds specialized ones
  tasks:
    train_models:
      type: "python"
      dependencies: ["validate_data"]
      operator:
        python_command: |
          train_classification_models(
            data_path='{{ upstream.validate_data.output_path }}',
            algorithms={{ .algorithms | json }},
            num_classes={{ .num_classes }}
          )
          
    evaluate_models:
      type: "python"
      dependencies: ["train_models"]  
      operator:
        python_command: "evaluate_classification('{{ upstream.train_models.model_path }}')"
```

### Template Instantiation and Management

Template instances are created through configuration files or API calls that specify the template name, version, and parameter values. The template engine validates parameters and generates concrete DAG definitions that integrate seamlessly with the existing DAG management system.

| Instantiation Method | Use Case | Configuration Format |
|---------------------|----------|---------------------|
| Static Files | Development and testing | YAML instance files alongside templates |
| API Endpoints | Dynamic generation | REST API with JSON parameter payload |
| CLI Commands | Deployment automation | Command-line tool with parameter files |
| UI Wizards | User-friendly creation | Web form with parameter validation |

Template instance management includes versioning, parameter tracking, and relationship mapping between templates and generated DAGs. This metadata enables impact analysis when templates change and supports automated instance updates.

The template instantiation workflow processes requests through these steps:

1. **Template Resolution**: Requested template name and version are resolved to specific template definition
2. **Parameter Validation**: Instance parameters are validated against template parameter schema
3. **Dependency Resolution**: Template inheritance chain and composition dependencies are resolved
4. **Template Expansion**: Parameters are substituted and inheritance/composition rules are applied
5. **DAG Generation**: Complete DAG definition is generated in standard format
6. **Validation**: Generated DAG undergoes normal cycle detection and validation processes
7. **Registration**: Valid DAG is registered in the system with template relationship metadata
8. **Instance Tracking**: Instance metadata is stored for management and impact analysis

### Template Library and Sharing

Template libraries enable organizations to share common workflow patterns across teams while maintaining version control and access policies. Libraries support both internal organizational templates and external community-contributed patterns.

| Library Feature | Purpose | Implementation |
|----------------|---------|----------------|
| Version Control | Template evolution tracking | Git-based storage with semantic versioning |
| Access Control | Template usage permissions | Role-based access with organization boundaries |
| Dependency Management | Template composition support | Dependency resolution with conflict detection |
| Documentation | Usage guidance and examples | Integrated docs with parameter descriptions |
| Testing Framework | Template validation and examples | Automated testing of template instantiation |

Template discovery and documentation systems help developers find appropriate templates for their use cases while understanding parameter requirements and customization options.

### Common Pitfalls in Template Implementation

⚠️ **Pitfall: Parameter Injection Security**
Template systems that don't properly escape parameter values can enable injection attacks when parameters contain SQL queries, shell commands, or script code. This occurs when user-provided parameters are directly substituted into executable contexts. The fix involves implementing context-aware escaping and parameter sanitization based on the target operator type.

⚠️ **Pitfall: Circular Template Dependencies**
Template inheritance or composition can create circular dependencies where template A extends template B which extends template A. This creates infinite expansion loops that crash the template engine. The solution involves dependency graph analysis during template loading with cycle detection similar to DAG validation.

⚠️ **Pitfall: Template Version Conflicts**
Organizations with multiple template versions can encounter conflicts when templates depend on different versions of base templates or when parameter schemas change incompatibly. The fix requires implementing semantic versioning with compatibility checking and automated migration tools for parameter schema changes.

## Performance Optimizations

> **Decision: Multi-Layer Optimization Strategy**
> - **Context**: As workflow complexity and execution volume increase, the orchestrator faces performance bottlenecks in database queries, DAG parsing, task scheduling, and web UI responsiveness
> - **Options Considered**: Reactive optimization addressing bottlenecks as they occur, comprehensive performance redesign, targeted optimization of known bottlenecks
> - **Decision**: Implement targeted optimization with caching, database improvements, and architectural enhancements
> - **Rationale**: Targeted optimization provides measurable improvements without architectural disruption while establishing patterns for future scaling
> - **Consequences**: Increases system complexity but enables handling larger workflow volumes and provides better user experience

Performance optimization becomes critical as workflow orchestrator usage scales beyond initial proof-of-concept deployments. Real-world usage patterns reveal bottlenecks in database queries, memory usage, task scheduling efficiency, and user interface responsiveness that require systematic optimization approaches.

### Optimization Categories and Impact Areas

Performance optimizations span multiple system layers, each addressing different bottleneck types and scaling constraints. The optimization strategy prioritizes high-impact, low-risk improvements while establishing infrastructure for more advanced optimizations.

| Optimization Layer | Primary Bottlenecks | Impact Areas | Implementation Complexity |
|-------------------|-------------------|--------------|-------------------------|
| Database Layer | Query performance, connection pooling | Task scheduling, state updates, history queries | Medium |
| Caching Layer | Repeated data access, computation results | DAG parsing, worker discovery, UI responsiveness | Medium |
| Scheduling Algorithm | Task queue efficiency, dependency resolution | Workflow execution latency, system throughput | High |
| Memory Management | Object caching, garbage collection | System stability, response times | Medium |
| Network Optimization | Worker communication, result collection | Distributed execution overhead | High |

### Database Performance Optimization

Database optimization focuses on query performance, indexing strategy, and connection management to handle increased workflow volume and complexity. The database layer becomes a critical bottleneck as DAG complexity and execution frequency increase.

#### Query Optimization and Indexing Strategy

Database performance improvements begin with analyzing common query patterns and implementing targeted indexes to accelerate frequent operations. The orchestrator's workload includes time-series queries for scheduling, graph traversal for dependency resolution, and aggregation queries for monitoring dashboards.

| Query Type | Frequency | Optimization Strategy | Index Requirements |
|------------|-----------|----------------------|-------------------|
| Ready Task Lookup | Every scheduling cycle | Composite index on state and dependencies | `(state, dag_run_id, dependencies_met)` |
| DAG Run History | Dashboard and API requests | Time-range partitioning and covering indexes | `(dag_id, execution_date DESC)` |
| Worker Health Queries | Heartbeat processing | Clustered index on heartbeat timestamps | `(worker_id, last_heartbeat DESC)` |
| Dependency Resolution | Task state transitions | Graph-optimized storage with adjacency indexes | `(upstream_task_id, downstream_task_id)` |
| Log Aggregation | Debugging and monitoring | Partitioned storage by date and task | `(task_instance_id, log_level, timestamp)` |

Advanced query optimization includes implementing materialized views for complex aggregations, query plan analysis for identifying bottlenecks, and connection pooling optimization for concurrent access patterns.

#### Connection Pool and Transaction Management

Database connection management becomes critical under high concurrency as multiple scheduler instances, workers, and web clients access shared state. Proper connection pooling prevents connection exhaustion while maintaining transaction isolation for state updates.

| Pool Configuration | Purpose | Recommended Settings | Monitoring Metrics |
|-------------------|---------|---------------------|-------------------|
| Maximum Connections | Prevent database overload | 2x CPU cores + disk count | Active connections, wait time |
| Idle Timeout | Resource cleanup | 10-15 minutes | Idle connection count, cleanup frequency |
| Connection Lifetime | Prevent stale connections | 1-2 hours | Connection age distribution |
| Prepared Statement Cache | Query performance | 100-500 statements | Cache hit ratio, memory usage |
| Transaction Timeout | Prevent deadlocks | 30-60 seconds | Long transaction count, deadlock frequency |

Connection pool optimization follows these implementation steps:

1. **Workload Analysis**: Monitor connection usage patterns, peak concurrency, and transaction duration
2. **Pool Sizing**: Configure maximum connections based on database capacity and application concurrency
3. **Timeout Configuration**: Set appropriate timeouts for idle connections and long transactions  
4. **Health Monitoring**: Implement connection health checks and automatic recovery for failed connections
5. **Load Balancing**: Distribute read queries across replicas while routing writes to primary database
6. **Circuit Breaker**: Implement circuit breaker pattern for database failures to prevent cascade failures
7. **Metrics Collection**: Track connection pool metrics for ongoing optimization and capacity planning

### Multi-Layer Caching Strategy

Caching optimization reduces database load and improves response times by storing frequently accessed data in memory. The caching strategy addresses different data access patterns with appropriate cache technologies and eviction policies.

#### DAG Definition Caching

DAG definitions represent parsed and validated workflow configurations that change infrequently but are accessed constantly during scheduling and execution. Caching parsed DAGs eliminates repeated file parsing and validation overhead.

| Cache Level | Storage Location | Cache Key | TTL | Invalidation Strategy |
|-------------|------------------|-----------|-----|----------------------|
| In-Memory | Scheduler process | `dag_id:version_hash` | 1 hour | File modification timestamp |
| Redis | Shared cache | `dag:definition:dag_id` | 4 hours | Explicit invalidation on updates |
| Application | Worker processes | `worker:dag:dag_id` | 30 minutes | Heartbeat-based refresh |

DAG caching implementation includes version tracking to ensure cache consistency when DAG definitions change and memory pressure management to prevent cache growth from affecting system stability.

#### Task State Caching

Task state information is frequently queried for dependency resolution, dashboard updates, and API responses. Caching task states reduces database load while maintaining consistency through appropriate invalidation policies.

```yaml
# Task state cache configuration
task_state_cache:
  backend: "redis"
  key_pattern: "task_state:{{ .dag_run_id }}:{{ .task_id }}"
  ttl: "15m"
  write_through: true
  invalidation:
    - event: "task_state_change" 
      keys: ["task_state:{{ .dag_run_id }}:{{ .task_id }}"]
    - event: "dag_run_complete"
      pattern: "task_state:{{ .dag_run_id }}:*"
  compression: "gzip"
  max_size: "1MB"
```

#### Worker Registry Caching

Worker registration and health status information requires fast access for task scheduling decisions but changes frequently as workers join, leave, or update their status. Multi-level caching balances freshness requirements with performance needs.

The worker registry caching strategy layers caches by update frequency and access pattern:

1. **Hot Cache**: In-memory cache for active worker list with 30-second TTL
2. **Warm Cache**: Redis cache for detailed worker status with 2-minute TTL  
3. **Cold Storage**: Database for historical worker data and audit trails
4. **Capacity Cache**: Aggregated capacity metrics with 5-minute refresh intervals
5. **Health Cache**: Worker health status with 1-minute TTL and background refresh

### Scheduling Algorithm Optimization

Scheduling algorithm optimization focuses on reducing the computational overhead of dependency resolution and task queue management as DAG complexity and volume increase. Advanced algorithms replace naive approaches with more efficient graph processing and queue management.

#### Dependency Resolution Optimization

The standard approach of checking each task's dependencies individually becomes inefficient for large DAGs with complex dependency graphs. Optimized dependency resolution uses graph algorithms and caching to minimize redundant computation.

| Algorithm Approach | Computational Complexity | Memory Usage | Best Use Case |
|-------------------|-------------------------|--------------|---------------|
| Naive Per-Task Check | O(T×D) per scheduling cycle | Low | Small DAGs (<50 tasks) |
| Topological Sort | O(T+E) per DAG run | Medium | Medium DAGs (50-200 tasks) |
| Incremental Updates | O(changed tasks) per update | High | Large DAGs (200+ tasks) |
| Cached Subgraphs | O(subgraph) per change | High | Complex DAGs with patterns |

Advanced dependency resolution maintains a dependency graph in memory with efficient update mechanisms:

1. **Graph Preprocessing**: Build adjacency lists and reverse dependency maps during DAG registration
2. **Incremental Updates**: When task states change, update only affected downstream dependencies
3. **Batch Processing**: Group state changes within scheduling cycles to minimize graph updates
4. **Subgraph Caching**: Cache dependency resolution results for repeated subgraph patterns
5. **Parallel Resolution**: Resolve independent subgraphs in parallel for large DAGs
6. **Lazy Evaluation**: Defer dependency checks until tasks are ready for scheduling

#### Priority Queue Implementation

Task scheduling benefits from priority-based queue management that considers task priority, resource requirements, and execution history. Advanced queue implementations optimize task selection for system throughput and resource utilization.

```go
type PriorityTaskQueue struct {
    queues          map[string]*heap.TaskHeap    // Priority-based heaps
    resourceTracker *ResourceTracker             // Available resources
    metrics         *QueueMetrics                // Performance tracking
    batchSize       int                          // Batch processing size
    affinityMap     map[string][]string          // Worker affinity rules
}

type TaskPriority struct {
    Level           int           // Explicit priority level
    SubmissionTime  time.Time     // Age-based priority boost
    ResourceWeight  float64       // Resource requirement factor
    DAGPriority     int           // DAG-level priority
    RetryPenalty    float64       // Retry count penalty
}
```

The optimized scheduling algorithm processes tasks in batches with multiple optimization strategies:

1. **Multi-Level Queues**: Separate queues for different priority levels with round-robin selection
2. **Resource-Aware Scheduling**: Consider available worker resources when selecting tasks
3. **Batch Processing**: Process multiple ready tasks simultaneously to reduce overhead
4. **Affinity Optimization**: Schedule related tasks on the same workers to leverage data locality
5. **Backpressure Management**: Throttle task submission when worker queues are full
6. **Load Balancing**: Distribute tasks across workers based on current load and capacity

### Memory Management and Resource Optimization

Memory optimization addresses object lifecycle management, garbage collection tuning, and resource pool management to maintain stable performance under high load. Proper memory management prevents performance degradation and system crashes as usage scales.

#### Object Pool Management

Object pooling reduces garbage collection pressure by reusing frequently allocated objects like task instances, execution contexts, and message payloads. Pool management balances memory usage with allocation overhead.

| Pool Type | Object Lifecycle | Pool Size Strategy | Monitoring Metrics |
|-----------|------------------|-------------------|-------------------|
| Task Execution Context | Per task execution | Dynamic based on worker count | Pool hits/misses, allocation rate |
| Message Payloads | Message queue operations | Fixed size with overflow handling | Pool utilization, overflow frequency |
| Database Connections | Query and transaction lifecycle | Configured maximums with health checks | Active connections, wait queue depth |
| HTTP Response Writers | API request handling | Dynamic with request volume | Response time, memory per request |

Object pool implementation includes lifecycle management, thread safety, and monitoring:

1. **Pool Initialization**: Pre-allocate pool with minimum number of objects based on expected load
2. **Object Acquisition**: Provide thread-safe object checkout with timeout handling
3. **Object Reset**: Clear object state before returning to pool to prevent data leakage
4. **Pool Expansion**: Dynamically grow pools under high load with configured maximums
5. **Health Monitoring**: Track pool utilization and object allocation patterns
6. **Leak Detection**: Monitor objects that aren't returned to pools and force cleanup
7. **Pool Cleanup**: Periodically shrink pools during low usage to free memory

#### Garbage Collection Optimization

Garbage collection tuning optimizes memory allocation patterns and collection frequency to minimize pause times and throughput impact. Configuration focuses on the specific allocation patterns of workflow orchestration.

```yaml
# GC optimization configuration
garbage_collection:
  # Heap sizing for steady-state operation
  heap_target: "2GB"
  max_heap: "4GB"
  
  # GC trigger thresholds
  gc_percent: 50        # Trigger GC at 50% heap growth
  
  # Allocation optimization
  object_pools:
    task_contexts: 1000
    message_buffers: 500
    response_writers: 200
    
  # Memory profiling
  memory_profiling:
    enabled: true
    heap_dump_threshold: "3GB"
    profile_frequency: "10m"
```

### Common Pitfalls in Performance Optimization

⚠️ **Pitfall: Premature Database Connection Pool Sizing**
Setting database connection pool sizes without understanding actual concurrency patterns can either waste resources with oversized pools or create bottlenecks with undersized pools. This occurs when developers guess at appropriate pool sizes instead of measuring real usage patterns. The fix involves monitoring actual concurrent connection usage during representative load and sizing pools based on measured peak concurrency plus headroom.

⚠️ **Pitfall: Cache Invalidation Race Conditions**
Multi-layer caching without proper invalidation ordering can create inconsistent cache states where different cache levels contain different versions of the same data. This happens when cache updates aren't coordinated across cache layers. The solution requires implementing ordered cache invalidation with write-through or write-behind patterns that ensure cache consistency.

⚠️ **Pitfall: Memory Leak in Object Pools**
Object pools that don't properly reset object state can accumulate memory leaks as objects carry references to previous usage contexts. This occurs when pooled objects aren't fully cleaned between uses. The fix involves implementing thorough object reset methods and periodic pool validation to detect objects that aren't properly cleaned.

### Implementation Guidance

The future extensions provide a roadmap for evolving the workflow orchestrator beyond basic DAG-based scheduling into a comprehensive enterprise workflow management platform. Each extension builds upon the core architecture while adding specialized capabilities for specific operational requirements.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| File System Monitoring | `fsnotify` library with polling fallback | `inotify`/`kqueue` with recursive watching |
| Database Sensors | Standard `database/sql` with polling | Change data capture with triggers |
| HTTP Sensors | `net/http` client with retries | Webhook receiver with event queue |
| Template Engine | `text/template` with custom functions | `Jinja2`-compatible processor |
| Caching Layer | In-memory `sync.Map` with TTL | Redis cluster with consistent hashing |
| Object Pooling | `sync.Pool` for simple objects | Custom pools with lifecycle management |

#### Recommended File Structure

```
project-root/
  internal/
    sensors/
      sensor.go                 ← Sensor interface and manager
      file_sensor.go           ← File system event monitoring
      database_sensor.go       ← Database change detection  
      http_sensor.go          ← HTTP endpoint monitoring
      sensor_manager.go       ← Sensor lifecycle coordination
    templates/
      template.go             ← Template engine and processor
      inheritance.go          ← Template inheritance logic
      parameters.go           ← Parameter validation and types
      library.go              ← Template library management
    optimization/
      cache/
        manager.go            ← Multi-layer cache coordination
        dag_cache.go         ← DAG definition caching
        task_cache.go        ← Task state caching
      pools/
        object_pools.go      ← Reusable object management
        connection_pools.go  ← Database connection pooling
      scheduler/
        priority_queue.go    ← Advanced task scheduling
        dependency_graph.go  ← Optimized dependency resolution
```

#### Infrastructure Starter Code

**Sensor Interface and Base Implementation:**

```go
package sensors

import (
    "context"
    "time"
    "encoding/json"
)

// Sensor interface for all event detection implementations
type Sensor interface {
    // Initialize sets up sensor with configuration and baseline state
    Initialize(ctx context.Context, config SensorConfig) error
    
    // Poll checks for events and returns triggers if conditions are met
    Poll(ctx context.Context) ([]TriggerEvent, error)
    
    // GetState returns current sensor state for persistence
    GetState() ([]byte, error)
    
    // SetState restores sensor from persisted state
    SetState(state []byte) error
    
    // Close cleans up sensor resources
    Close() error
}

type SensorConfig struct {
    ID          string                 `json:"id"`
    Type        string                 `json:"type"`
    DAGID       string                 `json:"dag_id"`
    Parameters  map[string]interface{} `json:"parameters"`
    PollInterval time.Duration         `json:"poll_interval"`
}

type TriggerEvent struct {
    SensorID    string                 `json:"sensor_id"`
    EventType   string                 `json:"event_type"`
    Timestamp   time.Time             `json:"timestamp"`
    Metadata    map[string]interface{} `json:"metadata"`
}

type SensorManager struct {
    sensors     map[string]Sensor
    db          *sql.DB
    logger      Logger
    stopChan    chan struct{}
}

// NewSensorManager creates sensor coordination system
func NewSensorManager(db *sql.DB, logger Logger) *SensorManager {
    return &SensorManager{
        sensors:  make(map[string]Sensor),
        db:       db,
        logger:   logger,
        stopChan: make(chan struct{}),
    }
}

// RegisterSensor adds new sensor to management system
func (sm *SensorManager) RegisterSensor(config SensorConfig) error {
    // TODO 1: Create sensor instance based on config.Type
    // TODO 2: Initialize sensor with config and load persisted state
    // TODO 3: Add sensor to internal registry for polling
    // TODO 4: Store sensor configuration in database
    return nil
}

// StartPolling begins sensor monitoring loops
func (sm *SensorManager) StartPolling(ctx context.Context) error {
    // TODO 1: Start goroutine for each registered sensor
    // TODO 2: Poll sensors according to their configured intervals
    // TODO 3: Process trigger events and create DAG runs
    // TODO 4: Persist sensor state updates atomically
    // TODO 5: Handle sensor errors with exponential backoff
    return nil
}
```

**Template Engine Core Implementation:**

```go
package templates

import (
    "text/template"
    "gopkg.in/yaml.v2"
)

type TemplateEngine struct {
    funcs       template.FuncMap
    cache       map[string]*Template
    loader      TemplateLoader
    validator   *TemplateValidator
}

type Template struct {
    Name        string                 `yaml:"name"`
    Version     string                 `yaml:"version"`
    Extends     string                 `yaml:"extends,omitempty"`
    Parameters  map[string]Parameter   `yaml:"parameters"`
    DAGTemplate DAGTemplate           `yaml:"dag"`
}

type Parameter struct {
    Type        string      `yaml:"type"`
    Required    bool        `yaml:"required"`
    Default     interface{} `yaml:"default,omitempty"`
    Description string      `yaml:"description"`
    Validation  Validation  `yaml:"validation,omitempty"`
}

type Validation struct {
    MinLength   *int     `yaml:"min_length,omitempty"`
    MaxLength   *int     `yaml:"max_length,omitempty"`
    Pattern     string   `yaml:"pattern,omitempty"`
    Minimum     *float64 `yaml:"minimum,omitempty"`
    Maximum     *float64 `yaml:"maximum,omitempty"`
}

// NewTemplateEngine creates template processing system
func NewTemplateEngine() *TemplateEngine {
    // TODO 1: Initialize template function map with utility functions
    // TODO 2: Create template cache with TTL and eviction policies
    // TODO 3: Set up template loader for file and registry sources
    // TODO 4: Initialize parameter validator with type checking
    return nil
}

// LoadTemplate reads and parses template definition
func (te *TemplateEngine) LoadTemplate(name, version string) (*Template, error) {
    // TODO 1: Check cache for existing parsed template
    // TODO 2: Load template definition from storage
    // TODO 3: Parse YAML and validate template structure
    // TODO 4: Resolve inheritance chain if template extends another
    // TODO 5: Cache parsed template for future use
    return nil, nil
}

// InstantiateTemplate generates concrete DAG from template and parameters
func (te *TemplateEngine) InstantiateTemplate(templateName, version string, params map[string]interface{}) (*DAGDefinition, error) {
    // TODO 1: Load and validate template definition
    // TODO 2: Validate provided parameters against template schema
    // TODO 3: Apply parameter defaults for missing values
    // TODO 4: Resolve template inheritance and composition
    // TODO 5: Execute template expansion with parameter substitution
    // TODO 6: Generate final DAG definition for standard parser
    return nil, nil
}
```

**Multi-Layer Cache Implementation:**

```go
package cache

import (
    "time"
    "sync"
)

type CacheManager struct {
    layers      []CacheLayer
    metrics     *CacheMetrics
    config      CacheConfig
}

type CacheLayer interface {
    Get(key string) (interface{}, bool)
    Set(key string, value interface{}, ttl time.Duration) error
    Delete(key string) error
    Clear() error
    Size() int
}

type CacheConfig struct {
    Layers []LayerConfig `json:"layers"`
}

type LayerConfig struct {
    Name     string        `json:"name"`
    Type     string        `json:"type"`        // "memory", "redis"
    Size     int           `json:"size"`
    TTL      time.Duration `json:"ttl"`
    Servers  []string      `json:"servers,omitempty"`
}

// NewCacheManager creates multi-layer caching system
func NewCacheManager(config CacheConfig) *CacheManager {
    // TODO 1: Initialize cache layers based on configuration
    // TODO 2: Set up cache metrics collection and monitoring
    // TODO 3: Configure cache invalidation policies and TTL handling
    // TODO 4: Implement cache warming strategies for critical data
    return nil
}

// Get retrieves value from cache layers in order
func (cm *CacheManager) Get(key string) (interface{}, error) {
    // TODO 1: Check each cache layer starting with fastest
    // TODO 2: On cache hit, promote value to faster layers
    // TODO 3: Record cache hit/miss metrics for monitoring
    // TODO 4: Return value and indicate which layer provided it
    return nil, nil
}

// Set stores value in all appropriate cache layers
func (cm *CacheManager) Set(key string, value interface{}, ttl time.Duration) error {
    // TODO 1: Determine which cache layers should store this value
    // TODO 2: Store value in each selected layer with appropriate TTL
    // TODO 3: Handle storage failures gracefully without losing data
    // TODO 4: Update cache metrics for capacity and performance tracking
    return nil
}
```

#### Core Logic Skeleton Code

**File Sensor Implementation:**

```go
// Poll checks directory for file changes and generates trigger events
func (fs *FileSensor) Poll(ctx context.Context) ([]TriggerEvent, error) {
    // TODO 1: Scan configured directory using filepath.Walk
    // TODO 2: Filter files based on configured pattern (glob matching)
    // TODO 3: Compare current file metadata against stored state
    // TODO 4: Identify new files, modified files, and deleted files
    // TODO 5: Check file stability using size and modification time
    // TODO 6: Generate trigger events for files meeting trigger conditions
    // TODO 7: Update internal state with new file metadata
    // Hint: Use os.Stat() for file metadata and filepath.Match() for patterns
    return nil, nil
}
```

**Template Parameter Validation:**

```go
// ValidateParameters checks instance parameters against template schema
func (tv *TemplateValidator) ValidateParameters(template *Template, params map[string]interface{}) error {
    // TODO 1: Iterate through template parameter definitions
    // TODO 2: Check that all required parameters are provided
    // TODO 3: Validate parameter types match declared types
    // TODO 4: Apply validation rules (min/max, patterns, ranges)
    // TODO 5: Verify no unexpected parameters are provided
    // TODO 6: Return detailed validation errors with parameter names
    // Hint: Use type assertions and regexp.MatchString() for validation
    return nil
}
```

**Priority Task Queue:**

```go
// SelectNextTask chooses highest priority ready task for execution
func (pq *PriorityTaskQueue) SelectNextTask(workerCapacity ResourceRequirements) (*TaskInstance, error) {
    // TODO 1: Check each priority queue from highest to lowest
    // TODO 2: Filter tasks by worker resource capacity
    // TODO 3: Consider worker affinity preferences for task placement
    // TODO 4: Apply load balancing across available workers
    // TODO 5: Remove selected task from queue and update metrics
    // TODO 6: Return task instance with scheduling metadata
    // Hint: Use heap.Pop() for priority queue operations
    return nil, nil
}
```

#### Milestone Checkpoints

**Sensor Implementation Verification:**
```bash
# Test file sensor functionality
echo "test data" > /tmp/dag_trigger/test_file.csv
# Check logs for sensor detection: "File sensor detected new file"
# Verify DAG run creation: "Created DAG run for sensor trigger"

# Test database sensor
psql -c "INSERT INTO sensor_table (data) VALUES ('test');"
# Check logs for database sensor: "Database sensor detected 1 new records"
```

**Template System Validation:**
```bash
# Test template instantiation
./orchestrator template create --template=etl_pipeline --params=params.yaml
# Expected: Generated DAG definition with substituted parameters
# Verify: ./orchestrator dag validate --file=generated_dag.yaml returns success

# Test template inheritance
./orchestrator template validate --template=ml_classification
# Expected: "Template validation successful, inherits from ml_base"
```

**Performance Optimization Verification:**
```bash
# Monitor cache hit rates
./orchestrator metrics cache --interval=1m
# Expected: Cache hit rates above 80% for DAG definitions
# Expected: Response times under 100ms for cached queries

# Test connection pool scaling
ab -n 1000 -c 50 http://localhost:8080/api/v1/dags
# Expected: No connection timeouts, steady response times
# Check: Database connection pool metrics show no wait queue buildup
```

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|----------------|-----|
| Sensor not triggering | File permissions or pattern mismatch | Check sensor logs for access errors | Fix directory permissions, validate glob patterns |
| Template expansion fails | Parameter type mismatch or missing required values | Enable template debug logging | Validate parameter JSON schema, check required fields |
| Cache inconsistency | Race condition in multi-layer invalidation | Monitor cache invalidation order | Implement ordered cache updates with write-through |
| Memory growth | Object pool leaks or missing cache eviction | Profile heap usage and object counts | Add object reset methods, implement TTL-based eviction |
| Slow dependency resolution | Large DAG with inefficient algorithms | Profile scheduling cycle performance | Implement incremental dependency updates |

The future extensions transform the basic workflow orchestrator into an enterprise-grade platform capable of handling complex operational requirements. Each extension addresses specific scalability and usability challenges while maintaining the architectural foundation established in the core system. Implementation should follow the milestone approach, validating each extension thoroughly before proceeding to the next enhancement.


## Glossary

> **Milestone(s):** All milestones (comprehensive terminology reference for DAG Definition, Scheduler, Task Execution, Worker & Executor, Web UI & Monitoring)

The glossary serves as a comprehensive reference for all technical terms, acronyms, and domain-specific vocabulary used throughout the workflow orchestrator design. Understanding this terminology is crucial for effectively implementing and maintaining the system, as it establishes a shared vocabulary between developers, operations teams, and stakeholders.

### Mental Model: The Technical Dictionary

Think of this glossary as a technical dictionary for the workflow orchestration domain. Just as a foreign language dictionary helps you understand unfamiliar words and their precise meanings, this glossary clarifies the specific meaning of terms within the context of workflow orchestration. Each term has a precise definition that may differ from its general usage in other domains.

The terminology is organized into several categories: core workflow concepts, system architecture components, execution and scheduling terms, monitoring and management concepts, and implementation-specific vocabulary. This organization helps you understand not just what each term means, but how it relates to other concepts in the system.

### Core Workflow Orchestration Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **workflow orchestration** | The coordination of complex task dependencies and execution across distributed systems, ensuring tasks run in the correct order based on their interdependencies | Core concept describing the primary purpose of the entire system |
| **directed acyclic graph (DAG)** | A workflow representation that ensures no circular dependencies exist between tasks, allowing for deterministic execution ordering | Fundamental data structure used to model all workflows in the system |
| **task dependencies** | Upstream requirements that must complete successfully before a task can begin execution, defining the execution order within a workflow | Critical concept for understanding task scheduling and execution flow |
| **idempotency** | The ability to safely restart operations without causing side effects or duplicate work, ensuring consistent results regardless of execution count | Essential property for reliable workflow execution and recovery |
| **backfill** | The process of executing workflows for historical time periods to process missed data or catch up on delayed processing | Important operation for data pipeline management and recovery scenarios |

### Scheduling and Execution Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **scheduler** | The component responsible for triggering workflow runs based on cron schedules or manual requests, managing the timing of workflow execution | Core system component that drives all workflow execution |
| **executor** | The component that distributes and manages task execution across worker pools, coordinating the actual running of workflow tasks | Central component for distributed task execution and resource management |
| **worker** | A process that executes individual tasks and reports results back to the executor, providing the computational resources for task execution | Basic unit of task execution in the distributed system |
| **cron expression** | A time-based schedule format with five fields (minute, hour, day of month, month, day of week) used to define when workflows should run | Standard scheduling format for automated workflow triggering |
| **execution date** | The logical timestamp representing the data processing period for a workflow run, which may differ from the actual execution time | Important concept for data pipeline consistency and backfill operations |
| **scheduler drift** | Gradual timing error accumulation causing workflows to trigger late or early relative to their intended schedule | Performance issue that can affect data pipeline reliability |
| **timezone handling** | Correct interpretation of local time schedules across daylight saving transitions and geographic regions | Critical consideration for global deployments and schedule accuracy |

### Task and Execution Context Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **operator** | A task implementation for specific execution types like Python scripts, Bash commands, or SQL queries | Pluggable component that defines how different types of work get executed |
| **xcom** | Cross-task communication mechanism for passing small data payloads between task instances within a workflow run | Essential feature for sharing results and coordinating between dependent tasks |
| **execution context** | Environmental information and services provided to tasks during execution, including parameters, logging, and resource access | Runtime environment that tasks operate within during execution |
| **task isolation** | Separation of tasks during execution to prevent interference and ensure security between different task executions | Security and reliability mechanism for multi-tenant and robust execution |
| **retry mechanism** | System for handling transient failures through repeated execution with configurable delays and backoff strategies | Fault tolerance feature for handling temporary failures and improving reliability |
| **timeout handling** | Enforcement of maximum execution time limits for tasks to prevent runaway processes and resource exhaustion | Resource management feature preventing tasks from consuming excessive resources |

### Distributed System Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **message queue** | Communication system for distributing tasks and collecting results between scheduler and workers | Infrastructure component enabling distributed task execution |
| **heartbeat mechanism** | Periodic worker health reporting system used for failure detection and worker management | Monitoring system for detecting and responding to worker failures |
| **worker registry** | System for tracking available workers and their capacity, managing the pool of execution resources | Resource management component for efficient task distribution |
| **task distribution** | The process of sending tasks from scheduler to available workers using load balancing and capacity management | Core function of the executor for coordinating distributed work |
| **dependency resolution** | The process of determining when tasks are ready for execution based on upstream task completion status | Critical algorithm for maintaining correct execution order in workflows |
| **load balancing** | Distributing tasks evenly across available workers to optimize resource utilization and execution performance | Performance optimization technique for efficient resource usage |
| **fault tolerance** | The system's ability to handle worker failures and network issues gracefully without losing work or corrupting state | Reliability characteristic essential for production workflow orchestration |

### State Management Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **DAGRunState** | Enumeration of possible states for workflow runs: queued, running, success, failed, up_for_retry | State machine governing workflow execution lifecycle |
| **TaskInstanceState** | Enumeration of possible states for individual tasks: none, queued, running, success, failed, up_for_retry, upstream_failed, skipped | State machine governing task execution lifecycle |
| **trigger rule evaluation** | The process of checking if upstream dependencies satisfy execution prerequisites for a task | Algorithm determining when downstream tasks can begin execution |
| **upstream failure propagation** | The process of marking downstream tasks as blocked when their dependencies fail | Error handling mechanism preventing execution of tasks with failed prerequisites |
| **state reconciliation** | The process of detecting and repairing data inconsistencies after failures or network partitions | Recovery mechanism ensuring system state remains coherent |
| **eventual consistency** | Design approach allowing delayed consistency to achieve better availability for non-critical operations | Distributed system pattern balancing consistency and availability |

### Monitoring and Management Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **server-sent events (SSE)** | One-way real-time communication protocol from server to browser used for live workflow monitoring | Web technology enabling real-time dashboard updates |
| **event hub** | Central distribution point for real-time state change events across the system | Communication infrastructure for coordinating real-time updates |
| **circuit breaker patterns** | Design pattern for preventing cascading failures through isolation of external dependency failures | Reliability pattern for handling external service failures gracefully |
| **health monitoring** | Continuous assessment of component availability and performance across the distributed system | Operational practice for maintaining system reliability |
| **diagnostic queries** | Pre-built database queries designed for troubleshooting common workflow orchestration issues | Operational tool for system debugging and performance analysis |

### Architecture and Design Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **cycle detection** | Algorithm to validate that a DAG has no circular dependencies using graph traversal techniques | Fundamental validation ensuring workflow definitions are executable |
| **topological sort** | Algorithm producing linear ordering of vertices in a directed graph, used for task execution ordering | Core algorithm for determining valid task execution sequences |
| **template engine** | Component that processes parameterized expressions in workflow definitions, enabling dynamic configuration | Feature enabling reusable and configurable workflow patterns |
| **file discovery** | Mechanism for scanning directories and finding DAG definition files for automatic workflow registration | System feature for automatic workflow deployment and management |
| **adjacency list** | Graph representation mapping each vertex to its connected neighbors, used for efficient DAG storage | Data structure choice for representing workflow dependencies |
| **in-degree** | Number of incoming edges to a vertex in a directed graph, used in topological sorting algorithms | Graph theory concept used in dependency resolution algorithms |

### Implementation and Infrastructure Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **polling interval** | Frequency at which the scheduler checks for pending work and schedule triggers | Configuration parameter affecting system responsiveness and resource usage |
| **event-driven scheduling** | Immediate response to notifications rather than periodic polling for improved responsiveness | Alternative scheduling approach for better performance |
| **serialization** | Conversion of data structures to byte format for storage or transmission between system components | Technical requirement for persisting and communicating workflow state |
| **resource requirements** | Specification of CPU, memory, and other resources needed for task execution | Resource management concept for efficient task scheduling |
| **graceful shutdown** | Proper cleanup of connections and resources during server stop to prevent data loss | Operational procedure for safe system maintenance |
| **middleware stack** | Composable request processing chain for HTTP servers enabling modular functionality | Web architecture pattern for organizing request processing logic |

### Performance and Optimization Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **multi-layer caching** | Caching strategy with multiple storage levels for different access patterns and performance characteristics | Performance optimization technique for reducing database load |
| **dependency resolution optimization** | Efficient algorithms for determining task readiness that minimize computational overhead | Performance optimization for large-scale workflow execution |
| **object pooling** | Reusing frequently allocated objects to reduce garbage collection pressure and improve performance | Memory management optimization for high-throughput systems |
| **cache invalidation** | Coordinated updates across cache layers to maintain consistency while preserving performance benefits | Cache management strategy balancing performance and data consistency |
| **resource slot management** | System for limiting concurrent task execution based on worker capacity and resource availability | Resource management technique preventing system overload |

### Testing and Quality Assurance Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **integration testing** | Validation of component interactions with minimal mocking to ensure real-world behavior | Testing strategy for validating system-wide functionality |
| **unit testing** | Testing individual components in isolation to verify specific functionality | Testing strategy for validating component-level correctness |
| **test fixtures** | Pre-defined test data and system states for consistent and repeatable testing | Testing infrastructure for reliable test execution |
| **load testing** | Performance validation under high throughput conditions to ensure system scalability | Testing practice for validating system performance characteristics |
| **milestone verification** | Concrete validation steps after completing development milestones to ensure progress | Development practice for ensuring implementation quality |

### Error Handling and Recovery Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **failure mode analysis** | Systematic catalog of potential system failures and their corresponding recovery strategies | Engineering practice for designing robust systems |
| **exponential backoff** | Retry delay strategy that increases wait time after each failure to reduce system load during outages | Retry pattern for handling transient failures efficiently |
| **cascade failure** | Failures spreading from one component to others through dependencies, potentially causing system-wide outages | Failure pattern that must be prevented through careful system design |
| **split-brain scenarios** | Situations where multiple components believe they are the primary coordinator, leading to inconsistent state | Distributed system problem requiring careful coordination protocol design |
| **optimistic locking** | Concurrency control approach that detects conflicts using version fields without holding locks | Concurrency pattern for high-performance systems with low conflict rates |

### Workflow Definition and Configuration Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **YAML_CONFIG** | Workflow definition format using YAML syntax for human-readable configuration | Configuration format choice for workflow definitions |
| **template inheritance** | Hierarchical workflow patterns where specialized templates extend base templates | Design pattern for reusable workflow components |
| **workflow templating** | Template system for creating reusable workflow patterns with parameterization | Feature enabling efficient workflow development and maintenance |
| **sensor-based triggering** | Event-driven workflow triggering based on external events like file changes or API notifications | Advanced triggering mechanism for responsive data pipelines |

### Acronyms and Abbreviations

| Acronym | Full Form | Definition |
|---------|-----------|------------|
| **DAG** | Directed Acyclic Graph | Graph structure representing workflow dependencies without cycles |
| **SSE** | Server-Sent Events | Web standard for real-time server-to-client communication |
| **API** | Application Programming Interface | Programmatic interface for system interaction |
| **REST** | Representational State Transfer | Architectural style for web APIs |
| **CORS** | Cross-Origin Resource Sharing | Web security mechanism for cross-domain requests |
| **SQL** | Structured Query Language | Standard language for database operations |
| **TTL** | Time To Live | Duration for which cached data remains valid |
| **UUID** | Universally Unique Identifier | Standard format for unique object identification |

### State Enumerations

The system uses several enumerated types to represent state throughout the workflow lifecycle:

| State Type | Possible Values | Transitions |
|------------|----------------|-------------|
| **DAGRunState** | queued → running → success/failed/up_for_retry | Linear progression with retry capability |
| **TaskInstanceState** | none → queued → running → success/failed/up_for_retry/upstream_failed/skipped | Complex state machine with multiple terminal states |
| **CircuitBreakerState** | CircuitClosed → CircuitOpen → CircuitHalfOpen | Failure detection and recovery state machine |
| **WorkerStatus** | healthy → degraded → failed → recovering | Worker health management state machine |

### Configuration and Infrastructure Terms

| Term | Definition | Context and Usage |
|------|------------|-------------------|
| **connection pooling** | Reusing database connections to improve performance and resource utilization | Database optimization technique for high-concurrency systems |
| **migration management** | Systematic approach to evolving database schema while preserving data | Database management practice for system evolution |
| **environment configuration** | System for managing configuration differences between development, staging, and production | Deployment practice for consistent system behavior across environments |
| **structured logging** | Consistent log format with searchable fields and correlation IDs for effective troubleshooting | Operational practice for system observability and debugging |

> **Key Design Principle**: This glossary reflects the principle of precise terminology throughout the workflow orchestrator. Each term has a specific meaning within the context of workflow orchestration, and using consistent vocabulary across documentation, code, and team communication reduces confusion and improves system maintainability.

Understanding these terms and their relationships is essential for effective implementation and operation of the workflow orchestrator. The terminology provides a foundation for clear communication between team members and accurate system documentation.

### Implementation Guidance

When implementing the workflow orchestrator, maintain consistency in terminology across all code, documentation, and user interfaces. This consistency reduces cognitive load for developers and operators working with the system.

**Terminology Usage Guidelines:**

| Context | Recommendation | Example |
|---------|---------------|---------|
| **Code Identifiers** | Use exact terms from naming conventions | `TaskInstanceState`, `DAGRunState`, `ExecutionContext` |
| **API Responses** | Use consistent field names matching internal terminology | `execution_date`, `dag_run_id`, `task_instance_state` |
| **User Interface** | Use human-friendly versions while maintaining accuracy | "Workflow" for DAG, "Task Status" for TaskInstanceState |
| **Documentation** | Always use precise technical terms with definitions | Reference this glossary for authoritative definitions |
| **Log Messages** | Include technical terms for searchability | "TaskInstance abc123 transitioned to running state" |

**Common Terminology Mistakes:**

⚠️ **Pitfall: Inconsistent State Names**
Using different names for the same concept across components leads to confusion and bugs. For example, using both "completed" and "success" to refer to successful task completion creates ambiguity.

⚠️ **Pitfall: Overloaded Terms**
Using generic terms like "job" or "process" instead of specific terms like "DAGRun" or "TaskInstance" reduces precision and makes communication less effective.

⚠️ **Pitfall: Missing Context**
Using abbreviated terms without context, such as "DAG" without specifying whether you mean the definition, instance, or run, creates confusion in documentation and discussions.

This glossary serves as the authoritative source for all terminology used throughout the workflow orchestrator system, ensuring consistent understanding and communication across all stakeholders.

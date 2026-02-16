# AUDIT & FIX: chaos-engineering

## CRITIQUE
- **Milestone 1 and Milestone 5 Redundantly Cover the Same Ground**: M1 implements 'Network latency injection' and 'Packet loss simulation'. M5 implements 'Network partition simulation', 'Configurable packet loss, latency injection'. These are the same capabilities described twice. This wastes 10 hours of the learner's time.
- **Blast Radius Control is Scattered, Not Centralized**: Blast radius is mentioned in M1 concepts, M4 deliverables, and M2 concepts, but no single milestone owns it as a first-class feature with rigorous ACs. In real chaos engineering, blast radius control is THE safety mechanism—it deserves dedicated treatment.
- **Milestone Ordering is Wrong**: M4 (Steady-State Hypothesis & Metrics Validation) should come BEFORE M2 (Experiment Orchestration) or at least be part of it. You can't orchestrate experiments without first defining what 'steady state' means and how to measure it. The current order has you building the orchestrator (M2) and then retroactively bolting on hypothesis validation (M4).
- **'GameDay Automation' (M3) is Too Process-Oriented, Not Technical Enough**: Most of M3's ACs are about running experiments in sequence and generating reports—which is essentially what M2 already does. The concepts mention 'observer briefing' and 'communication protocols', which are organizational practices, not technical implementations.
- **Missing: Target Discovery and Experiment Registry**: There's no mechanism for discovering available targets (services, pods, nodes) or for storing and versioning experiment definitions. Real chaos engineering tools (Chaos Monkey, Litmus, ChaosBlade) all have an experiment catalog.
- **tc/iptables Requires Root/CAP_NET_ADMIN but This is Only in a Pitfall**: This is a hard blocker, not a 'pitfall'. If the learner doesn't have the right privileges, nothing works. It should be in the prerequisites or the first AC.
- **50 Hours is Overestimated Given the Redundancy**: With M1 and M5 merged, and M3/M4 rationalized, this is closer to 35-45 hours.
- **Statistical Significance Mentioned but Never Defined**: M4 says 'Post-experiment report comparing pre/during/post metrics with statistical significance' but never specifies what statistical test or confidence level. This makes the AC unverifiable.

## FIXED YAML
```yaml
id: chaos-engineering
name: Chaos Engineering Framework
description: Fault injection and resilience testing with safety controls
difficulty: expert
estimated_hours: "40-55"
essence: >
  Controlled failure injection into distributed systems with automated
  steady-state hypothesis validation, experiment orchestration with blast
  radius controls and safety abort boundaries, and systematic measurement
  of system resilience to discover weaknesses before they cause production
  outages.
why_important: >
  Building this develops critical skills for designing highly available
  distributed systems. You learn to think probabilistically about failure
  modes, implement safety mechanisms for controlled experimentation, and
  systematically validate resilience—skills essential for senior engineering
  roles in infrastructure, SRE, and platform engineering.
learning_outcomes:
  - Implement fault injection primitives (latency, packet loss, process kill, resource exhaustion, network partition, DNS failure)
  - Design steady-state hypothesis definitions with measurable metric thresholds
  - Build experiment orchestration with pre-validation, fault injection, continuous monitoring, and automatic rollback
  - Implement blast radius controls limiting experiment scope to specific targets and traffic percentages
  - Build safety abort mechanisms that halt experiments when metrics breach thresholds
  - Design a GameDay automation system running multi-experiment sequences with reporting
  - Implement experiment versioning and a catalog for reproducible chaos testing
skills:
  - Distributed Systems Resilience
  - Fault Injection (network, process, resource)
  - Steady-State Hypothesis Testing
  - Blast Radius Management
  - Safety Abort Mechanisms
  - Observability and Metrics Integration
  - Experiment Orchestration
  - Linux Networking (tc, iptables, cgroups)
tags:
  - chaos
  - experiments
  - expert
  - fault-injection
  - framework
  - reliability
  - resilience
  - testing
architecture_doc: architecture-docs/chaos-engineering/index.md
languages:
  recommended:
    - Go
    - Python
    - Java
  also_possible: []
resources:
  - name: "Principles of Chaos Engineering"
    url: "https://principlesofchaos.org/"
    type: documentation
  - name: "Chaos Monkey by Netflix"
    url: "https://netflix.github.io/chaosmonkey/"
    type: tool
  - name: "LitmusChaos"
    url: "https://litmuschaos.io/"
    type: tool
  - name: "Google Cloud Chaos Engineering Guide"
    url: "https://cloud.google.com/blog/products/devops-sre/getting-started-with-chaos-engineering"
    type: article
prerequisites:
  - type: skill
    name: "Linux system administration (tc, iptables, cgroups, CAP_NET_ADMIN)"
  - type: skill
    name: "HTTP server and client programming"
  - type: skill
    name: "Metrics and monitoring basics (Prometheus or equivalent)"
  - type: project
    id: container-runtime
    name: "Container Runtime (recommended for understanding namespaces/cgroups)"
milestones:
  - id: chaos-engineering-m1
    name: "Fault Injection Primitives"
    description: >
      Implement a library of fault injection primitives covering network
      faults (latency, packet loss, partition, DNS failure), process faults
      (kill, pause), and resource faults (CPU, memory, disk exhaustion).
      Each fault must have a clean rollback mechanism.
    acceptance_criteria:
      - "Network latency injection: uses tc netem to add configurable delay (mean and jitter) to traffic on a specified network interface; verified by measuring RTT increase with ping"
      - "Packet loss injection: uses tc netem to drop a configurable percentage of packets on a specified interface; verified by measuring packet loss with ping -c 100"
      - "Network partition: uses iptables DROP rules to block all traffic between specified IP pairs; verified by confirming connection timeout between partitioned services"
      - "DNS failure injection: intercepts DNS resolution and returns NXDOMAIN, SERVFAIL, or wrong IP for specified domains; implemented via /etc/resolv.conf manipulation, local DNS proxy, or iptables DNS redirect"
      - "Process kill: sends SIGKILL to a specified process by name or PID and verifies the process is terminated; optionally supports SIGSTOP/SIGCONT for pause/resume"
      - "CPU stress: launches CPU-bound workers (or uses cgroup cpu.max) to consume a configurable percentage of CPU on the target; verified by measuring CPU utilization via /proc/stat or cgroup metrics"
      - "Memory pressure: allocates memory (or sets cgroup memory.max) to simulate OOM conditions at configurable utilization; verified by checking memory metrics or OOM kill events"
      - "Disk exhaustion: fills a specified filesystem to a configurable percentage using fallocate; verified by checking df output"
      - "EVERY fault has a corresponding rollback function that removes the fault completely (deletes tc rules, removes iptables rules, kills stress processes, removes files); rollback is verified by re-measuring the affected metric"
      - "Privilege check: each fault verifier tests for required capabilities (CAP_NET_ADMIN for tc/iptables, CAP_SYS_RESOURCE for cgroups) and fails with a clear error message if missing"
    pitfalls:
      - "tc and iptables commands require CAP_NET_ADMIN or root; without this capability, all network faults fail silently or with cryptic errors. Check capabilities at startup."
      - "CPU stress tool running in the same cgroup as the chaos framework consumes resources needed by the framework itself; always isolate stress workloads in a separate cgroup"
      - "Process kill without a restart mechanism means the test target stays dead; ensure the target has a supervisor (systemd, kubelet) or the test framework restarts it"
      - "Rollback that fails (e.g., iptables rule deletion fails due to changed rule index) leaves persistent faults; implement rollback verification that re-checks the system state"
      - "Disk exhaustion with fallocate on the root filesystem can make the system unbootable; always target a specific non-root mountpoint"
    concepts:
      - Network fault injection via tc netem and iptables
      - Process signal injection (SIGKILL, SIGSTOP)
      - Resource exhaustion via cgroups and stress tools
      - DNS interception and manipulation
      - Rollback verification for fault cleanup
    deliverables:
      - "Fault library with implementations for: latency, packet loss, partition, DNS failure, process kill, CPU stress, memory pressure, disk exhaustion"
      - "Rollback function for each fault type with post-rollback verification"
      - "Capability checker validating required Linux capabilities before fault execution"
      - "Fault configuration schema: target (IP/interface/process/path), parameters (duration, intensity), and rollback config"
    estimated_hours: "12-16"

  - id: chaos-engineering-m2
    name: "Steady-State Hypothesis & Blast Radius"
    description: >
      Implement steady-state hypothesis definition with measurable metric
      thresholds and blast radius controls that limit experiment scope.
      The hypothesis engine validates system health before, during, and
      after fault injection.
    acceptance_criteria:
      - "Hypothesis definition specifies metric queries (e.g., Prometheus PromQL), expected thresholds (e.g., p99_latency < 200ms, error_rate < 1%), and evaluation interval (e.g., every 5s)"
      - "Baseline validation: before any fault is injected, all hypothesis metrics are evaluated for a configurable warm-up period (e.g., 30s); if baseline is unhealthy, the experiment is aborted before injection"
      - "During-fault validation: metrics are continuously evaluated during fault injection; if any metric breaches its threshold, the experiment is flagged as hypothesis-violated"
      - "Post-fault validation: after fault rollback, metrics are re-evaluated for a recovery period; the system must return to baseline within a configurable recovery timeout"
      - "Blast radius control - target scope: experiments specify affected targets as a percentage of instances (e.g., 10% of pods) or specific named instances; the framework ensures only specified targets receive faults"
      - "Blast radius control - traffic scope: for HTTP/gRPC faults, only a configurable percentage of requests are affected (e.g., inject latency on 5% of requests)"
      - "Safety abort: if any hypothesis metric exceeds a stricter 'abort threshold' (separate from the hypothesis threshold), the experiment immediately rolls back all faults and terminates"
      - "Abort test: run an experiment with abort threshold at error_rate > 5%; inject a fault that causes 10% error rate; verify the experiment auto-aborts within 2 evaluation intervals and faults are rolled back"
    pitfalls:
      - "Metrics collection lag (Prometheus scrape interval, metric pipeline latency) means the hypothesis engine sees stale data; account for at least 2x scrape interval delay in abort decisions"
      - "Confounding variables: other system changes during the experiment can affect metrics; always document what changed and consider running a control group"
      - "Blast radius percentage calculation requires accurate knowledge of total instances; stale instance count means wrong percentage"
      - "Abort threshold too close to hypothesis threshold causes false aborts; abort threshold should represent actual danger, not expected experiment behavior"
    concepts:
      - Steady-state hypothesis with metric thresholds
      - Baseline, during-fault, and post-fault validation phases
      - Blast radius control (target scope and traffic scope)
      - Safety abort with hard thresholds
      - Metrics lag and its impact on abort latency
    deliverables:
      - "Hypothesis definition schema: metrics, thresholds, evaluation interval, abort thresholds"
      - "Hypothesis validator: baseline check, continuous monitoring, post-recovery validation"
      - "Blast radius controller: target percentage selection and traffic percentage limiting"
      - "Safety abort mechanism: threshold breach detection and automatic fault rollback"
      - "Abort latency test verifying rollback occurs within bounded time of threshold breach"
    estimated_hours: "10-14"

  - id: chaos-engineering-m3
    name: "Experiment Orchestration"
    description: >
      Build the experiment engine that combines fault injection, hypothesis
      validation, and blast radius into a single orchestrated workflow.
      Experiments are defined declaratively and executed through a state
      machine: BASELINE -> INJECTION -> MONITORING -> ROLLBACK -> VALIDATION -> REPORT.
    acceptance_criteria:
      - "Experiment definition (YAML or JSON) specifies: name, hypothesis, fault(s) to inject, target(s), blast radius, duration, abort conditions, and metadata (owner, description)"
      - "Experiment state machine executes phases in order: INIT -> BASELINE_CHECK -> FAULT_INJECTION -> MONITORING -> ROLLBACK -> POST_VALIDATION -> REPORT"
      - "BASELINE_CHECK phase validates all hypothesis metrics are within thresholds for the warm-up period; experiment aborts if baseline is unhealthy"
      - "FAULT_INJECTION phase activates the configured faults on the specified targets within blast radius limits"
      - "MONITORING phase continuously validates hypothesis metrics and safety abort thresholds for the configured experiment duration"
      - "ROLLBACK phase removes all injected faults and verifies rollback via rollback verification checks"
      - "POST_VALIDATION phase checks that the system returns to baseline within recovery timeout"
      - "REPORT phase generates a structured report containing: experiment ID, start/end timestamps, each phase's duration and outcome, metric values (baseline average, during-fault average, post-recovery average), hypothesis result (maintained/violated), and a pass/fail verdict"
      - "Experiment catalog: experiments are stored with version numbers; re-running an experiment uses the same definition for reproducibility"
      - "Dry-run mode: experiment executes all phases except FAULT_INJECTION (faults are logged but not applied) to validate the experiment definition and metric connectivity"
    pitfalls:
      - "Always verify steady state BEFORE injecting faults; skipping baseline validation means you can't distinguish pre-existing issues from experiment-caused issues"
      - "Automatic rollback is critical; manual cleanup after a failed experiment is error-prone and may leave persistent faults"
      - "Abort conditions should check error rates and latency, not just availability; a system can be 'available' but degraded"
      - "Experiment definitions without version control lead to unreproducible results; always version and store experiment configs"
    concepts:
      - Declarative experiment definition
      - State machine-based experiment execution
      - Phase-ordered orchestration with abort at any phase
      - Experiment versioning and reproducibility
      - Dry-run validation
    deliverables:
      - "Experiment definition schema (YAML/JSON) with all required fields"
      - "Experiment state machine executing phases in order with abort capability"
      - "Experiment runner integrating fault library, hypothesis validator, and blast radius controller"
      - "Experiment report generator producing structured pass/fail results with metrics"
      - "Experiment catalog for storing, versioning, and retrieving experiment definitions"
      - "Dry-run mode for experiment validation without actual fault injection"
    estimated_hours: "12-16"

  - id: chaos-engineering-m4
    name: "GameDay Automation & Reporting"
    description: >
      Build a GameDay runner that executes multiple experiments in sequence
      (a scenario), with configurable pauses between experiments, continuous
      monitoring, and a comprehensive final report summarizing all results.
    acceptance_criteria:
      - "GameDay scenario definition lists multiple experiments in sequence with configurable pause durations between each"
      - "GameDay runner executes experiments one at a time, waiting for each to complete (including post-validation) before starting the next"
      - "If any experiment triggers a safety abort, the GameDay halts all remaining experiments and generates a partial report"
      - "System-wide health check runs between experiments; if the system hasn't recovered to baseline from the previous experiment, the GameDay pauses until recovery or times out and halts"
      - "Final GameDay report aggregates all experiment reports into a single document with: overall pass/fail, per-experiment summary, timeline of events, and recommendations for failed hypotheses"
      - "Schedule GameDays for recurring execution (e.g., weekly) with configurable notification (webhook, email) on completion"
      - "Approval gate: optionally require manual approval before each experiment in the sequence (for initial adoption)"
    pitfalls:
      - "Running experiments back-to-back without recovery verification causes cascading failures where each experiment starts from a degraded baseline"
      - "GameDays without team notification mean nobody is watching when things go wrong; always notify the team before automated GameDays"
      - "Recording observations is critical: automated metrics capture misses qualitative observations like 'UI was slow but not measured'"
      - "Scheduling GameDays during off-hours when nobody can respond defeats the purpose of validating operational readiness"
    concepts:
      - Multi-experiment scenario sequencing
      - Inter-experiment health gating
      - GameDay scheduling and notifications
      - Aggregated reporting across experiments
      - Approval gates for safety
    deliverables:
      - "GameDay scenario definition with ordered experiment list and inter-experiment pauses"
      - "GameDay runner with sequential execution, health gating, and abort-on-failure"
      - "Aggregated GameDay report combining all experiment results"
      - "Scheduling system for recurring GameDay execution with notifications"
      - "Optional approval gate for manual experiment-by-experiment approval"
    estimated_hours: "8-10"
```
# DOMAIN PROFILE: DevOps & Software Engineering
# Applies to: software-engineering
# Projects: CI/CD, test framework, observability, APM, metrics, logging, load testing, etc.

## Fundamental Tension Type
Operational constraints. VELOCITY (ship fast) vs RELIABILITY (don't break production). DevOps tools make this tradeoff less painful — speed WITHOUT sacrificing reliability.

Secondary: observability depth vs cost, blast radius vs deploy speed, abstraction vs debuggability, alert sensitivity vs fatigue, test thoroughness vs feedback loop speed.

## Three-Level View
- **Level 1 — Developer Experience**: CLI, config files, dashboard, PR checks
- **Level 2 — Pipeline Architecture**: Build stages, test runners, deploy strategies, artifacts, notifications
- **Level 3 — Infrastructure**: Container orchestration, cloud resources, networking, storage, scaling

## Soul Section: "Reliability Soul"
- MTTR if this fails?
- Blast radius? Cascading failures?
- Self-healing or manual intervention?
- Rollback strategy? How fast?
- Detection method? (metrics threshold, health checks, log patterns, user reports)
- On-call experience? 3am wakeup or can wait?
- Graceful degradation? (circuit breaker, feature flags, fallback)

## Alternative Reality Comparisons
GitHub Actions/GitLab CI/Jenkins, ArgoCD/Flux, Prometheus/Grafana, ELK/Loki, Jaeger/Zipkin, Datadog/New Relic, PagerDuty, Terraform/Pulumi, pytest/Jest/JUnit, Chaos Monkey.

## TDD Emphasis
- Pipeline spec: MANDATORY — stages, deps, parallelism, failure handling, retries
- Config schema: MANDATORY — every field with type, default, validation, examples
- Plugin/extension interface: if extensible, specify contract
- Dashboard/query spec: metrics collected, queries per panel
- Storage schema: metric/log/trace storage, retention, aggregation
- API: REST/gRPC endpoints for programmatic access
- Memory layout: SKIP
- Cache line: SKIP
- Lock ordering: SKIP (use queues + event-driven)
- Benchmarks: pipeline time, ingestion rate events/sec, query latency, storage cost

## Cross-Domain Notes
Borrow from systems-lowlevel when: CI runner process management (fork/exec/wait), container isolation (namespaces, cgroups), resource limits.
Borrow from distributed when: distributed pipeline execution, log aggregation consistency.
Borrow from security when: pipeline security (secrets management, supply chain, RBAC).
Borrow from web-app when: dashboard frontend, API design.

# AUDIT & FIX: alerting-system

## CRITIQUE
- **State Persistence**: This is the most critical gap. The entire alerting system's correctness depends on tracking 'pending' durations (for-duration). If the service restarts, all pending timers reset to zero, causing alerts that were about to fire to be silently delayed by the full for-duration again. This can mask real incidents. The original has NO mention of persisting alert state.
- **Flapping Protection**: Metrics hovering exactly on the threshold cause rapid firing->resolved->firing cycles. The original mentions 'Flapping alerts from noisy metrics need hysteresis' as a pitfall but provides no AC requiring hysteresis implementation. Hysteresis means using a different (lower) threshold to resolve than to fire.
- **Inhibition Logic**: The M3 milestone mentions inhibition but the AC is shallow. Real inhibition requires: (1) defining source-target relationships, (2) matching on label subsets, (3) preventing infinite inhibition loops (A inhibits B, B inhibits A). The audit correctly identifies this gap.
- **Missing heartbeat/dead-man's-switch alerts**: No mention of alerting on the ABSENCE of data (e.g., 'no metrics received from service X in 5 minutes'). This is critical—a crashed service sends no metrics, so threshold-based alerts never fire.
- **M1 Template rendering errors**: The pitfall mentions template rendering errors crashing the evaluation loop but no AC requires error isolation so that one bad template doesn't prevent other alerts from being evaluated.
- **M2 group_wait for critical alerts**: The pitfall mentions 'Long group_wait delays critical alerts' but the AC doesn't require a way to bypass group_wait for high-severity alerts.
- **M4 Escalation is a deliverable but has no AC**: Escalation policies are listed as a deliverable but no AC specifies the escalation behavior (time-to-escalate, escalation chain, acknowledgment).
- **Missing recording rules / pre-computed aggregations**: Complex alert expressions evaluated every 15 seconds against raw time series can be expensive. Recording rules pre-compute and store aggregations for alert evaluation.

## FIXED YAML
```yaml
id: alerting-system
name: Alerting System
description: >-
  Metric-based alerting with state machine transitions, label-based grouping,
  inhibition, silencing, notification routing with escalation, persistent state,
  and flapping protection.
difficulty: intermediate
estimated_hours: "35-45"
essence: >-
  Time-series metric evaluation with persistent state machine transitions
  (pending -> firing -> resolved) with hysteresis for flapping protection,
  label-based alert deduplication and grouping, inhibition rules preventing
  cascading notifications, and policy-driven routing trees with escalation
  that fan-out notifications to multiple receivers.
why_important: >-
  Alerting is critical for on-call reliability. Understanding alert fatigue,
  grouping, flapping protection, inhibition, and state persistence helps design
  alert systems that wake people up for the right reasons and stay quiet for the
  wrong ones.
learning_outcomes:
  - Design an alert rule evaluation engine with persistent state across restarts
  - Implement alert state machine with hysteresis for flapping protection
  - Build notification grouping with configurable group_wait and group_interval
  - Implement alert silencing for maintenance windows and inhibition for cascading alerts
  - Design notification routing trees with escalation policies
  - Handle dead-man's-switch alerts that fire on absence of data
skills:
  - Time-series querying
  - Alert state machines
  - State persistence
  - Notification routing
  - Label-based grouping
  - Silence management
  - Webhook integration
  - Template rendering
  - Inhibition rules
  - Hysteresis
tags:
  - devops
  - distributed-systems
  - escalation
  - intermediate
  - notifications
  - observability
  - thresholds
  - webhooks
architecture_doc: architecture-docs/alerting-system/index.md
languages:
  recommended:
    - Go
    - Python
  also_possible:
    - Java
resources:
  - name: Prometheus Alerting Overview
    url: https://prometheus.io/docs/alerting/latest/overview/
    type: documentation
  - name: Alertmanager Configuration
    url: https://prometheus.io/docs/alerting/latest/configuration/
    type: documentation
  - name: Google SRE Monitoring
    url: https://sre.google/sre-book/monitoring-distributed-systems/
    type: book
  - name: Alerting Tutorial
    url: https://prometheus.io/docs/tutorials/alerting_based_on_metrics/
    type: tutorial
prerequisites:
  - type: project
    id: metrics-collector
  - type: skill
    name: State machine design
  - type: skill
    name: HTTP webhooks
milestones:
  - id: alerting-system-m1
    name: Alert Rule Evaluation with Persistent State
    description: >-
      Build a rule evaluation engine that periodically queries metrics, evaluates
      threshold conditions with for-duration, and persists alert state across
      service restarts.
    acceptance_criteria:
      - "Rule engine evaluates PromQL-like expressions against current metric data at a configurable interval (default 15 seconds)"
      - "Comparison operators >, <, >=, <=, ==, != are supported for threshold-based alerting"
      - "for-duration keeps alerts in 'pending' state until the condition holds continuously for the specified period; a brief return to normal resets the pending timer"
      - Alert states transition correctly: inactive -> pending (condition met) -> firing (for-duration elapsed) -> resolved (condition no longer met)
      - "Alert state (including pending start time and firing start time) is persisted to disk or database; a service restart does NOT reset pending timers or lose firing state"
      - Hysteresis (flapping protection): alert fires when metric exceeds 'fire_threshold' and resolves only when metric drops below 'resolve_threshold' (where resolve_threshold < fire_threshold); configurable per rule
      - Dead-man's-switch alert: a rule can be configured to fire when NO data is received for a metric within a configurable window (e.g., 5 minutes), detecting crashed services
      - "Template rendering errors for one alert rule do not crash the evaluation loop or prevent other rules from being evaluated; errors are logged and the affected alert uses a fallback message"
    pitfalls:
      - State loss on restart: pending durations reset, causing alerts to be delayed by the full for-duration again after every restart—persist state
      - Flapping without hysteresis: metric oscillating at exactly the threshold causes rapid firing/resolved cycles, flooding notification channels
      - "for-duration timer resets on ANY brief recovery, even a single scrape—consider configurable tolerance (e.g., 'condition must hold for 4 out of 5 evaluations')"
      - "Template rendering error crashing the entire evaluation goroutine—isolate template rendering with panic recovery"
      - "Dead-man's-switch false positives during planned maintenance—integrate with silencing"
    concepts:
      - Periodic metric evaluation
      - Alert state machine with persistence
      - Hysteresis for flapping protection
      - Dead-man's-switch alerts
      - Error isolation in evaluation loops
    skills:
      - Time-series database querying
      - State machine implementation with persistence
      - Expression parsing and evaluation
      - Template engine integration with error handling
    deliverables:
      - "Rule expression parser converting PromQL-like alert expressions into evaluable objects"
      - "Periodic evaluation loop checking all active rules at configurable interval"
      - "State persistence layer saving alert state (pending start, firing start) to disk/database"
      - "Hysteresis implementation with configurable fire and resolve thresholds"
      - "Dead-man's-switch rule type firing on absence of data"
      - "Error-isolated template renderer with fallback message on failure"
    estimated_hours: "10-12"

  - id: alerting-system-m2
    name: Alert Grouping & Batching
    description: >-
      Group related alerts to reduce notification noise with configurable
      grouping keys, wait times, and severity-based bypass.
    acceptance_criteria:
      - "Alerts are grouped by user-configurable label sets (e.g., alertname, cluster, service); all alerts with the same group key are batched into a single notification"
      - "group_wait delays the first notification for a new group to collect more alerts (configurable, default 30s)"
      - "group_interval controls re-send frequency for groups with changed alerts (configurable, default 5m)"
      - "Group key generation produces a stable, deterministic key from the sorted set of grouping label values; label order does not affect grouping"
      - "Critical severity alerts can bypass group_wait and send immediately (configurable per severity level)"
      - "Resolved notifications are sent when all alerts in a group have cleared; the notification includes the list of resolved alerts"
      - "Empty groups (all alerts resolved and notification sent) are cleaned up after a configurable idle period to prevent memory leaks"
    pitfalls:
      - "Group key change (e.g., label value changes) orphans the alert in the old group and creates a new group—handle gracefully by moving the alert"
      - "Long group_wait delays critical alerts—allow severity-based bypass for urgent alerts"
      - "Memory leak from groups that never get cleaned up after all alerts resolve—implement group garbage collection"
      - "Notification batching that produces messages too large for the notification channel (e.g., Slack message length limit)—truncate or paginate"
    concepts:
      - Label-based grouping keys
      - Time-based batching windows
      - Group lifecycle management
      - Severity-based routing bypass
    skills:
      - Hash-based grouping algorithms
      - Timer management for delayed processing
      - Memory management for long-lived collections
      - Notification message formatting
    deliverables:
      - "Group-by-labels logic aggregating alerts into groups with deterministic key generation"
      - "group_wait and group_interval timers controlling notification timing"
      - "Severity-based bypass allowing critical alerts to skip group_wait"
      - "Group resolution handler sending resolved notifications when all alerts clear"
      - "Group garbage collector cleaning up idle empty groups"
    estimated_hours: "8-10"

  - id: alerting-system-m3
    name: Silencing & Inhibition
    description: >-
      Implement alert silencing for maintenance windows and inhibition rules
      that suppress child alerts when parent alerts are firing.
    acceptance_criteria:
      - "Silence rules suppress notifications for alerts whose labels match ALL specified matchers (exact match, regex match, not-equal)"
      - "Time-based silence windows activate at start time and expire at end time; expired silences are automatically removed"
      - Inhibition rules define: source alert matchers, target alert matchers, and equal labels; when a source alert is firing, target alerts with matching equal labels are suppressed
      - "Inhibition cycle detection prevents rules where A inhibits B and B inhibits A; cycles are rejected at configuration time with an error listing the cycle"
      - "Suppressed alerts maintain their state (pending/firing) but do NOT trigger notifications; when the silence or inhibition condition ends, notifications resume without re-triggering for-duration"
      - Race condition between silence creation and a currently firing alert is handled: if a silence is created while an alert is firing, the next notification cycle respects the silence
    pitfalls:
      - Inhibition loops: A inhibits B, B inhibits A—both get suppressed and nobody gets notified. Detect cycles at configuration time.
      - "Silence with wrong matchers (e.g., regex typo) doesn't match intended alerts but silences unrelated ones—provide a 'preview' API showing which alerts a silence would affect"
      - Race between silence creation and in-flight notification: the notification was already queued before the silence was created—check silences at send time, not just evaluation time
      - Suppressed alerts not being tracked: if an inhibition source resolves, target alerts must be re-evaluated and notifications sent if they're still firing
    concepts:
      - Label matcher syntax and evaluation
      - Time-based silence activation
      - Inhibition dependency graphs with cycle detection
      - Concurrent state access safety
    skills:
      - Label selector implementation
      - Cycle detection in directed graphs
      - Time range validation
      - Atomic state updates
    deliverables:
      - "Silence matcher suppressing alerts by label matchers with time-based activation/expiry"
      - "Silence preview API showing which currently firing alerts would be affected by a proposed silence"
      - "Inhibition rule engine suppressing target alerts when source alerts with matching labels are firing"
      - "Cycle detection rejecting inhibition configurations that create circular suppression"
      - "Suppression state tracker maintaining alert state while suppressing notifications"
    estimated_hours: "8-10"

  - id: alerting-system-m4
    name: Notification Routing & Escalation
    description: >-
      Route alerts to notification channels based on label matching with
      escalation policies for unacknowledged alerts.
    acceptance_criteria:
      - "Routing tree matches alerts to notification channels based on label matchers; a default route catches all unmatched alerts"
      - Multiple notification channels are supported: email, Slack webhook, PagerDuty, and generic HTTP webhook
      - "Notification templates use Go-template (or Jinja2) syntax with access to alert labels, annotations, and group information"
      - "Rate limiting enforces a maximum notification frequency per channel (e.g., max 1 per minute per channel) to prevent notification storms"
      - Escalation policy: if an alert is not acknowledged within a configurable time (e.g., 15 minutes), it is re-routed to the next escalation level (e.g., from Slack to PagerDuty)
      - "Acknowledgment API allows on-call responders to acknowledge an alert, stopping escalation and suppressing repeat notifications for a configurable period"
      - "Notification delivery failure (HTTP timeout, 5xx response) triggers retry with exponential backoff (max 3 retries); persistent failure is logged and the alert is routed to fallback channel"
    pitfalls:
      - Missing default route: alerts not matching any route are silently dropped—always require a catch-all default route
      - "continue=true on every route sends duplicate notifications to multiple channels unintentionally—document the continue flag behavior clearly"
      - "Rate limiting by channel prevents important alerts when mixed with noisy alerts—rate limit per (channel, alert_group) tuple, not just per channel"
      - "Escalation timer starts from first notification, not from alert firing time—if group_wait delays the first notification by 30s, escalation is also delayed"
      - "Notification channel API changes (Slack API v2, PagerDuty Events API v2) break integrations—use versioned client implementations"
    concepts:
      - Tree-based routing with label matchers
      - Escalation chain management
      - Notification delivery reliability
      - Rate limiting per destination
    skills:
      - Routing tree traversal and matching
      - HTTP client with retry and backoff
      - Template rendering
      - Rate limiter implementation
    deliverables:
      - "Routing tree matching alerts to channels based on label matchers with default route"
      - "Notification channel integrations for email, Slack, PagerDuty, and webhook"
      - "Customizable notification templates with alert context variables"
      - "Escalation policy engine re-routing unacknowledged alerts after timeout"
      - "Acknowledgment API stopping escalation and suppressing repeat notifications"
      - "Notification retry with exponential backoff and fallback channel on persistent failure"
    estimated_hours: "10-12"
```
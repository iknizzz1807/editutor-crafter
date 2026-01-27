#!/usr/bin/env python3
"""Add orphaned expert_projects to their appropriate domains."""

import yaml
from pathlib import Path

# Mapping of orphaned projects to their domains and levels
ORPHANED_PROJECTS = {
    # Security domain
    "security": {
        "intermediate": [
            {"id": "session-management", "name": "Session Management", "description": "Secure session handling with tokens and cookies", "detailed": True},
            {"id": "audit-logging", "name": "Audit Logging System", "description": "Immutable audit trail for compliance", "detailed": True},
        ],
        "advanced": [
            {"id": "oauth2-provider", "name": "OAuth2/OIDC Provider", "description": "Identity provider with authorization code flow, PKCE, JWT", "detailed": True},
            {"id": "rbac-system", "name": "RBAC/ABAC Authorization", "description": "Role and attribute based access control system", "detailed": True},
            {"id": "secret-management", "name": "Secret Management Vault", "description": "Secure storage for API keys, credentials, certificates", "detailed": True},
        ],
    },
    # Application Development domain
    "app-dev": {
        "intermediate": [
            {"id": "file-upload-service", "name": "File Upload Service", "description": "Chunked uploads, resumable, virus scanning", "detailed": True},
            {"id": "websocket-server", "name": "WebSocket Server", "description": "Real-time bidirectional communication", "detailed": True},
            {"id": "notification-service", "name": "Notification Service", "description": "Push notifications, email, SMS delivery", "detailed": True},
        ],
        "advanced": [
            {"id": "realtime-chat", "name": "Real-time Chat System", "description": "Scalable chat with presence, typing indicators", "detailed": True},
            {"id": "collaborative-editor", "name": "Collaborative Editor", "description": "Real-time collaborative editing like Google Docs", "detailed": True},
            {"id": "media-processing", "name": "Media Processing Pipeline", "description": "Image/video transcoding, thumbnails, CDN", "detailed": True},
        ],
    },
    # Data & Storage domain
    "data-storage": {
        "intermediate": [
            {"id": "etl-pipeline", "name": "ETL Pipeline", "description": "Extract, transform, load data processing", "detailed": True},
            {"id": "cdc-system", "name": "Change Data Capture", "description": "Database change streaming and replication", "detailed": True},
        ],
        "advanced": [
            {"id": "search-engine", "name": "Search Engine", "description": "Full-text search with inverted index, ranking", "detailed": True},
            {"id": "event-sourcing", "name": "Event Sourcing System", "description": "Event store with projections and snapshots", "detailed": True},
        ],
        "expert": [
            {"id": "time-series-db", "name": "Time-Series Database", "description": "Optimized for time-stamped data with compression", "detailed": True},
            {"id": "graph-db", "name": "Graph Database", "description": "Native graph storage with traversal algorithms", "detailed": True},
        ],
    },
    # Distributed & Cloud domain
    "distributed": {
        "intermediate": [
            {"id": "feature-flags", "name": "Feature Flag System", "description": "Dynamic feature toggles with targeting rules", "detailed": True},
            {"id": "job-scheduler", "name": "Job Scheduler", "description": "Distributed task scheduling with retries", "detailed": True},
        ],
        "advanced": [
            {"id": "service-mesh", "name": "Service Mesh", "description": "Sidecar proxy for service-to-service communication", "detailed": True},
            {"id": "rate-limiter-distributed", "name": "Distributed Rate Limiter", "description": "Rate limiting across multiple nodes", "detailed": True},
            {"id": "saga-orchestrator", "name": "Saga Orchestrator", "description": "Distributed transactions with compensating actions", "detailed": True},
            {"id": "chaos-engineering", "name": "Chaos Engineering Framework", "description": "Fault injection and resilience testing", "detailed": True},
            {"id": "container-runtime", "name": "Container Runtime", "description": "OCI-compliant container execution", "detailed": True},
        ],
        "expert": [
            {"id": "infrastructure-as-code", "name": "Infrastructure as Code Engine", "description": "Terraform-like resource provisioning", "detailed": True},
        ],
    },
    # Software Engineering Practices domain
    "software-engineering": {
        "intermediate": [
            {"id": "ci-cd-pipeline", "name": "CI/CD Pipeline Builder", "description": "Automated build, test, deploy workflows", "detailed": True},
        ],
        "advanced": [
            {"id": "load-testing-framework", "name": "Load Testing Framework", "description": "Performance testing with distributed load", "detailed": True},
            {"id": "log-aggregator", "name": "Log Aggregation System", "description": "Centralized logging with search and alerts", "detailed": True},
            {"id": "alerting-system", "name": "Alerting System", "description": "Metric-based alerts with escalation", "detailed": True},
            {"id": "apm-system", "name": "APM System", "description": "Application performance monitoring", "detailed": True},
            {"id": "metrics-collector", "name": "Metrics Collector", "description": "Time-series metrics collection and aggregation", "detailed": True},
        ],
    },
    # Specialized domain
    "specialized": {
        "advanced": [
            {"id": "payment-gateway", "name": "Payment Gateway", "description": "Payment processing with PCI compliance", "detailed": True},
            {"id": "subscription-billing", "name": "Subscription Billing", "description": "Recurring payments, invoicing, dunning", "detailed": True},
            {"id": "webhook-delivery", "name": "Webhook Delivery System", "description": "Reliable webhook dispatch with retries", "detailed": True},
            {"id": "cdn-implementation", "name": "CDN Implementation", "description": "Content delivery with edge caching", "detailed": True},
        ],
        "expert": [
            {"id": "multiplayer-game-server", "name": "Multiplayer Game Server", "description": "Real-time game state synchronization", "detailed": True},
        ],
    },
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    added_count = 0

    for domain in data['domains']:
        domain_id = domain['id']
        if domain_id in ORPHANED_PROJECTS:
            projects_to_add = ORPHANED_PROJECTS[domain_id]

            for level, projects in projects_to_add.items():
                if level not in domain['projects']:
                    domain['projects'][level] = []

                existing_ids = {p['id'] for p in domain['projects'][level]}

                for project in projects:
                    if project['id'] not in existing_ids:
                        domain['projects'][level].append(project)
                        print(f"Added: {project['id']} to {domain_id}/{level}")
                        added_count += 1
                    else:
                        print(f"Skipped (exists): {project['id']}")

    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nAdded {added_count} orphaned projects to domains")


if __name__ == "__main__":
    main()

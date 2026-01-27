#!/usr/bin/env python3
"""Update projects.yaml with missing real-world projects and fix duplicates."""

import yaml
from pathlib import Path

# New projects to add
NEW_PROJECTS = {
    # Backend Enterprise - app-dev domain
    "app-dev": {
        "intermediate": [
            {
                "id": "graphql-server",
                "name": "GraphQL Server",
                "description": "Schema-first API with resolvers, dataloaders, subscriptions",
                "detailed": True
            },
            {
                "id": "background-job-processor",
                "name": "Background Job Processor",
                "description": "Async task queue like Sidekiq/Celery with retries, scheduling",
                "detailed": True
            },
        ],
        "advanced": [
            {
                "id": "multi-tenant-saas",
                "name": "Multi-tenant SaaS Backend",
                "description": "Tenant isolation, row-level security, billing integration",
                "detailed": True
            },
        ],
        "expert": [
            {
                "id": "build-graphql-engine",
                "name": "Build Your Own GraphQL Engine",
                "description": "Query parsing, execution, schema stitching like Hasura",
                "detailed": True
            },
        ],
    },
    # Data Engineering - data-storage domain
    "data-storage": {
        "intermediate": [
            {
                "id": "data-quality-framework",
                "name": "Data Quality Framework",
                "description": "Schema validation, anomaly detection, data profiling",
                "detailed": True
            },
        ],
        "advanced": [
            {
                "id": "workflow-orchestrator",
                "name": "Workflow Orchestrator",
                "description": "DAG-based task scheduling like Airflow with dependencies",
                "detailed": True
            },
            {
                "id": "vector-database",
                "name": "Vector Database",
                "description": "Similarity search with HNSW/IVF indexes for embeddings",
                "detailed": True
            },
        ],
        "expert": [
            {
                "id": "stream-processing-engine",
                "name": "Stream Processing Engine",
                "description": "Real-time data processing like Flink with windowing, state",
                "detailed": True
            },
            {
                "id": "data-lakehouse",
                "name": "Data Lakehouse",
                "description": "Delta Lake-like ACID transactions on object storage",
                "detailed": True
            },
        ],
    },
    # Cloud Native - distributed domain
    "distributed": {
        "advanced": [
            {
                "id": "kubernetes-operator",
                "name": "Kubernetes Operator",
                "description": "Custom controller with CRDs for automated app management",
                "detailed": True
            },
            {
                "id": "gitops-deployment",
                "name": "GitOps Deployment System",
                "description": "Git-driven deployments like ArgoCD with sync, rollback",
                "detailed": True
            },
        ],
        "expert": [
            {
                "id": "serverless-runtime",
                "name": "Serverless Function Runtime",
                "description": "Function-as-a-Service with cold start optimization, scaling",
                "detailed": True
            },
        ],
    },
    # AI/ML Practical - ai-ml domain
    "ai-ml": {
        "intermediate": [
            {
                "id": "ml-model-serving",
                "name": "ML Model Serving API",
                "description": "Model inference service with batching, versioning, A/B testing",
                "detailed": True
            },
        ],
        "advanced": [
            {
                "id": "llm-finetuning-pipeline",
                "name": "LLM Fine-tuning Pipeline",
                "description": "LoRA/QLoRA fine-tuning with dataset preparation, evaluation",
                "detailed": True
            },
            {
                "id": "mlops-platform",
                "name": "MLOps Platform",
                "description": "End-to-end ML lifecycle: training, versioning, deployment, monitoring",
                "detailed": True
            },
        ],
    },
    # Fintech/Enterprise - specialized domain
    "specialized": {
        "advanced": [
            {
                "id": "order-matching-engine",
                "name": "Order Matching Engine",
                "description": "Low-latency trading engine with order book, price-time priority",
                "detailed": True
            },
            {
                "id": "ledger-system",
                "name": "Double-entry Ledger System",
                "description": "Accounting system with journal entries, balance sheets, audit trail",
                "detailed": True
            },
        ],
    },
}

# Duplicates to remove (keep only one instance)
DUPLICATES_TO_REMOVE = {
    # Remove from software-engineering, keep in app-dev
    "software-engineering": ["distributed-tracing"],
    # Remove ci-pipeline (keep ci-cd-pipeline which is more comprehensive)
    # Actually let's keep ci-pipeline as beginner-friendly and ci-cd-pipeline as more advanced
}

# Projects to consolidate (remove one, rename other if needed)
CONSOLIDATE = {
    # realtime-chat and chat-app are similar - keep chat-app (intermediate), remove realtime-chat (advanced)
    "app-dev": {
        "advanced": ["realtime-chat"],  # Remove this, chat-app covers it
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    changes = []

    # 1. Remove duplicates
    for domain in data['domains']:
        domain_id = domain['id']

        # Remove specified duplicates
        if domain_id in DUPLICATES_TO_REMOVE:
            for level_name, level_projects in domain.get('projects', {}).items():
                if isinstance(level_projects, list):
                    original_len = len(level_projects)
                    domain['projects'][level_name] = [
                        p for p in level_projects
                        if p['id'] not in DUPLICATES_TO_REMOVE[domain_id]
                    ]
                    removed = original_len - len(domain['projects'][level_name])
                    if removed > 0:
                        changes.append(f"Removed {removed} duplicate(s) from {domain_id}/{level_name}")

        # Consolidate similar projects
        if domain_id in CONSOLIDATE:
            for level_name, ids_to_remove in CONSOLIDATE[domain_id].items():
                if level_name in domain.get('projects', {}):
                    level_projects = domain['projects'][level_name]
                    if isinstance(level_projects, list):
                        original_len = len(level_projects)
                        domain['projects'][level_name] = [
                            p for p in level_projects
                            if p['id'] not in ids_to_remove
                        ]
                        removed = original_len - len(domain['projects'][level_name])
                        if removed > 0:
                            changes.append(f"Consolidated: removed {ids_to_remove} from {domain_id}/{level_name}")

    # 2. Add new projects
    for domain in data['domains']:
        domain_id = domain['id']

        if domain_id in NEW_PROJECTS:
            projects_to_add = NEW_PROJECTS[domain_id]

            for level, new_projects in projects_to_add.items():
                if level not in domain.get('projects', {}):
                    domain['projects'][level] = []

                existing_ids = {p['id'] for p in domain['projects'][level]}

                for project in new_projects:
                    if project['id'] not in existing_ids:
                        domain['projects'][level].append(project)
                        changes.append(f"Added: {project['id']} to {domain_id}/{level}")
                    else:
                        changes.append(f"Skipped (exists): {project['id']}")

    # Save
    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print("Changes made:")
    for change in changes:
        print(f"  - {change}")
    print(f"\nTotal changes: {len(changes)}")


if __name__ == "__main__":
    main()

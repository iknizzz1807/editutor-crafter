#!/usr/bin/env python3
"""Add Data Storage projects to the curriculum."""

import yaml
from pathlib import Path

# Data Storage Projects
data_storage_projects = {
    "time-series-db": {
        "name": "Time-Series Database",
        "description": "Build a specialized database optimized for time-stamped data with compression, downsampling, and efficient range queries - essential for metrics, IoT, and financial data",
        "category": "Data Storage",
        "difficulty": "advanced",
        "estimated_hours": 180,
        "skills": [
            "Time-series data modeling",
            "Columnar storage",
            "Delta encoding and compression",
            "Write-ahead logging",
            "Retention policies",
            "Continuous queries"
        ],
        "prerequisites": [
            "database-engine",
            "distributed-cache"
        ],
        "learning_outcomes": [
            "Understand time-series data characteristics and access patterns",
            "Implement columnar storage with compression algorithms",
            "Build efficient time-range query execution",
            "Design downsampling and aggregation pipelines",
            "Create retention and compaction policies"
        ],
        "milestones": [
            {
                "name": "Storage Engine",
                "description": "Time-series optimized storage with compression",
                "skills": ["Columnar storage", "Delta encoding", "Run-length encoding"],
                "deliverables": [
                    "Time-structured merge tree (TSM) implementation",
                    "Delta-of-delta timestamp compression",
                    "Gorilla float compression algorithm",
                    "Dictionary encoding for tags/labels",
                    "Block-based storage with index",
                    "Memory-mapped file access"
                ]
            },
            {
                "name": "Write Path",
                "description": "High-throughput ingestion with buffering",
                "skills": ["Write batching", "WAL", "Memory management"],
                "deliverables": [
                    "Write-ahead log for durability",
                    "In-memory buffer (memtable) for writes",
                    "Batch point ingestion API",
                    "Out-of-order write handling",
                    "Series cardinality tracking",
                    "Backpressure mechanisms"
                ]
            },
            {
                "name": "Query Engine",
                "description": "Efficient time-range queries and aggregations",
                "skills": ["Query planning", "Aggregations", "Downsampling"],
                "deliverables": [
                    "Time-range predicate pushdown",
                    "Tag-based filtering and indexing",
                    "Built-in aggregation functions (sum, avg, min, max, count)",
                    "Windowed aggregations (tumbling, sliding)",
                    "GROUP BY time buckets",
                    "Last/first value queries optimization"
                ]
            },
            {
                "name": "Retention & Compaction",
                "description": "Automatic data lifecycle management",
                "skills": ["Retention policies", "Compaction", "Downsampling"],
                "deliverables": [
                    "TTL-based retention policies",
                    "Automatic data expiration and deletion",
                    "Background compaction process",
                    "Level-based compaction strategy",
                    "Continuous downsampling queries",
                    "Rollup aggregation storage"
                ]
            },
            {
                "name": "Query Language & API",
                "description": "Expressive query interface for time-series",
                "skills": ["Query parsing", "API design", "PromQL/InfluxQL"],
                "deliverables": [
                    "SQL-like query language with time extensions",
                    "Flux-style functional query pipeline",
                    "HTTP write API (Line Protocol compatible)",
                    "Query API with multiple output formats",
                    "Prometheus remote read/write API",
                    "Grafana data source compatibility"
                ]
            }
        ]
    },
    "graph-db": {
        "name": "Graph Database",
        "description": "Build a graph database with native graph storage, traversal algorithms, and query language - fundamental for social networks, recommendations, and knowledge graphs",
        "category": "Data Storage",
        "difficulty": "advanced",
        "estimated_hours": 200,
        "skills": [
            "Graph data modeling",
            "Index-free adjacency",
            "Graph traversal algorithms",
            "Query optimization",
            "Pattern matching",
            "Graph partitioning"
        ],
        "prerequisites": [
            "database-engine",
            "btree-implementation"
        ],
        "learning_outcomes": [
            "Understand graph storage models and their trade-offs",
            "Implement index-free adjacency for O(1) traversals",
            "Build graph traversal and pathfinding algorithms",
            "Design a graph query language parser and executor",
            "Create efficient pattern matching for subgraph queries"
        ],
        "milestones": [
            {
                "name": "Graph Storage Engine",
                "description": "Native graph storage with index-free adjacency",
                "skills": ["Node/edge storage", "Property storage", "Adjacency lists"],
                "deliverables": [
                    "Node store with fixed-size records",
                    "Relationship store with double-linked lists",
                    "Property store with dynamic records",
                    "Label and relationship type indexes",
                    "Index-free adjacency implementation",
                    "Transaction log and recovery"
                ]
            },
            {
                "name": "Graph Traversal",
                "description": "Efficient graph exploration algorithms",
                "skills": ["BFS/DFS", "Pathfinding", "Pattern matching"],
                "deliverables": [
                    "Breadth-first and depth-first traversal",
                    "Shortest path (Dijkstra, A*)",
                    "All paths between nodes",
                    "Variable-length path patterns",
                    "Bidirectional search optimization",
                    "Traversal result streaming"
                ]
            },
            {
                "name": "Query Language (Cypher-like)",
                "description": "Declarative graph query language",
                "skills": ["Query parsing", "AST", "Pattern matching"],
                "deliverables": [
                    "MATCH clause for pattern specification",
                    "WHERE clause for filtering",
                    "CREATE/MERGE for graph mutations",
                    "RETURN with aggregations",
                    "WITH clause for query chaining",
                    "OPTIONAL MATCH for outer joins"
                ]
            },
            {
                "name": "Query Optimization",
                "description": "Cost-based query planning for graphs",
                "skills": ["Query planning", "Statistics", "Join ordering"],
                "deliverables": [
                    "Pattern matching query planner",
                    "Cardinality estimation for patterns",
                    "Join order optimization",
                    "Index selection for label scans",
                    "Eager vs lazy evaluation",
                    "Query plan caching"
                ]
            },
            {
                "name": "Graph Algorithms",
                "description": "Built-in graph analytics algorithms",
                "skills": ["Centrality", "Community detection", "Similarity"],
                "deliverables": [
                    "PageRank algorithm",
                    "Betweenness/closeness centrality",
                    "Community detection (Louvain, Label Propagation)",
                    "Node similarity (Jaccard, Cosine)",
                    "Triangle counting and clustering coefficient",
                    "Streaming/approximate algorithms for scale"
                ]
            },
            {
                "name": "Full-text & Spatial",
                "description": "Extended indexing capabilities",
                "skills": ["Full-text search", "Spatial indexing", "Composite indexes"],
                "deliverables": [
                    "Full-text index on node properties",
                    "Fuzzy matching and relevance scoring",
                    "Spatial index (R-tree) for location data",
                    "Distance and bounding box queries",
                    "Composite property indexes",
                    "Index-backed ORDER BY"
                ]
            }
        ]
    }
}

def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    if 'projects' not in data:
        data['projects'] = {}

    for key, project in data_storage_projects.items():
        data['projects'][key] = project
        print(f"Added: {key} - {project['name']}")

    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nAdded {len(data_storage_projects)} Data Storage projects")

if __name__ == "__main__":
    main()

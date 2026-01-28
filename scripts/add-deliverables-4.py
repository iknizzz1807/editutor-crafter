#!/usr/bin/env python3
"""Add deliverables to remaining 53 milestones with mismatched names (part 4)."""

import yaml
from pathlib import Path

# Deliverables keyed by exact milestone names in the YAML
DELIVERABLES = {
    "build-kafka": {
        "Topic and Partitions": [
            "Topic creation and configuration",
            "Partition management",
            "Partition assignment to brokers",
            "Topic metadata storage"
        ],
        "Producer": [
            "Message serialization",
            "Partition selection",
            "Batch accumulation",
            "Delivery acknowledgment"
        ]
    },
    "build-raft": {
        "Safety Properties": [
            "Election safety (one leader per term)",
            "Log matching property",
            "Leader completeness",
            "State machine safety proof"
        ],
        "Cluster Membership Changes": [
            "AddServer operation",
            "RemoveServer operation",
            "Joint consensus (optional)",
            "Configuration change safety"
        ]
    },
    "build-react": {
        "Reconciliation (Diffing)": [
            "Tree diff algorithm",
            "Key-based element matching",
            "Minimal DOM updates",
            "Component lifecycle during reconciliation"
        ],
        "Fiber Architecture": [
            "Fiber node structure",
            "Work-in-progress tree",
            "Interruptible rendering",
            "Priority-based scheduling"
        ]
    },
    "build-spreadsheet": {
        "Grid & Cell Rendering": [
            "Grid rendering with rows/columns",
            "Cell editing UI",
            "Selection highlighting",
            "Scrolling support"
        ],
        "Dependency Graph & Recalculation": [
            "Cell dependency tracking",
            "Topological sort for evaluation",
            "Circular reference detection",
            "Incremental recalculation"
        ],
        "Advanced Features": [
            "Copy/paste support",
            "Multiple sheets",
            "Cell formatting",
            "Undo/redo"
        ]
    },
    "build-tls": {
        "Key Exchange": [
            "ECDHE parameters",
            "Shared secret computation",
            "Key derivation (HKDF)",
            "Finished message verification"
        ],
        "Certificate Verification": [
            "Certificate chain parsing",
            "Signature verification",
            "Expiration checking",
            "Common name/SAN validation"
        ]
    },
    "build-web-framework": {
        "Routing": [
            "Route registration",
            "Path parameter extraction",
            "Method-based dispatch",
            "Route matching"
        ],
        "Request/Response Enhancement": [
            "Request body parsing",
            "Response builder",
            "Cookie handling",
            "File upload support"
        ]
    },
    "bytecode-compiler": {
        "Expression Compilation": [
            "Literal emission",
            "Binary operations",
            "Unary operations",
            "Precedence handling"
        ],
        "Variables and Assignment": [
            "Local variable slots",
            "Variable declaration",
            "Assignment compilation",
            "Scope tracking"
        ],
        "Functions": [
            "Function compilation",
            "Argument handling",
            "Return instruction",
            "Closure upvalues"
        ]
    },
    "chat-app": {
        "WebSocket Server Setup": [
            "WebSocket server implementation",
            "Connection handling",
            "Message routing",
            "Connection state tracking"
        ],
        "Message Broadcasting": [
            "Room-based broadcasting",
            "Message format (sender, content, timestamp)",
            "User join/leave events",
            "Typing indicators"
        ],
        "User Authentication & Persistence": [
            "User login/registration",
            "Session management",
            "Message persistence",
            "Chat history loading"
        ]
    },
    "diff-tool": {
        "Line Tokenization": [
            "File line splitting",
            "Line normalization",
            "Empty line handling",
            "Encoding detection"
        ],
        "CLI and Color Output": [
            "Argument parsing",
            "ANSI color output",
            "Context line count",
            "Side-by-side option"
        ]
    },
    "json-parser": {
        "Tokenizer": [
            "Token types (string, number, bool, null)",
            "String with escape sequences",
            "Number parsing (int, float)",
            "Delimiter tokens"
        ],
        "Error Handling & Edge Cases": [
            "Syntax error messages",
            "Trailing comma handling",
            "Unicode escape sequences",
            "Nested depth limits"
        ]
    },
    "mini-shell": {
        "Job Control": [
            "Foreground/background jobs",
            "SIGTSTP handling (Ctrl+Z)",
            "fg and bg commands",
            "Job table management"
        ]
    },
    "query-optimizer": {
        "Query Plan Representation": [
            "Plan tree structure",
            "Physical operators",
            "Plan cost annotation",
            "Plan visualization"
        ],
        "Join Ordering": [
            "Dynamic programming approach",
            "Cross product elimination",
            "Join cost estimation",
            "Bushy vs left-deep plans"
        ]
    },
    "rate-limiter": {
        "Token Bucket Implementation": [
            "Bucket capacity and rate",
            "Token consumption",
            "Token refill logic",
            "Burst handling"
        ],
        "Per-Client Rate Limiting": [
            "Client identification",
            "Per-client bucket storage",
            "Memory-efficient storage",
            "Client cleanup"
        ],
        "HTTP Middleware Integration": [
            "Middleware function",
            "Rate limit headers",
            "429 Too Many Requests",
            "Configurable limits"
        ]
    },
    "service-discovery": {
        "Service Registry": [
            "Registration API",
            "Service metadata storage",
            "TTL-based deregistration",
            "Health check integration"
        ],
        "HTTP API": [
            "Service lookup endpoint",
            "Health status endpoint",
            "Registration endpoint",
            "Query filtering"
        ]
    },
    "type-checker": {
        "Basic Type Checking": [
            "Type annotation parsing",
            "Expression type inference",
            "Type compatibility checking",
            "Error reporting"
        ],
        "Polymorphism": [
            "Generic type parameters",
            "Type variable instantiation",
            "Constraint solving",
            "Generic function calls"
        ]
    },
    "wal-impl": {
        "Log Record Format": [
            "Record header (LSN, type, length)",
            "Record types (redo, undo, checkpoint)",
            "CRC for integrity",
            "Record serialization"
        ],
        "Log Writer": [
            "Sequential log writing",
            "Force write (fsync)",
            "Buffer management",
            "Log segment rotation"
        ],
        "Crash Recovery": [
            "Log scanning from checkpoint",
            "Redo pass (replay committed)",
            "Undo pass (rollback uncommitted)",
            "Recovery completion"
        ]
    },
    "recommendation-engine": {
        "Matrix Factorization": [
            "User-item matrix construction",
            "SVD or ALS decomposition",
            "Latent factor learning",
            "Rating prediction"
        ],
        "Content-Based Filtering": [
            "Item feature extraction",
            "User preference modeling",
            "Similarity computation",
            "Hybrid scoring"
        ],
        "Production System": [
            "Recommendation API",
            "Real-time scoring",
            "A/B testing support",
            "Feedback integration"
        ]
    },
    "websocket-server": {
        "HTTP Upgrade Handshake": [
            "Upgrade request detection",
            "Sec-WebSocket-Key processing",
            "Accept header generation",
            "Protocol upgrade response"
        ],
        "Frame Parsing": [
            "Frame header parsing",
            "Payload masking/unmasking",
            "Frame type dispatch",
            "Fragmented message assembly"
        ],
        "Ping/Pong Heartbeat": [
            "Ping frame sending",
            "Pong frame response",
            "Connection timeout detection",
            "Keep-alive interval"
        ]
    },
    "collaborative-editor": {
        "Operation-based CRDT": [
            "Insert/delete operation types",
            "Causal ordering",
            "Convergence guarantees",
            "Operation merging"
        ],
        "Cursor Presence": [
            "Cursor position sharing",
            "User color assignment",
            "Remote cursor rendering",
            "Cursor position transformation"
        ],
        "Undo/Redo with Collaboration": [
            "Local undo stack",
            "Selective undo (own operations)",
            "Redo after undo",
            "Undo against concurrent edits"
        ]
    },
    "alerting-system": {
        "Alert Rule Evaluation": [
            "Rule expression parsing",
            "Periodic evaluation loop",
            "Threshold comparison",
            "Duration-based firing"
        ],
        "Alert Grouping": [
            "Group by labels",
            "Aggregated notifications",
            "Group wait/interval configuration",
            "Group resolution"
        ],
        "Silencing & Inhibition": [
            "Silence matcher creation",
            "Time-based silencing",
            "Inhibition rules",
            "Priority-based suppression"
        ]
    },
    "apm-system": {
        "Service Map": [
            "Service dependency extraction",
            "Call graph construction",
            "Map visualization",
            "Real-time updates"
        ],
        "Trace Sampling": [
            "Head-based sampling",
            "Tail-based sampling",
            "Sampling rate configuration",
            "Consistent sampling"
        ]
    },
    "build-vpn": {
        "Routing and NAT": [
            "Route table manipulation",
            "Default gateway setup",
            "NAT for client traffic",
            "Split tunneling configuration"
        ]
    },
    "vulnerability-scanner": {
        "Service Fingerprinting": [
            "Banner grabbing",
            "Service version detection",
            "OS fingerprinting",
            "Technology stack identification"
        ],
        "Report Generation": [
            "Vulnerability report structure",
            "Severity classification (CVSS)",
            "Remediation recommendations",
            "Export formats (JSON, HTML)"
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    updated = 0

    for proj_id, proj in data.get('expert_projects', {}).items():
        if proj_id not in DELIVERABLES:
            continue

        proj_deliverables = DELIVERABLES[proj_id]

        for milestone in proj.get('milestones', []):
            name = milestone.get('name')
            if name in proj_deliverables and not milestone.get('deliverables'):
                milestone['deliverables'] = proj_deliverables[name]
                updated += 1

    # Save
    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"Updated: {updated} milestones with deliverables")

    # Final check
    missing = 0
    for proj_id, proj in data.get('expert_projects', {}).items():
        for m in proj.get('milestones', []):
            if not m.get('deliverables'):
                missing += 1
                print(f"  Still missing: {proj_id}: {m.get('name')}")

    if missing == 0:
        print("\nALL MILESTONES NOW HAVE DELIVERABLES!")
    else:
        print(f"\nStill missing: {missing}")


if __name__ == "__main__":
    main()

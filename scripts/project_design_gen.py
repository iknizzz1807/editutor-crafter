#!/usr/bin/env python3
"""
Project Design Generator - Technical Design Document Generator

This script generates detailed technical design documents for software projects,
focusing on proper microservices architecture, database-per-service pattern,
and production-ready infrastructure.

Unlike langgraph_arch_gen.py (which generates educational content),
this generates Technical Design Documents for architects and engineers.

Usage:
    python project_design_gen.py --prompt "URL Shortener with analytics"
    python project_design_gen.py --project url-shortener --output ./designs
"""

import os
import json
import re
import subprocess
import yaml
import argparse
from typing import Annotated, Dict, Any, Optional, List
from pathlib import Path
from string import Template

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / ".." / "data"
PROJECTS_DATA_DIR = DATA_DIR / "projects_data"
DEFAULT_OUTPUT = DATA_DIR / "project-designs"
D2_EXAMPLES_DIR = SCRIPT_DIR / ".." / "d2_examples"

# --- LLM SETUP ---
LLM_PROVIDER = None
LLM = None


def init_llm():
    """Initialize LLM provider."""
    global LLM_PROVIDER, LLM

    USE_CLAUDE_CLI = os.getenv("USE_CLAUDE_CLI", "false").lower() == "true"
    USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")

    if USE_CLAUDE_CLI:
        LLM_PROVIDER = "claude-cli"
        print(f">>> Provider: Claude CLI ({CLAUDE_MODEL})")
    elif USE_GEMINI:
        LLM = ChatOpenAI(
            base_url="http://127.0.0.1:7999/v1",
            api_key=SecretStr(os.getenv("GEMINI_PROXY_API_KEY", "mythong2005")),
            model="gemini_cli/gemini-3-flash-preview",
            temperature=0.7,
            max_completion_tokens=32000,
        )
        LLM_PROVIDER = "gemini"
        print(">>> Provider: Gemini (Local Proxy)")


def invoke_llm(messages: List, max_retries: int = 3) -> str:
    """Invoke LLM with retry logic."""
    global LLM_PROVIDER, LLM

    if LLM_PROVIDER == "claude-cli":
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
            elif isinstance(msg, HumanMessage):
                user_prompt = str(msg.content)

        cmd = ["claude", "-p", "--model", os.getenv("CLAUDE_MODEL", "sonnet"),
               "--dangerously-skip-permissions", "--tools", ""]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        result = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Strip ANSI codes
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_escape.sub("", result.stdout)
        raise Exception(f"Claude CLI failed: {result.stderr}")

    # Gemini/OpenAI path
    for attempt in range(max_retries):
        try:
            result = LLM.invoke(messages, timeout=600)
            return str(result.content)
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            raise e

    raise Exception("LLM invocation failed")


def extract_json(text: str) -> Optional[Any]:
    """Extract JSON from LLM response."""
    try:
        # Try to find JSON in code blocks first
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            return json.loads(match.group(1).strip())

        # Try to find raw JSON
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
    return None


# --- STATE ---
class DesignState(Dict[str, Any]):
    """State for the design generation pipeline."""
    project_id: str
    project_name: str
    project_description: str
    design_spec: Dict[str, Any]
    services: List[Dict[str, Any]]
    databases: List[Dict[str, Any]]
    apis: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    infrastructure: Dict[str, Any]
    antipatterns: List[str]
    accumulated_md: str
    diagrams_to_generate: List[Dict[str, Any]]
    current_diagram: Optional[Dict[str, Any]]
    current_diagram_code: Optional[str]
    diagram_attempt: int
    last_error: Optional[str]
    status: str


# --- DESIGN SPEC SYSTEM PROMPT ---
DESIGN_ARCHITECT_PROMPT = """You are a Principal Software Architect specializing in microservices architecture.

Your task is to design production-ready systems following these PRINCIPLES:

## MANDATORY PATTERNS (Must Include)
1. **Database per Service** - Each service has its own database. NO shared databases.
2. **API Gateway** - Single entry point for all external requests
3. **Service Discovery** - Services find each other dynamically (Docker DNS, Consul, K8s)
4. **Circuit Breaker** - Handle service failures gracefully
5. **Async Communication** - Use message brokers (RabbitMQ, Kafka) for decoupling
6. **Distributed Tracing** - Request tracing across services (Jaeger, Zipkin)
7. **Containerization** - Docker + Docker Compose (or Kubernetes)

## ANTI-PATTERNS (Must Avoid)
1. **Shared Database** - NEVER share databases between services
2. **Distributed Monolith** - NO long synchronous chains (A→B→C→D)
3. **God Service** - Services must have single responsibility
4. **Nano Services** - Don't split too small without reason
5. **Chatty Services** - Batch operations, avoid excessive network calls

## OUTPUT FORMAT
You MUST output valid JSON with this exact structure:

```json
{
  "project_name": "...",
  "project_description": "...",
  "services": [
    {
      "name": "service-name",
      "responsibility": "What this service does",
      "database": {
        "type": "postgresql|mongodb|redis|...",
        "schema": "Brief description of main tables/collections"
      },
      "endpoints": [
        {"method": "GET|POST|PUT|DELETE", "path": "/api/...", "description": "..."}
      ],
      "dependencies": ["other-service-1", "other-service-2"],
      "scales_for": "Why this service needs independent scaling"
    }
  ],
  "api_gateway": {
    "routes": [
      {"path": "/api/...", "service": "service-name", "methods": ["GET", "POST"]}
    ],
    "rate_limiting": "Strategy description",
    "authentication": "How auth is handled"
  },
  "async_events": [
    {
      "name": "event.name",
      "producer": "service-name",
      "consumers": ["service-1", "service-2"],
      "payload": {"field": "type"}
    }
  ],
  "infrastructure": {
    "message_broker": "rabbitmq|kafka",
    "service_discovery": "docker-dns|consul|kubernetes",
    "tracing": "jaeger|zipkin",
    "logging": "elk|loki",
    "monitoring": "prometheus"
  },
  "diagrams_needed": [
    {"id": "arch-overview", "title": "Architecture Overview", "type": "architecture"},
    {"id": "data-flow", "title": "Data Flow", "type": "sequence"},
    {"id": "db-schema", "title": "Database Schema", "type": "er-diagram"}
  ]
}
```

## DESIGN PHILOSOPHY
- **Simple is better** - Start minimal, add complexity only when needed
- **Deep not wide** - Fewer services with richer functionality
- **Practical** - Every service must have a clear reason to exist
- **Testable** - Each service can be tested independently
"""


# --- NODES ---
def architect_node(state: DesignState) -> Dict[str, Any]:
    """Generate the high-level system design."""
    print(f"  [Architect] Designing {state['project_name']}...")

    prompt = f"""{DESIGN_ARCHITECT_PROMPT}

--- PROJECT ---
Name: {state['project_name']}
Description: {state['project_description']}

Design a production-ready microservices architecture for this system.
Focus on: simplicity, clear boundaries, practical decomposition.

Output ONLY the JSON design specification."""

    response = invoke_llm([
        SystemMessage(content="You are a Principal Software Architect. Output ONLY valid JSON."),
        HumanMessage(content=prompt)
    ])

    design_spec = extract_json(response)
    if not design_spec:
        raise ValueError("Failed to parse design specification from LLM response")

    print(f"  [Architect] Design generated: {len(design_spec.get('services', []))} services")

    return {
        "design_spec": design_spec,
        "services": design_spec.get("services", []),
        "diagrams_to_generate": design_spec.get("diagrams_needed", []),
        "status": "writing"
    }


def service_detail_node(state: DesignState) -> Dict[str, Any]:
    """Generate detailed service specifications."""
    print(f"  [Service Detail] Writing service specifications...")

    services_md = "\n\n## Services\n\n"

    for svc in state.get("services", []):
        svc_name = svc.get("name", "unknown")
        print(f"    - {svc_name}")

        prompt = f"""Write a detailed technical specification for this microservice:

Service: {svc_name}
Responsibility: {svc.get('responsibility', 'N/A')}
Database: {svc.get('database', {}).get('type', 'N/A')} - {svc.get('database', {}).get('schema', 'N/A')}
Dependencies: {svc.get('dependencies', [])}

Include:
1. **API Endpoints** - Full REST API spec with request/response schemas
2. **Database Schema** - Detailed table/collection definitions
3. **Business Logic** - Key algorithms and workflows
4. **Error Handling** - Error codes and handling strategies
5. **Configuration** - Environment variables and config

Format as Markdown. Be specific and implementation-ready."""

        response = invoke_llm([HumanMessage(content=prompt)])
        services_md += f"### {svc_name}\n\n{response}\n\n"

    return {
        "accumulated_md": state.get("accumulated_md", "") + services_md,
        "status": "apis"
    }


def api_contracts_node(state: DesignState) -> Dict[str, Any]:
    """Generate API contracts (OpenAPI-style)."""
    print(f"  [API Contracts] Writing API specifications...")

    gateway = state.get("design_spec", {}).get("api_gateway", {})

    prompt = f"""Generate detailed API contracts for this system:

API Gateway Routes:
{json.dumps(gateway.get('routes', []), indent=2)}

Services:
{json.dumps([s['name'] for s in state.get('services', [])], indent=2)}

Output:
1. REST API endpoints with full request/response schemas
2. Authentication/Authorization flow
3. Rate limiting strategy
4. Error response format

Format as Markdown with code examples. Be specific enough for frontend developers to implement against."""

    response = invoke_llm([HumanMessage(content=prompt)])

    api_md = f"\n\n## API Contracts\n\n{response}"

    return {
        "accumulated_md": state.get("accumulated_md", "") + api_md,
        "status": "events"
    }


def events_node(state: DesignState) -> Dict[str, Any]:
    """Generate event contracts for async communication."""
    print(f"  [Events] Writing event specifications...")

    events = state.get("design_spec", {}).get("async_events", [])

    if not events:
        return {"status": "infrastructure"}

    prompt = f"""Generate detailed event contracts for async communication:

Events:
{json.dumps(events, indent=2)}

For each event, specify:
1. **Event Schema** - Exact JSON structure
2. **Producer Logic** - When and why this event is published
3. **Consumer Logic** - How each consumer handles the event
4. **Error Handling** - Dead letter queues, retries
5. **Ordering Guarantees** - Per-partition ordering, idempotency

Format as Markdown. Include JSON schema examples."""

    response = invoke_llm([HumanMessage(content=prompt)])

    events_md = f"\n\n## Event Contracts\n\n{response}"

    return {
        "accumulated_md": state.get("accumulated_md", "") + events_md,
        "status": "infrastructure"
    }


def infrastructure_node(state: DesignState) -> Dict[str, Any]:
    """Generate infrastructure specifications."""
    print(f"  [Infrastructure] Writing infrastructure specs...")

    infra = state.get("design_spec", {}).get("infrastructure", {})
    services = state.get("services", [])

    prompt = f"""Generate production infrastructure specification:

Infrastructure Requirements:
{json.dumps(infra, indent=2)}

Services to deploy:
{json.dumps([s['name'] for s in services], indent=2)}

Include:
1. **Docker Compose** - Full docker-compose.yml for local development
2. **Service Configuration** - Environment variables per service
3. **Network Topology** - How services communicate
4. **Volume Mounts** - Persistent storage
5. **Health Checks** - Liveness and readiness probes
6. **Resource Limits** - CPU and memory constraints

Format as Markdown with code blocks. Include actual docker-compose.yml."""

    response = invoke_llm([HumanMessage(content=prompt)])

    infra_md = f"\n\n## Infrastructure\n\n{response}"

    return {
        "accumulated_md": state.get("accumulated_md", "") + infra_md,
        "status": "antipatterns"
    }


def antipatterns_node(state: DesignState) -> Dict[str, Any]:
    """Document potential anti-patterns and how to avoid them."""
    print(f"  [Anti-patterns] Writing anti-pattern analysis...")

    prompt = f"""Analyze this system design for potential anti-patterns:

Services:
{json.dumps(state.get('services', []), indent=2)}

Events:
{json.dumps(state.get('design_spec', {}).get('async_events', []), indent=2)}

For each potential anti-pattern:
1. **What could go wrong** - Describe the anti-pattern
2. **How to detect it** - Warning signs
3. **How to prevent it** - Best practices
4. **Refactoring strategies** - If already present

Focus on: Distributed Monolith, Shared Database, God Service, Nano Services, Chatty Services.

Format as Markdown checklist."""

    response = invoke_llm([HumanMessage(content=prompt)])

    anti_md = f"\n\n## Anti-Pattern Checklist\n\n{response}"

    return {
        "accumulated_md": state.get("accumulated_md", "") + anti_md,
        "status": "visualizing"
    }


def visualizer_node(state: DesignState) -> Dict[str, Any]:
    """Generate D2 diagram code."""
    if not state.get("diagrams_to_generate"):
        return {"status": "done"}

    diag = state["diagrams_to_generate"][0]
    diag_id = diag.get("id", "unknown")
    diag_title = diag.get("title", "Diagram")
    diag_type = diag.get("type", "architecture")

    print(f"  [Visualizer] Drawing: {diag_title}...")

    # Build context from design spec
    design_context = json.dumps(state.get("design_spec", {}), indent=2)

    prompt = f"""Generate D2 diagram code for: {diag_title}

Type: {diag_type}

Design Context:
{design_context}

D2 Code Guidelines:
- Use direction: right or down for flow
- Use proper container syntax: service_name: {{ ... }}
- Label connections with: a -> b: "label"
- For architecture: show services, databases, message broker
- For sequence: show request/response flow
- For ER: show tables with columns

Output ONLY valid D2 code, no explanations."""

    response = invoke_llm([
        SystemMessage(content="Output ONLY raw D2 code. No markdown, no explanations."),
        HumanMessage(content=prompt)
    ])

    # Clean up the response
    code = re.sub(r'```d2\n?|```\n?', '', response).strip()

    return {
        "current_diagram": diag,
        "current_diagram_code": code,
        "diagram_attempt": 1,
        "status": "compiling"
    }


def compiler_node(state: DesignState) -> Dict[str, Any]:
    """Compile D2 diagram to SVG."""
    diag = state.get("current_diagram")
    code = state.get("current_diagram_code")

    if not diag or not code:
        return {"status": "visualizing"}

    proj_dir = DEFAULT_OUTPUT / state["project_id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "diagrams").mkdir(exist_ok=True)

    d2_path = proj_dir / "diagrams" / f"{diag['id']}.d2"
    d2_path.write_text(code)

    result = subprocess.run(
        ["d2", "--layout=elk", str(d2_path), str(d2_path.with_suffix(".svg"))],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"    ✓ Diagram compiled: {diag['id']}")

        # Add to markdown
        link = f"\n\n![{diag.get('title')}](./diagrams/{diag['id']}.svg)\n"

        return {
            "accumulated_md": state.get("accumulated_md", "") + link,
            "diagrams_to_generate": state["diagrams_to_generate"][1:],
            "current_diagram": None,
            "current_diagram_code": None,
            "diagram_attempt": 0,
            "status": "visualizing"
        }
    else:
        print(f"    ✗ Diagram failed: {result.stderr[:100]}")

        if state.get("diagram_attempt", 0) >= 3:
            # Skip after 3 attempts
            return {
                "diagrams_to_generate": state["diagrams_to_generate"][1:],
                "current_diagram": None,
                "current_diagram_code": None,
                "diagram_attempt": 0,
                "status": "visualizing"
            }

        return {
            "last_error": result.stderr,
            "diagram_attempt": state.get("diagram_attempt", 0) + 1,
            "status": "visualizing"
        }


def route_visualizer(state: DesignState) -> str:
    """Route after visualization."""
    if state.get("diagrams_to_generate"):
        return "visualize"
    return "done"


def route_compiler(state: DesignState) -> str:
    """Route after compilation."""
    if state.get("last_error") and state.get("diagram_attempt", 0) < 3:
        return "retry"
    if state.get("diagrams_to_generate"):
        return "next"
    return "done"


# --- GRAPH ---
def build_graph():
    """Build the LangGraph pipeline."""
    workflow = StateGraph(DesignState)

    workflow.add_node("architect", architect_node)
    workflow.add_node("service_detail", service_detail_node)
    workflow.add_node("api_contracts", api_contracts_node)
    workflow.add_node("events", events_node)
    workflow.add_node("infrastructure", infrastructure_node)
    workflow.add_node("antipatterns", antipatterns_node)
    workflow.add_node("visualizer", visualizer_node)
    workflow.add_node("compiler", compiler_node)

    workflow.add_edge(START, "architect")
    workflow.add_edge("architect", "service_detail")
    workflow.add_edge("service_detail", "api_contracts")
    workflow.add_edge("api_contracts", "events")
    workflow.add_edge("events", "infrastructure")
    workflow.add_edge("infrastructure", "antipatterns")
    workflow.add_edge("antipatterns", "visualizer")

    workflow.add_conditional_edges("visualizer", route_visualizer, {
        "visualize": "visualizer",
        "done": END
    })

    workflow.add_conditional_edges("compiler", route_compiler, {
        "retry": "visualizer",
        "next": "visualizer",
        "done": END
    })

    workflow.add_edge("visualizer", "compiler")

    return workflow.compile()


def generate_design(project_id: str, project_name: str, project_description: str, output_dir: Path):
    """Generate technical design document."""
    global DEFAULT_OUTPUT
    DEFAULT_OUTPUT = output_dir

    print(f"\n{'='*60}")
    print(f"PROJECT DESIGN GENERATOR")
    print(f"{'='*60}")
    print(f"Project: {project_name}")
    print(f"ID: {project_id}")
    print(f"{'='*60}\n")

    initial_state = {
        "project_id": project_id,
        "project_name": project_name,
        "project_description": project_description,
        "design_spec": {},
        "services": [],
        "databases": [],
        "apis": [],
        "events": [],
        "infrastructure": {},
        "antipatterns": [],
        "accumulated_md": f"# {project_name}\n\n## Overview\n\n{project_description}\n",
        "diagrams_to_generate": [],
        "current_diagram": None,
        "current_diagram_code": None,
        "diagram_attempt": 0,
        "last_error": None,
        "status": "starting"
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # Write output
    proj_dir = output_dir / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Write markdown
    md_path = proj_dir / "TECHNICAL_DESIGN.md"
    md_path.write_text(final_state.get("accumulated_md", ""))

    # Write design spec JSON
    spec_path = proj_dir / "design_spec.json"
    spec_path.write_text(json.dumps(final_state.get("design_spec", {}), indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"✓ DESIGN COMPLETE")
    print(f"  Output: {proj_dir}")
    print(f"  - TECHNICAL_DESIGN.md")
    print(f"  - design_spec.json")
    print(f"  - diagrams/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Technical Design Documents")
    parser.add_argument("--project", help="Project ID from projects_data/")
    parser.add_argument("--prompt", help="Direct prompt (e.g., 'URL Shortener with analytics')")
    parser.add_argument("--name", help="Project name (used with --prompt)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--claude-cli", action="store_true", help="Use Claude CLI")

    args = parser.parse_args()

    # Set LLM provider
    if args.claude_cli:
        os.environ["USE_CLAUDE_CLI"] = "true"

    init_llm()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.project:
        # Load from YAML
        yaml_path = PROJECTS_DATA_DIR / f"{args.project}.yaml"
        if not yaml_path.exists():
            print(f"Error: Project '{args.project}' not found in {PROJECTS_DATA_DIR}")
            return

        with open(yaml_path) as f:
            meta = yaml.safe_load(f)

        project_id = meta.get("id", args.project)
        project_name = meta.get("name", args.project)
        project_description = meta.get("description", meta.get("essence", "No description"))

    elif args.prompt:
        # Use direct prompt
        project_id = args.prompt.lower().replace(" ", "-").replace("/", "-")
        project_name = args.name or args.prompt
        project_description = args.prompt

    else:
        parser.print_help()
        return

    generate_design(project_id, project_name, project_description, output_dir)


if __name__ == "__main__":
    main()

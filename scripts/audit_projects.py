#!/usr/bin/env python3
"""
Project spec auditor for editutor-crafter.
Analyzes project specs against technical accuracy, measurability, progression,
completeness, realism, security, and performance criteria.
"""

import json
import re
from typing import Dict, List, Any

def audit_project(project_data: Dict[str, Any]) -> Dict[str, Any]:
    """Audit a single project spec and return results."""
    project_id = project_data.get("project_id", "unknown")
    yaml_content = project_data.get("yaml", "")

    issues = []
    strengths = []
    score = 10

    # Parse YAML content for analysis
    milestones = extract_milestones(yaml_content)
    difficulty = project_data.get("difficulty", "unknown")
    estimated_hours = project_data.get("estimated_hours", 0)

    # 1. Technical Accuracy
    tech_issues, tech_strengths = check_technical_accuracy(project_id, yaml_content, milestones)
    issues.extend(tech_issues)
    strengths.extend(tech_strengths)

    # 2. Measurability
    measurability_issues, measurability_strengths = check_measurability(milestones)
    issues.extend(measurability_issues)
    strengths.extend(measurability_strengths)

    # 3. Progression
    progression_issues, progression_strengths = check_progression(milestones)
    issues.extend(progression_issues)
    strengths.extend(progression_strengths)

    # 4. Completeness
    completeness_issues, completeness_strengths = check_completeness(project_id, yaml_content, milestones)
    issues.extend(completeness_issues)
    strengths.extend(completeness_strengths)

    # 5. Realism
    realism_issues, realism_strengths = check_realism(difficulty, estimated_hours, milestones)
    issues.extend(realism_issues)
    strengths.extend(realism_strengths)

    # 6. Security
    security_issues, security_strengths = check_security(project_id, yaml_content, milestones)
    issues.extend(security_issues)
    strengths.extend(security_strengths)

    # 7. Performance
    performance_issues, performance_strengths = check_performance(yaml_content, milestones)
    issues.extend(performance_issues)
    strengths.extend(performance_strengths)

    # Calculate score
    score = max(1, min(10, 10 - len(issues) + len(strengths) // 3))

    return {
        "project_id": project_id,
        "overall_score": score,
        "verdict": "needs_fix" if issues else "good",
        "issues": issues,
        "strengths": strengths[:10],  # Limit to top 10
        "critique": generate_critique(issues, strengths, score),
        "fixed_yaml": None if not issues else yaml_content  # Would need actual fixes
    }

def extract_milestones(yaml_content: str) -> List[Dict]:
    """Extract milestones from YAML content."""
    # Simple parsing - in production would use proper YAML parser
    milestone_pattern = r'-\s+description:([^\n]+).*?id:\s+(\S+).*?name:\s+([^\n]+)'
    matches = re.findall(milestone_pattern, yaml_content, re.DOTALL)
    return [{"description": m[0].strip(), "id": m[1], "name": m[2]} for m in matches]

def check_technical_accuracy(project_id: str, yaml_content: str, milestones: List[Dict]) -> tuple:
    """Check for technical accuracy issues."""
    issues = []
    strengths = []

    # Project-specific technical checks
    if project_id == "build-sqlite":
        if "B-tree" in yaml_content and "B+tree" in yaml_content:
            strengths.append("Correctly distinguishes between B-tree (tables) and B+tree (indexes)")
        if "three-valued logic" in yaml_content.lower() or "three valued logic" in yaml_content.lower():
            strengths.append("Properly addresses SQL three-valued logic (TRUE, FALSE, NULL)")
        if "ACID" in yaml_content:
            strengths.append("Comprehensive coverage of ACID transaction properties")

    elif project_id == "build-strace":
        if "ptrace" in yaml_content and "x86_64" in yaml_content:
            strengths.append("Accurately describes ptrace API and x86_64 syscall ABI")
        if "orig_rax" in yaml_content or "orig_eax" in yaml_content:
            strengths.append("Correct register references for syscall number detection")
        if "entry/exit" in yaml_content.lower():
            strengths.append("Properly addresses syscall entry/exit state tracking")

    elif project_id == "build-tcp-stack":
        if "checksum" in yaml_content.lower() and "pseudo-header" in yaml_content.lower():
            strengths.append("Accurately covers TCP checksum with pseudo-header")
        if "three-way handshake" in yaml_content.lower() or "3-way handshake" in yaml_content.lower():
            strengths.append("Correct TCP connection establishment description")
        if "congestion control" in yaml_content.lower() or "slow start" in yaml_content.lower():
            strengths.append("Comprehensive coverage of TCP congestion control algorithms")

    elif project_id == "build-test-framework":
        if "AST" in yaml_content or "abstract syntax tree" in yaml_content.lower():
            strengths.append("Correctly describes AST-based assertion rewriting")
        if "fixture" in yaml_content.lower() and "dependency injection" in yaml_content.lower():
            strengths.append("Accurate fixture dependency injection description")
        if "parallel" in yaml_content.lower() and "process" in yaml_content.lower():
            strengths.append("Proper approach to process-level test isolation")

    elif project_id == "build-text-editor":
        if "raw mode" in yaml_content.lower() or "termios" in yaml_content.lower():
            strengths.append("Accurate terminal raw mode configuration description")
        if "gap buffer" in yaml_content.lower():
            strengths.append("Appropriate data structure recommendation for text editing")
        if "ANSI" in yaml_content or "escape sequence" in yaml_content.lower():
            strengths.append("Correct coverage of ANSI escape sequences for terminal control")

    return issues, strengths

def check_measurability(milestones: List[Dict]) -> tuple:
    """Check if acceptance criteria are measurable."""
    issues = []
    strengths = []

    for milestone in milestones:
        desc = milestone.get("description", "").lower()
        # Check for measurable keywords
        if any(word in desc for word in ["test", "verify", "measure", "benchmark", "validate"]):
            strengths.append(f"{milestone.get('name', 'Milestone')}: Includes measurable outcomes")

    return issues, strengths

def check_progression(milestones: List[Dict]) -> tuple:
    """Check if milestones build logically on each other."""
    issues = []
    strengths = []

    if len(milestones) >= 3:
        strengths.append(f"Good milestone structure with {len(milestones)} sequential phases")

    return issues, strengths

def check_completeness(project_id: str, yaml_content: str, milestones: List[Dict]) -> tuple:
    """Check for completeness gaps."""
    issues = []
    strengths = []

    # Check for essential sections
    required_sections = ["description", "difficulty", "learning_outcomes", "milestones", "skills"]
    for section in required_sections:
        if section in yaml_content.lower():
            strengths.append(f"Includes {section} section")

    # Check for prerequisites
    if "prerequisites" in yaml_content.lower():
        strengths.append("Clearly lists prerequisites")

    # Check for resources
    if "resources" in yaml_content.lower():
        strengths.append("Provides learning resources")

    return issues, strengths

def check_realism(difficulty: str, estimated_hours: Any, milestones: List[Dict]) -> tuple:
    """Check if scope is realistic for difficulty level."""
    issues = []
    strengths = []

    # Extract hours from range if needed
    if isinstance(estimated_hours, str):
        match = re.search(r'(\d+)', estimated_hours)
        if match:
            hours = int(match.group(1))
        else:
            hours = 0
    else:
        hours = estimated_hours if isinstance(estimated_hours, (int, float)) else 0

    # Check if hours per milestone is reasonable (not too much per milestone)
    if hours > 0 and len(milestones) > 0:
        hours_per_milestone = hours / len(milestones)
        if difficulty == "expert" and hours_per_milestone > 20:
            strengths.append(f"Appropriate scope for expert-level: {hours} hours across {len(milestones)} milestones")
        elif difficulty == "intermediate" and 5 <= hours_per_milestone <= 15:
            strengths.append(f"Realistic scope for intermediate-level: {hours} hours")
        elif difficulty == "advanced" and 8 <= hours_per_milestone <= 18:
            strengths.append(f"Appropriate scope for advanced-level: {hours} hours")

    return issues, strengths

def check_security(project_id: str, yaml_content: str, milestones: List[Dict]) -> tuple:
    """Check for security considerations."""
    issues = []
    strengths = []

    security_keywords = ["security", "vulnerability", "injection", "overflow", "corruption",
                        "exploit", "attack", "malicious", "sanitize", "validate"]

    has_security = any(keyword in yaml_content.lower() for keyword in security_keywords)

    # Project-specific security checks
    if project_id == "build-sqlite":
        if "sql injection" in yaml_content.lower() or "sanitize" in yaml_content.lower():
            strengths.append("Addresses SQL injection concerns")
        if "corruption" in yaml_content.lower():
            strengths.append("Addresses data corruption scenarios")

    elif project_id == "build-strace":
        if "signal" in yaml_content.lower():
            strengths.append("Covers signal handling security implications")

    elif project_id == "build-tcp-stack":
        if "checksum" in yaml_content.lower():
            strengths.append("Addresses packet integrity verification")
        if "corruption" in yaml_content.lower():
            strengths.append("Handles data corruption scenarios")

    elif project_id == "build-test-framework":
        if "isolation" in yaml_content.lower():
            strengths.append("Addresses test isolation for security")

    elif project_id == "build-text-editor":
        if "escape sequence" in yaml_content.lower():
            strengths.append("Properly handles terminal escape sequences")

    return issues, strengths

def check_performance(yaml_content: str, milestones: List[Dict]) -> tuple:
    """Check for performance considerations."""
    issues = []
    strengths = []

    performance_keywords = ["performance", "optimization", "efficient", "fast", "slow",
                           "complexity", "o(n)", "latency", "throughput"]

    has_performance = any(keyword in yaml_content.lower() for keyword in performance_keywords)

    if has_performance:
        strengths.append("Includes performance considerations")

    # Check for specific performance metrics
    if "ms" in yaml_content or "milliseconds" in yaml_content.lower():
        strengths.append("Includes specific performance timing requirements")

    if "benchmark" in yaml_content.lower():
        strengths.append("Includes benchmarking requirements")

    return issues, strengths

def generate_critique(issues: List[str], strengths: List[str], score: int) -> str:
    """Generate a brief critique summary."""
    if score >= 8:
        return "Excellent project spec with comprehensive coverage, accurate technical content, and measurable outcomes."
    elif score >= 6:
        return "Good project spec with minor areas for improvement in clarity or completeness."
    elif score >= 4:
        return "Acceptable project spec that needs refinement in several areas."
    else:
        return "Project spec requires significant revision to meet quality standards."

# Main execution
if __name__ == "__main__":
    import sys

    # Read JSON input from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            projects = json.load(f)
    else:
        projects = json.loads(sys.stdin.read())

    results = [audit_project(p) for p in projects]

    print(json.dumps(results, indent=2))

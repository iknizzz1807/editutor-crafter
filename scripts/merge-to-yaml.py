#!/usr/bin/env python3
"""
Merge extracted JSON data into projects.yaml
This script:
1. Reads extracted JSON from HTML
2. Converts to YAML format
3. Outputs new projects.yaml
"""

import json
import yaml
import os
import re

class MyDumper(yaml.SafeDumper):
    pass

# Custom YAML representer for multiline strings
def str_representer(dumper, data):
    if '\n' in data or len(data) > 80:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    # Quote strings with special YAML characters
    if any(c in data for c in [':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`']):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

MyDumper.add_representer(str, str_representer)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

json_path = os.path.join(data_dir, 'extracted-projects.json')
output_path = os.path.join(data_dir, 'projects-merged.yaml')

# Read JSON
with open(json_path, 'r') as f:
    json_data = json.load(f)

def convert_hints(hints):
    """Convert hints object to YAML format"""
    if not hints:
        return None
    result = {}
    for key, value in hints.items():
        result[key] = value
    return result

def convert_milestone(m):
    """Convert a milestone from JSON to YAML format"""
    result = {
        'id': m.get('id', ''),
        'name': m.get('name', ''),
        'description': m.get('description', ''),
    }

    # Acceptance criteria
    if 'criteria' in m:
        result['acceptance_criteria'] = m['criteria']
    elif 'acceptance_criteria' in m:
        result['acceptance_criteria'] = m['acceptance_criteria']

    # Hints
    if 'hints' in m:
        hints = m['hints']
        result['hints'] = {}
        if 'level1' in hints:
            result['hints']['level1'] = hints['level1']
        if 'level2' in hints:
            result['hints']['level2'] = hints['level2']
        if 'level3' in hints:
            result['hints']['level3'] = hints['level3']

    # Pitfalls
    if 'pitfalls' in m and m['pitfalls']:
        result['pitfalls'] = m['pitfalls']

    # Concepts
    if 'concepts' in m and m['concepts']:
        result['concepts'] = m['concepts']

    # Estimated hours
    if 'estimatedHours' in m:
        result['estimated_hours'] = m['estimatedHours']
    elif 'estimated_hours' in m:
        result['estimated_hours'] = m['estimated_hours']

    return result

def convert_project(project_id, project):
    """Convert a project from JSON to YAML format"""
    result = {
        'id': project_id,
        'name': project.get('name', ''),
        'description': project.get('description', ''),
        'difficulty': project.get('difficulty', 'intermediate'),
        'estimated_hours': project.get('estimatedHours', project.get('estimated_hours', '')),
    }

    # Prerequisites
    if 'prerequisites' in project and project['prerequisites']:
        result['prerequisites'] = project['prerequisites']

    # Languages
    if 'languages' in project and project['languages']:
        langs = project['languages']
        if isinstance(langs, dict):
            result['languages'] = {}
            if 'recommended' in langs:
                result['languages']['recommended'] = langs['recommended']
            if 'also' in langs:
                result['languages']['also_possible'] = langs['also']
        else:
            result['languages'] = {'recommended': langs}

    # Resources
    if 'resources' in project and project['resources']:
        result['resources'] = project['resources']

    # Milestones
    if 'milestones' in project and project['milestones']:
        result['milestones'] = [convert_milestone(m) for m in project['milestones']]

    return result

# Convert all expert projects from JSON
expert_projects = {}
json_experts = json_data.get('expertProjects', {})

print(f"Converting {len(json_experts)} expert projects from JSON...")

for project_id, project in json_experts.items():
    expert_projects[project_id] = convert_project(project_id, project)

# Update domains from JSON (they have more detailed info)
json_domains = json_data.get('domains', [])

def convert_domain_project(p):
    """Convert a domain project entry"""
    result = {
        'id': p.get('id', ''),
        'name': p.get('name', ''),
        'description': p.get('description', ''),
    }
    if p.get('detailed'):
        result['detailed'] = True
    if p.get('bridge'):
        result['bridge'] = True
    if p.get('languages'):
        result['languages'] = p['languages']
    return result

converted_domains = []
for domain in json_domains:
    d = {
        'id': domain.get('id', ''),
        'name': domain.get('name', ''),
        'icon': domain.get('icon', ''),
    }

    if 'subdomains' in domain:
        # Convert string subdomains to proper format
        subdomains = domain['subdomains']
        if subdomains and isinstance(subdomains[0], str):
            d['subdomains'] = [{'name': s} for s in subdomains]
        else:
            d['subdomains'] = subdomains

    if 'projects' in domain:
        d['projects'] = {}
        for level in ['beginner', 'intermediate', 'advanced', 'expert']:
            if level in domain['projects']:
                d['projects'][level] = [convert_domain_project(p) for p in domain['projects'][level]]

    converted_domains.append(d)

# Build the final YAML structure
output_data = {}

# Use converted domains
output_data['domains'] = converted_domains

# Add all expert projects (as expert_projects dict)
output_data['expert_projects'] = {}
for project_id in sorted(expert_projects.keys()):
    output_data['expert_projects'][project_id] = expert_projects[project_id]

# Write output
print(f"Writing merged YAML with {len(expert_projects)} expert projects...")

with open(output_path, 'w') as f:
    # Write header comment
    f.write("""# EduTutor Crafter - Projects Data (Unified)
# Source of truth for all domains, projects, and milestones
# Version: 3.0.0 - Merged from HTML visualizer data
#
# Structure:
#   1. DOMAINS - Full taxonomy with all difficulty levels
#   2. EXPERT_PROJECTS - Detailed specifications (alphabetically sorted)

""")

    # Write YAML with proper formatting
    yaml.dump(output_data, f, Dumper=MyDumper, default_flow_style=False, allow_unicode=True, sort_keys=False, width=100)

print(f"Output written to: {output_path}")
print("Please review the merged file and rename to projects.yaml when ready")

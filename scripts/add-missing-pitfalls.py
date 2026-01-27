#!/usr/bin/env python3
"""
Add pitfalls to milestones that are missing them.
Based on common real-world issues encountered in these implementations.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

# Pitfalls to add - keyed by (project_id, milestone_name)
pitfalls_to_add = {
    ("build-git", "Repository Initialization"): [
        "Wrong permissions on .git directory (should be 0755)",
        "Missing directories cause later commands to fail silently",
        "HEAD file must have exact format 'ref: refs/heads/master' with newline",
        "Creating .git inside existing repo causes confusion"
    ],

    ("build-raytracer", "Output an Image"): [
        "RGB values must be clamped to 0-255 range",
        "Forgetting newline between pixel values in PPM",
        "Integer overflow when color values exceed 1.0",
        "Y-axis direction: most formats have origin at top-left, not bottom-left"
    ],

    ("build-raytracer", "Ray Class and Background"): [
        "Rays with zero-length direction vectors cause NaN",
        "Forgetting to normalize direction vector affects calculations",
        "Using int instead of float for coordinates loses precision",
        "Background color calculation sensitive to ray direction normalization"
    ],

    ("build-raytracer", "Surface Normals and Multiple Objects"): [
        "Normal must point outward from surface (flip if inside)",
        "t_min should be small positive (0.001) not zero to avoid self-intersection",
        "Floating point precision: comparing t == 0 fails",
        "Forgetting to return closest hit, not first hit"
    ],

    ("build-raytracer", "Antialiasing"): [
        "Not using random offsets within pixel causes visible patterns",
        "Too few samples causes noisy images",
        "Forgetting to average samples produces overly bright images",
        "Random number generator state affects reproducibility"
    ],

    ("build-raytracer", "Metal and Reflections"): [
        "Reflected ray pointing into surface (dot product check)",
        "Infinite recursion without max depth limit",
        "Fuzz parameter > 1 causes rays to pass through surface",
        "Not attenuating color with each bounce produces unrealistic brightness"
    ],

    ("build-raytracer", "Positionable Camera"): [
        "Field of view in radians vs degrees confusion",
        "Up vector parallel to look direction crashes",
        "Aspect ratio calculation wrong leads to stretched images",
        "Viewport height/width off by one pixel"
    ],

    ("build-raytracer", "Depth of Field"): [
        "Aperture too large causes everything to be blurry",
        "Focus distance wrong makes subject blurry",
        "Random disk sampling bias causes visible artifacts",
        "Thin lens approximation breaks at extreme apertures"
    ]
}

# Load YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})
updated = 0

for project_id, project in expert_projects.items():
    milestones = project.get('milestones', [])
    for milestone in milestones:
        key = (project_id, milestone.get('name', ''))
        if key in pitfalls_to_add:
            milestone['pitfalls'] = pitfalls_to_add[key]
            updated += 1
            print(f"Added pitfalls to: {project_id}/{milestone['name']}")

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nAdded pitfalls to {updated} milestones")

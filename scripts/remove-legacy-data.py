#!/usr/bin/env python3
"""
Remove the legacy embedded data from visualizer.html and update initialization.
"""

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(script_dir, '..', 'visualizer.html')

# Read the file
with open(html_path, 'r') as f:
    lines = f.readlines()

# Find start and end of legacy data
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if 'const _legacyData = {' in line:
        start_idx = i
    if start_idx is not None and line.strip() == '};' and end_idx is None:
        # Check if next line has // State
        if i + 2 < len(lines) and '// State' in lines[i + 2]:
            end_idx = i
            break

if start_idx is None or end_idx is None:
    print(f"Could not find legacy data boundaries: start={start_idx}, end={end_idx}")
    exit(1)

print(f"Found legacy data from line {start_idx + 1} to {end_idx + 1}")

# Find and update initialization section
init_idx = None
for i, line in enumerate(lines):
    if '// Initialize' in line and 'calculateStats()' in lines[i + 1]:
        init_idx = i
        break

if init_idx is None:
    print("Could not find initialization section")
    exit(1)

print(f"Found initialization at line {init_idx + 1}")

# Build new file content
new_lines = []

# Add everything before legacy data
new_lines.extend(lines[:start_idx])

# Add placeholder comment instead of legacy data
new_lines.append('    // Legacy embedded data removed - now loaded from YAML via loadProjectsData()\n')
new_lines.append('\n')

# Add everything after legacy data up to (but not including) initialization
new_lines.extend(lines[end_idx + 1:init_idx])

# Add new async initialization
new_lines.append('    // Initialize - load data from YAML first\n')
new_lines.append('    (async function init() {\n')
new_lines.append('        await loadProjectsData();\n')
new_lines.append('        calculateStats();\n')
new_lines.append('        renderDomains();\n')
new_lines.append('    })();\n')

# Add closing script tag and rest
# Find where </script> is after the old initialization
script_end_idx = None
for i in range(init_idx, len(lines)):
    if '</script>' in lines[i]:
        script_end_idx = i
        break

if script_end_idx:
    new_lines.extend(lines[script_end_idx:])
else:
    new_lines.append('    </script>\n')
    new_lines.append('</body>\n')
    new_lines.append('</html>\n')

# Write new file
with open(html_path, 'w') as f:
    f.writelines(new_lines)

print(f"Updated visualizer.html: removed {end_idx - start_idx + 1} lines of legacy data")
print(f"New file has {len(new_lines)} lines (was {len(lines)})")

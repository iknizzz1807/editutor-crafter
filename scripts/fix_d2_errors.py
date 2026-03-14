#!/usr/bin/env python3
"""
Fix failing D2 diagram files that have various syntax errors.
v2 - improved with more fix cases.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path("/home/ikniz/Work/Coding/AI_MachineLearning/editutor-crafter")
FAILING_FILES_LIST = Path("/tmp/failing_d2_with_errors.txt")

FIXED_COUNT = 0
STILL_FAILING = []
FIX_LOG = []


def run_d2(filepath: Path) -> tuple[bool, str]:
    """Run d2 compiler on a file, return (success, error_output)."""
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["d2", str(filepath), tmp_path],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        output = "timeout"
        success = False
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        # Clean up directory if d2 created one
        dir_path = tmp_path.replace('.svg', '')
        if os.path.isdir(dir_path):
            import shutil
            shutil.rmtree(dir_path, ignore_errors=True)

    return success, output


def compile_to_svg(filepath: Path) -> bool:
    """Compile a d2 file to its corresponding SVG."""
    svg_path = filepath.with_suffix(".svg")
    try:
        result = subprocess.run(
            ["d2", str(filepath), str(svg_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


# ============================================================================
# FIX FUNCTIONS
# ============================================================================

def fix_block_strings(content: str) -> str:
    """Fix block strings where content contains pipe characters."""
    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line opens a block string: ends with |+word optionally followed by whitespace
        opener_match = re.search(
            r'(\|+)(md|go|python|py|js|javascript|typescript|ts|ruby|rb|bash|sh|c|cpp|rust|java|latex|tex)\s*$',
            line
        )
        if opener_match:
            pipe_count = len(opener_match.group(1))
            lang = opener_match.group(2)
            i += 1
            content_lines = []
            terminator_found = False
            terminator_line = None
            terminator_idx = None

            while i < len(lines):
                curr = lines[i]
                # terminator: exactly pipe_count pipes at start of line (with optional leading ws)
                # followed by optional whitespace or { (for style block after block string)
                # But NOT pipe_count+1 or more pipes
                term_pattern = r'^\s*' + r'\|' * pipe_count + r'(\s*$|\s*\{)'
                next_pattern = r'^\s*' + r'\|' * (pipe_count + 1)
                if re.match(term_pattern, curr) and not re.match(next_pattern, curr):
                    terminator_found = True
                    terminator_line = curr
                    terminator_idx = i
                    break
                content_lines.append(curr)
                i += 1

            if terminator_found:
                # Check if content has pipes (which would need more pipes in delimiter)
                full_content = '\n'.join(content_lines)
                max_consecutive_pipes = max(
                    (len(m.group()) for m in re.finditer(r'\|+', full_content)),
                    default=0
                )

                if max_consecutive_pipes >= pipe_count:
                    # Need more pipes
                    new_pipe_count = max_consecutive_pipes + 1
                    new_pipes = '|' * new_pipe_count
                    # Replace opener - replace the pipe sequence before the lang
                    new_opener = re.sub(r'\|+(' + re.escape(lang) + r')', new_pipes + r'\1', line)
                    result.append(new_opener)
                    result.extend(content_lines)
                    # Replace terminator - change pipe count, keep anything after pipes
                    after_pipes = re.match(r'^\s*\|+(.*)$', terminator_line)
                    rest = after_pipes.group(1) if after_pipes else ''
                    leading_ws = re.match(r'^(\s*)', terminator_line).group(1)
                    result.append(leading_ws + new_pipes + rest)
                else:
                    result.append(line)
                    result.extend(content_lines)
                    result.append(terminator_line)
            else:
                # No terminator found - block was never closed
                full_content = '\n'.join(content_lines)
                max_consecutive_pipes = max(
                    (len(m.group()) for m in re.finditer(r'\|+', full_content)),
                    default=0
                )
                if max_consecutive_pipes >= pipe_count:
                    new_pipe_count = max_consecutive_pipes + 1
                    new_pipes = '|' * new_pipe_count
                    new_opener = re.sub(r'\|+(' + re.escape(lang) + r')', new_pipes + r'\1', line)
                    result.append(new_opener)
                    result.extend(content_lines)
                    result.append(new_pipes)
                else:
                    result.append(line)
                    result.extend(content_lines)
                    result.append('|' * pipe_count)
                i -= 1
        else:
            result.append(line)
        i += 1
    return '\n'.join(result)


def fix_backtick_pipe_opener(content: str) -> str:
    """Fix |`lang or `|lang openers -> |lang."""
    # Fix |`cpp -> ||cpp (backtick inside pipe opener)
    content = re.sub(
        r'\|`(md|go|python|py|js|javascript|typescript|ts|ruby|rb|bash|sh|c|cpp|rust|java|latex|tex)',
        r'||' + r'\1',
        content
    )
    # Fix `| terminator -> |
    content = re.sub(r'`\|', '|', content)
    return content


def fix_single_quote_pipe_opener(content: str) -> str:
    """Fix '| terminators -> |."""
    content = re.sub(r"'\|", '|', content)
    return content


def fix_curly_pipe_opener(content: str) -> str:
    """Fix }{ | patterns -> ||."""
    content = re.sub(r'\}\{\|', '||', content)
    return content


def fix_hat_pipe_opener(content: str) -> str:
    """Fix |^| patterns -> |||."""
    content = re.sub(r'\|\^\|', '|||', content)
    return content


def fix_near_elk_constant(content: str, errors: str) -> str:
    """Fix near set to object path (not constant) for elk/dagre layout."""
    valid_constants = {
        'top-left', 'top-center', 'top-right',
        'center-left', 'center-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    }

    # Find object names from error messages
    elk_re = re.compile(r'Object "([^"]+)" has "near" set to another object')
    bad_objects = elk_re.findall(errors)

    lines = content.split('\n')
    result = []

    # Track depth to find top-level near: vs nested near:
    depth = 0
    for line in lines:
        # Check for near: with a non-constant value
        near_match = re.match(r'^(\s*)near:\s*(.+)$', line)
        if near_match:
            val = near_match.group(2).strip()
            # Remove quotes if present
            val_unquoted = val.strip('"').strip("'")
            if val_unquoted not in valid_constants:
                # This is a non-constant near value - replace it
                indent = near_match.group(1)
                # Choose best constant based on context
                if 'annot' in val_unquoted.lower() or 'note' in val_unquoted.lower() or 'legend' in val_unquoted.lower():
                    new_val = 'top-right'
                elif 'bottom' in val_unquoted.lower():
                    new_val = 'bottom-right'
                else:
                    new_val = 'top-right'
                result.append(f'{indent}near: {new_val}')
                # Track opens/closes
                opens = len(re.findall(r'\{', line))
                closes = len(re.findall(r'\}', line))
                depth += opens - closes
                continue

        opens = len(re.findall(r'\{', line))
        closes = len(re.findall(r'\}', line))
        depth += opens - closes
        result.append(line)

    return '\n'.join(result)


def fix_near_non_root(content: str) -> str:
    """Remove near: from nested shapes (only works at root level)."""
    valid_constants = {
        'top-left', 'top-center', 'top-right',
        'center-left', 'center-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    }

    lines = content.split('\n')
    result = []
    depth = 0

    for line in lines:
        opens = len(re.findall(r'\{', line))
        closes = len(re.findall(r'\}', line))

        near_match = re.match(r'^(\s*)near:\s*(.+)$', line)
        if near_match and depth > 0:
            val = near_match.group(2).strip().strip('"')
            if val in valid_constants:
                # Nested shape with constant near - remove it
                depth += opens - closes
                continue

        depth += opens - closes
        result.append(line)

    return '\n'.join(result)


def fix_near_invalid_constant(content: str, errors: str) -> str:
    """Fix invalid near constants."""
    invalid_near_re = re.compile(r'near key "([^"]+)" must be')
    invalid_keys = invalid_near_re.findall(errors)

    valid_map = {
        'right-center': 'center-right',
        'left-center': 'center-left',
        'center': 'center-right',
        'middle': 'center-right',
        'top': 'top-center',
        'bottom': 'bottom-center',
        'right': 'center-right',
        'left': 'center-left',
    }

    for invalid_key in invalid_keys:
        lower_key = invalid_key.lower()
        replacement = None
        for k, v in valid_map.items():
            if k in lower_key:
                replacement = v
                break
        if replacement is None:
            replacement = 'top-right'

        content = re.sub(
            r'(near:\s*)"?' + re.escape(invalid_key) + r'"?',
            f'near: {replacement}',
            content
        )

    return content


def fix_near_ancestor(content: str) -> str:
    """Remove near: from shapes where it references ancestors/descendants."""
    valid_constants = {
        'top-left', 'top-center', 'top-right',
        'center-left', 'center-right',
        'bottom-left', 'bottom-center', 'bottom-right'
    }
    lines = content.split('\n')
    result = []
    for line in lines:
        if re.match(r'^\s*near:\s+', line):
            val = re.match(r'^\s*near:\s+(.+)$', line)
            if val:
                v = val.group(1).strip().strip('"')
                if v not in valid_constants:
                    continue
        result.append(line)
    return '\n'.join(result)


def fix_missing_vars_colors(content: str, errors: str) -> str:
    """Add missing color variables to vars.colors block."""
    missing_var_re = re.compile(r'could not resolve variable "vars\.colors\.([^"]+)"')
    missing_vars = list(dict.fromkeys(missing_var_re.findall(errors)))  # deduplicate

    if not missing_vars:
        return content

    # Default colors for common variable names
    def guess_color(name: str) -> str:
        n = name.lower().replace('_', ' ').replace('-', ' ')
        if any(w in n for w in ['red', 'error', 'bug', 'conflict', 'critical', 'danger']):
            return '#F44336'
        if any(w in n for w in ['green', 'success', 'safe', 'verified', 'fix', 'good']):
            return '#4CAF50'
        if any(w in n for w in ['blue', 'data', 'info', 'primary', 'main', 'leader']):
            return '#2196F3'
        if any(w in n for w in ['yellow', 'warning', 'flag', 'uncommitted']):
            return '#FFEB3B'
        if any(w in n for w in ['orange', 'pointer', 'hot']):
            return '#FF9800'
        if any(w in n for w in ['purple', 'header', 'intel']):
            return '#9C27B0'
        if any(w in n for w in ['gray', 'grey', 'padding', 'light', 'bg', 'background']):
            return '#9E9E9E'
        if any(w in n for w in ['white', 'free']):
            return '#FFFFFF'
        if any(w in n for w in ['black', 'dark']):
            return '#212121'
        if any(w in n for w in ['consumer', 'follower']):
            return '#03A9F4'
        if any(w in n for w in ['producer', 'node']):
            return '#8BC34A'
        if any(w in n for w in ['queue', 'channel']):
            return '#FF5722'
        if any(w in n for w in ['stack', 'barrier']):
            return '#607D8B'
        if any(w in n for w in ['frag', 'segment']):
            return '#795548'
        return '#607D8B'  # default blue-grey

    # Check if there's a colors block in vars
    colors_block_match = re.search(r'(colors:\s*\{)((?:[^{}]|\{[^{}]*\})*)\}', content, re.DOTALL)

    if colors_block_match:
        colors_content = colors_block_match.group(2)
        # Add missing vars
        new_entries = ''
        for var_name in missing_vars:
            # Check if already present
            if not re.search(r'\b' + re.escape(var_name) + r'\s*:', colors_content):
                color_val = guess_color(var_name)
                new_entries += f'\n    {var_name}: "{color_val}"'
        if new_entries:
            new_colors_block = colors_block_match.group(1) + colors_content + new_entries + '\n  }'
            content = content[:colors_block_match.start()] + new_colors_block + content[colors_block_match.end():]
    else:
        # No colors block at all - need to add vars section or add to existing vars
        vars_match = re.search(r'(vars:\s*\{)', content)
        if vars_match:
            entries = '\n  colors: {'
            for var_name in missing_vars:
                color_val = guess_color(var_name)
                entries += f'\n    {var_name}: "{color_val}"'
            entries += '\n  }'
            insert_pos = vars_match.end()
            content = content[:insert_pos] + entries + content[insert_pos:]
        else:
            # No vars block - add one
            entries = 'vars: {\n  colors: {'
            for var_name in missing_vars:
                color_val = guess_color(var_name)
                entries += f'\n    {var_name}: "{color_val}"'
            entries += '\n  }\n}\n'
            content = entries + content

    return content


def fix_missing_node_base_var(content: str, errors: str) -> str:
    """Fix could not resolve variable "node_base" and similar."""
    missing_var_re = re.compile(r'could not resolve variable "([^"]+)"')
    missing_vars = list(dict.fromkeys(missing_var_re.findall(errors)))

    # Filter out vars.colors.* (handled elsewhere) and keep root-level vars
    root_vars = [v for v in missing_vars if '.' not in v or v.startswith('vars.') == False]
    root_vars = [v for v in missing_vars if not v.startswith('vars.')]

    if not root_vars:
        return content

    for var_name in root_vars:
        # Check if it's already defined
        if re.search(r'\b' + re.escape(var_name) + r'\s*:', content):
            continue
        # Check if there's a vars block
        vars_match = re.search(r'(vars:\s*\{)', content)
        if vars_match:
            # Add to vars block
            insert_pos = vars_match.end()
            content = content[:insert_pos] + f'\n  {var_name}: "rectangle"' + content[insert_pos:]
        else:
            content = f'vars: {{\n  {var_name}: "rectangle"\n}}\n' + content

    return content


def fix_key_length(content: str, errors: str) -> str:
    """Truncate keys that exceed maximum allowed length."""
    lines = content.split('\n')
    result = []

    for line in lines:
        # Check if the line is too long
        stripped = line.strip()
        if len(stripped) <= 520:
            result.append(line)
            continue

        # Pattern: "very long quoted string": value
        m = re.match(r'^(\s*)"(.{200,})"\s*(.*)$', line)
        if m:
            truncated = m.group(2)[:100] + '...'
            result.append(f'{m.group(1)}"{truncated}" {m.group(3)}'.rstrip())
            continue

        # Pattern: key: "very long value"
        m2 = re.match(r'^(\s*)([^:]+):\s*"(.{200,})"(.*)$', line)
        if m2:
            truncated = m2.group(3)[:150] + '...'
            result.append(f'{m2.group(1)}{m2.group(2)}: "{truncated}"{m2.group(4)}')
            continue

        # Pattern: "key" | "col1" | "col2" (sql_table row)
        # Split and check each pipe-separated part
        if '|' in stripped and len(stripped) > 520:
            parts = stripped.split('|')
            new_parts = []
            for part in parts:
                p = part.strip()
                if len(p) > 150 and p.startswith('"') and p.endswith('"'):
                    p = '"' + p[1:151] + '..."'
                new_parts.append(p)
            indent = re.match(r'^(\s*)', line).group(1)
            result.append(indent + ' | '.join(new_parts))
            continue

        result.append(line)

    return '\n'.join(result)


def fix_unknown_shapes(content: str) -> str:
    """Replace unknown shapes with valid equivalents."""
    content = re.sub(r'\bshape:\s*capsule\b', 'shape: oval', content)
    content = re.sub(r'\bshape:\s*sticky_note\b', 'shape: rectangle', content)
    content = re.sub(r'\bshape:\s*line\b', 'shape: rectangle', content)
    return content


def fix_invalid_style_keywords(content: str) -> str:
    """Remove invalid style keywords."""
    lines = content.split('\n')
    result = []
    for line in lines:
        if re.match(r'^\s*text-align\s*:', line):
            continue
        if re.match(r'^\s*source-arrowhead\s*:', line):
            continue
        if re.match(r'^\s*target-arrowhead\s*:', line):
            continue
        if re.match(r'^\s*font:\s*"sans-serif"', line):
            continue
        if re.match(r'^\s*double-border\s*:', line):
            continue
        if re.match(r'^\s*stroke-dash-offset\s*:', line):
            continue
        result.append(line)
    return '\n'.join(result)


def fix_style_prefix(content: str) -> str:
    """Fix bare opacity/bold/font-color -> style.opacity/style.bold/style.font-color."""
    lines = content.split('\n')
    result = []
    for line in lines:
        # Skip lines already using style. prefix
        if 'style.' in line:
            result.append(line)
            continue
        # Fix bare opacity
        if re.match(r'^(\s*)opacity:\s*', line):
            m = re.match(r'^(\s*)opacity:\s*(.+)$', line)
            if m:
                result.append(f'{m.group(1)}style.opacity: {m.group(2)}')
                continue
        # Fix bare bold
        if re.match(r'^(\s*)bold:\s*', line):
            m = re.match(r'^(\s*)bold:\s*(.+)$', line)
            if m:
                result.append(f'{m.group(1)}style.bold: {m.group(2)}')
                continue
        # Fix bare font-color
        if re.match(r'^(\s*)font-color:\s*', line):
            m = re.match(r'^(\s*)font-color:\s*(.+)$', line)
            if m:
                result.append(f'{m.group(1)}style.font-color: {m.group(2)}')
                continue
        result.append(line)
    return '\n'.join(result)


def fix_indexed_edges(content: str, errors: str) -> str:
    """Remove (N) index from edge references."""
    # Pattern: (edge_src -> edge_dst)[N].something
    content = re.sub(r'\(([^)]+)\)\[(\d+)\]\.', r'\1.', content)
    return content


def fix_edge_map_keys(content: str, errors: str) -> str:
    """Fix edge declarations where the label contains {...} that get parsed as map."""
    line_re = re.compile(r':(\d+):\d+: edge map keys must be reserved keywords')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            # Pattern: actor -> actor: label {key: val}
            # The {...} after label gets parsed as a map - quote the content between {} as label
            m = re.match(r'^(\s*)(.*?->.*?:\s*)(\{[^}]*\})(.*)$', line)
            if m:
                label_map = m.group(3)
                inner = label_map[1:-1].strip()
                # Escape any quotes
                inner = inner.replace('"', '\\"')
                result[idx] = f'{m.group(1)}{m.group(2)}"{inner}"{m.group(4)}'
            else:
                # Try another pattern - just quote the whole RHS
                m2 = re.match(r'^(\s*)(.*?->.*?:)\s*(.+)$', line)
                if m2 and not m2.group(3).startswith('"'):
                    val = m2.group(3).strip()
                    if not val.startswith('{'):
                        result[idx] = f'{m2.group(1)}{m2.group(2)} "{val}"'

    return '\n'.join(result)


def fix_layers_in_class(content: str, errors: str) -> str:
    """Fix 'layers' used as field name in class shapes."""
    line_re = re.compile(r':(\d+):\d+: layers must be declared at a board root scope')
    error_lines = [int(m.group(1)) for m in line_re.finditer(errors)]

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            if re.match(r'^\s*layers\s*:', line):
                result[idx] = line.replace('layers:', '"layers":', 1)

    return '\n'.join(result)


def fix_scenarios_in_non_root(content: str, errors: str) -> str:
    """Fix 'scenarios' used as field name in nested shapes."""
    line_re = re.compile(r':(\d+):\d+: scenarios must be declared at a board root scope')
    error_lines = [int(m.group(1)) for m in line_re.finditer(errors)]

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            if re.match(r'^\s*scenarios\s*:', line):
                result[idx] = line.replace('scenarios:', '"scenarios":', 1)

    return '\n'.join(result)


def fix_edge_board_keyword(content: str, errors: str) -> str:
    """Fix 'edge with board keyword alone'."""
    line_re = re.compile(r':(\d+):\d+: edge with board keyword alone')
    error_lines = [int(m.group(1)) for m in line_re.finditer(errors)]

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            result[idx] = '# REMOVED (invalid edge): ' + result[idx]

    return '\n'.join(result)


def fix_substitutions(content: str) -> str:
    """Fix $varname -> ${varname} substitutions."""
    # Only fix $varname that's NOT inside a block string and NOT already ${...}
    # Strategy: process line by line, skip lines inside block strings
    lines = content.split('\n')
    result = []
    in_block = False
    block_depth = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Track block string state
        if not in_block:
            opener = re.search(
                r'(\|+)(md|go|python|py|js|javascript|typescript|ts|ruby|rb|bash|sh|c|cpp|rust|java|latex|tex)\s*$',
                line
            )
            if opener:
                in_block = True
                block_depth = len(opener.group(1))
                result.append(line)
                i += 1
                continue
        else:
            # Check for terminator
            term_pattern = r'^\s*' + r'\|' * block_depth + r'(\s*$|\s*\{)'
            next_pattern = r'^\s*' + r'\|' * (block_depth + 1)
            if re.match(term_pattern, line) and not re.match(next_pattern, line):
                in_block = False
                block_depth = 0
            result.append(line)
            i += 1
            continue

        # Fix $var -> ${var} only outside block strings
        # But be careful: $-1 or $5 are not vars; only $identifier
        new_line = re.sub(
            r'\$(?!\{)(?!-)([a-zA-Z_][a-zA-Z0-9_.]*)',
            r'${\1}',
            line
        )
        result.append(new_line)
        i += 1

    return '\n'.join(result)


def fix_class_method_args(content: str, errors: str) -> str:
    """Fix class/interface method declarations with problematic syntax."""
    line_re = re.compile(r':(\d+):\d+: unexpected text after map key')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            stripped = line.strip()

            # Skip empty lines, comments, or already valid lines
            if not stripped or stripped.startswith('#'):
                continue

            # Pattern 1: +method<T>(args): "return_type"
            # Pattern 2: +method(args): "return_type" where args have special chars
            # Pattern 3: field_name<T>: "type"
            # These need the key part quoted

            # Try to split on ': ' to find key: value
            colon_pos = -1
            quote_depth = 0
            paren_depth = 0
            angle_depth = 0
            for ci, ch in enumerate(stripped):
                if ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
                elif ch == '<':
                    angle_depth += 1
                elif ch == '>':
                    angle_depth -= 1
                elif ch == '"':
                    quote_depth = 1 - quote_depth
                elif ch == ':' and quote_depth == 0 and paren_depth == 0 and angle_depth == 0:
                    # Check if next char is space (value separator)
                    if ci + 1 < len(stripped) and stripped[ci+1] in (' ', '"', '|'):
                        colon_pos = ci
                        break

            if colon_pos >= 0:
                key_part = stripped[:colon_pos].strip()
                val_part = stripped[colon_pos+1:].strip()
                indent = re.match(r'^(\s*)', line).group(1)

                # Quote the key if it has special characters
                needs_quoting = bool(re.search(r'[<>()\[\]|*&,]|\.\.\.', key_part))
                if needs_quoting and not key_part.startswith('"'):
                    # Escape internal quotes
                    safe_key = key_part.replace('"', '\\"')
                    result[idx] = f'{indent}"{safe_key}": {val_part}'
            else:
                # No colon - might be a standalone method name or field
                indent = re.match(r'^(\s*)', line).group(1)
                key_part = stripped
                needs_quoting = bool(re.search(r'[<>()\[\]|*&,]|\.\.\.', key_part))
                if needs_quoting and not key_part.startswith('"'):
                    safe_key = key_part.replace('"', '\\"')
                    result[idx] = f'{indent}"{safe_key}"'

    return '\n'.join(result)


def fix_unquoted_string_errors(content: str, errors: str) -> str:
    """Fix unexpected text after unquoted string errors."""
    line_re = re.compile(r':(\d+):\d+: unexpected text after unquoted string')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            stripped = line.strip()
            indent = re.match(r'^(\s*)', line).group(1)

            # Find : separator
            colon_idx = stripped.find(': ')
            if colon_idx > 0:
                key = stripped[:colon_idx].strip()
                val = stripped[colon_idx+2:].strip()
                # If the value has spaces and isn't quoted/block, quote it
                if re.search(r'[\s<>()]', key) and not key.startswith('"'):
                    safe_key = key.replace('"', '\\"')
                    result[idx] = f'{indent}"{safe_key}": {val}'
                elif re.search(r'[\s<>()]', val) and not val.startswith('"') and \
                     not val.startswith('|') and not val.startswith('{'):
                    safe_val = val.replace('"', '\\"')
                    result[idx] = f'{indent}{key}: "{safe_val}"'
            else:
                # No colon - quote the whole thing if it has special chars
                if re.search(r'[<>()]', stripped) and not stripped.startswith('"'):
                    safe = stripped.replace('"', '\\"')
                    result[idx] = f'{indent}"{safe}"'

    return '\n'.join(result)


def fix_double_quoted_near(content: str, errors: str) -> str:
    """Fix near: set to a quoted path."""
    line_re = re.compile(r':(\d+):\d+: unexpected text after double quoted string')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            if re.search(r'\bnear\s*:', line):
                result[idx] = re.sub(r'near\s*:.*$', 'near: top-right', line)

    return '\n'.join(result)


def fix_sql_table_children(content: str, errors: str) -> str:
    """Flatten nested objects inside sql_table shapes."""
    line_re = re.compile(r':(\d+):\d+: sql_table columns cannot have children')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    # For each error line, find and remove the block
    # The error is on the first child key inside the nested object
    # We need to find the parent { ... } block and flatten it

    # We'll work from bottom to top to avoid index shifts
    processed = set()
    for lineno in sorted(error_lines, reverse=True):
        idx = lineno - 1
        if idx in processed or idx >= len(result):
            continue

        # Find the opening brace of this child block
        # The error is on a line like: child_key: "value"
        # The parent block was opened somewhere above

        # Look backward to find the parent key: { line
        parent_idx = None
        brace_count = 0
        for j in range(idx, -1, -1):
            line = result[j]
            # Count braces in reverse
            closes = line.count('}')
            opens = line.count('{')
            brace_count += opens - closes

            if brace_count > 0:
                # This might be the parent block opening
                # Check if it ends with { (it's a block opener)
                if re.search(r'\{', result[j]):
                    parent_idx = j
                    break

        if parent_idx is not None:
            # Check if this parent is inside a sql_table
            # Find the closing brace
            close_idx = None
            depth = 0
            for j in range(parent_idx, len(result)):
                for ch in result[j]:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                if depth == 0:
                    close_idx = j
                    break

            if close_idx is not None:
                # Remove the block: keep just the parent key as a simple value
                parent_line = result[parent_idx]
                # Extract just the key part (before {)
                key_match = re.match(r'^(\s*)(.+?)\s*\{.*$', parent_line)
                if key_match:
                    indent = key_match.group(1)
                    key = key_match.group(2).strip()
                    # Replace the whole block with just the key as a simple entry
                    result[parent_idx:close_idx+1] = [f'{indent}{key}']
                    processed.update(range(parent_idx, close_idx+1))

    return '\n'.join(result)


def fix_class_fields_children(content: str, errors: str) -> str:
    """Flatten nested objects inside class shapes."""
    line_re = re.compile(r':(\d+):\d+: class fields cannot have children')
    return fix_sql_table_children_generic(content, errors, line_re)


def fix_sql_table_children_generic(content: str, errors: str, line_re) -> str:
    """Generic fix for children inside table/class shapes."""
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in sorted(error_lines, reverse=True):
        idx = lineno - 1
        if idx >= len(result):
            continue

        # Look backward to find the parent block
        parent_idx = None
        depth = 0
        for j in range(idx - 1, -1, -1):
            line = result[j]
            closes = line.count('}')
            opens = line.count('{')
            depth += closes - opens  # going backward, so invert
            if depth == 0 and opens > 0:
                # Potential parent opener
                if re.search(r'\{', result[j]):
                    parent_idx = j
                    break

        if parent_idx is not None:
            # Find closing brace
            close_idx = None
            depth = 0
            for j in range(parent_idx, len(result)):
                for ch in result[j]:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                if depth <= 0:
                    close_idx = j
                    break

            if close_idx is not None:
                parent_line = result[parent_idx]
                key_match = re.match(r'^(\s*)(.+?)\s*\{.*$', parent_line)
                if key_match:
                    indent = key_match.group(1)
                    key = key_match.group(2).strip()
                    result[parent_idx:close_idx+1] = [f'{indent}{key}']

    return '\n'.join(result)


def fix_newlines_in_sql_label(content: str) -> str:
    """Fix sql_table labels that have newlines (\\n)."""
    # sql_table cannot have newlines in label
    lines = content.split('\n')
    result = []
    for line in lines:
        if 'shape: sql_table' not in line:
            # For lines inside sql_table that have \n in their key or label
            # Remove the \n
            if '\\n' in line and ('label:' in line or ':' in line):
                # If it's a label with \n, replace with space
                line = re.sub(r'(label:\s*"[^"]*?)\\n([^"]*")', r'\1 \2', line)
            result.append(line)
        else:
            result.append(line)
    return '\n'.join(result)


def fix_reserved_keywords_in_edges(content: str, errors: str) -> str:
    """Fix reserved keywords (like 'near', 'label') used in edge paths."""
    line_re = re.compile(r':(\d+):\d+: reserved keywords are prohibited in edges')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            # Comment out the problematic edge line
            result[idx] = '# REMOVED (reserved keyword in edge): ' + line

    return '\n'.join(result)


def fix_locked_position(content: str, errors: str) -> str:
    """Fix locked position (top/left) for elk layout."""
    # Remove top: and left: style attributes that are not supported by elk
    if 'attribute "top" and/or "left" set' not in errors and \
       'layout engine "elk" does not support locked positions' not in errors:
        return content

    lines = content.split('\n')
    result = []
    for line in lines:
        if re.match(r'^\s*top\s*:', line) or re.match(r'^\s*left\s*:', line):
            continue
        result.append(line)
    return '\n'.join(result)


def fix_unexpected_map_termination(content: str, errors: str) -> str:
    """Fix unexpected map termination character } in file map."""
    line_re = re.compile(r':(\d+):\d+: unexpected map termination character \} in file map')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in sorted(error_lines, reverse=True):
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            # This is an extra } at file root level - remove it
            if re.match(r'^\s*\}\s*$', line):
                result[idx] = '# REMOVED (extra closing brace): ' + line

    return '\n'.join(result)


def fix_arrowhead_shape(content: str) -> str:
    """Fix invalid arrowhead shapes."""
    # Remove arrowhead shape declarations that are invalid
    lines = content.split('\n')
    result = []
    for line in lines:
        if re.match(r'^\s*shape:\s*(capsule|hexagon|cylinder|cloud|callout|queue|book|stored_data|step|tape|page)', line):
            # Check if we're in an arrowhead context by looking at surrounding context
            # For safety, just keep it (it might be valid in non-arrowhead context)
            result.append(line)
        else:
            result.append(line)
    return '\n'.join(result)


def fix_fill_color(content: str, errors: str) -> str:
    """Fix invalid fill color values."""
    line_re = re.compile(r':(\d+):\d+: expected "fill" to be a valid named color')
    error_lines = sorted(set(int(m.group(1)) for m in line_re.finditer(errors)))

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            # Replace invalid fill value with a safe default
            result[idx] = re.sub(r'(fill:\s*)([^\n{]+)', r'\1"#FFFFFF"', line)

    return '\n'.join(result)


def fix_stroke_width(content: str, errors: str) -> str:
    """Fix stroke-width values out of range (0-15)."""
    if '"stroke-width" to be a number between 0 and 15' not in errors:
        return content
    # Replace stroke-width values > 15 with 15
    def clamp_stroke(m):
        try:
            val = float(m.group(1))
            if val > 15:
                return f'stroke-width: 15'
            return m.group(0)
        except:
            return m.group(0)
    content = re.sub(r'stroke-width:\s*(\d+(?:\.\d+)?)', clamp_stroke, content)
    return content


def apply_all_fixes(filepath: Path, content: str, errors: str) -> tuple[str, list[str]]:
    """Apply all applicable fixes to content based on errors."""
    fixes_applied = []
    original = content

    def apply(name, fn, *args):
        nonlocal content, original
        new = fn(*args) if args else fn(content)
        if new != content:
            content = new
            fixes_applied.append(name)
            original = content

    # Fix malformed terminators first (before block string analysis)
    if '`' in content:
        apply("backtick-pipe terminators", fix_backtick_pipe_opener, content)

    if "block string must be terminated with '|" in errors:
        apply("single-quote-pipe terminators", fix_single_quote_pipe_opener, content)

    if "}{\\" in errors or "block string must be terminated with }{" in errors:
        apply("curly-pipe terminators", fix_curly_pipe_opener, content)

    if "|^|" in errors:
        apply("hat-pipe terminators", fix_hat_pipe_opener, content)

    # Fix block strings with pipe content (most common issue)
    if ('block string must be terminated with |' in errors or
        'unexpected text after md block string' in errors or
        'unexpected text after latex block string' in errors or
        'unexpected text after go block string' in errors or
        'unexpected text after cpp block string' in errors or
        'unexpected text after map key' in errors or
        'maps must be terminated with }' in errors or
        'unexpected map termination' in errors):
        apply("block strings (pipe count)", fix_block_strings, content)

    # Fix key length issues
    if 'key length' in errors and 'exceeds maximum allowed length' in errors:
        apply("key lengths", fix_key_length, content, errors)

    # Fix near: elk constant issues (object paths)
    if ('layout engine "elk" only supports constant values' in errors or
        'layout engine "dagre" only supports constant values' in errors):
        apply("near: elk/dagre constant", fix_near_elk_constant, content, errors)

    # Fix constant near in nested shapes
    if 'constant near keys can only be set on root level shapes' in errors:
        apply("near: in nested shapes", fix_near_non_root, content)

    # Fix invalid near constants
    if 'must be the absolute path to a shape or one of the following constants' in errors:
        apply("invalid near constants", fix_near_invalid_constant, content, errors)

    # Fix near: ancestor / sequence diagram / grid cell
    if ('near keys cannot be set to an ancestor' in errors or
        'near keys cannot be set to an object within sequence diagrams' in errors or
        'near keys cannot be set to descendants of special objects' in errors or
        'near keys cannot be set to an object with a constant near key' in errors or
        'invalid "near" field' in errors):
        apply("invalid near (ancestor/sequence/grid)", fix_near_ancestor, content)

    # Fix double-quoted near path
    if 'unexpected text after double quoted string' in errors:
        apply("double-quoted near path", fix_double_quoted_near, content, errors)

    # Fix missing vars.colors
    if 'could not resolve variable "vars.colors.' in errors:
        apply("missing vars.colors", fix_missing_vars_colors, content, errors)

    # Fix missing non-colors vars
    if 'could not resolve variable "' in errors and '"vars.colors.' not in errors:
        apply("missing root vars", fix_missing_node_base_var, content, errors)

    # Fix unknown shapes
    if ('unknown shape "capsule"' in errors or 'unknown shape "sticky_note"' in errors or
        'unknown shape "line"' in errors):
        apply("unknown shapes", fix_unknown_shapes, content)

    # Fix invalid style keywords
    if ('invalid style keyword' in errors or '"sans-serif" is not a valid font' in errors or
        'key "double-border"' in errors):
        apply("invalid style keywords", fix_invalid_style_keywords, content)

    # Fix style prefix
    if ('opacity must be style.opacity' in errors or 'bold must be style.bold' in errors or
        'font-color must be style.font-color' in errors):
        apply("style prefix", fix_style_prefix, content)

    # Fix indexed edges
    if 'indexed edge does not exist' in errors:
        apply("indexed edges", fix_indexed_edges, content, errors)

    # Fix edge map keys
    if 'edge map keys must be reserved keywords' in errors:
        apply("edge map keys", fix_edge_map_keys, content, errors)

    # Fix layers/scenarios keyword in class shapes
    if 'layers must be declared at a board root scope' in errors:
        apply("'layers' field in class shape", fix_layers_in_class, content, errors)

    if 'scenarios must be declared at a board root scope' in errors:
        apply("'scenarios' field in shape", fix_scenarios_in_non_root, content, errors)

    # Fix edge board keyword
    if "edge with board keyword alone doesn't make sense" in errors:
        apply("edge board keyword", fix_edge_board_keyword, content, errors)

    # Fix substitutions
    if 'substitutions must begin on {' in errors:
        apply("substitution syntax", fix_substitutions, content)

    # Fix sql_table/class nested children
    if 'sql_table columns cannot have children' in errors:
        apply("sql_table nested children", fix_sql_table_children, content, errors)

    if 'class fields cannot have children' in errors:
        apply("class field children", fix_class_fields_children, content, errors)

    # Fix sql_table newlines in label
    if 'shape sql_table cannot have newlines in label' in errors:
        apply("sql_table label newlines", fix_newlines_in_sql_label, content)

    # Fix class method args (unexpected text after map key)
    if 'unexpected text after map key' in errors:
        apply("class method args", fix_class_method_args, content, errors)

    # Fix unquoted string errors
    if 'unexpected text after unquoted string' in errors:
        apply("unquoted string errors", fix_unquoted_string_errors, content, errors)

    # Fix reserved keywords in edges
    if 'reserved keywords are prohibited in edges' in errors:
        apply("reserved keywords in edges", fix_reserved_keywords_in_edges, content, errors)

    # Fix locked position for elk
    if 'does not support locked positions' in errors:
        apply("locked position (elk)", fix_locked_position, content, errors)

    # Fix unexpected map termination (extra })
    if 'unexpected map termination character }' in errors:
        apply("unexpected map termination", fix_unexpected_map_termination, content, errors)

    # Fix fill color
    if 'expected "fill" to be a valid named color' in errors:
        apply("invalid fill color", fix_fill_color, content, errors)

    # Fix stroke-width range
    if '"stroke-width" to be a number between 0 and 15' in errors:
        apply("stroke-width range", fix_stroke_width, content, errors)

    # Fix invalid arrowhead
    if 'invalid shape, can only set "arrow" for arrowheads' in errors:
        # Remove the invalid shape line entirely
        def fix_arrow(c):
            lines = c.split('\n')
            result = []
            for l in lines:
                if re.match(r'^\s*shape:\s+\w+', l):
                    # Check context - if this is an arrowhead shape, remove it
                    # For simplicity, remove shape declarations that are in source/target-arrowhead blocks
                    result.append(l)
                else:
                    result.append(l)
            return '\n'.join(result)
        # Actually, let's just remove arrowhead shape lines
        new_c = re.sub(r'(source-arrowhead|target-arrowhead)\s*:\s*\{[^}]*shape:\s*\w+[^}]*\}',
                       '', content, flags=re.DOTALL)
        if new_c != content:
            content = new_c
            fixes_applied.append("invalid arrowhead shape")
            original = content

    return content, fixes_applied


def iterative_fix(filepath: Path, max_iterations: int = 8) -> bool:
    """Iteratively fix a file until it compiles or max iterations reached."""

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    all_fixes = []
    last_errors = ""

    for iteration in range(max_iterations):
        success, output = run_d2(filepath)
        if success:
            if all_fixes:
                FIX_LOG.append(f"FIXED: {filepath.name} - fixes: {', '.join(all_fixes)}")
            return True

        errors = output
        if errors == last_errors and iteration > 0:
            # Same error, no progress
            break
        last_errors = errors

        new_content, fixes = apply_all_fixes(filepath, content, errors)

        if not fixes and not new_content != content:
            break

        all_fixes.extend(fixes)

        if new_content != content:
            content = new_content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # Content didn't change
            break

    # Final check
    success, final_output = run_d2(filepath)
    if success:
        FIX_LOG.append(f"FIXED: {filepath.name} - fixes: {', '.join(all_fixes)}")
        return True
    else:
        FIX_LOG.append(
            f"FAILED: {filepath.name} - "
            f"tried: {', '.join(all_fixes) if all_fixes else 'none'} - "
            f"remaining: {final_output[:300]}"
        )
        return False


def generate_svg(filepath: Path) -> bool:
    """Generate SVG for a file that compiles successfully."""
    svg_path = filepath.with_suffix('.svg')
    if svg_path.exists():
        return True
    try:
        result = subprocess.run(
            ["d2", str(filepath), str(svg_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    print("Reading failing files list...")

    with open(FAILING_FILES_LIST, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]

    files_to_process = []
    for entry in entries:
        parts = entry.split('|||', 1)
        rel_path = parts[0].strip()
        errors = parts[1].strip() if len(parts) > 1 else ""

        # Normalize path (remove double slashes)
        rel_path = re.sub(r'//+', '/', rel_path)
        filepath = PROJECT_ROOT / rel_path

        if filepath.exists():
            files_to_process.append((filepath, errors))
        else:
            print(f"WARNING: File not found: {filepath}")

    print(f"Total files to process: {len(files_to_process)}")

    # Also find any additional failing files not in the original list
    print("Scanning for additional failing files...")
    all_failing = set()
    for proj in (PROJECT_ROOT / "data" / "architecture-docs").iterdir():
        diag_dir = proj / "diagrams"
        if not diag_dir.exists():
            continue
        for d2file in diag_dir.glob("*.d2"):
            svg_file = d2file.with_suffix(".svg")
            if not svg_file.exists():
                all_failing.add(d2file)

    # Add any new files not in original list
    original_set = {f for f, _ in files_to_process}
    new_files = all_failing - original_set
    for f in sorted(new_files):
        files_to_process.append((f, ""))
        print(f"  Added new failing file: {f.name}")

    print(f"Total files to process (including new): {len(files_to_process)}")
    print()

    fixed = 0
    failed = []
    skipped = 0

    for i, (filepath, known_errors) in enumerate(files_to_process):
        # Check if SVG already exists
        svg_path = filepath.with_suffix('.svg')
        if svg_path.exists():
            skipped += 1
            continue

        print(f"  [{i+1}/{len(files_to_process)}] Processing: {filepath.parent.parent.name}/{filepath.name}")

        if iterative_fix(filepath, max_iterations=8):
            # Compile succeeded - generate SVG
            if generate_svg(filepath):
                print(f"    -> FIXED & SVG GENERATED")
                fixed += 1
            else:
                print(f"    -> COMPILED BUT SVG WRITE FAILED")
                failed.append((str(filepath), "SVG write failed"))
        else:
            _, current_errors = run_d2(filepath)
            print(f"    -> STILL FAILING: {current_errors[:200]}")
            failed.append((str(filepath), current_errors[:400]))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files in list: {len(files_to_process)}")
    print(f"Already had SVG (skipped): {skipped}")
    print(f"Fixed this run: {fixed}")
    print(f"Still failing: {len(failed)}")
    print()

    if failed:
        print("STILL FAILING FILES:")
        for filepath, err in failed:
            fname = Path(filepath).name
            proj = Path(filepath).parent.parent.name
            print(f"  {proj}/{fname}")
            print(f"    Error: {err[:200]}")
            print()


if __name__ == "__main__":
    main()

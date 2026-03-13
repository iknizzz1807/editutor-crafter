#!/usr/bin/env python3
"""
Fix failing D2 diagram files that have various syntax errors.
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
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=True) as tmp:
        tmp_path = tmp.name

    result = subprocess.run(
        ["d2", str(filepath), tmp_path],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )

    output = result.stdout + result.stderr
    success = result.returncode == 0

    # Clean up temp file if created
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return success, output


def compile_to_svg(filepath: Path) -> bool:
    """Compile a d2 file to its corresponding SVG."""
    svg_path = filepath.with_suffix(".svg")
    result = subprocess.run(
        ["d2", str(filepath), str(svg_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    return result.returncode == 0


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
        # Check if this line opens a block string: ends with |word or contains ": |word"
        opener_match = re.search(
            r'(\|+)(md|go|python|py|js|javascript|typescript|ts|ruby|rb|bash|sh|c|cpp|rust|java|latex|tex)\s*$',
            line
        )
        if opener_match:
            pipe_count = len(opener_match.group(1))
            lang = opener_match.group(2)
            # Collect block content until we find the matching terminator
            block_lines = [line]
            i += 1
            content_lines = []
            terminator_found = False
            terminator_line = None
            while i < len(lines):
                # terminator: exactly pipe_count pipes on a line (with optional whitespace)
                # but NOT pipe_count+1 pipes
                term_pattern = r'^\s*' + r'\|' * pipe_count + r'\s*$'
                next_pattern = r'^\s*' + r'\|' * (pipe_count + 1)
                if re.match(term_pattern, lines[i]) and not re.match(next_pattern, lines[i]):
                    terminator_found = True
                    terminator_line = lines[i]
                    break
                content_lines.append(lines[i])
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
                    new_opener = re.sub(r'\|+' + lang, new_pipes + lang, line)
                    result.append(new_opener)
                    result.extend(content_lines)
                    # Replace terminator (preserve leading whitespace)
                    leading_ws = re.match(r'^\s*', terminator_line).group()
                    result.append(leading_ws + new_pipes)
                else:
                    result.append(line)
                    result.extend(content_lines)
                    result.append(terminator_line)
            else:
                # No terminator found - block was never closed
                # Add current content_lines back and add a terminator
                full_content = '\n'.join(content_lines)
                max_consecutive_pipes = max(
                    (len(m.group()) for m in re.finditer(r'\|+', full_content)),
                    default=0
                )
                if max_consecutive_pipes >= pipe_count:
                    new_pipe_count = max_consecutive_pipes + 1
                    new_pipes = '|' * new_pipe_count
                    new_opener = re.sub(r'\|+' + lang, new_pipes + lang, line)
                    result.append(new_opener)
                    result.extend(content_lines)
                    result.append(new_pipes)  # Add missing terminator
                else:
                    # Add terminator without changing pipe count
                    result.append(line)
                    result.extend(content_lines)
                    result.append('|' * pipe_count)
                i -= 1  # will be incremented at end of loop
        else:
            result.append(line)
        i += 1
    return '\n'.join(result)


def fix_backtick_block_strings(content: str) -> str:
    """Fix malformed block string terminators like backtick-pipe."""
    # Handle: block string must be terminated with `|
    # This means there's a `| instead of just |
    # Replace backtick-pipe terminators
    content = re.sub(r'`\|', '|', content)
    return content


def fix_single_quote_block_strings(content: str) -> str:
    """Fix malformed block string terminators like single-quote-pipe."""
    # Handle: block string must be terminated with '|
    content = re.sub(r"'\|", '|', content)
    return content


def fix_curly_block_strings(content: str) -> str:
    """Fix malformed block string terminators like }{|."""
    # Handle: block string must be terminated with }{|
    content = re.sub(r'\}\{\|', '||', content)
    return content


def fix_hat_block_strings(content: str) -> str:
    """Fix malformed block string terminators like |^|."""
    # Handle: block string must be terminated with |^|
    # The |^| seems to be meant as a separator, replace opener ||md -> |||md etc.
    content = re.sub(r'\|\^\|', '|||', content)
    return content


def fix_near_elk_constant(content: str, errors: str) -> str:
    """Fix near set to object path (not constant) for elk/dagre layout."""
    # Find patterns like: near: some.object.path
    # Replace with: near: top-right
    lines = content.split('\n')
    result = []
    for line in lines:
        # near: some_path_not_a_constant
        m = re.match(r'^(\s*)near:\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*$', line)
        if m:
            val = m.group(2)
            # Valid constants
            valid_constants = {
                'top-left', 'top-center', 'top-right',
                'center-left', 'center-right',
                'bottom-left', 'bottom-center', 'bottom-right'
            }
            if val not in valid_constants:
                # This is a non-constant path - check if it looks like an object path
                if '.' in val or '_' in val:
                    indent = m.group(1)
                    # Choose top-right for annotation boxes, top-center for titles
                    if 'annot' in val.lower() or 'note' in val.lower() or 'legend' in val.lower():
                        new_val = 'top-right'
                    else:
                        new_val = 'top-right'
                    result.append(f'{indent}near: {new_val}')
                    continue
        result.append(line)
    return '\n'.join(result)


def fix_near_non_root(content: str) -> str:
    """Remove near: from nested shapes (only works at root level)."""
    lines = content.split('\n')
    result = []
    depth = 0
    for line in lines:
        # Track nesting depth
        # Count opens and closes (rough estimate)
        opens = len(re.findall(r'\{', line))
        closes = len(re.findall(r'\}', line))

        # Check if this line is a near: line inside a nested shape
        near_match = re.match(r'^(\s+)near:\s+', line)
        if near_match and depth > 0:
            # Skip this near: line (remove it)
            depth += opens - closes
            continue

        depth += opens - closes
        result.append(line)
    return '\n'.join(result)


def fix_near_invalid_constant(content: str, errors: str) -> str:
    """Fix invalid near constants like 'right-center', 'T500', etc."""
    # Extract invalid constants from error messages
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
        # Try to find a valid replacement
        replacement = None
        for k, v in valid_map.items():
            if k in lower_key:
                replacement = v
                break
        if replacement is None:
            replacement = 'top-right'

        # Replace in content: near: "invalid_key" or near: invalid_key
        content = re.sub(
            r'(near:\s*)"?' + re.escape(invalid_key) + r'"?',
            f'near: {replacement}',
            content
        )

    return content


def fix_near_absolute_path_in_quotes(content: str) -> str:
    """Fix near: "Path"."Sub" (quoted path) by replacing with constant."""
    # Pattern: near: "X"."Y" - this is a quoted absolute path, invalid for elk
    content = re.sub(
        r'(\.near:\s*)"[^"]*"\."[^"]*"',
        r'\1near: top-right',
        content
    )
    content = re.sub(
        r'^(\s*near:\s*)"[^"]*"\."[^"]*"',
        r'\1top-right',
        content,
        flags=re.MULTILINE
    )
    return content


def fix_missing_vars_colors(content: str, errors: str) -> str:
    """Add missing color variables to vars.colors block."""
    missing_var_re = re.compile(r'could not resolve variable "vars\.colors\.(\w+)"')
    missing_vars = missing_var_re.findall(errors)

    if not missing_vars:
        return content

    # Default colors for common variable names
    color_defaults = {
        'primary': '#2196F3',
        'secondary': '#9C27B0',
        'accent': '#FF9800',
        'success': '#4CAF50',
        'error': '#F44336',
        'warning': '#FF9800',
        'info': '#2196F3',
        'bg': '#FFFFFF',
        'text': '#333333',
        'border': '#CCCCCC',
        'highlight': '#FFEB3B',
        'dark': '#212121',
        'light': '#F5F5F5',
        'red': '#F44336',
        'blue': '#2196F3',
        'green': '#4CAF50',
        'yellow': '#FFEB3B',
        'orange': '#FF9800',
        'purple': '#9C27B0',
        'gray': '#9E9E9E',
        'grey': '#9E9E9E',
        'white': '#FFFFFF',
        'black': '#212121',
    }

    for var_name in missing_vars:
        # Find the vars.colors block and add the missing key
        lower_name = var_name.lower()
        color_val = color_defaults.get(lower_name, '#607D8B')

        # Try to find colors block and insert
        colors_block_pattern = re.compile(r'(colors:\s*\{[^}]*?)(\})', re.DOTALL)
        m = colors_block_pattern.search(content)
        if m:
            new_colors = m.group(1) + f'    {var_name}: "{color_val}"\n  ' + m.group(2)
            content = content[:m.start()] + new_colors + content[m.end():]

    return content


def fix_key_length(content: str, errors: str) -> str:
    """Truncate keys that exceed maximum allowed length of 518."""
    # Find lines that are being flagged as too long
    # These are usually SQL table column values or class field declarations
    # We need to truncate long strings to under ~500 chars
    MAX_KEY_LEN = 100  # Keep much shorter than 518 to be safe

    lines = content.split('\n')
    result = []
    for line in lines:
        # Check if the line has a long quoted string value
        # Pattern: key: "very long value"
        m = re.match(r'^(\s*\S+:\s*)"(.+)"(\s*)$', line)
        if m and len(m.group(2)) > 200:
            truncated = m.group(2)[:150] + '...'
            result.append(f'{m.group(1)}"{truncated}"{m.group(3)}')
            continue

        # Pattern: "long key" { ... } or "long key" (in sql_table etc)
        # Check for unquoted long keys or quoted long keys at start
        if len(line.strip()) > 520:
            # Find if there's a very long quoted string
            m2 = re.match(r'^(\s*)(\"[^\"]{200,}\"|[^\s\{:]{200,})(.*)', line)
            if m2:
                key_part = m2.group(2)
                if key_part.startswith('"') and key_part.endswith('"'):
                    truncated = key_part[1:151] + '...'
                    result.append(f'{m2.group(1)}"{truncated}"{m2.group(3)}')
                else:
                    truncated = key_part[:150] + '...'
                    result.append(f'{m2.group(1)}{truncated}{m2.group(3)}')
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
        # Remove text-align style
        if re.match(r'^\s*text-align\s*:', line):
            continue
        # Remove source-arrowhead style
        if re.match(r'^\s*source-arrowhead\s*:', line):
            continue
        # Remove invalid font
        if re.search(r'"sans-serif"\s+is\s+not', line):
            continue
        if re.match(r'^\s*font:\s*"sans-serif"', line):
            continue
        result.append(line)
    return '\n'.join(result)


def fix_style_prefix(content: str) -> str:
    """Fix opacity: X -> style.opacity: X and bold: true -> style.bold: true."""
    lines = content.split('\n')
    result = []
    for line in lines:
        # Fix bare opacity (not inside style block)
        if re.match(r'^(\s*)opacity:\s*', line):
            m = re.match(r'^(\s*)opacity:\s*(.+)$', line)
            if m:
                # Check if we're inside a style block by looking at indentation
                # Simple heuristic: just prefix with style.
                # D2 accepts style.opacity at any level
                result.append(f'{m.group(1)}style.opacity: {m.group(2)}')
                continue
        # Fix bare bold (not inside style block)
        if re.match(r'^(\s*)bold:\s*', line):
            m = re.match(r'^(\s*)bold:\s*(.+)$', line)
            if m:
                result.append(f'{m.group(1)}style.bold: {m.group(2)}')
                continue
        result.append(line)
    return '\n'.join(result)


def fix_indexed_edges(content: str, errors: str) -> str:
    """Remove (N) index from edge references."""
    # Pattern: (edge_src -> edge_dst)[N].something
    # Replace with edge_src -> edge_dst: (remove the indexing)
    # Or just remove the indexed edge reference lines
    content = re.sub(r'\(([^)]+)\)\[\d+\]\.', r'\1.', content)
    return content


def fix_sql_table_children(content: str) -> str:
    """Flatten nested structures under sql_table columns."""
    # This is complex - for now, remove nested blocks under column definitions
    # in sql_table shapes
    # Pattern: inside sql_table, a field has a { block } - flatten it
    # This requires context-aware parsing which is complex
    # For now, handled by iterative d2 compilation
    return content


def fix_edge_map_keys(content: str, errors: str) -> str:
    """Fix edge declarations with non-reserved keywords."""
    # The error is for lines like:
    # S1 -> S2: replicate {T2, idx2}  <- the label has {...} which gets parsed as map
    # Fix: quote the label

    # Parse error line numbers
    line_re = re.compile(r':(\d+):\d+: edge map keys must be reserved keywords')
    error_lines = [int(m.group(1)) for m in line_re.finditer(errors)]

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1  # 0-indexed
        if 0 <= idx < len(result):
            line = result[idx]
            # Pattern: actor -> actor: label {key: val}
            # The issue: {T2, idx2} after the label gets parsed as a map
            # Fix: quote the entire label
            m = re.match(r'^(\s*)(.+?->.*?:\s*)(\{[^}]*\})(.*)$', line)
            if m:
                # The {T2, idx2} is the label map - this is not valid
                # Just remove the map part or convert to quoted string
                label_map = m.group(3)
                # Extract content of the map as a string label
                inner = label_map[1:-1].strip()
                result[idx] = f'{m.group(1)}{m.group(2)}"{inner}"{m.group(4)}'

    return '\n'.join(result)


def fix_substitutions(content: str) -> str:
    """Fix $varname -> ${varname} substitutions."""
    # But only outside of block strings (md content) where $ is just text
    # This is tricky - we need context
    # Simple approach: replace $word that's not already ${...}
    # Pattern: $identifier (not ${ already)
    # Only in non-block-string areas

    # For safety, only fix obvious variable references in style/near contexts
    # like style.fill: $colors.something
    content = re.sub(
        r'\$([a-zA-Z_][a-zA-Z0-9_.]*)',
        lambda m: '${' + m.group(1) + '}' if not m.group(0).startswith('${') else m.group(0),
        content
    )
    return content


def fix_layers_in_class(content: str, errors: str) -> str:
    """Fix 'layers' used as a field name in class shapes (conflicts with D2 keyword)."""
    # The error: layers must be declared at a board root scope
    # This happens when a class shape field is named 'layers'
    # Fix: rename to 'layer_list' or similar

    # Parse error line numbers
    line_re = re.compile(r':(\d+):\d+: layers must be declared at a board root scope')
    error_lines = [int(m.group(1)) for m in line_re.finditer(errors)]

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1  # 0-indexed
        if 0 <= idx < len(result):
            line = result[idx]
            # Replace field named 'layers' with a quoted version
            # Pattern: layers: "type" inside class shape
            if re.match(r'^\s*layers\s*:', line):
                result[idx] = line.replace('layers:', '"layers":', 1)

    return '\n'.join(result)


def fix_edge_board_keyword(content: str, errors: str) -> str:
    """Fix 'edge with board keyword alone doesn't make sense'."""
    # Parse error line numbers
    line_re = re.compile(r':(\d+):\d+: edge with board keyword alone')
    error_lines = [int(m.group(1)) for m in line_re.finditer(errors)]

    if not error_lines:
        return content

    lines = content.split('\n')
    result = list(lines)

    for lineno in error_lines:
        idx = lineno - 1
        if 0 <= idx < len(result):
            line = result[idx]
            # Comment out the problematic line
            result[idx] = '# FIXED: ' + line

    return '\n'.join(result)


def fix_near_ancestor(content: str) -> str:
    """Remove 'near: ancestor' lines that reference ancestors."""
    # near keys cannot be set to an ancestor
    # near keys cannot be set to descendants of special objects
    # near keys cannot be set to an object within sequence diagrams
    # Just remove these near: lines that reference paths
    lines = content.split('\n')
    result = []
    for line in lines:
        if re.match(r'^\s*near:\s+', line):
            val = re.match(r'^\s*near:\s+(.+)$', line)
            if val:
                v = val.group(1).strip()
                valid_constants = {
                    'top-left', 'top-center', 'top-right',
                    'center-left', 'center-right',
                    'bottom-left', 'bottom-center', 'bottom-right'
                }
                if v not in valid_constants:
                    # Remove this near: line
                    continue
        result.append(line)
    return '\n'.join(result)


def fix_class_method_args(content: str, errors: str) -> str:
    """Fix class shape methods with problematic argument syntax."""
    # Error: unexpected text after map key
    # Caused by things like: +method<T>(arg, ...): "return"
    # or +method(arg, ...): "return"  (with ...)
    # Fix: quote the entire method signature

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
            # Pattern: optional +/-  method_name<T>(args): "return"
            # The problem is usually <T> template or ... or complex args
            m = re.match(r'^(\s*)([\+\-]?\w[\w\s]*[\w<>\[\](),.* &]+(?:\(.*\))?(?:\s*:\s*.+)?)$', line)
            if m:
                indent = m.group(1)
                key_part = m.group(2).strip()

                # If the key contains <, >, [, ], (, ) - these need quoting
                if re.search(r'[<>\[\]()]', key_part) and not key_part.startswith('"'):
                    # Split on ': ' to separate key from value
                    colon_match = re.match(r'^(.+?):\s*"(.+)"$', key_part)
                    if colon_match:
                        k = colon_match.group(1).strip()
                        v = colon_match.group(2)
                        if not k.startswith('"'):
                            k = f'"{k}"'
                        result[idx] = f'{indent}{k}: "{v}"'
                    else:
                        # Just quote the whole thing as a key
                        if not key_part.startswith('"'):
                            result[idx] = f'{indent}"{key_part}"'

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
            # This usually happens with things like: key: value extra_stuff
            # Or method_name<T>(args) without quotes
            m = re.match(r'^(\s*)(\S+(?:\s+\S+)*):\s*(.+)$', line)
            if m:
                indent = m.group(1)
                key = m.group(2)
                val = m.group(3)
                # Quote the key if it has spaces or special chars
                if re.search(r'[\s<>()\[\]]', key) and not key.startswith('"'):
                    key = f'"{key}"'
                # Quote the value if it's not already quoted and has spaces
                if re.search(r'\s', val) and not val.startswith('"') and not val.startswith('|') and not val.startswith('{'):
                    val = f'"{val}"'
                result[idx] = f'{indent}{key}: {val}'

    return '\n'.join(result)


def fix_double_quoted_near(content: str, errors: str) -> str:
    """Fix near: set to a quoted path like 'near: "A"."B"'."""
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
            # Check if this is a near: line with double-quoted path
            if 'near:' in line:
                result[idx] = re.sub(r'near:.*$', 'near: top-right', line)

    return '\n'.join(result)


def fix_arrowhead_shape(content: str) -> str:
    """Fix invalid arrowhead shapes."""
    # Remove invalid arrowhead shape declarations
    content = re.sub(r'shape:\s*box\b', 'shape: arrow', content)
    return content


def fix_near_grid_cell(content: str) -> str:
    """Remove near: from grid cell descendants."""
    # near keys cannot be set to descendants of special objects, like grid cells
    return fix_near_ancestor(content)


def apply_all_fixes(filepath: Path, content: str, errors: str) -> tuple[str, list[str]]:
    """Apply all applicable fixes to content based on errors."""
    fixes_applied = []
    original = content

    # Priority order: block strings first (they cause cascade errors)

    # Fix malformed terminators first
    if '`|' in content or "block string must be terminated with `|" in errors:
        content = fix_backtick_block_strings(content)
        if content != original:
            fixes_applied.append("fixed backtick-pipe terminators")
            original = content

    if "block string must be terminated with '|" in errors:
        content = fix_single_quote_block_strings(content)
        if content != original:
            fixes_applied.append("fixed single-quote-pipe terminators")
            original = content

    if "}{\|" in errors or "block string must be terminated with }{\|" in errors:
        content = fix_curly_block_strings(content)
        if content != original:
            fixes_applied.append("fixed curly-pipe terminators")
            original = content

    if "|^|" in errors or "block string must be terminated with |^|" in errors:
        content = fix_hat_block_strings(content)
        if content != original:
            fixes_applied.append("fixed hat-pipe terminators")
            original = content

    # Fix block strings with pipe content
    if ('block string must be terminated with |' in errors or
        'unexpected text after md block string' in errors or
        'unexpected text after latex block string' in errors or
        'unexpected text after map key' in errors or
        'maps must be terminated with }' in errors):
        new_content = fix_block_strings(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed block strings (pipe count)")
            original = content

    # Fix key length issues
    if 'key length' in errors and 'exceeds maximum allowed length' in errors:
        new_content = fix_key_length(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed key lengths")
            original = content

    # Fix near: elk constant issues
    if 'layout engine "elk" only supports constant values' in errors or \
       'layout engine "dagre" only supports constant values' in errors:
        new_content = fix_near_elk_constant(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed near: elk/dagre constant")
            original = content

    # Fix constant near in nested shapes
    if 'constant near keys can only be set on root level shapes' in errors:
        new_content = fix_near_non_root(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed near: in nested shapes (removed)")
            original = content

    # Fix invalid near constants
    if 'must be the absolute path to a shape or one of the following constants' in errors:
        new_content = fix_near_invalid_constant(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed invalid near constants")
            original = content

    # Fix near: ancestor / sequence diagram / grid cell
    if ('near keys cannot be set to an ancestor' in errors or
        'near keys cannot be set to an object within sequence diagrams' in errors or
        'near keys cannot be set to descendants of special objects' in errors or
        'invalid "near" field' in errors):
        new_content = fix_near_ancestor(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed invalid near (ancestor/sequence)")
            original = content

    # Fix double-quoted near path
    if 'unexpected text after double quoted string' in errors:
        new_content = fix_double_quoted_near(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed double-quoted near path")
            original = content

    # Fix missing vars.colors
    if 'could not resolve variable "vars.colors.' in errors:
        new_content = fix_missing_vars_colors(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed missing vars.colors")
            original = content

    # Fix unknown shapes
    if 'unknown shape "capsule"' in errors or 'unknown shape "sticky_note"' in errors or \
       'unknown shape "line"' in errors:
        new_content = fix_unknown_shapes(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed unknown shapes")
            original = content

    # Fix invalid style keywords
    if 'invalid style keyword' in errors or '"sans-serif" is not a valid font' in errors:
        new_content = fix_invalid_style_keywords(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed invalid style keywords")
            original = content

    # Fix style prefix
    if 'opacity must be style.opacity' in errors or 'bold must be style.bold' in errors:
        new_content = fix_style_prefix(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed style prefix (opacity/bold)")
            original = content

    # Fix indexed edges
    if 'indexed edge does not exist' in errors:
        new_content = fix_indexed_edges(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed indexed edges")
            original = content

    # Fix edge map keys
    if 'edge map keys must be reserved keywords' in errors:
        new_content = fix_edge_map_keys(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed edge map keys")
            original = content

    # Fix layers keyword in class shapes
    if 'layers must be declared at a board root scope' in errors:
        new_content = fix_layers_in_class(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed 'layers' field in class shape")
            original = content

    # Fix edge board keyword
    if "edge with board keyword alone doesn't make sense" in errors:
        new_content = fix_edge_board_keyword(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed edge board keyword")
            original = content

    # Fix substitutions
    if 'substitutions must begin on {' in errors:
        # Only fix $var patterns that aren't already ${var}
        new_content = re.sub(
            r'\$(?!\{)([a-zA-Z_][a-zA-Z0-9_.]*)',
            r'${\1}',
            content
        )
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed substitution syntax $var -> ${var}")
            original = content

    # Fix class method args (unexpected text after map key)
    if 'unexpected text after map key' in errors:
        new_content = fix_class_method_args(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed class method args")
            original = content

    # Fix unquoted string errors
    if 'unexpected text after unquoted string' in errors:
        new_content = fix_unquoted_string_errors(content, errors)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed unquoted string errors")
            original = content

    # Fix near: arrowhead shape
    if 'invalid shape, can only set "box" for arrowheads' in errors:
        new_content = fix_arrowhead_shape(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("fixed arrowhead shape")
            original = content

    return content, fixes_applied


def iterative_fix(filepath: Path, max_iterations: int = 5) -> bool:
    """Iteratively fix a file until it compiles or max iterations reached."""
    global FIXED_COUNT, FIX_LOG

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        original_content = f.read()

    content = original_content
    all_fixes = []

    for iteration in range(max_iterations):
        success, output = run_d2(filepath)
        if success:
            if all_fixes:
                # File was fixed in a previous iteration
                FIX_LOG.append(f"FIXED: {filepath} - fixes: {', '.join(all_fixes)}")
            return True

        errors = output
        new_content, fixes = apply_all_fixes(filepath, content, errors)

        if not fixes:
            # No fixes applied - can't fix this file
            break

        all_fixes.extend(fixes)

        if new_content != content:
            content = new_content
            # Write the fixed content back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # Content didn't change, stop iterating
            break

    # Final check
    success, final_output = run_d2(filepath)
    if success:
        FIX_LOG.append(f"FIXED: {filepath} - fixes: {', '.join(all_fixes)}")
        return True
    else:
        # Restore original if we made things worse
        # Actually, keep whatever we have since some fixes may be partial
        FIX_LOG.append(f"FAILED: {filepath} - tried: {', '.join(all_fixes)} - remaining: {final_output[:200]}")
        return False


def generate_svg(filepath: Path) -> bool:
    """Generate SVG for a file that compiles successfully."""
    svg_path = filepath.with_suffix('.svg')
    result = subprocess.run(
        ["d2", str(filepath), str(svg_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    return result.returncode == 0


def main():
    global FIXED_COUNT, STILL_FAILING

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

    # Separate files with no errors (just need SVG generation) from files with errors
    no_error_files = [(f, e) for f, e in files_to_process if not e]
    error_files = [(f, e) for f, e in files_to_process if e]

    print(f"Files needing only SVG generation: {len(no_error_files)}")
    print(f"Files with errors to fix: {len(error_files)}")
    print()

    # Process files that just need SVG generation
    print("=== Generating SVGs for files with no errors ===")
    svg_generated = 0
    svg_failed = []

    for filepath, _ in no_error_files:
        svg_path = filepath.with_suffix('.svg')
        if svg_path.exists():
            print(f"  SKIP (SVG exists): {filepath.name}")
            svg_generated += 1
            continue

        # First verify it compiles
        success, output = run_d2(filepath)
        if success:
            if generate_svg(filepath):
                print(f"  GENERATED: {filepath.name}")
                svg_generated += 1
            else:
                print(f"  FAILED TO WRITE SVG: {filepath.name}")
                svg_failed.append(str(filepath))
        else:
            # Has errors after all - add to error files for processing
            print(f"  HAS ERRORS: {filepath.name} - {output[:100]}")
            error_files.append((filepath, output))

    print(f"\nSVGs generated: {svg_generated}, failed: {len(svg_failed)}")
    print()

    # Process files with actual errors
    print("=== Fixing files with errors ===")
    fixed = 0
    failed = []

    for i, (filepath, known_errors) in enumerate(error_files):
        # Check if SVG already exists (might have been fixed by a previous run)
        svg_path = filepath.with_suffix('.svg')
        if svg_path.exists():
            print(f"  [{i+1}/{len(error_files)}] SKIP (SVG exists): {filepath.name}")
            fixed += 1
            continue

        print(f"  [{i+1}/{len(error_files)}] Processing: {filepath.name}")

        if iterative_fix(filepath, max_iterations=6):
            # Compile succeeded - generate SVG
            if generate_svg(filepath):
                print(f"    -> FIXED & SVG GENERATED")
                fixed += 1
            else:
                print(f"    -> FIXED BUT SVG WRITE FAILED")
                failed.append((str(filepath), "SVG write failed"))
        else:
            # Get current errors
            _, current_errors = run_d2(filepath)
            print(f"    -> STILL FAILING: {current_errors[:150]}")
            failed.append((str(filepath), current_errors[:300]))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(files_to_process)}")
    print(f"SVGs generated (no-error files): {svg_generated}")
    print(f"Error files fixed: {fixed}")
    print(f"Still failing: {len(failed)}")
    print()

    if failed:
        print("STILL FAILING FILES:")
        for filepath, err in failed:
            print(f"  {filepath}")
            print(f"    Error: {err[:200]}")
            print()

    print()
    print("FIX LOG:")
    for entry in FIX_LOG:
        print(f"  {entry}")


if __name__ == "__main__":
    main()

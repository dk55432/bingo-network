import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from winning_pattern import WinningPattern, MaskPattern, AnyPattern, AllPattern, ConstraintPattern
from bingo_card import BingoCard

# Leaves you already have:
# MaskPattern(name, mask)
# AnyPattern(patterns)
# AllPattern(patterns)
# ConstraintPattern.at_least(name, mask, n) / ConstraintPattern.at_most(name, mask, n)

class ParseError(Exception):
    pass

def strip_blank_and_comments(line: str) -> str:
    # treat lines starting with # as comments; ignore trailing comments if you want
    s = line.rstrip("\n")
    if not s.strip():
        return ""
    if s.lstrip().startswith("#"):
        return ""
    return s

def indent_of(line: str) -> int:
    # count leading spaces; disallow tabs or normalize them first
    return len(line) - len(line.lstrip(" "))

def parse_mask_row(line: str) -> List[bool]:
    s = line.strip()
    if len(s) != 5:
        raise ParseError(f"Mask row must be 5 chars, got {len(s)}: {line!r}")
    out = []
    for ch in s:
        if ch == '-':
            out.append(False)
        elif ch in ('X', 'x'):
            out.append(True)
        else:
            raise ParseError(f"Invalid mask char {ch!r}; use '-' or 'X'")
    return out

def parse_masks(lines: List[str], i: int) -> Tuple[Dict[str, List[List[bool]]], int]:
    masks = {}
    while i < len(lines):
        raw = strip_blank_and_comments(lines[i])
        if not raw:
            i += 1
            continue
        if raw.startswith("PATTERN "):
            break

        if raw.startswith("MASK "):
            name = raw[len("MASK "):].strip()
            grid = []
            # next 5 non-empty lines? (or exactly next 5 lines—pick one)
            j = i + 1
            while len(grid) < 5:
                if not (j < len(lines)):
                    raise ParseError(f"index j {j} must be < len(lines) {len(lines)}")
                cand = strip_blank_and_comments(lines[j])
                if cand:  # non-empty after comment/blank removal
                    grid.append(parse_mask_row(cand))
                j += 1
            if len(grid) != 5:
                raise ParseError(f"Mask {name} did not have 5 rows")
            if name in masks:
                raise ParseError(f"Duplicate mask name {name!r}")
            masks[name] = grid
            i = j
        else:
            # ignore other header lines until PATTERNS start
            i += 1
    return masks, i

# Matches "AtLeast(maskname, n)" or "AtMost(maskname, n)", with optional
# whitespace around the comma. Mask names can contain letters, digits,
# hyphens, and underscores (matches how MASK names are already written
# in the config, e.g. "diagonal-bottomleft-upperright").
CONSTRAINT_RE = re.compile(
    r'^(AtLeast|AtMost)\(\s*([A-Za-z0-9_\-]+)\s*,\s*(-?\d+)\s*\)$'
)

def parse_pattern_expr(lines: List[str], i: int, base_indent: int, masks: Dict[str, List[List[bool]]]):
    # expects current line at some indent > base_indent? We'll manage calls accordingly.
    # Return (WinningPattern, new_index)
    raw = strip_blank_and_comments(lines[i])
    if not raw:
        return None, i

    cur_indent = indent_of(lines[i])
    if cur_indent <= base_indent:
        raise ParseError("Unexpected indentation")

    # operator line?
    content = raw.strip()
    if content == "AND" or content == "OR":
        op = content
        op_indent = cur_indent
        i += 1

        children = []
        while i < len(lines):
            nxt_raw = strip_blank_and_comments(lines[i])
            if not nxt_raw:
                i += 1
                continue

            nxt_indent = indent_of(lines[i])
            if nxt_indent <= op_indent:
                break  # end of this operator block

            child_pat, i = parse_pattern_node(lines, i, op_indent, masks)
            children.append(child_pat)

        if op == "AND":
            return AllPattern(children), i
        else:
            return AnyPattern(children), i

    raise ParseError(f"Expected AND/OR at indent {cur_indent}, got: {raw!r}")

def parse_pattern_node(lines: List[str], i: int, parent_indent: int, masks):
    raw = strip_blank_and_comments(lines[i])
    if not raw:
        return None, i

    cur_indent = indent_of(lines[i])
    if cur_indent <= parent_indent:
        raise ParseError("Child line indentation must be greater than parent")

    content = raw.strip()

    # nested operator
    if content in ("AND", "OR"):
        return parse_pattern_expr(lines, i, parent_indent, masks)

    # constraint leaf: AtLeast(maskname, n) or AtMost(maskname, n)
    if content.startswith("AtLeast(") or content.startswith("AtMost("):
        constraint_match = CONSTRAINT_RE.match(content)
        if not constraint_match:
            raise ParseError(f"Bad AtLeast/AtMost syntax: {content!r}")
        func_name, maskname, n_str = constraint_match.groups()
        if maskname not in masks:
            raise ParseError(f"Unknown mask {maskname!r}")
        n = int(n_str)
        if n < 0 or n > 25:
            raise ParseError(f"Int {n} outside range 0..25")
        mask = masks[maskname]
        if func_name == "AtLeast":
            return ConstraintPattern.at_least(maskname, mask, n), i + 1
        else:
            return ConstraintPattern.at_most(maskname, mask, n), i + 1

    # mask leaf reference
    # whole remainder is mask name, so things like diagonal\ work fine
    maskname = content
    if maskname not in masks:
        raise ParseError(f"Unknown mask {maskname!r}")
    return MaskPattern(maskname, masks[maskname]), i + 1

def parse_patterns(lines: List[str], i: int, masks) -> Tuple[Dict[str, WinningPattern], int]:
    patterns = {}
    while i < len(lines):
        raw = strip_blank_and_comments(lines[i])
        if not raw:
            i += 1
            continue
        if raw.startswith("MASK "):
            raise ParseError("Found MASK in PATTERNS section unexpectedly")
        if raw.startswith("PATTERN "):
            name = raw[len("PATTERN "):].strip()
            i += 1
            # find next non-empty line: should be AND or OR
            while i < len(lines) and not strip_blank_and_comments(lines[i]):
                i += 1
            if i >= len(lines):
                raise ParseError(f"PATTERN {name} missing AND/OR")

            # next operator block starts at its indent; we use base_indent = that indent-1
            # simpler: use base_indent = indent_of(line) - 1 so child indent condition works.
            base_indent = indent_of(lines[i]) - 1
            expr, i = parse_pattern_expr(lines, i, base_indent, masks)
            if name in patterns:
                raise ParseError(f"Duplicate pattern name {name!r}")
            patterns[name] = expr
            continue
        i += 1
    return patterns, i


def load_patterns_from_file(path: str) -> Dict[str, WinningPattern]:
    """Read and parse a patterns config file in one call, returning just
    the finished {pattern_name: WinningPattern} dict — the masks dict is
    an internal parsing detail callers loading a whole config file don't
    need to see."""
    with open(path) as f:
        lines = f.read().splitlines()
    masks, i = parse_masks(lines, 0)
    patterns, _ = parse_patterns(lines, i, masks)
    return patterns

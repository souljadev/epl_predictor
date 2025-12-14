"""
Lightweight unused-Python-file detector.

Scans the repo (excluding certain folders), builds an import graph, and
reports Python files that are not reachable from the workflow entrypoint
`daily_agent.py` (and scripts invoked by `scripts/full_pipeline.py`).

This is best-effort (static). It looks at `import` / `from` statements and
also literal ``"*.py"`` filenames referenced in code (for subprocess calls).

Usage:
    python scripts/tools/find_unused.py

"""
import ast
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE = {".venv", "data", "logs", "processed", "metrics_dashboard.ipynb"}


def list_py_files(root: Path):
    files = []
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if parts & EXCLUDE:
            continue
        # ignore hidden dirs
        if any(part.startswith(".") for part in p.parts):
            continue
        files.append(p)
    return files


def module_name_for_path(root: Path, path: Path):
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def parse_imports(path: Path):
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return set(), set()

    imports = set()
    literal_files = set()

    try:
        tree = ast.parse(src)
    except Exception:
        # fallback: regex find "something.py"
        for m in re.finditer(r"[\'\"]([\w\-_/]+\.py)[\'\"]", src):
            literal_files.add(m.group(1))
        return imports, literal_files

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            if mod:
                imports.add(mod)
        elif isinstance(node, (ast.Constant, ast.Str)):
            # find literal .py filenames
            val = getattr(node, 's', None) or getattr(node, 'value', None)
            if isinstance(val, str) and val.endswith('.py'):
                literal_files.add(val)

    return imports, literal_files


def build_graph(py_files, root):
    mod_by_name = {module_name_for_path(root, p): p for p in py_files}
    path_by_name = {p: module_name_for_path(root, p) for p in py_files}

    edges = {p: set() for p in py_files}

    for p in py_files:
        imports, literal_files = parse_imports(p)
        for imp in imports:
            # match exact module or prefix
            for mod_name, target_path in mod_by_name.items():
                if mod_name == imp or mod_name.startswith(imp + '.') or imp.startswith(mod_name + '.'):
                    edges[p].add(target_path)
        # also match referenced literal filenames
        for lit in literal_files:
            try:
                cand = (p.parent / lit).resolve()
                if cand in edges:
                    edges[p].add(cand)
            except Exception:
                # also try root-relative
                try:
                    cand = (root / lit).resolve()
                    if cand in edges:
                        edges[p].add(cand)
                except Exception:
                    pass

    return edges, mod_by_name


def find_reachable(edges, start_paths):
    seen = set()
    stack = list(start_paths)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nb in edges.get(cur, []):
            if nb not in seen:
                stack.append(nb)
    return seen


def main():
    py_files = list_py_files(ROOT)
    edges, mod_by_name = build_graph(py_files, ROOT)

    # Starting entrypoints
    start = set()
    # daily_agent.py
    da = ROOT / 'daily_agent.py'
    if da.exists():
        start.add(da.resolve())
    # full_pipeline.py (subprocess invoked) - include it explicitly
    fp = ROOT / 'scripts' / 'full_pipeline.py'
    if fp.exists():
        start.add(fp.resolve())
    # Also mark any .py literal filenames referenced by full_pipeline as used
    _, lits = parse_imports(fp) if fp.exists() else (set(), set())
    for lit in lits:
        cand = (fp.parent / lit).resolve()
        if cand.exists():
            start.add(cand)
    
    # Also include top-level dashboard app if exists (user might use it)
    dash = ROOT / 'dashboard' / 'app.py'
    if dash.exists():
        start.add(dash.resolve())

    reachable = find_reachable(edges, start)

    all_set = set(p.resolve() for p in py_files)
    unreachable = sorted(list(all_set - reachable))

    print(f"Project root: {ROOT}")
    print(f"Scanned Python files: {len(py_files)}")
    print(f"Reachable from entrypoints: {len(reachable)}")
    print("")

    if not unreachable:
        print("No unreachable Python files detected (best-effort).")
        return

    print("Potentially unused Python files (best-effort):")
    for p in unreachable:
        print(f" - {p.relative_to(ROOT)}")

    print("")
    print("Notes:")
    print(" - This is static and best-effort. Review before deleting.")
    print(" - Files referenced only dynamically or used as CLI entrypoints may be false positives.")


if __name__ == '__main__':
    main()

"""Read-only project census; write documentation evidence inside this directory."""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, encoding='utf-8').strip()


def main():
    tracked = git('ls-files').splitlines()
    python = [f for f in tracked if f.endswith('.py')]
    groups = defaultdict(lambda: {'files': 0, 'physical_lines': 0})
    tests = 0
    for f in python:
        text = (ROOT / f).read_text(encoding='utf-8-sig')
        group = f.split('/')[0]
        groups[group]['files'] += 1
        groups[group]['physical_lines'] += len(text.splitlines())
        if f.startswith('tests/'):
            tests += sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_')
                         for n in ast.walk(ast.parse(text)))
    result = {
        'date': '2026-10-15', 'root': str(ROOT), 'head': git('rev-parse', 'HEAD'),
        'branch': git('branch', '--show-current'), 'tracked_files': len(tracked),
        'python_files': len(python), 'python_physical_lines': sum(x['physical_lines'] for x in groups.values()),
        'python_groups': dict(groups), 'static_test_function_definitions': tests,
        'test_count_note': 'Static definitions, not parametrized collected cases or a new test run.',
        'git_status_at_audit': git('status', '--short'), 'tracked_paths': tracked,
        'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in python},
    }
    (HERE / 'repository_snapshot.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tree = {}
    for f in tracked:
        node = tree
        parts = f.split('/')
        for p in parts:
            node = node.setdefault(p, {})
    out = ['RAGScope/']
    def walk(node, prefix=''):
        for i, (name, child) in enumerate(sorted(node.items())):
            last = i == len(node)-1
            out.append(prefix + ('└── ' if last else '├── ') + name + ('/' if child else ''))
            if child:
                walk(child, prefix + ('    ' if last else '│   '))
    walk(tree)
    (HERE / 'repository_tree.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'source_sha256','tracked_paths','git_status_at_audit'}}, ensure_ascii=True))


if __name__ == '__main__':
    main()

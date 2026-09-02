#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Reports how far the numpydoc conversion has got.

Every function in ``grand/`` is to carry a numpydoc docstring with a
description, ``Parameters``, ``Returns`` where it returns something, and a
worked ``.. jupyter-execute::`` example.  This counts what is there::

    python quality/docstring_coverage.py            # summary
    python quality/docstring_coverage.py --by-file  # per module

A module counts as *converted* when every function in it satisfies all four,
at which point its entry can be deleted from ``per-file-ignores`` in
``pyproject.toml``.  That list is the ratchet: it may shrink and must never
grow.
"""

import argparse
import ast
import pathlib
import sys


def audit(path):
    r"""Returns ``(total, described, parameters, returns, examples)`` for a file."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return (0, 0, 0, 0, 0)
    total = described = params = returns = examples = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        total += 1
        doc = ast.get_docstring(node) or ''
        if doc.strip():
            described += 1
        if 'Parameters\n' in doc or 'Parameters\r\n' in doc:
            params += 1
        if 'Returns\n' in doc:
            returns += 1
        if 'jupyter-execute' in doc:
            examples += 1
    return (total, described, params, returns, examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--by-file', action='store_true')
    parser.add_argument('root', nargs='?', default='grand')
    args = parser.parse_args()

    rows = []
    for path in sorted(pathlib.Path(args.root).rglob('*.py')):
        counts = audit(path)
        if counts[0]:
            rows.append((str(path), counts))

    columns = ('functions', 'described', 'Parameters', 'Returns', 'Examples')
    totals = [sum(r[1][i] for r in rows) for i in range(5)]

    if args.by_file:
        print('%-46s %s' % ('module', '  '.join('%10s' % c for c in columns)))
        for name, counts in sorted(rows, key=lambda r: -r[1][0]):
            print('%-46s %s' % (name[:46],
                                '  '.join('%10d' % c for c in counts)))
        print()

    print('%-46s %s' % ('TOTAL', '  '.join('%10d' % t for t in totals)))
    print('%-46s %s' % ('', '  '.join(
        '%9.0f%%' % (100.0 * t / totals[0]) for t in totals)))
    done = [n for n, c in rows if c[0] and c[1] == c[0] and c[2] == c[0]
            and c[4] == c[0]]
    print('\nfully converted modules: %d of %d' % (len(done), len(rows)))
    for name in done:
        print('  %s' % name)
    return 0


if __name__ == '__main__':
    sys.exit(main())

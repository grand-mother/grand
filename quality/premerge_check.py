#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Reports what a branch would add, before it is merged.

A clean textual merge is not a compatible merge.  The three ways branches in
this repository have been found to be incompatible without conflicting are:

1. **Two names for one quantity.**  ``dev_nutrig_fields`` adds
   ``nutrig_rhox``/``nutrig_rhoy`` to ``TADC``; ``dev_fix_root_warnings_lwp_new_fields``
   adds ``correlation_x``/``correlation_y`` for the same measurement.  They
   collided only because they touched neighbouring lines.  Had they been in
   different classes, both would have merged and the schema would carry the
   quantity twice.

2. **Two implementations of one thing, in different files.**  ``refact_galaxy``
   adds ``galaxy_new.py`` beside the existing ``galaxy.py``.  Nothing
   conflicts; merging leaves two Galactic-noise models with no deprecation
   path and no signal about which is authoritative.

3. **A change of meaning under an unchanged name.**  ``dev_snonis`` alters the
   noise normalisation by a factor of about 1.41 without touching a
   signature.  Only running the code detects this; see ``--numeric`` below,
   which is not implemented here because it needs a fixture and a built
   environment.

This script covers 1 and 2, which are static.  Run it on every branch before
merging::

    python quality/premerge_check.py dev_snonis refact_galaxy

Exit status is 1 if anything was flagged, so it can gate a merge.
"""

import argparse
import collections
import re
import subprocess
import sys


def sh(cmd):
    r"""Returns the stdout of a shell command, as text."""
    return subprocess.run(cmd, shell=True, capture_output=True,
                          text=True).stdout


FIELD_RE = re.compile(
    r'^\+\s*(\w+):\s*(?:StdVectorListDesc|TTreeScalarDesc|TTreeArrayDesc'
    r'|StdStringDesc)[^\n]*\n\+\s*"""([^"]*)"""', re.M)

STOP = {'the', 'a', 'of', 'in', 'for', 'from', 'to', 'and', 'value', 'with',
        'this', 'is', 'each', 'per', 'number'}


def words(text):
    r"""Returns the meaningful lower-case words of a docstring."""
    return {w for w in re.findall(r'[a-z]+', text.lower()) if w not in STOP}


def trunk_fields():
    r"""Returns ``{field name: (class, docstring)}`` for the current checkout."""
    out = {}
    for path in ('grand/dataio/event_trees.py', 'grand/dataio/run_trees.py'):
        try:
            src = open(path).read()
        except OSError:
            continue
        cls = None
        for line in src.split('\n'):
            m = re.match(r'class (\w+)', line)
            if m:
                cls = m.group(1)
            m = re.match(r'\s*(\w+):\s*(?:StdVectorListDesc|TTreeScalarDesc'
                         r'|TTreeArrayDesc|StdStringDesc)', line)
            if m and cls:
                out[m.group(1)] = (cls, '')
        for name, doc in re.findall(
                r'\s(\w+):\s*(?:StdVectorListDesc|TTreeScalarDesc'
                r'|TTreeArrayDesc|StdStringDesc)[^\n]*\n\s*"""([^"]*)"""', src):
            if name in out:
                out[name] = (out[name][0], doc)
    return out


def check(branch, existing):
    r"""Reports what `branch` adds, and returns the number of flags raised."""
    base = sh('git merge-base HEAD origin/%s' % branch).strip()
    if not base:
        print('  ! unknown branch')
        return 1

    flags = 0
    diff = sh("git diff %s..origin/%s -- 'grand/*.py'" % (base, branch))

    # 1. New tree fields, and whether the trunk already describes the same thing.
    for name, doc in FIELD_RE.findall(diff):
        near = [(n, d) for n, (c, d) in existing.items()
                if d and words(d) & words(doc)
                and len(words(d) & words(doc)) >= 2 and n != name]
        if near:
            flags += 1
            print('  ! field %-18s "%s"' % (name, doc.strip()[:44]))
            for n, d in near[:2]:
                print('      may duplicate %-14s "%s"' % (n, d.strip()[:40]))
        else:
            print('    field %-18s "%s"' % (name, doc.strip()[:44]))

    # 2. New modules that shadow an existing one.
    added = sh("git diff --name-only --diff-filter=A %s..origin/%s "
               "-- 'grand/*.py'" % (base, branch)).split()
    for path in added:
        stem = path.split('/')[-1][:-3]
        root = re.sub(r'(_new|_v\d+|\d+)$', '', stem)
        sibling = sh('git ls-files "grand/**/%s.py"' % root).strip()
        if root != stem and sibling:
            flags += 1
            print('  ! module %s shadows %s -- merging keeps both' %
                  (path, sibling))
        else:
            print('    module %s' % path)

    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branches', nargs='+')
    args = parser.parse_args()

    existing = trunk_fields()
    total = 0
    for branch in args.branches:
        print('%s' % branch)
        total += check(branch, existing)
        print()

    if total:
        print('%d item(s) need a human decision before merging.' % total)
    else:
        print('nothing flagged (this does not establish numerical equivalence)')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())

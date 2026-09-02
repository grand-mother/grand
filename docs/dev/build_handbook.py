#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Converts the GRANDlib Handbook from LaTeX into the Sphinx tree.

    python docs/dev/build_handbook.py

Reads ``resources/GRANDlib_Handbook.zip``, converts
``GRANDlib_Handbook_Expanded.tex`` with pandoc, splits it at the top-level
sections, and writes one ``.rst`` per section into ``docs/source/handbook/``
together with the figures.

Why convert rather than link the PDF.  The Handbook is the only prose that
covers ``sim2root``, ``granddb`` and the example scripts at any length, and a
PDF sitting next to the documentation is not searchable from it, does not
cross-link to the API reference, and cannot carry the errata below where a
reader will see them.

**The output is generated.  Do not edit** ``docs/source/handbook/*.rst`` --
run this script instead.  Corrections belong either upstream in the Handbook
source or in the ``ERRATA`` table below, which is rendered into the landing
page and into the affected sections.

Requires pandoc.  If it is absent the script says so and exits non-zero; the
handbook pages are committed, so a checkout without pandoc still builds the
documentation.
"""

import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ZIP = ROOT / 'resources' / 'GRANDlib_Handbook.zip'
OUT = ROOT / 'docs' / 'source' / 'handbook'
TEX = 'GRANDlib_Handbook_Expanded.tex'

#: Statements in the Handbook that the code contradicts.  Each was checked
#: against the package rather than assumed; the "checked" column names the test
#: or the measurement that settles it.
ERRATA = [
    ('Antenna arms',
     'The "East-West arm" is "denoted as EW or X arm", and South-North as '
     '"SN or Y arm".',
     'The opposite. X is the south-north arm and Y is the east-west arm, '
     'which is what the code does and what GRANDCS implies.',
     ':ref:`issue-handbook-arm-naming`'),
    ('Air showers',
     'GRANDlib "allows [us] to simulate air showers" and to "simulate air '
     'showers and generate corresponding electric field traces".',
     'It does not. Air showers and their radio emission come from ZHAireS or '
     'CoREAS; ``grand/sim/shower/gen_shower.py`` holds ``ShowerEvent``, a '
     'container with ``load`` and ``dump``, not a generator.',
     ':doc:`../simulation`'),
    ('Module names',
     'Lists ``grand.io`` for file I/O and ``grand.topography`` for terrain.',
     'Neither exists. They are ``grand.dataio`` and ``grand.geo.topography``.',
     ':doc:`../api`'),
    ('Docker install',
     'Presents Docker as the first installation route, naming the published '
     'images grandlib/dev:1.x and 2.0.',
     'Unmaintained. Those images are the newest published — 1.2 dates from '
     '2023-01-14 — and pin ROOT 6.26.02 against 6.36 in the conda environment '
     'and 6.36/6.38 in CI, on an Ubuntu 20.04 base out of support since April '
     '2025. Nothing builds them and no CI covers them. The supported route is '
     'the conda environment.',
     ':ref:`issue-docker-unmaintained`'),
    ('Reconstruction',
     'Describes ``grand/recon/`` as "Reconstruction Algorithms" and '
     '``grand.aoi`` as providing "filtering and reconstruction".',
     '``grand.recon`` is a placeholder: two classes with a constructor and no '
     'other method. No reconstruction is implemented in GRANDlib.',
     ':doc:`../api`'),
]


#: The underline characters pandoc emits, in heading-level order.
LEVELS = '=-~^"'


def normalise_headings(text):
    r"""Returns `text` with heading levels made contiguous.

    The Handbook uses ``\paragraph`` without an intervening
    ``\subsubsection`` in several places, so pandoc emits a level-4 heading
    directly under a level-2 one.  Sphinx rejects that with "Inconsistent title
    style: skip from level 2 to 4", 149 times in the directory-structure
    section alone.

    Promotes each heading to at most one level deeper than the one before it,
    which preserves the nesting the author intended and removes the gaps.

    Parameters
    ----------
    text : str
        A converted section.

    Returns
    -------
    str
        The same text with its underlines rewritten.
    """
    lines = text.split('\n')
    out, previous, i = [], 0, 0
    while i < len(lines):
        title, under = lines[i], lines[i + 1] if i + 1 < len(lines) else ''
        is_heading = (title.strip() and under and len(set(under)) == 1
                      and under[0] in LEVELS
                      and len(under) >= len(title.strip()))
        if not is_heading:
            out.append(title)
            i += 1
            continue
        level = LEVELS.index(under[0]) + 1
        level = min(level, previous + 1)
        previous = level
        out.append(title)
        out.append(LEVELS[level - 1] * max(len(title.strip()), 3))
        i += 2
    return '\n'.join(out)


def errata_table():
    r"""Returns the errata as a reStructuredText list-table."""
    lines = ['.. list-table:: Errata',
             '   :header-rows: 1',
             '   :widths: 12 30 38 20',
             '',
             '   * - Topic',
             '     - The Handbook says',
             '     - The code says',
             '     - Where']
    for topic, says, actual, where in ERRATA:
        lines += ['   * - %s' % topic,
                  '     - %s' % says,
                  '     - %s' % actual,
                  '     - %s' % where]
    return '\n'.join(lines)


STAMP = 'source.txt'


def source_stamp():
    r"""Returns a line identifying the source these pages were generated from.

    Contains the SHA-256 of ``resources/GRANDlib_Handbook.zip`` and the pandoc
    version that converted it.

    Returns
    -------
    str
        One line, ``<sha256>  pandoc <version>``.
    """
    import hashlib

    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    try:
        version = subprocess.run(['pandoc', '--version'], capture_output=True,
                                 text=True).stdout.split('\n')[0].strip()
    except (OSError, subprocess.SubprocessError):
        version = 'pandoc unavailable'
    return '%s  %s' % (digest, version)


def check_current():
    r"""Returns 0 if the committed pages were generated from the current zip.

    Compares the recorded SHA-256 of the source archive, **not** the converted
    text.  Comparing the text does not work: pandoc's output differs between
    versions in whitespace and in whether it emits ``:width:`` on an image, so
    a byte comparison fails on any runner whose pandoc differs from the one
    that last generated the pages -- which is a false alarm about the pandoc
    version, not a report that the pages are stale.

    What actually matters is whether the *source* changed without the pages
    being rebuilt, and the hash answers exactly that.

    Returns
    -------
    int
        0 if current, 1 otherwise.
    """
    marker = OUT / STAMP
    if not marker.exists():
        print('  %s is missing; run docs/dev/build_handbook.py' % marker)
        return 1
    recorded = marker.read_text().strip().split('  ')[0]
    import hashlib

    current = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    if recorded != current:
        print('  the Handbook source has changed since the pages were '
              'generated:\n    recorded %s\n    current  %s\n'
              '  run docs/dev/build_handbook.py' % (recorded[:16], current[:16]))
        return 1
    print('  handbook pages were generated from the current source (%s)'
          % current[:16])
    return 0


def convert():
    r"""Returns the Handbook as reStructuredText, and extracts its figures.

    Returns
    -------
    str
        The converted document.
    """
    if shutil.which('pandoc') is None:
        sys.exit('pandoc is not installed; the committed handbook pages are '
                 'still usable, but they cannot be regenerated without it')

    work = OUT / '_tex'
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as archive:
        archive.extractall(work)

    # Figures live beside the pages so that the ``.. image::`` paths pandoc
    # emits resolve without rewriting.
    for image in work.glob('*.png'):
        shutil.copy2(image, OUT / image.name)

    result = subprocess.run(
        ['pandoc', TEX, '-f', 'latex', '-t', 'rst', '--wrap=preserve'],
        cwd=work, capture_output=True, text=True)
    if result.returncode:
        sys.exit('pandoc failed:\n%s' % result.stderr[:2000])
    shutil.rmtree(work, ignore_errors=True)
    return result.stdout


def split(text):
    r"""Splits the converted document at its top-level sections.

    Parameters
    ----------
    text : str
        Output of :func:`convert`.

    Returns
    -------
    list of (str, str)
        Section title and body, in document order.
    """
    # pandoc underlines a top-level section with '=' on the following line.
    lines = text.split('\n')
    marks = [i for i in range(len(lines) - 1)
             if lines[i].strip() and set(lines[i + 1]) == {'='}
             and len(lines[i + 1]) >= len(lines[i].strip())]
    out = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)
        out.append((lines[start].strip(), '\n'.join(lines[start + 2:end])))
    return out


def repair(text):
    r"""Repairs artifacts of the source document that produce invalid reStructuredText.

    Both of these come from the Handbook's LaTeX rather than from pandoc, and
    both are fixed here rather than upstream because this repository does not
    own that document.

    ``**Initialize Grand Library:** ```` \ cd grand\|``
        An empty inline literal followed by a stray pipe, from a line in the
        source that was evidently meant to be two commands.  docutils reports
        "Inline literal start-string without end-string".

    ``.. code:: bash`` blocks holding ``argparse`` output
        Pygments cannot lex a usage message as shell -- an apostrophe in
        "don't" reads as an unterminated string -- and Sphinx falls back with a
        warning.  They are terminal output, not commands, so they are relabelled
        ``text``.

    Parameters
    ----------
    text : str
        A converted section.

    Returns
    -------
    str
        The same text, repaired.
    """
    text = text.replace('````\\ cd grand\\| ``source env/setup.sh``',
                        '``cd grand`` then ``source env/setup.sh``')

    out, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip() == '.. code:: bash':
            # Look ahead over the block to see whether it is a usage message.
            j = i + 1
            body = []
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith('   ')):
                body.append(lines[j])
                j += 1
            joined = '\n'.join(body)
            if ('positional arguments:' in joined or 'usage:' in joined
                    or '\n  -h, --help' in joined):
                out[-1] = lines[i].replace('.. code:: bash', '.. code:: text')
        i += 1
    return '\n'.join(out)


SUBSTITUTION = re.compile(r'^\.\. \|([^|]+)\| image:: .*$')


def redistribute_substitutions(sections):
    r"""Moves pandoc's image substitution definitions into the sections that use them.

    Pandoc renders an inline figure as a ``|image3|`` substitution reference and
    collects every definition at the end of the document.  Splitting the
    document into pages therefore separates most references from their
    definitions, and docutils reports "Undefined substitution referenced" for
    each one.

    Copies each definition into every section that references it, and drops the
    orphaned block from wherever pandoc happened to put it.  Definitions are
    copied rather than moved because a figure may be referenced from more than
    one section.

    Parameters
    ----------
    sections : list of (str, str)
        Title and body of each section.

    Returns
    -------
    list of (str, str)
        The same sections, each carrying the definitions it needs.
    """
    definitions = {}
    for _, body in sections:
        for line in body.split('\n'):
            match = SUBSTITUTION.match(line)
            if match:
                definitions[match.group(1)] = line

    out = []
    for title, body in sections:
        kept = [line for line in body.split('\n') if not SUBSTITUTION.match(line)]
        body = '\n'.join(kept)
        needed = [name for name in definitions
                  if re.search(r'\|%s\|' % re.escape(name), body)]
        if needed:
            body += '\n\n' + '\n'.join(definitions[name] for name in sorted(needed))
        out.append((title, body))
    return out


def slug(title):
    r"""Returns a file-name slug for a section title."""
    return re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')


def main():
    r"""Converts the Handbook and writes the Sphinx pages."""
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob('*.rst'):
        stale.unlink()

    sections = redistribute_substitutions(
        [(title, repair(normalise_headings(body)))
         for title, body in split(convert())])
    print('  %d sections' % len(sections))

    names = []
    for title, body in sections:
        name = slug(title)
        names.append((name, title))
        header = ('.. This page is generated from resources/GRANDlib_Handbook.zip\n'
                  '   by docs/dev/build_handbook.py.  Do not edit it by hand.\n\n')
        banner = ''
        if name in ('introduction', 'directory_structure'):
            banner = ('.. warning::\n\n'
                      '   This section contains statements the code '
                      'contradicts.  See :doc:`index` for the errata.\n\n')
        (OUT / ('%s.rst' % name)).write_text(
            '%s%s\n%s\n\n%s%s\n' % (header, title, '=' * len(title), banner, body))

    index = ['.. This page is generated by docs/dev/build_handbook.py.\n',
             'GRANDlib Handbook',
             '=================',
             '',
             'The GRANDlib Handbook, converted from the LaTeX source in',
             '``resources/GRANDlib_Handbook.zip`` and included here so that it is',
             'searchable alongside the rest of the documentation.',
             '',
             '.. admonition:: Download the PDF',
             '   :class: tip',
             '',
             '   :download:`GRANDlib Handbook (PDF, 78 pages)',
             '   <../_static/GRANDlib_Handbook.pdf>` -- compiled from the same',
             '   source by ``docs/dev/build_handbook_pdf.py``, with the errata',
             '   below reproduced after its table of contents and a build',
             '   provenance block on the title page naming the commit it came',
             '   from.',
             '',
             '.. note::',
             '',
             '   This is somebody else\'s document, reproduced.  It is **not**',
             '   maintained as part of this documentation, it describes a Docker-based',
             '   workflow this repository no longer uses, and where it disagrees with',
             '   the code the code is right.  Read :doc:`../quickstart` and',
             '   :doc:`../installation` first; come here for the material nothing else',
             '   covers, chiefly ``sim2root``, ``granddb`` and the example scripts.',
             '',
             # The toctree goes *before* the Errata heading, not after it.
             # reStructuredText has no way to close a section, so anything
             # following "Errata" belongs to it -- putting the toctree there
             # nests every handbook page one level under Errata in the sidebar.
             # Placed here, the pages are its siblings, which is what the
             # navigation should show.  No "Contents" heading either: it would
             # add a level that carries no content of its own.
             '.. toctree::',
             '   :maxdepth: 2',
             '']
    index += ['   %s' % name for name, _ in names]
    index += ['',
              'Errata',
              '------',
              '',
              'Every entry below was checked against the package rather than '
              'assumed.',
              '',
              errata_table()]
    (OUT / 'index.rst').write_text('\n'.join(index) + '\n')
    (OUT / STAMP).write_text(source_stamp() + '\n')
    print('  wrote %s' % OUT)


if __name__ == '__main__':
    import sys

    if '--check' in sys.argv:
        raise SystemExit(check_current())
    main()

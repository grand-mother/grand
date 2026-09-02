#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Builds the GRANDlib Handbook PDF from its LaTeX source.

    python docs/dev/build_handbook_pdf.py
    python docs/dev/build_handbook_pdf.py --check   # compile only, keep nothing

Extracts ``resources/GRANDlib_Handbook.zip``, patches the LaTeX, compiles it
with ``pdflatex`` and writes the result to
``docs/source/_static/GRANDlib_Handbook.pdf`` so the documentation can link to
it.

Three patches, all applied to a copy in a temporary directory -- the shipped
zip is never modified:

**The logo** on the title page.  The Handbook had none; the artwork is in
``docs/source/_static/grandlib_logo_trimmed.png``.

**A provenance block**, so a printed copy can be traced back.  It carries the
build date and time, the git commit and branch, and whether the tree was dirty
when it was built.  A PDF that circulates by email otherwise says only "(2025)".

**The errata**, immediately after the table of contents.  The Handbook contains
statements the code contradicts, and a reader who meets them 40 pages in has
already been misled.  The table is the same one
:mod:`docs.dev.build_handbook` renders into the HTML, imported from there so
the two cannot disagree.

The compile runs twice: the first pass writes the table of contents, the second
resolves it.  ``--check`` compiles and reports, without installing the result,
which is what CI uses.
"""

import argparse
import datetime
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ZIP = ROOT / 'resources' / 'GRANDlib_Handbook.zip'
LOGO = ROOT / 'docs' / 'source' / '_static' / 'grandlib_logo_trimmed.png'
OUT = ROOT / 'docs' / 'source' / '_static' / 'GRANDlib_Handbook.pdf'
TEX = 'GRANDlib_Handbook_Expanded.tex'

sys.path.insert(0, str(HERE))
from build_handbook import ERRATA  # noqa: E402  (path is set immediately above)


def git(*args, default='unknown'):
    r"""Returns the output of a git command, or `default` if it cannot run.

    Parameters
    ----------
    *args : str
        Arguments after ``git``.
    default : str, optional
        Returned when git is absent or the command fails, so that a build from
        a source tarball still produces a PDF.

    Returns
    -------
    str
        The trimmed output.
    """
    try:
        result = subprocess.run(('git',) + args, cwd=str(ROOT),
                                capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or default if result.returncode == 0 else default
    except (OSError, subprocess.SubprocessError):
        return default


def texify(text):
    r"""Returns `text` with reStructuredText inline markup rendered as LaTeX.

    The errata are written once, for Sphinx.  This converts the handful of
    constructs they use rather than keeping a second copy in LaTeX, which would
    drift.

    Parameters
    ----------
    text : str
        A cell of the errata table.

    Returns
    -------
    str
        LaTeX source.
    """
    # Cross-references carry no meaning in a standalone PDF; keep the target's
    # name as plain text.
    text = re.sub(r':ref:`([^`]+)`', lambda m: m.group(1).replace('-', ' '), text)
    text = re.sub(r':doc:`[^`]*?([\w./]+)`', r'\1', text)
    text = re.sub(r'``([^`]+)``', r'\\texttt{\1}', text)
    for char, escaped in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_')):
        text = text.replace(char, escaped)
    return text


def errata_tex():
    r"""Returns the errata section as LaTeX."""
    rows = []
    for topic, says, actual, where in ERRATA:
        rows.append('%s & %s & %s \\\\ \\hline'
                    % (texify(topic), texify(says), texify(actual)))
        del where          # the HTML cross-reference has no PDF equivalent
    return r"""
\section*{Errata}
\addcontentsline{toc}{section}{Errata}

\noindent
This page is not part of the original Handbook.  It records statements in the
text that the code contradicts, each checked against the package rather than
assumed, and is generated from the same source as the errata in the online
documentation.

\vspace{0.4cm}
\noindent
\renewcommand{\arraystretch}{1.35}
\begin{longtable}{|p{2.7cm}|p{4.3cm}|p{6.4cm}|}
\hline
\textbf{Topic} & \textbf{The Handbook says} & \textbf{The code says} \\ \hline
\endhead
%s
\end{longtable}

\vspace{0.3cm}
\noindent
The online documentation carries the measurements behind each entry, under
\texttt{Known issues}.

\newpage
""" % '\n'.join(rows)


def provenance_tex():
    r"""Returns the provenance block as LaTeX.

    Returns
    -------
    str
        A centred block naming when and from what the PDF was built.
    """
    # UTC, and stated as such: a build stamp in an unnamed local time is not
    # traceable.
    now = datetime.datetime.now(datetime.timezone.utc)
    commit = git('rev-parse', '--short', 'HEAD')
    branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    described = git('describe', '--tags', '--always', '--dirty', default=commit)
    dirty = bool(git('status', '--porcelain', default=''))
    remote = git('config', '--get', 'remote.origin.url', default='')
    remote = re.sub(r'^git@github\.com:', 'https://github.com/', remote)
    remote = re.sub(r'\.git$', '', remote)

    lines = [
        r'\vspace{0.6cm}',
        r'\begin{center}',
        r'\small',
        r'\textbf{Build provenance}\\[2pt]',
        r'Compiled %s UTC\\' % now.strftime('%Y-%m-%d %H:%M:%S'),
        r'Commit \texttt{%s} on branch \texttt{%s}\\' % (commit, texify(branch)),
        r'Describes as \texttt{%s}%s\\' % (texify(described),
                                           r' \textbf{(tree dirty)}' if dirty else ''),
    ]
    if remote and remote != 'unknown':
        lines.append(r'\url{%s}\\' % remote)
    lines += [
        r'\vspace{2pt}',
        r'\footnotesize Built from \texttt{resources/GRANDlib\_Handbook.zip} by',
        r'\texttt{docs/dev/build\_handbook\_pdf.py}.',
        r'\end{center}',
    ]
    return '\n'.join(lines)


def patch(source):
    r"""Returns the Handbook LaTeX with the logo, provenance and errata added.

    Parameters
    ----------
    source : str
        The original document.

    Returns
    -------
    str
        The patched document.
    """
    # The logo goes above the title, so \maketitle renders it as part of the
    # title block rather than on a page of its own.
    title = r'\title{GRANDlib Handbook (dev branch) \\\large(2025)}'
    if title not in source:
        raise SystemExit('the title has changed; the logo patch needs updating')
    source = source.replace(
        title,
        r'\title{%s\\[10pt] GRANDlib Handbook (dev branch) \\\large(2025)}'
        % r'\includegraphics[width=0.30\textwidth]{grandlib_logo.png}')

    # Provenance after the title, errata after the table of contents.
    anchor = '\\maketitle\n\\tableofcontents\n\\newpage'
    if anchor not in source:
        raise SystemExit('the title/contents block has changed; patch needs updating')
    return source.replace(
        anchor,
        '\\maketitle\n%s\n\\newpage\n\\tableofcontents\n\\newpage\n%s'
        % (provenance_tex(), errata_tex()))


def build(check_only=False):
    r"""Compiles the Handbook and installs the PDF.

    Parameters
    ----------
    check_only : bool, optional
        Compile and report, but do not write the PDF into the documentation.

    Returns
    -------
    int
        A process exit status.
    """
    if shutil.which('pdflatex') is None:
        # The documentation links to the PDF with :download:, so the file has
        # to exist or Sphinx warns and the build stops being clean.  Fall back
        # to the copy shipped in resources/ -- the original, without the errata
        # or the provenance block, but a real Handbook rather than nothing.
        print('pdflatex is not installed; skipping the Handbook PDF')
        if not check_only:
            shipped = ROOT / 'resources' / 'GRANDlib_Handbook.pdf'
            if shipped.exists() and not OUT.exists():
                OUT.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shipped, OUT)
                print('installed the shipped PDF instead: %s' % OUT)
        return 0

    work = pathlib.Path(tempfile.mkdtemp(prefix='grand-handbook-'))
    try:
        with zipfile.ZipFile(ZIP) as archive:
            archive.extractall(work)
        shutil.copy2(LOGO, work / 'grandlib_logo.png')

        text = (work / TEX).read_text(encoding='utf-8', errors='replace')
        (work / TEX).write_text(patch(text), encoding='utf-8')

        # Twice: the first pass writes the table of contents, the second
        # resolves it and the errata's entry in it.
        for run in (1, 2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', TEX],
                cwd=str(work), capture_output=True, text=True)
            if result.returncode:
                tail = '\n'.join(result.stdout.strip().split('\n')[-40:])
                print('pdflatex failed on pass %d:\n%s' % (run, tail))
                return 1

        pdf = work / TEX.replace('.tex', '.pdf')
        size = pdf.stat().st_size
        # pdflatex reports the page count on its last line as "(N pages...".
        # Counting /Type /Page in the file does not work: pdfTeX compresses the
        # page objects into object streams, so the pattern finds nothing.
        found = re.search(r'\((\d+) pages?', result.stdout)
        pages = found.group(1) if found else '?'
        print('compiled: %.1f MB, %s pages' % (size / 1e6, pages))

        if not check_only:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf, OUT)
            print('wrote %s' % OUT)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    r"""Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='compile without installing the result')
    args = parser.parse_args()
    raise SystemExit(build(check_only=args.check))


if __name__ == '__main__':
    main()

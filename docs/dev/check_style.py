# -*- coding: utf-8 -*-
r"""Checks the prose of the documentation against the house style.

The documentation is written in American English, in plain sentences.  This
script reports, file by file, what does not follow that::

    python docs/dev/check_style.py          # exits 1 if anything is found

It reads the pages under ``docs/source/`` and ``README.md``, leaving out the
changelog, the generated pages and the Handbook, and ignores code: literal
blocks, code and executed blocks, and inline ``literals``.  The rules, with
the reason for each, are in :data:`RULES`; ``contributing.rst`` describes the
style they enforce.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: British spelling -> American spelling.
BRITISH = {
    "behaviour": "behavior", "normalisation": "normalization", "normalise": "normalize",
    "normalised": "normalized", "organise": "organize", "organised": "organized",
    "organisation": "organization", "digitisation": "digitization", "digitise": "digitize",
    "digitised": "digitized", "digitiser": "digitizer", "quantisation": "quantization",
    "realisation": "realization", "modelled": "modeled", "modelling": "modeling",
    "metre": "meter", "metres": "meters", "centre": "center", "centred": "centered",
    "analogue": "analog", "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "characterised": "characterized", "optimise": "optimize", "optimised": "optimized",
    "optimisation": "optimization", "colour": "color", "catalogue": "catalog",
    "initialise": "initialize", "initialised": "initialized", "polarisation": "polarization",
    "travelling": "traveling", "labelled": "labeled", "towards": "toward", "whilst": "while",
    "recognise": "recognize", "recognised": "recognized", "summarise": "summarize",
    "artefact": "artifact", "licence": "license", "neighbour": "neighbor", "grey": "gray",
    "programme": "program", "fibre": "fiber", "acknowledgement": "acknowledgment",
}

#: (name, pattern, why) -- each pattern is matched against prose only.
RULES = [
    ("comma-and", re.compile(r",\s+and\b"),
     'no comma before "and": write "A, B and C", or make two sentences'),
    ("worth", re.compile(r"\b(worth (knowing|stating|noting|saying|recording)|it is worth)\b", re.I),
     "say the thing rather than that it is worth saying"),
    ("plainly", re.compile(r"\b(said plainly|to say it plainly|plainly)\b", re.I),
     "state it; the adverb adds nothing"),
    ("why-it-matters", re.compile(r"\bwhy (it|this) matters\b", re.I),
     "use a heading that names the subject"),
    ("measured-not-assumed", re.compile(r"\b(measured|verified), not assumed\b", re.I),
     "give the measurement instead"),
    ("not-obvious", re.compile(r"\b(that|this) (was|is) not obvious\b", re.I),
     "explain the point instead"),
    ("internal-phase", re.compile(r"\b(before|after|until|in) Phase \d+\b"),
     "readers do not know the recovery plan's phases; describe the change"),
]

SKIP = {"changelog.rst", "data_format.rst"}

#: Rules that do not apply to a page, with the reason.
EXEMPT = {("roadmap.rst", "internal-phase"): "the page is about the plan's phases"}

#: Proper names that contain a British spelling.
PROPER = re.compile(r"National Science Centre|Centre Poland")
CODE_DIRECTIVES = re.compile(r"^\s*\.\. (code-block|jupyter-execute|code|literalinclude|math|"
                             r"bibliography|toctree|image|figure|list-table|csv-table|contents)::")
INLINE = re.compile(r"``.*?``|`[^`]*`_{0,2}|:[a-z:]+:`[^`]*`")


def files():
    r"""Returns the files the style applies to.

    Returns
    -------
    list of pathlib.Path
    """
    source = ROOT / "docs" / "source"
    pages = [p for p in sorted(source.glob("*.rst")) + sorted(source.glob("api/*.rst"))
             if p.name not in SKIP]
    return pages + [ROOT / "README.md"]


def prose(lines):
    r"""Yields ``(line_number, text)`` for the prose lines of a page, with inline code masked.

    Parameters
    ----------
    lines : list of str
        The page.

    Yields
    ------
    tuple
        The 1-based line number and the line, inline literals replaced by spaces.
    """
    block_indent = None
    fenced = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if block_indent is not None:
            if stripped and indent <= block_indent:
                block_indent = None
            else:
                continue
        if CODE_DIRECTIVES.match(line) or stripped.endswith("::") and not stripped.startswith(".."):
            block_indent = indent
            if not stripped.endswith("::") or CODE_DIRECTIVES.match(line):
                continue
            line = line.rstrip()[:-2]
        if stripped.startswith(".. ") or line.startswith("    "):
            continue
        # Letters, not spaces: "``a``, ``b`` and" must not read as ", and"
        yield number, INLINE.sub(lambda m: "x" * len(m.group(0)), line)


def check(path):
    r"""Returns the problems in one file.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    list of str
        ``path:line: rule: why``, one per problem.
    """
    found = []
    lines = path.read_text(encoding="utf-8").splitlines()
    # Join each prose line with the next, so a phrase broken across two lines is seen
    text = list(prose(lines))
    british = re.compile(r"(?<![\w\-/.])(%s)(?![\w\-/])" % "|".join(BRITISH), re.I)
    for index, (number, line) in enumerate(text):
        following = text[index + 1][1] if index + 1 < len(text) and text[index + 1][0] == number + 1 else ""
        joined = line + " " + following.lstrip()
        for match in british.finditer(PROPER.sub(lambda m: "x" * len(m.group(0)), line)):
            word = match.group(1)
            found.append("%s:%d: british: %r, write %r"
                         % (path.relative_to(ROOT), number, word, BRITISH[word.lower()]))
        for name, pattern, why in RULES:
            if (path.name, name) in EXEMPT:
                continue
            match = pattern.search(joined)
            if match and match.start() < len(line):
                found.append("%s:%d: %s: %s" % (path.relative_to(ROOT), number, name, why))
    return found


def main():
    r"""Checks every file and prints what it finds.

    Returns
    -------
    int
        0 when the documentation follows the style, 1 otherwise.
    """
    problems = [p for path in files() for p in check(path)]
    for problem in problems:
        print(problem)
    print("%d style problem%s" % (len(problems), "" if len(problems) == 1 else "s"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

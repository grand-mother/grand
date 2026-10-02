# -*- coding: utf-8 -*-
r"""Adds a new or edited notebook to ``make_notebooks.py``, and checks that it runs.

The tutorial notebooks are written by ``make_notebooks.py``: a notebook edited
in Jupyter is overwritten the next time the generator runs.  This script turns
such a notebook back into generator source, so that the edit is kept::

    python notebooks/import_notebook.py notebooks/05_galactic_noise.ipynb   # edited
    python notebooks/import_notebook.py ~/my_analysis.ipynb --name 13_my_analysis.ipynb

For an existing notebook, its block in ``make_notebooks.py`` is replaced; for a
new one, a block is added.  Then the notebook is built and executed by
``make_notebooks.py --only``, which also checks that it stored its outputs and
that every notebook still matches the generator.  If any step fails, the
generator and the notebook are restored as they were, and nothing is lost.

Before anything is written, the notebook is checked:

* it is a valid notebook with a Python kernel;
* its first cell is a level-1 heading, ``# NN — Title``, whose number matches
  the file name ``NN_name.ipynb``;
* every code cell compiles (IPython magics and shell lines are allowed);
* no cell names an absolute path such as ``/home/...``, which would not exist
  on another machine or in CI.  Paths are relative to ``notebooks/``, where the
  notebook runs.

Outputs and cell metadata are not imported: the build executes the notebook
and stores fresh outputs.  The generator's closing provenance cell is added
again by the build.

Options
-------
``--name NN_name.ipynb``
    The file name in ``notebooks/``.  By default, the input's own name.
``--no-execute``
    Write the generator block and the notebook without executing it.  For
    drafting only: CI executes every notebook and fails on one that does not
    run.
``--dry-run``
    Print the generator block and the checks' results, writing nothing.
"""

import argparse
import ast
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent
GENERATOR = HERE / "make_notebooks.py"
DOCS_INDEX = HERE.parent / "docs" / "source" / "notebooks.rst"

#: Marks the generator's provenance cell, which the build adds by itself.
PROVENANCE_MARKER = "<!-- provenance -->"

NAME = re.compile(r"^(\d\d)_[a-z0-9_]+\.ipynb$")
TITLE = re.compile(r"^#\s+(\d\d)\s*[—–-]\s*(.+)$")
ABSOLUTE_PATH = re.compile(r"""(?:^|[\s'"(=])(/home/|/Users/|/tmp/|/root/|[A-Za-z]:\\)""")


class NotebookImportError(Exception):
    """A notebook that cannot be imported, with the reason."""


def literal(text):
    r"""Returns Python source for the string `text`, as a raw triple-quoted literal when possible.

    Parameters
    ----------
    text : str
        The cell source.

    Returns
    -------
    str
        A literal that evaluates to `text` exactly.
    """
    for quote in ("'''", '"""'):
        if quote not in text and not text.endswith((quote[0], "\\")):
            candidate = "r%s%s%s" % (quote, text, quote)
            if ast.literal_eval(candidate) == text:
                return candidate
    return repr(text)


def read_notebook(path):
    r"""Reads and validates the notebook at `path`.

    Parameters
    ----------
    path : pathlib.Path
        The notebook to import.

    Returns
    -------
    nbformat.NotebookNode
        The notebook, in format version 4.

    Raises
    ------
    NotebookImportError
        If the file is missing, is not a valid notebook, or is not Python.
    """
    if not path.is_file():
        raise NotebookImportError("no such file: %s" % path)
    try:
        nb = nbf.read(str(path), as_version=4)
        nbf.validate(nb)
    except Exception as error:  # nbformat raises several unrelated types
        raise NotebookImportError("%s is not a valid notebook: %s" % (path, error))
    language = nb.metadata.get("kernelspec", {}).get("language",
                                                     nb.metadata.get("language_info", {}).get("name", "python"))
    if language != "python":
        raise NotebookImportError("%s uses a %s kernel; the notebooks run Python" % (path, language))
    return nb


def split_cells(nb, name):
    r"""Separates the title, the introduction and the remaining cells.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        The notebook.
    name : str
        Its file name in ``notebooks/``, ``NN_name.ipynb``.

    Returns
    -------
    title : str
        The heading, without the ``#``.
    intro : str
        The rest of the first cell.
    cells : list of tuple
        ``(cell_type, source)`` for every later cell, the provenance cell left out.

    Raises
    ------
    NotebookImportError
        If the name or the title does not follow the convention.
    """
    match = NAME.match(name)
    if not match:
        raise NotebookImportError("the file name must be NN_name.ipynb, with a two-digit number and "
                           "lowercase letters, digits and underscores; got %s" % name)
    cells = [c for c in nb.cells if PROVENANCE_MARKER not in c.source]
    if not cells or cells[0].cell_type != "markdown":
        raise NotebookImportError("the first cell must be markdown, starting with '# %s — Title'"
                           % match.group(1))
    first = cells[0].source.strip()
    heading, _, intro = first.partition("\n")
    title = TITLE.match(heading.strip())
    if not title:
        raise NotebookImportError("the first cell must start with '# %s — Title', got %r"
                           % (match.group(1), heading[:80]))
    if title.group(1) != match.group(1):
        raise NotebookImportError("the title is numbered %s but the file name %s"
                           % (title.group(1), match.group(1)))
    body = []
    for cell in cells[1:]:
        if cell.cell_type not in ("markdown", "code"):
            continue                                    # raw cells do not render on GitHub
        body.append((cell.cell_type, cell.source.rstrip()))
    return heading.lstrip("#").strip(), intro.strip(), body


def check_cells(cells):
    r"""Checks that every code cell compiles and that no cell names an absolute path.

    Parameters
    ----------
    cells : list of tuple
        ``(cell_type, source)`` pairs.

    Returns
    -------
    list of str
        Problems found, empty when there are none.
    """
    problems = []
    for index, (kind, source) in enumerate(cells, start=2):
        for line in source.splitlines():
            if ABSOLUTE_PATH.search(line):
                problems.append("cell %d names an absolute path, which will not exist elsewhere: %s"
                                % (index, line.strip()[:100]))
                break
        if kind != "code":
            continue
        # Magics (%matplotlib) and shell lines (!ls) are IPython, not Python:
        # IPython's own transformer turns them into calls, as the kernel does
        try:
            from IPython.core.inputtransformer2 import TransformerManager
            python = TransformerManager().transform_cell(source)
        except ImportError:
            python = source
        try:
            compile(python, "cell %d" % index, "exec")
        except SyntaxError as error:
            problems.append("cell %d does not compile: %s (line %s)"
                            % (index, error.msg, error.lineno))
    return problems


def generator_block(name, title, intro, cells):
    r"""Returns the ``make_notebooks.py`` source that builds this notebook.

    Parameters
    ----------
    name : str
        The file name, the key in ``books``.
    title, intro : str
        The heading and the introduction.
    cells : list of tuple
        ``(cell_type, source)`` pairs.

    Returns
    -------
    str
        The block, starting with its separator comment and ending with a newline.
    """
    rule = "# " + "-" * max(4, 79 - 3 - len(name)) + " " + name
    lines = [rule,
             "books[%r] = notebook(" % name,
             "    %s," % literal(title),
             "    %s," % literal(intro),
             "    ["]
    for kind, source in cells:
        lines.append("    %s(%s)," % ("md" if kind == "markdown" else "code", literal(source)))
    lines += ["    ])", ""]
    return "\n".join(lines)


def blocks(source):
    r"""Returns where each notebook's block lies in the generator source.

    Parameters
    ----------
    source : str
        The text of ``make_notebooks.py``.

    Returns
    -------
    dict
        File name to ``(first_line, last_line)``, 1-based and inclusive, the
        separator comment above the assignment included.
    """
    lines = source.splitlines()
    found = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "books"):
            key = ast.literal_eval(node.targets[0].slice)
            first = node.lineno
            if first > 1 and lines[first - 2].startswith("# ---") and key in lines[first - 2]:
                first -= 1
            found[key] = (first, node.end_lineno)
    return found


def place(source, name, block):
    r"""Returns the generator source with `block` in place of `name`'s, or added.

    Parameters
    ----------
    source : str
        The text of ``make_notebooks.py``.
    name : str
        The notebook's file name.
    block : str
        From :func:`generator_block`.

    Returns
    -------
    text : str
        The new source.
    replaced : bool
        True if a block for `name` existed.
    """
    lines = source.splitlines(keepends=True)
    existing = blocks(source)
    if name in existing:
        first, last = existing[name]
        if lines[first - 1].startswith("# ---"):
            # Keep the separator as it was, so the diff shows only the cells
            block = lines[first - 1] + block.split("\n", 1)[1]
        return "".join(lines[:first - 1]) + block + "".join(lines[last:]), True
    # A new notebook goes just before the command-line entry point
    main = next(node for node in ast.parse(source).body
                if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test))
    at = main.lineno - 1
    return "".join(lines[:at]) + block + "\n\n" + "".join(lines[at:]), False


def built_sources(generator, name):
    r"""Returns the cells the generator now builds for `name`, as ``(type, source)``.

    Runs in a separate interpreter, so that the generator is read as written.

    Parameters
    ----------
    generator : pathlib.Path
        ``make_notebooks.py``.
    name : str
        The notebook's file name.

    Returns
    -------
    list of tuple
        ``(cell_type, source)`` for each cell, the provenance cell left out.
    """
    script = ("import json, runpy, sys\n"
              "books = runpy.run_path(sys.argv[1])['books']\n"
              "print(json.dumps([(c.cell_type, c.source.rstrip()) for c in books[sys.argv[2]].cells\n"
              "                  if %r not in c.source]))\n" % PROVENANCE_MARKER)
    done = subprocess.run([sys.executable, "-c", script, str(generator), name],
                          capture_output=True, text=True, cwd=str(generator.parent), timeout=300)
    if done.returncode != 0:
        raise NotebookImportError("the generator no longer loads:\n%s" % done.stderr[-2000:])
    import json
    return [tuple(cell) for cell in json.loads(done.stdout.strip().splitlines()[-1])]


def main(argv=None):
    r"""Runs the import from the command line.

    Parameters
    ----------
    argv : list of str, optional
        The arguments; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Add a new or edited notebook to make_notebooks.py and check that it runs.")
    parser.add_argument("notebook", type=pathlib.Path, help="the .ipynb to import")
    parser.add_argument("--name", help="its file name in notebooks/, NN_name.ipynb "
                                       "(default: the input's name)")
    parser.add_argument("--no-execute", action="store_true",
                        help="write without executing (drafting only; CI executes every notebook)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the generator block and the checks, writing nothing")
    args = parser.parse_args(argv)

    source_path = args.notebook.resolve()
    name = args.name or source_path.name
    target = HERE / name
    try:
        nb = read_notebook(source_path)
        title, intro, cells = split_cells(nb, name)
        problems = check_cells(cells)
        if problems:
            raise NotebookImportError("the notebook cannot be imported:\n  - " + "\n  - ".join(problems))
        block = generator_block(name, title, intro, cells)
        original = GENERATOR.read_text()
        updated, replaced = place(original, name, block)
    except NotebookImportError as error:
        print("import_notebook: %s" % error, file=sys.stderr)
        return 1

    print("%s %s: %d cells (%d code)" % ("replacing" if replaced else "adding", name,
                                         len(cells) + 1, sum(k == "code" for k, _ in cells)))
    if args.dry_run:
        print(block)
        return 0

    # Everything this run may overwrite is kept until it has succeeded
    backup = pathlib.Path(tempfile.mkdtemp(prefix="import_notebook_"))
    shutil.copy2(source_path, backup / ("input_" + source_path.name))
    if target.exists():
        shutil.copy2(target, backup / name)
    shutil.copy2(GENERATOR, backup / GENERATOR.name)

    def restore(reason):
        GENERATOR.write_text(original)
        if (backup / name).exists():
            shutil.copy2(backup / name, target)
        elif target.exists() and target != source_path:
            target.unlink()
        shutil.copy2(backup / ("input_" + source_path.name), source_path)
        print("import_notebook: %s\nNothing was changed; a copy of your notebook is in %s"
              % (reason, backup), file=sys.stderr)
        return 1

    GENERATOR.write_text(updated)
    try:
        rebuilt = built_sources(GENERATOR, name)
    except NotebookImportError as error:
        return restore(str(error))
    expected = [("markdown", ("# %s\n\n%s" % (title, intro)).rstrip())] + list(cells)
    if rebuilt != expected:
        return restore("the generator block does not reproduce the notebook's cells")

    command = [sys.executable, str(GENERATOR), "--only", name]
    if args.no_execute:
        command.append("--no-execute")
    print("building: %s" % " ".join(command[1:]))
    done = subprocess.run(command, cwd=str(HERE.parent))
    if done.returncode != 0:
        return restore("the build failed (see above); fix the notebook and import it again")

    shutil.rmtree(backup, ignore_errors=True)
    print("done: %s is now built by make_notebooks.py%s"
          % (name, "" if args.no_execute else ", executed, with its outputs stored"))
    if not replaced:
        todo = ["add it to the list in docs/source/notebooks.rst"
                if name not in DOCS_INDEX.read_text() else None,
                "link it from the 'Where next' cell of the notebook before it",
                "commit notebooks/make_notebooks.py and notebooks/%s together" % name]
        print("for a new notebook, also:\n  - " + "\n  - ".join(t for t in todo if t))
    else:
        print("commit notebooks/make_notebooks.py and notebooks/%s together" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

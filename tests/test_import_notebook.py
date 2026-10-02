# -*- coding: utf-8 -*-
r"""``notebooks/import_notebook.py``: a notebook in, the generator block that rebuilds it out.

These tests check the conversion and the checks without executing anything;
the execution is ``make_notebooks.py``'s own, which CI runs on every notebook.
"""

import importlib.util
import pathlib
import runpy

import nbformat as nbf
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

_spec = importlib.util.spec_from_file_location("import_notebook", NOTEBOOKS / "import_notebook.py")
im = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(im)


def _cells(book):
    return [(c.cell_type, c.source.rstrip()) for c in book.cells
            if im.PROVENANCE_MARKER not in c.source]


@pytest.mark.parametrize("text", ["plain", "a'''b", 'x"""y', "ends\\", "q'", "both ''' and \"\"\"",
                                  "unicode — µV", "line\nbreaks\n\n"])
def test_a_literal_evaluates_to_its_text(text):
    assert eval(im.literal(text)) == text


@pytest.mark.parametrize("path", sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb")), ids=lambda p: p.name)
def test_every_notebook_round_trips(path):
    r"""Importing a notebook unchanged rebuilds exactly the cells the generator builds now."""
    generator = runpy.run_path(str(NOTEBOOKS / "make_notebooks.py"))
    want = _cells(generator["books"][path.name])

    title, intro, cells = im.split_cells(im.read_notebook(path), path.name)
    assert im.check_cells(cells) == []
    exec(compile(im.generator_block(path.name, title, intro, cells), "block", "exec"), generator)
    assert _cells(generator["books"][path.name]) == want


def test_an_edited_notebook_replaces_its_block_and_nothing_else():
    source = (NOTEBOOKS / "make_notebooks.py").read_text()
    path = NOTEBOOKS / "02_data_model.ipynb"
    title, intro, cells = im.split_cells(im.read_notebook(path), path.name)
    cells.insert(1, ("markdown", "An added note."))

    updated, replaced = im.place(source, path.name, im.generator_block(path.name, title, intro, cells))
    assert replaced
    before, after = im.blocks(source), im.blocks(updated)
    assert sorted(before) == sorted(after)
    untouched = [n for n in before if n != path.name]
    old_lines, new_lines = source.splitlines(), updated.splitlines()
    for name in untouched:
        (a, b), (c, d) = before[name], after[name]
        assert old_lines[a - 1:b] == new_lines[c - 1:d], name


def test_a_new_notebook_is_added_before_the_entry_point():
    source = (NOTEBOOKS / "make_notebooks.py").read_text()
    block = im.generator_block("99_new.ipynb", "99 — New", "Intro.", [("code", "x = 1")])
    updated, replaced = im.place(source, "99_new.ipynb", block)
    assert not replaced
    assert "99_new.ipynb" in im.blocks(updated)
    assert updated.index("books['99_new.ipynb']") < updated.index("if __name__ == '__main__':")


def _notebook(tmp_path, name, first, *cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(first)] + list(cells)
    nb.metadata = {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}
    path = tmp_path / name
    nbf.write(nb, str(path))
    return path


@pytest.mark.parametrize("name, first, message", [
    ("my notebook.ipynb", "# 13 — Title", "NN_name.ipynb"),
    ("13_title.ipynb", "Some text", "must start with '# 13"),
    ("13_title.ipynb", "# 14 — Title", "numbered 14"),
])
def test_names_and_titles_are_checked(tmp_path, name, first, message):
    path = _notebook(tmp_path, "x.ipynb", first)
    with pytest.raises(im.NotebookImportError, match=message):
        im.split_cells(im.read_notebook(path), name)


def test_magics_compile_and_errors_and_absolute_paths_are_reported():
    problems = im.check_cells([
        ("code", "%matplotlib inline\n!ls\nx = (1 +\n     2)\nprint('%d' %\n      x)"),
        ("code", "def f(:\n    pass"),
        ("markdown", "the file is /home/me/data.root"),
    ])
    assert len(problems) == 2
    assert "cell 3 does not compile" in problems[0]
    assert "cell 4 names an absolute path" in problems[1]


def test_a_dry_run_writes_nothing(tmp_path, capsys):
    before = (NOTEBOOKS / "make_notebooks.py").read_text()
    path = _notebook(tmp_path, "13_scratch.ipynb", "# 13 — Scratch\n\nA test.",
                     nbf.v4.new_code_cell("x = 1"))
    assert im.main([str(path), "--dry-run"]) == 0
    assert "books['13_scratch.ipynb']" in capsys.readouterr().out
    assert (NOTEBOOKS / "make_notebooks.py").read_text() == before
    assert not (NOTEBOOKS / "13_scratch.ipynb").exists()

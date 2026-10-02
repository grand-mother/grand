# -*- coding: utf-8 -*-
r"""``docs/dev/check_style.py``: the documentation follows the house style, and the check works."""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_style", ROOT / "docs" / "dev" / "check_style.py")
style = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(style)


def test_the_documentation_follows_the_style():
    problems = [p for path in style.files() for p in style.check(path)]
    assert problems == [], "\n".join(problems)


def test_the_check_finds_what_it_should(tmp_path):
    page = tmp_path / "page.rst"
    page.write_text("Title\n=====\n\n"
                    "The behaviour is fixed, and it is worth knowing why.\n"
                    "Said plainly, this is code: ``a, and b``.\n\n"
                    ".. code-block:: python\n\n"
                    "    colour = 1, and 2\n")
    original_root = style.ROOT
    style.ROOT = tmp_path
    try:
        found = "\n".join(style.check(page))
    finally:
        style.ROOT = original_root
    assert "'behaviour'" in found
    assert "comma-and" in found
    assert "worth" in found and "plainly" in found
    assert "colour" not in found                    # code is not prose
    assert found.count("comma-and") == 1            # nor are inline literals

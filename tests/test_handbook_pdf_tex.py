"""The Handbook PDF's LaTeX escapes what the errata contain (the build broke on a '$')."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "dev"))

import build_handbook_pdf as builder  # noqa: E402


def test_errata_cells_have_no_unescaped_specials():
    for row in builder.ERRATA:
        for cell in row[:3]:
            tex = builder.texify(cell)
            assert not re.search(r"(?<!\\)[$~^]", tex), tex


def test_dollar_tilde_caret_are_escaped():
    assert builder.texify("``a $PWD ~ ^``") == r"\texttt{a \$PWD \textasciitilde{} \textasciicircum{}}"

# -*- coding: utf-8 -*-
r"""``docs/dev/check_links.py``: which links it collects, without touching the network."""

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_links", ROOT / "docs" / "dev" / "check_links.py")
check_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_links)


def test_the_links_of_the_documentation_are_collected():
    found = check_links.links()
    assert "https://arxiv.org/abs/2408.10926" in found
    assert any(url.startswith("https://grand-mother.github.io/grand/") for url in found)
    assert all(not check_links.IGNORE.search(url) for url in found)
    assert all(not url.endswith((".", ",", ")")) for url in found)


def test_no_link_points_at_the_retired_documentation_site():
    assert not [url for url in check_links.links() if "grand-docs" in url]

# -*- coding: utf-8 -*-
r"""Checks that the external links of the documentation still resolve.

Collects every ``http(s)://`` link from the documentation pages, the README,
the citation and packaging metadata, the issue templates and the notebook
generator, and requests each once::

    python docs/dev/check_links.py          # exits 1 if a link is broken

A link is broken when the server answers 404 or 410, or the host does not
resolve.  Other failures (403, 429, time-outs) are reported as warnings: many
sites refuse automated requests, and a site that is down for an hour is not a
broken link.  The weekly ``linkcheck.yml`` workflow runs this.
"""

import concurrent.futures
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]

SOURCES = ["README.md", "CONTRIBUTING.md", "CITATION.cff", "pyproject.toml",
           ".github/ISSUE_TEMPLATE/*.yml", "docs/source/*.rst", "docs/source/api/*.rst",
           "docs/source/refs.bib", "notebooks/make_notebooks.py"]

URL = re.compile(r"https?://[^\s<>`'\"\]\)}|\\]+")

#: Links that are examples or placeholders, not destinations.
IGNORE = re.compile(r"<you>|localhost|127\.0\.0\.1|example\.(com|org)|\{|\$")

BROKEN = {404, 410}


def links():
    r"""Returns every external link, with the files that use it.

    Returns
    -------
    dict
        URL to a sorted list of the files it appears in.
    """
    found = {}
    for pattern in SOURCES:
        for path in sorted(ROOT.glob(pattern)):
            for url in URL.findall(path.read_text(encoding="utf-8", errors="replace")):
                url = url.rstrip(".,;:")
                if IGNORE.search(url):
                    continue
                found.setdefault(url, set()).add(str(path.relative_to(ROOT)))
    return {url: sorted(files) for url, files in sorted(found.items())}


def status(url, timeout=20):
    r"""Requests `url` and returns what happened.

    Parameters
    ----------
    url : str
    timeout : float, optional
        Seconds to wait for an answer.

    Returns
    -------
    tuple
        ``(code, message)``: the HTTP status, or None when there was no answer.
    """
    headers = {"User-Agent": "GRANDlib link check (+https://github.com/grand-mother/grand)"}
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                return answer.status, "ok"
        except urllib.error.HTTPError as error:
            # Some servers refuse HEAD but answer GET
            if method == "HEAD" and error.code in (403, 405, 501):
                continue
            return error.code, error.reason
        except (urllib.error.URLError, OSError) as error:
            return None, str(getattr(error, "reason", error))
    return None, "no answer"


def main():
    r"""Checks every link and prints the broken ones and the warnings.

    Returns
    -------
    int
        0 when no link is broken, 1 otherwise.
    """
    found = links()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(found, pool.map(status, found)))
    broken = 0
    for url, (code, message) in results.items():
        if code is not None and code < 400:
            continue
        unresolved = code is None and "Name or service not known" in message
        level = "BROKEN" if code in BROKEN or unresolved else "warning"
        broken += level == "BROKEN"
        print("%-7s %s  (%s %s)  in %s" % (level, url, code or "", message, ", ".join(found[url])))
    print("%d links checked, %d broken" % (len(found), broken))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

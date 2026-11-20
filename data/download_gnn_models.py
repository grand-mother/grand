#! /usr/bin/env python3
"""Downloads the GNN reconstruction networks into <GRAND_DATA_PATH>/gnn.

The networks are listed in gnn_model_version.flag, one per line:

    <name> <version> <forge attachment id>

and each is fetched from

    https://forge.in2p3.fr/attachments/download/<id>/gnn_<name>_<version>.tar.gz

An archive holds <name>/<version>/gnn_model_card.json and the seed_XX.pt files
it lists; the SHA-256 of each file is checked against the GNN model card after
extraction.  The training repository writes the archive.

An id of "-" means the network is not uploaded yet: it is skipped with a
message, and this script still succeeds -- as it does when a network is
already installed.  Which networks grand.analysis.gnn uses is set in
grand/analysis/gnn/settings.json.

Usage: download_gnn_models.py [--force] [name ...]
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import tarfile
import time
from urllib import request
from urllib.error import HTTPError, URLError

from grand import GRAND_DATA_PATH

FLAG_FILE = osp.join(osp.dirname(osp.abspath(__file__)), "gnn_model_version.flag")
GNN_DIR = osp.join(GRAND_DATA_PATH, "gnn")
FORGE_BASE_URL = "https://forge.in2p3.fr/attachments/download"
RETRIES = 4
BACKOFF_SECONDS = 5


def read_flag(path=FLAG_FILE):
    """Returns [(name, version, id)] from the flag file, comments and blank lines skipped."""
    entries = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{number}: expected '<name> <version> <forge id>', got {line!r}")
            entries.append(tuple(parts))
    return entries


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory):
    """Returns the problems of an installed network (empty if none): its GNN model card and the files' SHA-256."""
    gnn_model_card_path = osp.join(directory, "gnn_model_card.json")
    if not osp.isfile(gnn_model_card_path):
        return [f"{gnn_model_card_path} is missing"]
    with open(gnn_model_card_path) as f:
        gnn_model_card = json.load(f)
    problems = []
    for member in gnn_model_card.get("ensemble", []):
        path = osp.join(directory, member["file"])
        if not osp.isfile(path):
            problems.append(f"{path} is missing")
        elif member.get("sha256") and _sha256(path) != member["sha256"]:
            problems.append(f"{path} does not match the SHA-256 of its GNN model card")
    return problems


def _download(url, destination):
    """Fetches `url`, retrying, and checks the length the server announced."""
    for attempt in range(1, RETRIES + 1):
        try:
            path, headers = request.urlretrieve(url, destination)
            expected = headers.get("Content-Length")
            if expected is not None and osp.getsize(path) != int(expected):
                raise URLError(f"retrieval incomplete: got {osp.getsize(path)} of {expected} bytes")
            return
        except (HTTPError, URLError) as error:
            if osp.exists(destination):
                os.remove(destination)
            print(f"Download attempt {attempt} of {RETRIES} failed: {error}")
            if attempt == RETRIES:
                raise
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))


def install(name, version, forge_id, force=False):
    """Downloads and checks one network; returns True on success or when there is nothing to do."""
    directory = osp.join(GNN_DIR, name, version)
    if forge_id == "-":
        print(f"Skip GNN {name}/{version}: not uploaded to the forge yet")
        return True
    if not force and osp.isdir(directory) and not verify(directory):
        print(f"Skip GNN {name}/{version}: installed")
        return True
    os.makedirs(GNN_DIR, exist_ok=True)
    file_name = f"gnn_{name}_{version}.tar.gz"
    url = f"{FORGE_BASE_URL}/{forge_id}/{file_name}"
    tmp_file = osp.join(GNN_DIR, file_name + ".part")
    print(f"Downloading GNN {name}/{version} from {url}")
    try:
        _download(url, tmp_file)
        with tarfile.open(tmp_file) as archive:
            prefix = f"{name}/{version}/"
            outside = [m.name for m in archive.getmembers() if not (m.name + "/").startswith(prefix)]
            if outside:
                print(f"Refusing {file_name}: files outside {prefix}: {outside[:3]}")
                return False
            archive.extractall(GNN_DIR, filter="data")
    except Exception as error:      # network, archive or disk: report and go on with the others
        print(f"GNN {name}/{version} failed: {error}")
        return False
    finally:
        if osp.exists(tmp_file):
            os.remove(tmp_file)
    problems = verify(directory)
    for problem in problems:
        print(f"GNN {name}/{version}: {problem}")
    if not problems:
        print(f"GNN {name}/{version} installed in {directory}")
    return not problems


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the GNN reconstruction networks.")
    parser.add_argument("names", nargs="*", help="only these networks (default: all in the flag file)")
    parser.add_argument("--force", action="store_true", help="download again even if installed")
    args = parser.parse_args(argv)
    print("==============================")
    entries = [e for e in read_flag() if not args.names or e[0] in args.names]
    ok = all([install(*entry, force=args.force) for entry in entries])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

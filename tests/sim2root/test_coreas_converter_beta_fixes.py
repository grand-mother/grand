# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to the CoREAS converter.

Each test names its GitHub issue and fails without the fix.  The converter
writes into its working folder, so it runs on a copy of
``sim2root/CoREASRawRoot`` in a temporary directory.
"""

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
COREAS = ROOT / "sim2root" / "CoREASRawRoot"

pytestmark = pytest.mark.skipif(not (COREAS / "proton" / "SIM004100.reas").exists(),
                                reason="the committed CoREAS sample is not present")

#: Lines of the full .reas that carry the shower parameters; the committed
#: SIM004100.reas lacks them, which sends the converter to the CORSIKA .inp.
EVENT_KEYS = ("ShowerZenithAngle", "ShowerAzimuthAngle", "PrimaryParticleEnergy", "PrimaryParticleType",
              "DepthOfShowerMaximum", "DistanceOfShowerMaximum", "MagneticFieldStrength",
              "MagneticFieldInclinationAngle", "GeomagneticAngle")


def _workdir(tmp_path, with_event_block):
    work = tmp_path / "CoREASRawRoot"
    shutil.copytree(COREAS, work, ignore=shutil.ignore_patterns("*.rawroot", "__pycache__"))
    if with_event_block:
        full = (work / "proton" / "SIM004100-001004105-000000001.reas").read_text().splitlines()
        block = [line for line in full if line.split("=")[0].strip() in EVENT_KEYS]
        reas = work / "proton" / "SIM004100.reas"
        reas.write_text(reas.read_text().rstrip("\n") + "\n" + "\n".join(block) + "\n")
    return work


def _convert(work):
    from sim2root.Common.raw_root_trees import RawShowerTree

    done = subprocess.run([sys.executable, "CoreasToRawROOT.py", "--file", "proton/SIM004100.reas"],
                          cwd=work, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    shower = RawShowerTree(str(work / "Coreas_004100.rawroot"))
    shower.get_entry(0)
    angles = float(shower.zenith), float(shower.azimuth)
    shower.stop_using()
    return angles


@pytest.mark.parametrize("with_event_block", [False, True], ids=["inp_fallback", "reas_block"])
def test_azimuth_is_where_the_shower_comes_from(tmp_path, with_event_block):
    r"""#209: the .inp fallback gave 180 - PHIP = -13.57 (mirrored); both paths must give +13.57.

    CORSIKA PHIP = 193.57 and the .reas ShowerAzimuthAngle = -166.43 are both the
    direction of travel; GRAND's "comes from" azimuth is 13.57 degrees, which is
    also what a plane-wave fit to the sample's antenna timing gives.
    """
    zenith, azimuth = _convert(_workdir(tmp_path, with_event_block))
    assert zenith == pytest.approx(55.0, abs=1e-3)
    assert azimuth == pytest.approx(13.57, abs=1e-3)

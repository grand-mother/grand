# -*- coding: utf-8 -*-
r"""Utility scripts: argument parsing and where they write (#246)."""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _run(*argv, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    return subprocess.run([sys.executable, *map(str, argv)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)


def test_get_version_from_any_folder(tmp_path):
    done = _run(SCRIPTS / "get_version.py", cwd=tmp_path)
    assert done.returncode == 0
    assert done.stdout.strip() == "version=" + (ROOT / "grand" / "dataio" / "version").read_text().strip()


def test_help_does_not_run_or_write(tmp_path):
    for script in ("extract_rf_chain.py", "plot_tmax_vmax.py"):
        done = _run(SCRIPTS / script, "-h", cwd=tmp_path)
        assert done.returncode == 0 and "usage" in done.stdout, (script, done.stderr[-500:])
    assert not list(tmp_path.iterdir())


def test_bad_arguments_are_refused(tmp_path):
    missing = _run(SCRIPTS / "plot_tmax_vmax.py", tmp_path / "none.root", cwd=tmp_path)
    assert missing.returncode != 0 and "GRANDlib: plot_tmax_vmax: no such file" in missing.stderr
    for args in (("--lst", "24"), ("--lst", "3.7"), ("--du_type", "nope")):
        done = _run(SCRIPTS / "plot_noise.py", *args, cwd=tmp_path)
        assert done.returncode == 2, (args, done.stderr[-300:])
    for script in ("plot_rf_chain.py", "plot_Vout_AT_Device.py"):
        done = _run(SCRIPTS / script, "galactic", cwd=tmp_path)
        assert done.returncode == 2 and "invalid choice" in done.stderr


def test_extract_rf_chain_writes_where_asked(tmp_path):
    out = tmp_path / "tf.npy"
    assert _run(SCRIPTS / "extract_rf_chain.py", "-o", out, cwd=tmp_path).returncode == 0
    assert out.is_file()
    again = _run(SCRIPTS / "extract_rf_chain.py", "-o", out, cwd=tmp_path)
    assert again.returncode != 0 and "--overwrite" in again.stderr


def test_snakemake_report_refuses_a_non_log(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("nothing here\n")
    done = _run(SCRIPTS / "pipeline" / "snakemake_report.py", log, cwd=tmp_path)
    assert done.returncode != 0 and "no Snakemake log lines" in done.stderr

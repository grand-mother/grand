"""
Install script for grand.
Usage: python install.py
"""

import os
import subprocess
import shutil
import sys
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import urllib.request
import tarfile


TURTLE_URL = "https://github.com/niess/turtle.git"
GULL_URL = "https://github.com/niess/gull.git"
TURTLE_TAG = "v0.8"
GULL_TAG = "286ace5"

ZIG_VERSION = "0.16.0"
ZIG_DL_URL = "https://ziglang.org/download/"


@dataclass
class Config:
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    zig_bin: Optional[str] = None

    @property
    def src_dir(self) -> Path:
        return self.root_dir / "src"

    @property
    def build_dir(self) -> Path:
        return self.src_dir / "build"

    @property
    def grand_dir(self) -> Path:
        return self.root_dir / "grand"

    @property
    def lib_dir(self) -> Path:
        return self.root_dir / "lib"

    @property
    def zig_path(self) -> Path:
        return self.root_dir / ".zig"


def clone_repos(cfg: Config):
    cfg.build_dir.mkdir(parents=True, exist_ok=True)

    # turtle — clone specific tag directly
    turtle_dir = cfg.build_dir / "turtle"
    if not turtle_dir.exists():
        print("Cloning turtle...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    TURTLE_URL,
                    str(turtle_dir),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(turtle_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to clone turtle: {e}") from e
    else:
        print("turtle already cloned, skipping.")

    # gull — clone full then checkout specific commit
    gull_dir = cfg.build_dir / "gull"
    if not gull_dir.exists():
        print("Cloning gull...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    GULL_URL,
                    str(gull_dir),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(gull_dir),
                    "checkout",
                    GULL_TAG,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(gull_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to clone gull: {e}") from e
    else:
        print("gull already cloned, skipping.")


def install_zig(cfg: Config):

    system = platform.system()
    machine = platform.machine()

    # check if zig is already available
    zig_bin = shutil.which("zig")
    if zig_bin is not None:
        print(f"Found zig executable at {zig_bin}...")
        version = (
            subprocess.run([zig_bin, "version"], check=True, capture_output=True)
            .stdout.decode()
            .strip()
        )
        print(f"Zig version: {version}")
        if version == ZIG_VERSION:
            print(f"Zig version {version} matches the requirement.")
            cfg.zig_bin = zig_bin
            return
        print(
            f"Found zig version: {version}, but requires zig version {ZIG_VERSION}..."
        )

    # also check if we already downloaded it
    existing = list(cfg.zig_path.glob("zig-*/zig")) if cfg.zig_path.exists() else []
    if existing:
        cfg.zig_bin = str(existing[0])
        print(f"Using previously downloaded Zig at {cfg.zig_bin}")
        return

    # download zig
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    if system == "Darwin":
        url = f"{ZIG_DL_URL}{ZIG_VERSION}/zig-{arch}-macos-{ZIG_VERSION}.tar.xz"
    elif system == "Linux":
        url = f"{ZIG_DL_URL}{ZIG_VERSION}/zig-{arch}-linux-{ZIG_VERSION}.tar.xz"
    elif system == "Windows":
        arch = "aarch64" if machine == "ARM64" else "x86_64"
        url = f"{ZIG_DL_URL}{ZIG_VERSION}/zig-{arch}-windows-{ZIG_VERSION}.zip"
    else:
        raise RuntimeError(f"Please install Zig manually from {ZIG_DL_URL}")

    cfg.zig_path.mkdir(exist_ok=True)
    print(f"Downloading Zig from {url}...")
    if system == "Windows":
        tarball = cfg.zig_path / "zig.zip"
    else:
        tarball = cfg.zig_path / "zig.tar.xz"
    urllib.request.urlretrieve(url, str(tarball))

    verify_zig_signature(cfg, tarball, url)

    print("Extracting Zig...")
    if system == "Windows":
        import zipfile

        with zipfile.ZipFile(str(tarball)) as zf:
            zf.extractall(str(cfg.zig_path))
    else:
        with tarfile.open(str(tarball)) as tar:
            tar.extractall(str(cfg.zig_path))
    tarball.unlink()

    cfg.zig_bin = str(next(cfg.zig_path.glob("zig-*/zig")))
    if system == "Windows":
        cfg.zig_bin = str(next(cfg.zig_path.glob("zig-*/zig.exe")))
    else:
        cfg.zig_bin = str(next(cfg.zig_path.glob("zig-*/zig")))
    print(f"Zig installed at {cfg.zig_bin}")


def verify_zig_signature(cfg: Config, tarball: Path, url: str):
    import minisign

    ZIG_PUBLIC_KEY = "RWSGOq2NVecA2UPNdBUZykf1CCb147pkmdtYxgb3Ti+JO/wCYvhbAb/U"

    sig_url = url + ".minisig"
    sig_file = tarball.parent / (tarball.name + ".minisig")
    print(f"Downloading signature from {sig_url}...")
    urllib.request.urlretrieve(sig_url, str(sig_file))

    print("Verifying signature...")
    try:
        pk = minisign.PublicKey.from_base64(ZIG_PUBLIC_KEY)
        with open(tarball, "rb") as f:
            data = f.read()
        with open(sig_file, "rb") as f:
            sig = minisign.Signature.from_bytes(f.read())
        pk.verify(data, sig)
        print("Signature verified successfully.")
    except Exception as e:
        tarball.unlink(missing_ok=True)
        raise RuntimeError(
            f"Zig signature verification failed: {e}\n"
            "The download may be corrupted or tampered with."
        ) from e
    finally:
        sig_file.unlink(missing_ok=True)


def build_c_libs(cfg: Config):
    # Note: Zig targets the current OS version by default which may differ
    # from cffi's deployment target, causing linker warnings. These are
    # harmless for static linking and can be ignored.
    print("Building C libraries with Zig...")
    subprocess.run(
        [
            cfg.zig_bin,
            "build",
            "-Doptimize=ReleaseFast",
            "--prefix",
            str(cfg.build_dir),
        ],
        cwd=str(cfg.src_dir),
        check=True,
    )


def build_python_ext(cfg: Config):
    print("Building Python extension with cffi...")
    env = os.environ.copy()
    env["LDFLAGS"] = ""
    env["CFLAGS"] = ""
    env["CXXFLAGS"] = ""

    subprocess.run(
        [
            sys.executable,
            str(cfg.src_dir / "build_core.py"),
            str(cfg.build_dir),
        ],
        cwd=str(cfg.src_dir),
        env=env,
        check=True,
    )


def copy_files(cfg: Config):
    print("Copying `_core.abi3.so` to grand/...")
    shutil.copy(str(cfg.build_dir / "grand" / "_core.abi3.so"), str(cfg.grand_dir))

    print("Copying library files to lib/...")
    for lib_file in (cfg.build_dir / "lib").iterdir():
        shutil.copy(str(lib_file), str(cfg.lib_dir))


def install_package(cfg: Config):
    print("Installing grand package...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
        ],
        cwd=str(cfg.root_dir),
        check=True,
    )


def cleanup(cfg: Config):
    print("Cleaning up build artifacts...")
    shutil.rmtree(cfg.build_dir, ignore_errors=True)
    shutil.rmtree(cfg.src_dir / ".zig-cache", ignore_errors=True)
    print("Done.")


if __name__ == "__main__":
    print("=== Installing grand ===")
    cfg = Config()
    try:
        clone_repos(cfg)
        install_zig(cfg)
        build_c_libs(cfg)
        build_python_ext(cfg)
        copy_files(cfg)
        install_package(cfg)
    except Exception:
        cleanup(cfg)
        raise
    cleanup(cfg)
    print("=== Done ===")

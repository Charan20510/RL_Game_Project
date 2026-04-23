from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def rust_binary_path() -> Path:
    root = Path(__file__).parent.parent
    bin_name = "bobby-carrot"
    if os.name == "nt":
        bin_name += ".exe"
    return root / "rust" / "target" / "release" / bin_name


def build_rust():
    """Invoke `cargo build --release` in the rust folder."""
    root = Path(__file__).parent.parent
    print("Building Rust game (this may take a minute)...")
    res = subprocess.run(["cargo", "build", "--release"], cwd=root / "rust")
    res.check_returncode()


def main():
    binary = rust_binary_path()
    if not binary.exists():
        build_rust()
        if not binary.exists():
            print("error: failed to build rust binary", file=sys.stderr)
            sys.exit(1)

    # forward any command line args to the game
    args = [str(binary)] + sys.argv[1:]
    try:
        subprocess.run(args)
    except FileNotFoundError:
        print("error: rust binary not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

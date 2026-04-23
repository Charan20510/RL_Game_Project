# Bobby Carrot – Python Folder

The `python/` directory provides two ways to run the classic Bobby Carrot game
from a Python environment:

1. **Rust launcher** (`run.py`) – builds and executes the existing Rust
game.  This variant requires a Rust toolchain but does not depend on `pygame`.
2. **Pure‑Python port** (`bobby_carrot` package) – a faithful re‑implementation
of the game in Python using `pygame`.  All assets (tiles, audio, level files)
are bundled under `python/assets` so the port can run standalone.

The remainder of the repository is the original Rust source; the Python port
shares its assets and level data but lives completely in `python/`.

---

## Prerequisites

* **Python 3.7+** (3.13 recommended – `pygame` wheels currently available for
  up through 3.13).  If you are using a newer interpreter, install `pygame`
  manually or use a different version.
* **pygame** (only required for the Python implementation).
* **Rust** and SDL2 development libraries if you intend to use the Rust
  launcher or modify the Rust code.

---

## Setup

Create a virtual environment and install the package:

```powershell
cd python
python -m venv ../.venv313   # or use your preferred interpreter
& ..\.venv313\Scripts\Activate.ps1
pip install -e .
```

Install `pygame` in the activated environment:

```powershell
pip install pygame      # should succeed on Python 3.13 or earlier
```

Problems building `pygame` usually indicate an incompatible Python version;
see the note above.

---

## Running

*Rust launcher (no pygame needed):*
```powershell
python run.py [map]
```

*Python port:*
```powershell
python -m bobby_carrot [map]
# or, after installing, use the `run-python` console script
```

  * The Python game plays MIDI audio if supported; otherwise a system beep
    is used for feedback.
  * Press **F** during play to toggle full-screen mode.

`[map]` may be a plain number (e.g. `5`) or `normal-#`/`egg-#`; leave it out
for the default level.

If `pygame` is not installed the Python port will warn and exit, in which case
you can fall back to the Rust launcher.

---

## Notes for developers

* Source code: `python/bobby_carrot`.
* Assets: `python/assets` (maps, images, audio).
* The Python package is intentionally simple and is installed in editable mode
  by default.
* The Rust code remains untouched in `rust/src/main.rs`; the Python port is a
  line‑by‑line translation with identical gameplay.

---

This README now serves as the primary documentation for the Python portion of
the project and is ready for display on GitHub.  Adjust or extend as needed.
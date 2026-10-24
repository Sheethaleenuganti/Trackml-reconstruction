#!/usr/bin/env python3
"""Environment verification script.

Run this after installing dependencies to confirm what is actually present
in your environment, and to catch the "wrong trackml package" trap early.

Usage:
    python scripts/check_environment.py
"""
import importlib
import sys


def _version_of(module_name):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "unknown (no __version__ attribute)")
    except ImportError as exc:
        return f"NOT INSTALLED ({exc})"


def _check_trackml_identity():
    """Distinguish the real TrackML utility library from the unrelated
    PyPI package that happens to share the name 'trackml'.

    The real library (github.com/LAL/trackml-library) exposes
    trackml.dataset.load_event and trackml.score.score_event.
    The unrelated PyPI package (an experiment-tracking client) does not.
    """
    try:
        import trackml  # noqa: F401
    except ImportError as exc:
        return f"NOT INSTALLED ({exc})"

    has_dataset = importlib.util.find_spec("trackml.dataset") is not None
    has_score = importlib.util.find_spec("trackml.score") is not None

    if has_dataset and has_score:
        try:
            from trackml.dataset import load_event  # noqa: F401
            from trackml.score import score_event  # noqa: F401
        except ImportError as exc:
            return f"trackml package found but submodules broken ({exc})"
        version = getattr(trackml, "__version__", "unknown")
        location = getattr(trackml, "__file__", "unknown location")
        return f"OK — official library confirmed (version={version}, at {location})"
    else:
        location = getattr(trackml, "__file__", "unknown location")
        return (
            f"WRONG PACKAGE INSTALLED at {location} — this is NOT the "
            "official TrackML library. It is missing trackml.dataset and/or "
            "trackml.score. Uninstall it and install from "
            "https://github.com/LAL/trackml-library instead. "
            "See docs/DATA_ACCESS.md."
        )


def main():
    print("=" * 70)
    print("TrackML Project — Environment Verification")
    print("=" * 70)
    print(f"Python version:      {sys.version.split()[0]}")
    print(f"Python executable:   {sys.executable}")
    print("-" * 70)
    print(f"numpy version:       {_version_of('numpy')}")
    print(f"pandas version:      {_version_of('pandas')}")
    print(f"scipy version:       {_version_of('scipy')}")
    print(f"scikit-learn version:{_version_of('sklearn')}")
    print(f"matplotlib version:  {_version_of('matplotlib')}")
    print("-" * 70)
    print(f"trackml identity check: {_check_trackml_identity()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

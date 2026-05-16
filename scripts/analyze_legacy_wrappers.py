from pathlib import Path
import runpy


if __name__ == "__main__":
    _target = Path(__file__).resolve().parent / "dev/analyze_legacy_wrappers.py"
    runpy.run_path(str(_target), run_name="__main__")

from pathlib import Path
import runpy


if __name__ == "__main__":
    _target = Path(__file__).resolve().parent / "dev/audit_core_wrapper_lifecycle.py"
    runpy.run_path(str(_target), run_name="__main__")

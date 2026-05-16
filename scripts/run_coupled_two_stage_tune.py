from pathlib import Path
import runpy


if __name__ == "__main__":
    _target = Path(__file__).resolve().parent / "evaluate/run_coupled_two_stage_tune.py"
    runpy.run_path(str(_target), run_name="__main__")

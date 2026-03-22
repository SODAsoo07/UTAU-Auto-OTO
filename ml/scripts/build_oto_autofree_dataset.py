from __future__ import annotations

import os
import runpy


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "autofree", "build_dataset.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()

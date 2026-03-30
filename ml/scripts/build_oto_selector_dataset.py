"""Redirect - see coupled/build_selector_dataset.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if __name__ == "__main__":
    from ml.scripts.coupled.build_selector_dataset import main

    main()

"""Redirect - see coupled/evaluate_selector.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if __name__ == "__main__":
    from ml.scripts.coupled.evaluate_selector import main

    main()

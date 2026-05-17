from __future__ import annotations

"""Removed direct-parameter OTO CRNN top-level entrypoint."""

import sys


def main() -> int:
    print(
        "Direct-parameter OTO CRNN is deprecated. Use "
        "`ml.scripts.coarse_crnn.generate_boundary_oto` / `auto_oto` instead. "
        "For archived diagnostics, call "
        "`ml.scripts.coarse_crnn.deprecated.direct_param.predict_oto` explicitly.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

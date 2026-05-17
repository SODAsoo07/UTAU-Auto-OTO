from __future__ import annotations

"""Removed direct-parameter OTO CRNN top-level entrypoint."""

import sys


def main() -> int:
    print(
        "Direct-parameter OTO CRNN training is deprecated. Use "
        "`ml.scripts.coarse_crnn.train_boundary_oto` instead. "
        "For archived diagnostics, call "
        "`ml.scripts.coarse_crnn.deprecated.direct_param.train_oto` explicitly.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

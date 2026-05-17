from __future__ import annotations

"""Removed direct-parameter OTO CRNN top-level entrypoint."""

import sys


def main() -> int:
    print(
        "Direct-parameter OTO CRNN evaluation is deprecated. Use "
        "`ml.scripts.coarse_crnn.evaluate_boundary_oto` or "
        "`ml.scripts.coarse_crnn.evaluate_boundary_manifest` instead. "
        "For archived diagnostics, call "
        "`ml.scripts.coarse_crnn.deprecated.direct_param.evaluate_oto` explicitly.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

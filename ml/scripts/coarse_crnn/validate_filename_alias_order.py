from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from core.oto_file_utils import parse_oto_line, read_text_with_fallback


_TOKEN_RE = re.compile(r"[A-Za-z0-9\uAC00-\uD7A3]+")


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(str(text or ""))]


def _ordered_match(expected: list[str], observed: list[str]) -> tuple[int, float]:
    if not expected:
        return 0, 1.0
    cursor = 0
    matched = 0
    for tok in observed:
        if cursor >= len(expected):
            break
        if tok == expected[cursor]:
            matched += 1
            cursor += 1
    return matched, float(matched) / float(len(expected))


def run(source_oto: str, out_path: str, min_ratio: float) -> dict:
    per_wav: dict[str, list[str]] = defaultdict(list)
    for raw in read_text_with_fallback(source_oto).splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav = os.path.basename(str(parsed.get("wav", "") or "").replace("\\", "/").strip()).lower()
        alias = str(parsed.get("alias", "") or "").strip()
        if not wav or not alias:
            continue
        per_wav[wav].append(alias)

    mismatches = []
    for wav, aliases in sorted(per_wav.items()):
        expected = _tokens(os.path.splitext(wav)[0])
        observed = []
        for alias in aliases:
            observed.extend(_tokens(alias))
        matched, ratio = _ordered_match(expected, observed)
        if len(expected) >= 2 and ratio < float(min_ratio):
            mismatches.append(
                {
                    "wav": wav,
                    "expected_tokens": expected,
                    "observed_tokens_head": observed[:32],
                    "row_aliases_head": aliases[:12],
                    "matched": int(matched),
                    "expected_count": int(len(expected)),
                    "ratio": float(ratio),
                }
            )

    report = {
        "source_oto": os.path.abspath(source_oto),
        "wav_files": int(len(per_wav)),
        "mismatch_count": int(len(mismatches)),
        "mismatch_rate": (float(len(mismatches)) / float(len(per_wav))) if per_wav else 0.0,
        "min_ratio": float(min_ratio),
        "mismatches": mismatches,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate whether filename token order matches alias order per wav file."
    )
    parser.add_argument("--source-oto", required=True, help="Path to source/base oto.ini")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--min-ratio", type=float, default=0.60, help="Ordered match ratio threshold")
    args = parser.parse_args()

    report = run(
        source_oto=str(args.source_oto),
        out_path=str(args.out),
        min_ratio=float(args.min_ratio),
    )
    print(json.dumps({k: report[k] for k in ("wav_files", "mismatch_count", "mismatch_rate", "min_ratio")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

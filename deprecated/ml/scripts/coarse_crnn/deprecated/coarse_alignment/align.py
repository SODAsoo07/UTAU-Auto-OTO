from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.deprecated.coarse_alignment.inference import align_wav
from core.coarse_crnn.lang import normalize_language
from core.coarse_crnn.deprecated.coarse_alignment.workflow import resolve_coarse_crnn_model_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run coarse_crnn alignment for one wav or a top-level wav folder.")
    parser.add_argument("--audio", default="")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--language", required=True, choices=["ko", "ja", "korean", "japanese"])
    parser.add_argument("--transcript", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0. auto prefers CUDA when available.")
    args = parser.parse_args()

    model_path = resolve_coarse_crnn_model_path(args.model)
    lang = normalize_language(args.language)
    wavs = []
    if args.audio:
        wavs.append(os.path.abspath(args.audio))
    if args.input_dir:
        root = os.path.abspath(args.input_dir)
        wavs.extend(os.path.join(root, name) for name in sorted(os.listdir(root)) if name.lower().endswith(".wav"))
    if not wavs:
        raise SystemExit("--audio or --input-dir is required")

    results = []
    for wav in wavs:
        result = align_wav(
            wav_path=wav,
            output_dir=args.out,
            model_path=model_path,
            language=lang,
            transcript=args.transcript if len(wavs) == 1 else "",
            device=args.device,
        )
        results.append(result.__dict__)
    print(json.dumps({"model": model_path, "results": results}, ensure_ascii=False, indent=2))
    return 0 if all(r.get("success") for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())


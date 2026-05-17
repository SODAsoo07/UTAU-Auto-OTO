"""Extract frame-level phone alignment from voicebank wavs using MFA.

Phase 1 of the multi-head phone-aware boundary decoder migration
(Docs/multi_head_phone_aware_design_2026-05-17.md).

Workflow per voicebank:

  1. Locate wav files + their slot/alias metadata (from training manifest).
  2. Generate a per-wav ``.lab`` transcription (syllable sequence) alongside
     the wav, derived from the wav filename (apostrophe-separated KO tokens
     or hiragana/katakana characters for JA).
  3. Run ``mfa align`` with the pre-trained ``korean_mfa`` / ``japanese_mfa``
     acoustic model + dictionary.
  4. Parse each resulting TextGrid into frame-level label arrays at the
     project's standard hop (20 ms).
  5. Write a JSONL alignment file per voicebank with one row per wav:
        {wav, n_frames, cvs:[…], c_id:[…], v_id:[…], conf:[…]}

Usage::

    python ml/scripts/coarse_crnn/extract_phone_alignment.py \
        --manifest ml_workspace/coarse_crnn/oto_splits_full/oto_train.jsonl \
        --voicebanks "dokgo_ina_2xbinCVC,Yeo-Myeong Original_CVC,..." \
        --mfa-bin .env/Scripts/mfa.exe \
        --out-dir ml_workspace/coarse_crnn/phone_alignments_poc \
        --hop-ms 20

The script is intentionally idempotent — re-running with the same args
skips voicebanks whose output JSONL already exists. Use ``--force`` to
regenerate.

Decisions per Docs/multi_head_phone_aware_design_2026-05-17.md §11:
  - MFA already installed in project ``.env`` (venv activation handled
    via PATH prefix in subprocess calls).
  - Phone vocabulary is language-separated (KO and JA have distinct ID
    spaces); written into ``phone_vocab_v1_<lang>.json`` for downstream
    use.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KO_MFA_MODEL = "korean_mfa"
KO_MFA_DICT = "korean_mfa"
JA_MFA_MODEL = "japanese_mfa"
JA_MFA_DICT = "japanese_mfa"

# Hangul jamo unicode block for splitting raw Hangul filenames.
_HANGUL_RE = re.compile(r"[가-힣]")
_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")


# ---------------------------------------------------------------------------
# Filename → orthographic transcription
# ---------------------------------------------------------------------------


def filename_to_syllable_tokens(wav_name: str, *, language: str) -> list[str]:
    """Extract the orthographic syllable sequence from a voicebank wav name.

    Returns tokens in the form MFA expects:
    - **KO** → Hangul syllables (the pre-trained ``korean_mfa`` dictionary is
      keyed on Hangul, not romanized). Romanized tokens like ``geuNG`` are
      converted via ``parse_romaji_syllable``.
    - **JA** → hiragana/katakana glyphs (matches ``japanese_mfa`` dictionary
      directly).

    Splitting heuristic (per voicebank filename convention):
      1. Apostrophe-separated (KO CVVC/CVC: ``geuNG'geuN'geuM'…``)
      2. Underscore-separated (Yeo-Myeong / generic: ``a_i_u_e_o_eu_eo``)
      3. Native script (Hangul/kana characters one-per-syllable)
      4. Concatenated romanized (``_vavaveovavevavova``) — returns whole
         stem; caller may fall back to manifest aliases when this fails
         to produce a usable transcription.
    """
    stem = os.path.splitext(os.path.basename(str(wav_name or "")))[0]
    if stem.startswith("_"):
        stem = stem[1:]
    if not stem:
        return []

    # Native-script fast path (JA hiragana/katakana, KO Hangul)
    if language == "japanese" and _KANA_RE.search(stem):
        return [ch for ch in stem if _KANA_RE.match(ch)]
    if language == "korean" and _HANGUL_RE.search(stem):
        return [ch for ch in stem if _HANGUL_RE.match(ch)]

    # Romanized — try apostrophe then underscore as syllable separator
    raw_tokens: list[str] = []
    if "'" in stem:
        raw_tokens = [tok for tok in stem.split("'") if tok]
    elif "_" in stem:
        raw_tokens = [tok for tok in stem.split("_") if tok]
    else:
        # No separator (concatenated). Best-effort whole-stem token —
        # the converter below may fail, caller can fall back.
        raw_tokens = [stem]

    # For KO romanized, convert each syllable to Hangul so it matches the
    # pre-trained MFA dictionary keys.
    if language == "korean":
        try:
            from core.alignment.lab_generator import parse_romaji_syllable
        except Exception:
            parse_romaji_syllable = None
        if parse_romaji_syllable is not None:
            converted: list[str] = []
            for tok in raw_tokens:
                hangul = parse_romaji_syllable(tok)
                # parse_romaji_syllable returns the original token on failure;
                # filter those out so MFA doesn't see un-aligned junk.
                if hangul and _HANGUL_RE.match(hangul):
                    converted.append(hangul)
            if converted:
                return converted
    return raw_tokens


# ---------------------------------------------------------------------------
# Manifest scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WavTask:
    voicebank: str
    language: str
    wav_path: str
    syllable_tokens: tuple[str, ...]


def collect_wav_tasks(
    manifest_path: str,
    *,
    target_voicebanks: set[str],
) -> dict[str, list[WavTask]]:
    """Walk the manifest once and group wav paths by voicebank.

    Each wav appears in many manifest rows (one per alias slot); we keep
    only the first encounter per wav to build a deduplicated task list.
    """
    seen: set[tuple[str, str]] = set()
    by_vb: dict[str, list[WavTask]] = defaultdict(list)
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            vb = str(row.get("voicebank_id", "") or "")
            if vb not in target_voicebanks:
                continue
            audio = str(row.get("audio", "") or "")
            if not audio:
                continue
            key = (vb, audio)
            if key in seen:
                continue
            seen.add(key)
            lang = str(row.get("language", "") or "").lower()
            tokens = filename_to_syllable_tokens(audio, language=lang)
            by_vb[vb].append(
                WavTask(
                    voicebank=vb,
                    language=lang,
                    wav_path=audio,
                    syllable_tokens=tuple(tokens),
                )
            )
    return dict(by_vb)


# ---------------------------------------------------------------------------
# MFA wrapper
# ---------------------------------------------------------------------------


def _mfa_env(mfa_bin: str) -> dict[str, str]:
    """Augment os.environ so the MFA subprocess finds libsndfile + scripts."""
    env = os.environ.copy()
    bin_dir = os.path.dirname(os.path.abspath(mfa_bin))
    library_bin = os.path.join(os.path.dirname(bin_dir), "Library", "bin")
    extra_path = os.pathsep.join([library_bin, bin_dir])
    env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
    return env


def build_mfa_corpus(
    corpus_dir: Path,
    tasks: list[WavTask],
) -> int:
    """Stage wavs + .lab transcriptions in a flat corpus directory."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for idx, task in enumerate(tasks):
        if not task.syllable_tokens:
            continue
        # MFA prefers ASCII filenames — use a stable index-based slug.
        slug = f"wav_{idx:06d}"
        src_wav = Path(task.wav_path)
        if not src_wav.is_file():
            continue
        dst_wav = corpus_dir / f"{slug}.wav"
        dst_lab = corpus_dir / f"{slug}.lab"
        # Hard-link if same drive (Windows hard-links are cheap), else copy.
        if not dst_wav.exists():
            try:
                os.link(src_wav, dst_wav)
            except OSError:
                import shutil
                shutil.copy2(src_wav, dst_wav)
        # Transcription: tokens joined by space.
        dst_lab.write_text(" ".join(task.syllable_tokens), encoding="utf-8")
        staged += 1
    return staged


def run_mfa_align(
    mfa_bin: str,
    corpus_dir: Path,
    output_dir: Path,
    *,
    dictionary: str,
    acoustic_model: str,
    n_jobs: int = 2,
) -> tuple[int, str]:
    """Invoke ``mfa align`` and return (exit_code, last_stderr_lines)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.path.abspath(mfa_bin),
        "align",
        "--clean",
        "--num_jobs",
        str(int(n_jobs)),
        "--output_format",
        "long_textgrid",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
    ]
    proc = subprocess.run(
        cmd,
        env=_mfa_env(mfa_bin),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = "\n".join((proc.stderr or "").splitlines()[-20:])
    return int(proc.returncode), tail


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------


_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*xmin\s*=\s*([0-9.]+)\s*xmax\s*=\s*([0-9.]+)\s*text\s*=\s*\"([^\"]*)\"",
    re.DOTALL,
)
_TIER_RE = re.compile(r'item \[(\d+)\]:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"([^"]+)"')


def parse_textgrid_phone_intervals(path: Path) -> list[tuple[float, float, str]]:
    """Return ``(start_s, end_s, phone)`` for the 'phones' tier."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Find the phones tier section
    tiers = list(_TIER_RE.finditer(text))
    phones_start = None
    phones_end = len(text)
    for i, m in enumerate(tiers):
        if m.group(2).lower() == "phones":
            phones_start = m.end()
            if i + 1 < len(tiers):
                phones_end = tiers[i + 1].start()
            break
    if phones_start is None:
        return []
    section = text[phones_start:phones_end]
    return [
        (float(m.group(1)), float(m.group(2)), m.group(3).strip())
        for m in _INTERVAL_RE.finditer(section)
    ]


def phone_to_class(phone: str, language: str) -> str:
    """Coarse C/V/Sil classification used for the first multi-head."""
    p = str(phone or "").strip().lower()
    if not p or p in {"sil", "sp", "spn", "<unk>", ""}:
        return "Sil"
    # KO/JA vowel inventories (approximate — MFA-specific phone symbols vary)
    ko_vowels = {"a", "ɐ", "e", "ɛ", "i", "o", "ɔ", "u", "ʌ", "ɯ", "ɨ", "ø", "y"}
    ja_vowels = {"a", "i", "u", "e", "o", "ɯ"}
    base = re.sub(r"[ːˑʰʷ]", "", p)
    base = base.split("_")[0]
    if language == "korean" and base in ko_vowels:
        return "V"
    if language == "japanese" and base in ja_vowels:
        return "V"
    return "C"


def textgrid_to_frame_labels(
    intervals: list[tuple[float, float, str]],
    *,
    language: str,
    hop_ms: float,
    duration_ms: float,
) -> dict[str, list]:
    """Discretize phone intervals to per-frame labels.

    Returns a dict with:
      cvs:   list[int]  (0=Sil, 1=V, 2=C)
      phone: list[str]  (raw MFA phone symbol; consumer maps to vocab id)
    """
    n_frames = max(0, int(round(float(duration_ms) / float(hop_ms))))
    cvs = [0] * n_frames  # default Sil
    phone = [""] * n_frames
    for start_s, end_s, p in intervals:
        f_start = max(0, int(round(start_s * 1000.0 / hop_ms)))
        f_end = min(n_frames, int(round(end_s * 1000.0 / hop_ms)))
        cls = phone_to_class(p, language)
        cls_id = {"Sil": 0, "V": 1, "C": 2}[cls]
        for f in range(f_start, f_end):
            cvs[f] = cls_id
            phone[f] = p
    return {"cvs": cvs, "phone": phone}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_voicebank(
    vb_name: str,
    tasks: list[WavTask],
    *,
    mfa_bin: str,
    out_dir: Path,
    hop_ms: float,
    n_jobs: int,
    force: bool,
) -> dict:
    """Run MFA alignment + parsing for one voicebank. Returns summary stats."""
    safe_vb = re.sub(r"[^A-Za-z0-9_-]+", "_", vb_name)
    out_jsonl = out_dir / f"{safe_vb}.alignment.jsonl"
    if out_jsonl.exists() and not force:
        return {"voicebank": vb_name, "status": "skipped", "rows": 0}
    if not tasks:
        return {"voicebank": vb_name, "status": "empty", "rows": 0}
    language = tasks[0].language
    if language == "korean":
        dictionary, acoustic = KO_MFA_DICT, KO_MFA_MODEL
    elif language == "japanese":
        dictionary, acoustic = JA_MFA_DICT, JA_MFA_MODEL
    else:
        return {"voicebank": vb_name, "status": f"unsupported_lang_{language}", "rows": 0}

    work_root = Path(tempfile.mkdtemp(prefix=f"mfa_{safe_vb}_"))
    corpus_dir = work_root / "corpus"
    tg_dir = work_root / "textgrids"
    try:
        n_staged = build_mfa_corpus(corpus_dir, tasks)
        if n_staged == 0:
            return {"voicebank": vb_name, "status": "no_wavs_staged", "rows": 0}
        rc, tail = run_mfa_align(
            mfa_bin=mfa_bin,
            corpus_dir=corpus_dir,
            output_dir=tg_dir,
            dictionary=dictionary,
            acoustic_model=acoustic,
            n_jobs=n_jobs,
        )
        if rc != 0:
            return {
                "voicebank": vb_name,
                "status": f"mfa_failed_rc{rc}",
                "rows": 0,
                "stderr_tail": tail,
            }
        # Parse TextGrids back to per-wav alignment rows
        out_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        n_missing = 0
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for idx, task in enumerate(tasks):
                slug = f"wav_{idx:06d}"
                tg_path = tg_dir / f"{slug}.TextGrid"
                if not tg_path.exists():
                    n_missing += 1
                    continue
                intervals = parse_textgrid_phone_intervals(tg_path)
                if not intervals:
                    n_missing += 1
                    continue
                duration_ms = intervals[-1][1] * 1000.0
                labels = textgrid_to_frame_labels(
                    intervals, language=language, hop_ms=hop_ms, duration_ms=duration_ms,
                )
                handle.write(
                    json.dumps(
                        {
                            "voicebank": vb_name,
                            "language": language,
                            "wav": task.wav_path,
                            "hop_ms": hop_ms,
                            "n_frames": len(labels["cvs"]),
                            "cvs": labels["cvs"],
                            "phone": labels["phone"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_written += 1
        return {
            "voicebank": vb_name,
            "status": "ok",
            "rows": n_written,
            "missing_textgrids": n_missing,
            "staged_wavs": n_staged,
        }
    finally:
        # Cleanup staging dir to save space (TextGrids preserved? no — we only
        # need the JSONL output)
        import shutil
        shutil.rmtree(work_root, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="MFA-based phone alignment extraction.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--voicebanks", required=True, help="Comma-separated voicebank_id list.")
    ap.add_argument("--mfa-bin", default=".env/Scripts/mfa.exe")
    ap.add_argument("--out-dir", default="ml_workspace/coarse_crnn/phone_alignments_poc")
    ap.add_argument("--hop-ms", type=float, default=20.0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    target_vbs = {s.strip() for s in str(args.voicebanks).split(",") if s.strip()}
    if not target_vbs:
        print("No voicebanks specified", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_by_vb = collect_wav_tasks(args.manifest, target_voicebanks=target_vbs)

    summaries = []
    for vb in target_vbs:
        tasks = tasks_by_vb.get(vb, [])
        print(f"[align] {vb}: {len(tasks)} wavs", flush=True)
        summary = process_voicebank(
            vb,
            tasks,
            mfa_bin=args.mfa_bin,
            out_dir=out_dir,
            hop_ms=float(args.hop_ms),
            n_jobs=int(args.n_jobs),
            force=bool(args.force),
        )
        print(f"[align] {vb}: {summary}", flush=True)
        summaries.append(summary)

    summary_path = out_dir / "extraction_summary.json"
    summary_path.write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[align] wrote summary {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

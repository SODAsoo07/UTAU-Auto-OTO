"""
WFL-ASR aligner engine.

Drop-in alternative to the MFA aligner: runs the WFL-ASR forced-alignment
inference (MLo7Ghinsan/WFL-ASR `refactor` branch + Archivoice pretrained model)
and emits the same 3-tier TextGrid (`phones` / `words` / `utoa_meta`) the
sequence aligner produces, so the existing OTO ingest path consumes it unchanged.

Design (language-agnostic):
  * The forced-align phoneme sequence is built from the per-wav transcript
    `<base>.lab` (space-separated tokens) expanded through the alignment
    `dictionary_path` (token -> phones).  This reuses *exactly* the phones MFA
    would have used, so no per-language decomposition lives here.  The
    token->phones grouping recovered from the dictionary lookup is what lets us
    rebuild the `words` tier (e.g. recompose Korean jamo segments back to the
    original syllable span).
  * WFL inference is invoked as a subprocess against a separate Python env
    (one with torch + transformers), configured via env vars.

Configuration (all optional; sensible PoC defaults):
  UTOA_WFL_PYTHON              python.exe of an env with torch/transformers
  UTOA_WFL_REPO               WFL-ASR checkout dir (contains infer.py)
  UTOA_WFL_MODEL_DIR          model dir (config.yaml,last.ckpt,phonemes.txt,...)
  UTOA_WFL_MODEL_DIR_<LANG>   per-language override (LANG = KOREAN / JAPANESE)
  UTOA_WFL_LANG_ID            integer lang id passed to infer (-l); default 0
  UTOA_WFL_LANG_ID_<LANG>     per-language override
  UTOA_WFL_INFER_TIMEOUT_SEC  subprocess timeout (default 1800)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave
from typing import Dict, List, Sequence, Tuple

from core.pipeline_status import (
    ALIGN_EXEC_MISSING,
    ALIGN_MODEL_MISSING,
    ALIGN_NOT_READY,
    ALIGN_OUTPUT_EMPTY,
    ALIGN_RUN_FAILED,
    OK,
    make_runtime_report,
)

_LAST_WFL_ALIGN_META: Dict[str, object] = {}
_SILENCE_LABELS = {"sp", "sil", "ap", "pau", "spn", ""}


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _set_last_meta(meta: Dict[str, object]) -> None:
    _LAST_WFL_ALIGN_META.clear()
    _LAST_WFL_ALIGN_META.update(meta)


def get_last_wfl_align_meta() -> Dict[str, object]:
    return dict(_LAST_WFL_ALIGN_META)


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
def _env(name: str) -> str:
    return str(os.environ.get(name, "") or "").strip()


def _resolve_wfl_python() -> str:
    override = _env("UTOA_WFL_PYTHON")
    if override and os.path.isfile(override):
        return override
    # Known PoC env (torch + transformers + cuda). Override via UTOA_WFL_PYTHON.
    candidates = [
        os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "soulxsinger", "python.exe"),
        os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "soulxsinger", "bin", "python"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return sys.executable or ""


def _resolve_wfl_repo() -> str:
    override = _env("UTOA_WFL_REPO")
    if override and os.path.isdir(override):
        return override
    candidates = [
        os.path.join(os.path.expanduser("~"), "SODAsoo1", "Devs", "WFL_PoC", "WFL-ASR"),
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "infer.py")):
            return cand
    return override


def _resolve_wfl_model_dir(language: str) -> str:
    lang = str(language or "").strip().lower()
    per_lang = _env(f"UTOA_WFL_MODEL_DIR_{lang.upper()}")
    if per_lang and os.path.isdir(per_lang):
        return per_lang
    generic = _env("UTOA_WFL_MODEL_DIR")
    if generic and os.path.isdir(generic):
        return generic
    poc_root = os.path.join(os.path.expanduser("~"), "SODAsoo1", "Devs", "WFL_PoC")
    defaults = {
        "japanese": os.path.join(poc_root, "model_jp", "japanese"),
        "korean": os.path.join(poc_root, "model_ko", "korean"),
    }
    return defaults.get(lang, "")


def _resolve_wfl_lang_id(language: str) -> int:
    lang = str(language or "").strip().lower()
    raw = _env(f"UTOA_WFL_LANG_ID_{lang.upper()}") or _env("UTOA_WFL_LANG_ID")
    try:
        return int(raw) if raw else 0
    except Exception:
        return 0


def _infer_timeout_sec() -> int:
    raw = _env("UTOA_WFL_INFER_TIMEOUT_SEC")
    try:
        return max(60, int(float(raw))) if raw else 1800
    except Exception:
        return 1800


def _model_files(model_dir: str) -> Dict[str, str]:
    return {
        "config": os.path.join(model_dir, "config.yaml"),
        "ckpt": os.path.join(model_dir, "last.ckpt"),
        "phonemes": os.path.join(model_dir, "phonemes.txt"),
    }


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _ensure_infer_compat(repo: str, callback=None) -> None:
    """Make a clean WFL checkout tolerate the released checkpoints.

    The pretrained ckpts predate the refactor branch's `envelope_head`, so a
    strict `load_state_dict` raises on a fresh `infer.py`.  We idempotently relax
    that single call to `strict=False` (envelope_head is unused at inference).
    """
    infer_path = os.path.join(repo, "infer.py")
    try:
        with open(infer_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return
    needle = "model.load_state_dict(new_state_dict)"
    if needle not in src:
        return  # already patched or different revision
    patched = src.replace(needle, "model.load_state_dict(new_state_dict, strict=False)")
    try:
        with open(infer_path, "w", encoding="utf-8") as f:
            f.write(patched)
        _emit(callback, "[WFL] applied envelope_head compat shim (load_state_dict strict=False)")
    except Exception as exc:
        _emit(callback, f"[WFL] could not apply compat shim ({exc}); a clean checkpoint may fail to load")


def _ensure_encoder(repo: str, callback=None) -> bool:
    """Seed `repo/encoder` (the Whisper encoder model.py loads from cwd) so WFL
    runs fully offline.  Returns True if an encoder is present after this call.

    model.py downloads openai/whisper-base into `<cwd>/encoder` on first run; we
    invoke inference with cwd=repo, so the cache lives at `repo/encoder`.  To
    pre-bundle for offline use, set UTOA_WFL_ENCODER_DIR to a prepared encoder
    dir and it is copied in when missing.
    """
    enc = os.path.join(repo, "encoder")
    if os.path.isdir(enc) and os.listdir(enc):
        return True
    src = _env("UTOA_WFL_ENCODER_DIR")
    if src and os.path.isdir(src) and os.listdir(src):
        try:
            shutil.copytree(src, enc, dirs_exist_ok=True)
            _emit(callback, f"[WFL] seeded offline Whisper encoder from {src}")
            return True
        except Exception as exc:
            _emit(callback, f"[WFL] encoder seed failed ({exc}); will attempt online download")
    return os.path.isdir(enc) and bool(os.listdir(enc))


def _import_textgrid_module():
    try:
        import textgrid  # type: ignore

        return textgrid, ""
    except Exception as exc:
        return None, str(exc)


def _list_top_level_wavs(wav_folder: str) -> List[str]:
    root = os.path.abspath(str(wav_folder or "").strip())
    if not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if str(name).lower().endswith(".wav")
    ]


def _wav_duration_sec(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            frames = int(wf.getnframes() or 0)
            rate = int(wf.getframerate() or 0)
        if rate > 0:
            return float(frames) / float(rate)
    except Exception:
        return 0.0
    return 0.0


def _load_wfl_vocab(phonemes_path: str) -> set:
    """Load the model's phoneme inventory from phonemes.txt (BIO labels)."""
    vocab: set = set()
    try:
        with open(phonemes_path, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.strip()
                if not tok or tok == "O":
                    continue
                if tok.startswith("B-") or tok.startswith("I-"):
                    tok = tok[2:]
                vocab.add(tok)
    except Exception:
        return set()
    return vocab


def _load_dictionary(dict_path: str) -> Dict[str, List[str]]:
    """MFA-style dictionary: `<token>\\t<p1> <p2> ...` per line."""
    out: Dict[str, List[str]] = {}
    if not dict_path or not os.path.isfile(dict_path):
        return out
    try:
        with open(dict_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                row = line.rstrip("\n")
                if not row.strip() or row.lstrip().startswith("#"):
                    continue
                # token is first whitespace-delimited field; rest are phones.
                parts = row.split("\t", 1)
                if len(parts) == 2:
                    token, rest = parts[0].strip(), parts[1]
                else:
                    fields = row.split()
                    token, rest = fields[0], " ".join(fields[1:])
                phones = [p for p in rest.split() if p.strip()]
                if token and phones and token not in out:
                    out[token] = phones
    except Exception:
        return out
    return out


def _read_transcript_tokens(wav_path: str) -> List[str]:
    lab_path = os.path.splitext(os.path.abspath(wav_path))[0] + ".lab"
    if not os.path.isfile(lab_path):
        return []
    try:
        with open(lab_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
    except Exception:
        return []
    return [t for t in content.split() if t.strip()]


def _build_forced_sequence(
    tokens: Sequence[str],
    dict_map: Dict[str, List[str]],
    vocab: set,
) -> Tuple[List[str], List[Tuple[str, int]], List[str]]:
    """Return (flat_phones, groups[(token, phone_count)], missing_tokens)."""
    flat: List[str] = []
    groups: List[Tuple[str, int]] = []
    missing: List[str] = []
    for token in tokens:
        phones = dict_map.get(token)
        if not phones:
            missing.append(token)
            continue
        # Skip if any phone is outside the model vocabulary (would break
        # forced alignment).  Reported by the caller for diagnostics.
        if vocab and any(p not in vocab for p in phones):
            missing.append(token)
            continue
        flat.extend(phones)
        groups.append((token, len(phones)))
    return flat, groups, missing


# --------------------------------------------------------------------------- #
# Korean phonemizer (Hangul syllable -> WFL romanized phones)
#
# The project's MFA dictionary uses IPA (ɐ, ʌ, tɕ, k͈, …) which is NOT in the
# WFL KO vocabulary (romanized: a, eo, j, kk, …).  So for Korean we bypass the
# dictionary and decompose Hangul directly to WFL phones, position-aware
# (chosung=onset, jungsung=vowel, jongsung=coda).  Onsets are identity (every
# CHOSUNG roman is already in the WFL vocab); vowels split off the y/w glide;
# codas use Korean coda-neutralization to the WFL uppercase coda tokens.
# --------------------------------------------------------------------------- #
_KO_VOWEL_TO_WFL: Dict[str, List[str]] = {
    "a": ["a"], "ae": ["e"], "ya": ["y", "a"], "yae": ["y", "e"],
    "eo": ["eo"], "e": ["e"], "yeo": ["y", "eo"], "ye": ["y", "e"],
    "o": ["o"], "wa": ["w", "a"], "wae": ["w", "e"], "oe": ["w", "e"],
    "yo": ["y", "o"], "u": ["u"], "wo": ["w", "o"], "we": ["w", "e"],
    "wi": ["w", "i"], "yu": ["y", "u"], "eu": ["eu"], "ui": ["eu", "i"],
    "i": ["i"],
}
_KO_CODA_TO_WFL: Dict[str, List[str]] = {
    "k": ["K"], "kk": ["K"], "gs": ["K"], "lg": ["L", "K"],
    "n": ["N"], "nj": ["N"], "nh": ["N"],
    "d": ["T"], "s": ["T"], "ss": ["T"], "j": ["T"], "ch": ["T"], "t": ["T"], "h": ["T"],
    "l": ["L"], "lm": ["L", "M"], "lb": ["L", "P"], "ls": ["L"], "lt": ["L"], "lp": ["L", "P"], "lh": ["L"],
    "m": ["M"], "b": ["P"], "bs": ["P"], "p": ["P"], "ng": ["NG"],
}
# Standalone (non-Hangul) tokens that appear in KO reclists.
_KO_STANDALONE_TO_WFL: Dict[str, List[str]] = {
    "l": ["L"], "m": ["M"], "n": ["N"], "ng": ["NG"], "r": ["r"],
    "vin": ["vf"], "ving": ["vf"],
}


def _korean_token_to_wfl_phones(token: str) -> Tuple[List[str], bool]:
    """Return (wfl_phones, ok). ok=False if any part is unmappable."""
    from core.alignment.lab_generator import (
        CHOSUNG_LIST_ROMAN,
        JONGSUNG_LIST_ROMAN,
        JUNGSUNG_LIST_ROMAN,
    )

    text = str(token or "").strip()
    if not text:
        return [], False

    # Non-Hangul standalone token (coda/breath markers).
    if not any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in text):
        mapped = _KO_STANDALONE_TO_WFL.get(text.lower())
        return (list(mapped), True) if mapped else ([], False)

    phones: List[str] = []
    for ch in text:
        if not (0xAC00 <= ord(ch) <= 0xD7A3):
            mapped = _KO_STANDALONE_TO_WFL.get(ch.lower())
            if not mapped:
                return [], False
            phones.extend(mapped)
            continue
        code = ord(ch) - 0xAC00
        jong = code % 28
        jung = (code // 28) % 21
        cho = code // 588
        onset = CHOSUNG_LIST_ROMAN[cho]
        if onset:
            phones.append(onset)  # every chosung roman is in WFL vocab
        vowel_roman = JUNGSUNG_LIST_ROMAN[jung]
        vmapped = _KO_VOWEL_TO_WFL.get(vowel_roman)
        if not vmapped:
            return [], False
        phones.extend(vmapped)
        coda_roman = JONGSUNG_LIST_ROMAN[jong]
        if coda_roman:
            cmapped = _KO_CODA_TO_WFL.get(coda_roman)
            if not cmapped:
                return [], False
            phones.extend(cmapped)
    return phones, True


def _build_forced_sequence_korean(
    tokens: Sequence[str],
    vocab: set,
) -> Tuple[List[str], List[Tuple[str, int]], List[str]]:
    """KO path: Hangul syllable -> WFL phones (dictionary bypassed)."""
    flat: List[str] = []
    groups: List[Tuple[str, int]] = []
    missing: List[str] = []
    for token in tokens:
        phones, ok = _korean_token_to_wfl_phones(token)
        if not ok or not phones:
            missing.append(token)
            continue
        if vocab and any(p not in vocab for p in phones):
            missing.append(token)
            continue
        flat.extend(phones)
        groups.append((token, len(phones)))
    return flat, groups, missing


# --------------------------------------------------------------------------- #
# Japanese: the project MFA dictionary expands to IPA (ɯ, ɕ, tɕ, ɴ, bʲ, …),
# but the WFL JP vocab is romanized (u, sh, ch, N, by, …).  Map IPA -> WFL after
# dictionary expansion; tokens already in romaji pass through unchanged.
# --------------------------------------------------------------------------- #
_JA_IPA_TO_WFL: Dict[str, str] = {
    "ɯ": "u", "ɴ": "N", "ɾ": "r", "ɾʲ": "ry",
    "ɕ": "sh", "tɕ": "ch", "tɕʲ": "ch", "dʑ": "j", "dʑʲ": "j",
    "ç": "h", "ɸ": "f", "ɡ": "g", "ɡʲ": "gy", "ɲ": "ny",
    "bʲ": "by", "dʲ": "dy", "kʲ": "ky", "mʲ": "my", "pʲ": "py",
    "spn": "SP", "sil": "SP", "pau": "SP",
}


def _map_ja_phone(p: str) -> str:
    return _JA_IPA_TO_WFL.get(p, p)


def _build_forced_sequence_japanese(
    tokens: Sequence[str],
    dict_map: Dict[str, List[str]],
    vocab: set,
) -> Tuple[List[str], List[str], List[Tuple[str, int]], List[str]]:
    """JA path: dictionary expansion + IPA->WFL romaji mapping.

    Returns (wfl_phones, canonical_phones, groups, missing).  `wfl_phones` feed
    the WFL forced-align .txt; `canonical_phones` are the project's original dict
    phones (IPA) used to LABEL the output TextGrid so the downstream OTO mapper
    (built around MFA's IPA inventory) recognizes vowels/consonants instead of
    falling back to words-based synthesis.  They are 1:1 (one mapped phone per
    dict phone), so WFL boundaries carry over directly onto canonical labels.
    """
    flat: List[str] = []
    canonical: List[str] = []
    groups: List[Tuple[str, int]] = []
    missing: List[str] = []
    for token in tokens:
        phones = dict_map.get(token)
        if not phones:
            missing.append(token)
            continue
        mapped = [_map_ja_phone(p) for p in phones]
        if vocab and any(p not in vocab for p in mapped):
            missing.append(token)
            continue
        flat.extend(mapped)
        canonical.extend(phones)
        groups.append((token, len(mapped)))
    return flat, canonical, groups, missing


def _parse_wfl_lab(path: str) -> List[Tuple[float, float, str]]:
    """Parse HTK lab written by WFL (`<start_100ns> <end_100ns> <phone>`)."""
    out: List[Tuple[float, float, str]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    s = int(parts[0]) / 1e7
                    e = int(parts[1]) / 1e7
                except Exception:
                    continue
                out.append((s, e, parts[2]))
    except Exception:
        return []
    return out


def _align_output_to_sequence(
    segments: List[Tuple[float, float, str]],
    flat_phones: List[str],
) -> List[Tuple[float, float, str]]:
    """Trim WFL's auto-inserted leading/trailing SP so output lines up with the
    forced phone sequence.  Returns rows carrying the canonical phone labels."""
    segs = list(segments)
    # Drop a leading silence segment WFL prepends when audio starts after 0.
    while segs and flat_phones and segs[0][2] != flat_phones[0] and segs[0][2].strip().lower() in _SILENCE_LABELS:
        segs.pop(0)
    while segs and flat_phones and segs[-1][2] != flat_phones[-1] and segs[-1][2].strip().lower() in _SILENCE_LABELS:
        segs.pop()
    rows: List[Tuple[float, float, str]] = []
    n = min(len(segs), len(flat_phones))
    for i in range(n):
        s, e, _ = segs[i]
        rows.append((s, e, flat_phones[i]))
    return rows


def _regroup_words(
    phone_rows: List[Tuple[float, float, str]],
    groups: List[Tuple[str, int]],
) -> List[Tuple[float, float, str]]:
    """Merge the phone spans belonging to each original token into one interval."""
    words: List[Tuple[float, float, str]] = []
    idx = 0
    for token, count in groups:
        if idx >= len(phone_rows):
            break
        end_idx = min(len(phone_rows), idx + count)
        start = phone_rows[idx][0]
        end = phone_rows[end_idx - 1][1]
        words.append((start, end, token))
        idx = end_idx
    return words


def _write_textgrid(
    path: str,
    *,
    phone_rows: Sequence[Tuple[float, float, str]],
    word_rows: Sequence[Tuple[float, float, str]],
    duration_sec: float,
) -> None:
    textgrid_mod, err = _import_textgrid_module()
    if textgrid_mod is None:
        raise RuntimeError(f"python-textgrid import failed: {err}")

    max_t = max(float(duration_sec or 0.0), 0.0)
    if phone_rows:
        max_t = max(max_t, float(phone_rows[-1][1]))
    if word_rows:
        max_t = max(max_t, float(word_rows[-1][1]))

    tg = textgrid_mod.TextGrid(minTime=0.0, maxTime=max_t)

    phone_tier = textgrid_mod.IntervalTier(name="phones", minTime=0.0, maxTime=max_t)
    if not phone_rows and max_t > 0.0:
        phone_rows = [(0.0, max_t, "pau")]
    prev_end = 0.0
    for start, end, mark in phone_rows:
        s = max(prev_end, float(start))
        e = max(s, float(end))
        phone_tier.add(s, e, str(mark or "").strip())
        prev_end = e
    tg.append(phone_tier)

    word_tier = textgrid_mod.IntervalTier(name="words", minTime=0.0, maxTime=max_t)
    if not word_rows and max_t > 0.0:
        word_rows = [(0.0, max_t, "spn")]
    prev_end = 0.0
    for start, end, mark in word_rows:
        s = max(prev_end, float(start))
        e = max(s, float(end))
        word_tier.add(s, e, str(mark or "").strip())
        prev_end = e
    tg.append(word_tier)

    meta_max_t = max(float(max_t), 0.001)
    meta_tier = textgrid_mod.IntervalTier(name="utoa_meta", minTime=0.0, maxTime=meta_max_t)
    meta_tier.add(0.0, meta_max_t, "aligner=wfl")
    tg.append(meta_tier)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tg.write(path)


# --------------------------------------------------------------------------- #
# Public engine API
# --------------------------------------------------------------------------- #
def check_wfl_ready(
    *,
    language: str = "korean",
    wav_folder: str = "",
    callback=None,
) -> Dict[str, object]:
    lang = str(language or "").strip().lower() or "korean"

    python_exe = _resolve_wfl_python()
    if not python_exe or not os.path.isfile(python_exe):
        return make_runtime_report(
            "align", ALIGN_NOT_READY,
            "WFL python executable not found (set UTOA_WFL_PYTHON).",
            engine="wfl", language=lang, ready=False,
        )

    repo = _resolve_wfl_repo()
    if not repo or not os.path.isfile(os.path.join(repo, "infer.py")):
        return make_runtime_report(
            "align", ALIGN_EXEC_MISSING,
            "WFL repo (infer.py) not found (set UTOA_WFL_REPO).",
            engine="wfl", language=lang, ready=False,
        )

    model_dir = _resolve_wfl_model_dir(lang)
    files = _model_files(model_dir) if model_dir else {}
    if not model_dir or not os.path.isfile(files.get("ckpt", "")) or not os.path.isfile(files.get("config", "")):
        return make_runtime_report(
            "align", ALIGN_MODEL_MISSING,
            f"WFL model for '{lang}' not found (set UTOA_WFL_MODEL_DIR_{lang.upper()}).",
            engine="wfl", language=lang, ready=False,
        )
    if not os.path.isfile(files.get("phonemes", "")):
        return make_runtime_report(
            "align", ALIGN_MODEL_MISSING,
            f"WFL phonemes.txt missing in model dir: {model_dir}",
            engine="wfl", language=lang, ready=False,
        )

    textgrid_mod, textgrid_err = _import_textgrid_module()
    if textgrid_mod is None:
        return make_runtime_report(
            "align", ALIGN_NOT_READY,
            f"python-textgrid import failed: {textgrid_err}",
            engine="wfl", language=lang, ready=False,
        )

    if wav_folder and not os.path.isdir(wav_folder):
        return make_runtime_report(
            "align", ALIGN_NOT_READY, f"WAV folder not found: {wav_folder}",
            engine="wfl", language=lang, ready=False,
        )
    wav_files = _list_top_level_wavs(wav_folder)
    if wav_folder and not wav_files:
        return make_runtime_report(
            "align", ALIGN_NOT_READY, f"No WAV files found: {wav_folder}",
            engine="wfl", language=lang, ready=False,
        )

    _emit(callback, f"[WFL] ready: model={model_dir} wav_files={len(wav_files)}")
    return make_runtime_report(
        "align", OK, "WFL alignment ready",
        engine="wfl", language=lang, ready=True, wav_count=int(len(wav_files)),
    )


def run_wfl_align(
    _wfl_path: str,
    wav_folder: str,
    dictionary_path: str,
    output_folder: str,
    *,
    language: str = "korean",
    format_hint: str = "",
    callback=None,
) -> Tuple[bool, str]:
    _set_last_meta({})
    lang = str(language or "").strip().lower() or "korean"

    ready = check_wfl_ready(language=lang, wav_folder=wav_folder, callback=callback)
    if str(ready.get("code", "")).upper() != OK:
        return False, str(ready.get("message", "wfl aligner not ready") or "wfl aligner not ready")

    python_exe = _resolve_wfl_python()
    repo = _resolve_wfl_repo()
    model_dir = _resolve_wfl_model_dir(lang)
    files = _model_files(model_dir)
    lang_id = _resolve_wfl_lang_id(lang)

    vocab = _load_wfl_vocab(files["phonemes"])
    dict_map = _load_dictionary(dictionary_path)
    if not dict_map and lang != "korean":
        # Korean bypasses the IPA dictionary (uses the Hangul->WFL phonemizer).
        return False, f"Alignment dictionary empty or unreadable: {dictionary_path}"

    wav_files = _list_top_level_wavs(wav_folder)
    if not wav_files:
        return False, f"No WAV files found: {wav_folder}"

    work_dir = tempfile.mkdtemp(prefix="utoa_wfl_")
    staged: Dict[str, Dict[str, object]] = {}  # base -> {wav, flat, groups}
    skipped: List[str] = []
    try:
        for wav_path in wav_files:
            base = os.path.splitext(os.path.basename(wav_path))[0]
            tokens = _read_transcript_tokens(wav_path)
            if not tokens:
                skipped.append(base)
                continue
            if lang == "korean":
                flat, groups, missing = _build_forced_sequence_korean(tokens, vocab)
                canonical = list(flat)
            elif lang == "japanese":
                flat, canonical, groups, missing = _build_forced_sequence_japanese(tokens, dict_map, vocab)
            else:
                flat, groups, missing = _build_forced_sequence(tokens, dict_map, vocab)
                canonical = list(flat)
            if not flat or missing:
                skipped.append(base)
                if missing:
                    _emit(callback, f"[WFL] skip {base}: tokens not in dict/vocab: {missing[:6]}")
                continue
            # Pad leading/trailing silence so WFL forced-align locks the first/last
            # REAL phone to its true acoustic onset instead of absorbing the
            # leading breath/silence into it.  Without this, the first phone gets
            # stretched back to t=0 (observed ~500ms early vs MFA).  Requires SP in
            # the model vocab (true for the JP/KO models).
            if "SP" in vocab:
                flat = ["SP"] + flat + ["SP"]
                canonical = ["sil"] + canonical + ["sil"]
                groups = [("sil", 1)] + groups + [("sil", 1)]
            staged_wav = os.path.join(work_dir, base + ".wav")
            try:
                os.link(wav_path, staged_wav)
            except Exception:
                shutil.copy2(wav_path, staged_wav)
            with open(os.path.join(work_dir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write(" ".join(flat))
            staged[base] = {"wav": wav_path, "flat": flat, "canonical": canonical, "groups": groups}

        if not staged:
            return False, "No WAV had a usable transcript/dictionary phone sequence (vocab mismatch?)."

        _ensure_infer_compat(repo, callback=callback)
        if not _ensure_encoder(repo, callback=callback):
            _emit(callback, "[WFL] Whisper encoder not bundled; first run will download it (needs internet once).")
        cmd = [
            python_exe, "infer.py",
            "-i", work_dir,
            "-ckpt", files["ckpt"],
            "-c", files["config"],
            "-l", str(lang_id),
        ]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        _emit(callback, f"[WFL] infer start: staged={len(staged)} skipped={len(skipped)} lang_id={lang_id}")
        try:
            proc = subprocess.run(
                cmd, cwd=repo, env=env, capture_output=True, text=False,
                timeout=_infer_timeout_sec(),
            )
        except subprocess.TimeoutExpired:
            return False, "WFL inference timed out (raise UTOA_WFL_INFER_TIMEOUT_SEC)."
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")[-800:]
            return False, f"WFL inference failed (code {proc.returncode}): {tail}"

        written = 0
        low_coverage = 0
        for base, info in staged.items():
            lab_out = os.path.join(work_dir, base + ".lab")
            if not os.path.isfile(lab_out):
                _emit(callback, f"[WFL] no output lab for {base}")
                continue
            segments = _parse_wfl_lab(lab_out)
            flat = list(info["flat"])  # type: ignore[arg-type]
            canonical = list(info["canonical"])  # type: ignore[arg-type]
            groups = list(info["groups"])  # type: ignore[arg-type]
            phone_rows = _align_output_to_sequence(segments, flat)
            if len(phone_rows) < len(flat):
                low_coverage += 1
            # Relabel phone rows with the project's canonical phones (IPA) at the
            # WFL-predicted boundaries, so the OTO mapper recognizes them.
            phone_rows = [
                (s, e, canonical[i] if i < len(canonical) else mark)
                for i, (s, e, mark) in enumerate(phone_rows)
            ]
            word_rows = _regroup_words(phone_rows, groups)
            duration = _wav_duration_sec(str(info["wav"]))
            tg_path = os.path.join(output_folder, base + ".TextGrid")
            try:
                _write_textgrid(tg_path, phone_rows=phone_rows, word_rows=word_rows, duration_sec=duration)
                written += 1
            except Exception as exc:
                _emit(callback, f"[WFL] TextGrid write failed for {base}: {exc}")

        if written <= 0:
            return False, "WFL produced no TextGrid output."

        coverage = float(written) / float(max(1, len(wav_files)))
        confidence = max(0.0, min(1.0, 0.9 * coverage))
        warnings: List[str] = []
        if skipped:
            warnings.append(f"skipped_wavs:{len(skipped)}")
        if low_coverage:
            warnings.append(f"partial_phone_coverage:{low_coverage}")
        _set_last_meta({
            "confidence": confidence,
            "warnings": warnings,
            "fallback_hint": "mfa" if skipped else "",
            "written": written,
            "skipped": len(skipped),
        })
        _emit(callback, f"[WFL] done: textgrids={written} skipped={len(skipped)} conf={confidence:.2f}")
        return True, ""
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


__all__ = [
    "check_wfl_ready",
    "run_wfl_align",
    "get_last_wfl_align_meta",
]

from __future__ import annotations

import audioop
import contextlib
import math
import os
import re
import urllib.request
import wave
from typing import Callable


FELINE_BASE_URL = "https://utau.felinewasteland.com"
_SCRIPT_TIMEOUT_SEC = 30
_DEFAULT_LIST_ONLY_TIMING = (0.0, 120.0, -240.0, 60.0, 20.0)


def _log(callback: Callable[[str], None] | None, msg: str) -> None:
    if callable(callback):
        callback(msg)


def _normalize_pack(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"full", "fullga", "full_ga"}:
        return "full_ga"
    return "lite"


def _normalize_beat(value: str) -> str:
    text = str(value or "").strip().lower()
    return "4" if text.startswith("4") else "8"


def _normalize_preset(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"core_plus", "core+", "plus", "recommended"}:
        return "core_plus"
    if text in {"all", "all_with_alt", "all+alt", "full"}:
        return "all"
    return "core"


def _normalize_alias_suffix(suffix: str) -> str:
    val = str(suffix or "").strip()
    if not val:
        return ""
    return val[1:] if val.startswith("_") else val


def _apply_suffix_to_oto_line(line: str, suffix: str) -> str:
    suf = _normalize_alias_suffix(suffix)
    if not suf:
        return line
    if not line or "=" not in line:
        return line
    left, right = line.split("=", 1)
    if "," in right:
        alias, rest = right.split(",", 1)
        alias = alias.strip()
        if alias:
            alias = f"{alias}_{suf}"
        return f"{left}={alias},{rest}"
    alias = right.strip()
    if alias:
        alias = f"{alias}_{suf}"
    return f"{left}={alias}"


def _build_catalog(pack: str, beat: str) -> dict[str, tuple[str, str]]:
    if pack == "full_ga":
        if beat == "4":
            return {
                "core": ("/en/reclist/fullGa4Core.js", "FULL_4_core_oto"),
                "cluster": ("/en/reclist/fullGa4ClBr.js", "FULL_4_cluster_oto"),
                "breath": ("/en/reclist/fullGa4ClBr.js", "FULL_4_breath_oto"),
                "cvc": ("/en/reclist/fullGa4CVC.js", "FULL_4_cvc_oto"),
                "vv": ("/en/reclist/fullGa4VVCC.js", "FULL_4_vv_oto"),
                "cc": ("/en/reclist/fullGa4VVCC.js", "FULL_4_cc_oto"),
                "alt": ("/en/reclist/fullGa4Alt.js", "FULL_4_alt_oto"),
            }
        return {
            "core": ("/en/reclist/fullGa8Core.js", "FULL_8_core_oto"),
            "cluster": ("/en/reclist/fullGa8ClBr.js", "FULL_8_cluster_oto"),
            "breath": ("/en/reclist/fullGa8ClBr.js", "FULL_8_breath_oto"),
            "cvc": ("/en/reclist/fullGa8CVC.js", "FULL_8_cvc_oto"),
            "vv": ("/en/reclist/fullGa8VVCC.js", "FULL_8_vv_oto"),
            "cc": ("/en/reclist/fullGa8VVCC.js", "FULL_8_cc_oto"),
            "alt": ("/en/reclist/fullGa8Alt.js", "FULL_8_alt_oto"),
        }

    if beat == "4":
        return {
            "core": ("/en/reclist/lite4Core.js", "LITE_4_core_oto"),
            "cluster": ("/en/reclist/lite4ClBrCVC.js", "LITE_4_cluster_oto"),
            "breath": ("/en/reclist/lite4ClBrCVC.js", "LITE_4_breath_oto"),
            "cvc": ("/en/reclist/lite4ClBrCVC.js", "LITE_4_cvc_oto"),
            "vv": ("/en/reclist/lite4VVCC.js", "LITE_4_vv_oto"),
            "cc": ("/en/reclist/lite4VVCC.js", "LITE_4_cc_oto"),
            "alt": ("/en/reclist/lite4Alt.js", "LITE_4_alt_oto"),
        }
    return {
        "core": ("/en/reclist/lite8Core.js", "LITE_8_core_oto"),
        "cluster": ("/en/reclist/lite8ClBrCVC.js", "LITE_8_cluster_oto"),
        "breath": ("/en/reclist/lite8ClBrCVC.js", "LITE_8_breath_oto"),
        "cvc": ("/en/reclist/lite8ClBrCVC.js", "LITE_8_cvc_oto"),
        "vv": ("/en/reclist/lite8VVCC.js", "LITE_8_vv_oto"),
        "cc": ("/en/reclist/lite8VVCC.js", "LITE_8_cc_oto"),
        "alt": ("/en/reclist/lite8Alt.js", "LITE_8_alt_oto"),
    }


def _sections_for_preset(preset: str) -> list[str]:
    if preset == "core_plus":
        return ["core", "cluster", "breath", "cvc", "vv", "cc"]
    if preset == "all":
        return ["core", "cluster", "breath", "cvc", "vv", "cc", "alt"]
    return ["core"]


def _download_text(path: str) -> str:
    url = f"{FELINE_BASE_URL}{path}"
    with urllib.request.urlopen(url, timeout=_SCRIPT_TIMEOUT_SEC) as resp:
        body = resp.read()
    return body.decode("utf-8", errors="replace")


def _extract_js_block(js_text: str, var_name: str) -> str:
    pattern = rf"{re.escape(var_name)}\s*=\s*`(.*?)`;"
    m = re.search(pattern, js_text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1)


def _companion_list_var(var_name: str) -> str:
    name = str(var_name or "").strip()
    if name.endswith("_oto"):
        return name[: -len("_oto")]
    return ""


def _split_oto_lines(raw_block: str) -> list[str]:
    text = str(raw_block or "")
    text = text.replace("<br/>", "\n").replace("<br>", "\n")
    rows = []
    for line in text.splitlines():
        row = line.strip()
        if not row:
            continue
        if "=" not in row:
            continue
        rows.append(row)
    return rows


def _split_list_rows(raw_block: str) -> list[str]:
    text = str(raw_block or "")
    text = text.replace("<br/>", "\n").replace("<br>", "\n")
    rows: list[str] = []
    for line in text.splitlines():
        row = line.strip()
        if not row:
            continue
        rows.append(row)
    return rows


def _fmt_num(value: float) -> str:
    try:
        f = float(value)
    except Exception:
        return "0"
    i = int(f)
    if abs(f - float(i)) <= 1e-9:
        return str(i)
    return f"{f:.3f}".rstrip("0").rstrip(".")


def _aliases_from_list_row(row: str) -> list[str]:
    token = str(row or "").strip()
    if not token:
        return []
    parts = [p.strip() for p in token.split("__") if p.strip()]
    if not parts:
        parts = [token]

    aliases: list[str] = []
    seen: set[str] = set()
    for part in parts:
        alias = re.sub(r"\s+", " ", part.replace("-", " ").replace("_", " ").strip())
        if not alias:
            alias = part
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        aliases.append(alias)
    return aliases


def _synthesize_oto_from_list_rows(
    rows: list[str],
    timing: tuple[float, float, float, float, float] | None = None,
) -> list[str]:
    offset, consonant, cutoff, preutter, overlap = timing or _DEFAULT_LIST_ONLY_TIMING
    p_offset = _fmt_num(offset)
    p_consonant = _fmt_num(consonant)
    p_cutoff = _fmt_num(cutoff)
    p_preutter = _fmt_num(preutter)
    p_overlap = _fmt_num(overlap)

    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        stem = str(raw or "").strip()
        if not stem:
            continue
        wav_name = stem if stem.lower().endswith(".wav") else f"{stem}.wav"
        aliases = _aliases_from_list_row(stem)
        if not aliases:
            aliases = [stem]
        for alias in aliases:
            key = (wav_name.lower(), alias.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                f"{wav_name}={alias},{p_offset},{p_consonant},{p_cutoff},{p_preutter},{p_overlap}"
            )
    return out


def _build_wav_lookup(wav_dir: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not os.path.isdir(wav_dir):
        return lookup
    for name in os.listdir(wav_dir):
        if not str(name).lower().endswith(".wav"):
            continue
        key = str(name).lower()
        lookup[key] = name
    return lookup


def _stem_without_ext(name: str) -> str:
    text = str(name or "").strip()
    if text.lower().endswith(".wav"):
        return text[:-4]
    return text


def _normalize_stem_for_compound(stem: str) -> str:
    text = _stem_without_ext(stem).strip().strip("_")
    if not text:
        return ""
    if text.startswith("_+_"):
        text = text[3:]
    elif text.startswith("___"):
        text = text[3:]
    elif text.startswith("+__"):
        text = text[3:]
    if "__" in text:
        left, right = text.split("__", 1)
        if left and right:
            text = right
    return text.strip("_").lower()


def _contains_token_with_boundaries(haystack: str, needle: str) -> bool:
    hay = str(haystack or "")
    ned = str(needle or "")
    if not hay or not ned:
        return False
    if hay == ned:
        return True
    start = 0
    n_len = len(ned)
    while True:
        idx = hay.find(ned, start)
        if idx < 0:
            return False
        left_ok = idx == 0 or hay[idx - 1] in {"_", "-"}
        end = idx + n_len
        right_ok = end == len(hay) or hay[end] in {"_", "-"}
        if left_ok and right_ok:
            return True
        start = idx + 1


def _split_compound_stem(compound: str, known_tokens: set[str]) -> list[str]:
    target = str(compound or "").strip().lower()
    if not target:
        return []
    if target in known_tokens:
        return [target]

    token_by_head: dict[str, list[str]] = {}
    for token in known_tokens:
        if not token:
            continue
        token_by_head.setdefault(token[0], []).append(token)
    for head in token_by_head:
        token_by_head[head].sort(key=len, reverse=True)

    memo: dict[int, list[str] | None] = {}
    n = len(target)

    def _best(pos: int) -> list[str] | None:
        if pos in memo:
            return memo[pos]
        if pos >= n:
            return []
        head = target[pos]
        candidates = token_by_head.get(head, [])
        best_seq: list[str] | None = None
        for tok in candidates:
            if not target.startswith(tok, pos):
                continue
            end = pos + len(tok)
            if end < n and target[end] != "_":
                continue
            next_pos = end + 1 if end < n else end
            rest = _best(next_pos)
            if rest is None:
                continue
            seq = [tok] + rest
            if best_seq is None:
                best_seq = seq
                continue
            # 합본 녹음 대응을 위해 가능한 경우 더 많은 분절을 우선합니다.
            if len(seq) > len(best_seq):
                best_seq = seq
            elif len(seq) == len(best_seq):
                curr_len = sum(len(x) for x in seq)
                best_len = sum(len(x) for x in best_seq)
                if curr_len > best_len:
                    best_seq = seq
        memo[pos] = best_seq
        return best_seq

    out = _best(0)
    return out or []


def _build_compound_match_index(
    wav_lookup: dict[str, str],
    known_stems: set[str],
) -> tuple[dict[str, list[tuple[str, int, int]]], dict[str, str]]:
    token_hits: dict[str, list[tuple[str, int, int]]] = {}
    direct_norm_map: dict[str, str] = {}
    for _wav_key, actual in wav_lookup.items():
        base = _stem_without_ext(actual)
        norm = _normalize_stem_for_compound(base)
        if not norm:
            continue
        if norm in known_stems:
            direct_norm_map.setdefault(norm, actual)
        seq = _split_compound_stem(norm, known_stems)
        if len(seq) < 2:
            continue
        total = len(seq)
        for idx, token in enumerate(seq):
            token_hits.setdefault(token, []).append((actual, idx, total))
    return token_hits, direct_norm_map


def _parse_oto_line_with_params(row: str):
    text = str(row or "").strip()
    if "=" not in text:
        return None
    left, right = text.split("=", 1)
    if "," not in right:
        return None
    alias, rest = right.split(",", 1)
    parts = [p.strip() for p in rest.split(",")]
    if len(parts) < 5:
        return None
    try:
        offset = float(parts[0])
        consonant = float(parts[1])
        cutoff = float(parts[2])
        preutter = float(parts[3])
        overlap = float(parts[4])
    except Exception:
        return None
    return {
        "wav": left.strip(),
        "alias": alias.strip(),
        "params": [offset, consonant, cutoff, preutter, overlap],
        "tail_parts": parts[5:],
    }


def _build_oto_line_from_params(parsed) -> str:
    wav_name = parsed["wav"]
    alias = parsed["alias"]
    params = [_fmt_num(v) for v in parsed["params"]]
    payload = ",".join(params + list(parsed.get("tail_parts") or []))
    return f"{wav_name}={alias},{payload}"


def _estimate_segment_boundaries_ms(wav_path: str, segment_count: int) -> list[float]:
    seg = int(segment_count or 0)
    if seg <= 1:
        return [0.0, 0.0]
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            channels = int(wf.getnchannels() or 1)
            sampwidth = int(wf.getsampwidth() or 2)
            framerate = int(wf.getframerate() or 44100)
            nframes = int(wf.getnframes() or 0)
            if nframes <= 0 or framerate <= 0:
                raise ValueError("invalid wav header")
            raw = wf.readframes(nframes)
    except Exception:
        return [0.0] * (seg + 1)

    try:
        frame_bytes = channels * sampwidth
        window_frames = max(1, int(framerate * 0.01))
        step_frames = window_frames
        total_windows = int(math.ceil(float(nframes) / float(step_frames)))
        energy: list[float] = []
        for wi in range(total_windows):
            start_f = wi * step_frames
            end_f = min(nframes, start_f + window_frames)
            if end_f <= start_f:
                energy.append(0.0)
                continue
            s = start_f * frame_bytes
            e = end_f * frame_bytes
            chunk = raw[s:e]
            if channels > 1:
                chunk = audioop.tomono(chunk, sampwidth, 0.5, 0.5)
            rms = float(audioop.rms(chunk, sampwidth))
            energy.append(rms)
        if not energy:
            raise ValueError("no windows")

        smooth: list[float] = []
        radius = 2
        w_n = len(energy)
        for i in range(w_n):
            lo = max(0, i - radius)
            hi = min(w_n - 1, i + radius)
            smooth.append(sum(energy[lo : hi + 1]) / float(hi - lo + 1))

        duration_ms = float(nframes) * 1000.0 / float(framerate)
        boundaries_idx = [0]
        prev = 0
        for i in range(1, seg):
            target = int(round(float(i) * float(w_n) / float(seg)))
            search_radius = max(3, int(round(float(w_n) / float(seg * 3))))
            lo = max(prev + 1, target - search_radius)
            hi = min(w_n - 2, target + search_radius)
            if hi < lo:
                cut = max(prev + 1, min(w_n - 2, target))
            else:
                local = smooth[lo : hi + 1]
                min_local = min(local)
                min_pos = local.index(min_local)
                cut = lo + min_pos
            boundaries_idx.append(cut)
            prev = cut
        boundaries_idx.append(w_n - 1)

        out = [0.0]
        step_ms = float(step_frames) * 1000.0 / float(framerate)
        for i in range(1, len(boundaries_idx) - 1):
            out.append(max(0.0, min(duration_ms, float(boundaries_idx[i]) * step_ms)))
        out.append(duration_ms)
        if len(out) != seg + 1:
            raise ValueError("boundary count mismatch")
        for i in range(1, len(out)):
            if out[i] <= out[i - 1]:
                out[i] = min(duration_ms, out[i - 1] + 1.0)
        return out
    except Exception:
        try:
            with contextlib.closing(wave.open(wav_path, "rb")) as wf:
                framerate = int(wf.getframerate() or 44100)
                nframes = int(wf.getnframes() or 0)
            duration_ms = float(nframes) * 1000.0 / float(max(1, framerate))
        except Exception:
            duration_ms = 0.0
        if duration_ms <= 0.0:
            return [0.0] * (seg + 1)
        return [duration_ms * float(i) / float(seg) for i in range(seg + 1)]


def _apply_segment_timing_adjustment(
    row: str,
    wav_path: str,
    segment_idx: int,
    segment_count: int,
    boundary_cache: dict[tuple[str, int], list[float]],
) -> str:
    if segment_count <= 1:
        return row
    parsed = _parse_oto_line_with_params(row)
    if not parsed:
        return row
    key = (wav_path, int(segment_count))
    boundaries = boundary_cache.get(key)
    if boundaries is None:
        boundaries = _estimate_segment_boundaries_ms(wav_path, segment_count)
        boundary_cache[key] = boundaries
    if not boundaries or len(boundaries) < segment_count + 1:
        return row
    i = max(0, min(int(segment_idx), segment_count - 1))
    seg_start = float(boundaries[i])
    seg_end = float(boundaries[i + 1])
    wav_end = float(boundaries[-1])
    if seg_end <= seg_start + 2.0 or wav_end <= 0.0:
        return row

    offset, consonant, cutoff, preutter, overlap = parsed["params"]
    new_offset = seg_start + float(offset)
    max_offset = max(seg_start, seg_end - 2.0)
    new_offset = max(seg_start, min(max_offset, new_offset))

    # consonant/preutter/overlap는 상대 길이 성격을 가지므로 유지합니다.
    # cutoff는 파일 끝 상대값(음수)이 일반적이므로 세그먼트 끝 기준으로 보정합니다.
    new_cutoff = float(cutoff)
    if float(cutoff) < 0.0:
        right_blank = abs(float(cutoff))
        target_cut_ms = max(seg_start + 4.0, seg_end - right_blank)
        new_cutoff = target_cut_ms - wav_end
        if new_cutoff >= -1.0:
            new_cutoff = -1.0

    parsed["params"] = [new_offset, consonant, new_cutoff, preutter, overlap]
    return _build_oto_line_from_params(parsed)


def generate_en_cvvc_oto(
    *,
    wav_dir: str,
    out_path: str,
    pack: str = "lite",
    beat: str = "8",
    preset: str = "core",
    alias_suffix: str = "",
    include_list_only_synthesis: bool = False,
    list_only_timing: tuple[float, float, float, float, float] | None = None,
    callback: Callable[[str], None] | None = None,
) -> tuple[int, int, list[str]]:
    normalized_pack = _normalize_pack(pack)
    normalized_beat = _normalize_beat(beat)
    normalized_preset = _normalize_preset(preset)
    sections = _sections_for_preset(normalized_preset)
    catalog = _build_catalog(normalized_pack, normalized_beat)

    _log(
        callback,
        f"[EN-CVVC] source=FelineWasteland pack={normalized_pack} beat={normalized_beat} preset={normalized_preset}",
    )

    wav_lookup = _build_wav_lookup(wav_dir)
    if not wav_lookup:
        return 0, 0, [f"WAV not found in folder: {wav_dir}"]

    script_cache: dict[str, str] = {}
    collected: list[str] = []
    errors: list[str] = []

    for section in sections:
        script_path, var_name = catalog.get(section, ("", ""))
        if not script_path or not var_name:
            errors.append(f"unsupported section: {section}")
            continue
        try:
            if script_path not in script_cache:
                script_cache[script_path] = _download_text(script_path)
            block = _extract_js_block(script_cache[script_path], var_name)
            if not block:
                list_var = _companion_list_var(var_name)
                if list_var:
                    list_block = _extract_js_block(script_cache[script_path], list_var)
                    if list_block:
                        list_rows = _split_list_rows(list_block)
                        if include_list_only_synthesis:
                            synth_lines = _synthesize_oto_from_list_rows(
                                list_rows,
                                timing=list_only_timing,
                            )
                            collected.extend(synth_lines)
                            _log(
                                callback,
                                f"[EN-CVVC] section={section} list-only synth={len(synth_lines)} rows ({list_var})",
                            )
                        else:
                            _log(
                                callback,
                                f"[EN-CVVC] section={section} list-only data detected ({list_var}); skipped in preview mode",
                            )
                        continue
                errors.append(f"missing var in script: {var_name} ({script_path})")
                continue
            lines = _split_oto_lines(block)
            if not lines:
                if include_list_only_synthesis:
                    list_var = _companion_list_var(var_name)
                    if list_var:
                        list_rows = _split_list_rows(_extract_js_block(script_cache[script_path], list_var))
                        synth_lines = _synthesize_oto_from_list_rows(
                            list_rows,
                            timing=list_only_timing,
                        )
                        if synth_lines:
                            collected.extend(synth_lines)
                            _log(
                                callback,
                                f"[EN-CVVC] section={section} list-only synth={len(synth_lines)} rows ({list_var})",
                            )
                            continue
                _log(callback, f"[EN-CVVC] section={section} has no direct OTO rows; skipped")
                continue
            collected.extend(lines)
            _log(callback, f"[EN-CVVC] section={section} lines={len(lines)}")
        except Exception as e:
            errors.append(f"download/parse failed ({section}): {e}")

    total = len(collected)
    if total <= 0:
        if not errors:
            errors.append("No OTO rows collected from source scripts.")
        return 0, total, errors

    expected_stems: set[str] = set()
    for row in collected:
        if "=" not in row:
            continue
        left, _right = row.split("=", 1)
        stem = left.strip()
        expected_stems.add(_normalize_stem_for_compound(stem))
    expected_stems.discard("")
    compound_hits, direct_norm_map = _build_compound_match_index(wav_lookup, expected_stems)
    wav_norm_items: list[tuple[str, str]] = []
    for _k, actual_name in wav_lookup.items():
        wav_norm_items.append((actual_name, _normalize_stem_for_compound(actual_name)))

    output_lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    missing_wavs: list[str] = []
    timing_boundary_cache: dict[tuple[str, int], list[float]] = {}
    exact_match_count = 0
    norm_match_count = 0
    compound_match_count = 0

    for row in collected:
        left, right = row.split("=", 1)
        wav_name_raw = left.strip()
        key = wav_name_raw.lower()
        actual_wav_name = wav_lookup.get(key, "")
        segment_ctx: tuple[int, int] | None = None
        if actual_wav_name:
            exact_match_count += 1
        else:
            norm = _normalize_stem_for_compound(wav_name_raw)
            if norm:
                cand = direct_norm_map.get(norm, "")
                if cand:
                    actual_wav_name = cand
                    norm_match_count += 1
                else:
                    hits = compound_hits.get(norm, [])
                    if hits:
                        ranked = sorted(hits, key=lambda x: (x[2], x[1], x[0].lower()))
                        actual_wav_name, seg_i, seg_n = ranked[0]
                        segment_ctx = (int(seg_i), int(seg_n))
                        compound_match_count += 1
                    else:
                        # 분절 해석이 안 되는 합본 파일명은 경계 부분문자열로 보수 매칭합니다.
                        boundary_candidates: list[tuple[int, str]] = []
                        for actual_name, actual_norm in wav_norm_items:
                            if not actual_norm:
                                continue
                            is_match = _contains_token_with_boundaries(actual_norm, norm)
                            if is_match:
                                boundary_candidates.append((len(actual_norm), actual_name))
                        if boundary_candidates:
                            boundary_candidates.sort(key=lambda x: (x[0], x[1].lower()))
                            actual_wav_name = boundary_candidates[0][1]
                            norm_match_count += 1
        if not actual_wav_name:
            missing_wavs.append(wav_name_raw)
            continue
        row_resolved = f"{actual_wav_name}={right.strip()}"
        if segment_ctx is not None:
            wav_path = os.path.join(wav_dir, actual_wav_name)
            row_resolved = _apply_segment_timing_adjustment(
                row_resolved,
                wav_path=wav_path,
                segment_idx=segment_ctx[0],
                segment_count=segment_ctx[1],
                boundary_cache=timing_boundary_cache,
            )
        row_resolved = _apply_suffix_to_oto_line(row_resolved, alias_suffix)

        alias_key = ""
        if "=" in row_resolved:
            _l, _r = row_resolved.split("=", 1)
            alias_key = _r.split(",", 1)[0].strip().lower()
        dedupe_key = (actual_wav_name.lower(), alias_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        output_lines.append(row_resolved)

    if missing_wavs:
        unique_missing = sorted(set(missing_wavs))
        preview = ", ".join(unique_missing[:6])
        if len(unique_missing) > 6:
            preview += ", ..."
        errors.append(
            f"missing wav file: {len(unique_missing)} entries "
            f"(e.g. {preview})"
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    processed = len(output_lines)
    _log(
        callback,
        f"[EN-CVVC] written={processed} skipped_missing={max(0, total - processed)} "
        f"(exact={exact_match_count}, norm={norm_match_count}, compound={compound_match_count})",
    )
    return processed, total, errors


__all__ = ["generate_en_cvvc_oto"]

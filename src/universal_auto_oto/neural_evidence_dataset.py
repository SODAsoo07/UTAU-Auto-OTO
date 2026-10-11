from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


NEURAL_LABEL_SCHEMA_VERSION = 1
PUBLICDATA_LABEL_SOURCE = "publicdata_verified_lab"
SUPPORTED_PUBLICDATA_LANGUAGES = {"ko": "_한국어", "ja": "_일본어"}
EVENT_TRACKS = (
    "silence_likelihood",
    "active_likelihood",
    "closure_likelihood",
    "burst_likelihood",
    "frication_likelihood",
    "consonant_likelihood",
    "vowel_onset_likelihood",
    "vowel_stable_likelihood",
    "periodic_likelihood",
    "aperiodic_likelihood",
    "breath_likelihood",
    "transition_likelihood",
    "boundary_likelihood",
)


class NeuralEvidenceDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class PhoneInterval:
    start_ms: float
    end_ms: float
    phone: str


@dataclass(frozen=True)
class PublicDataItem:
    language: str
    wav_path: Path
    lab_path: Path
    corpus_id: str
    speaker_id: str
    duration_ms: float
    intervals: tuple[PhoneInterval, ...]


_SILENCE = frozenset({"sil", "sp", "pau", "pause", "silence"})
_BREATH = frozenset({"br", "breath", "exh", "inh", "asp"})
_APERIODIC = frozenset({"ap", "noise", "cl", "closure"})
_VOWELS = frozenset(
    {
        "a", "e", "i", "o", "u", "ae", "eo", "eu", "oe", "wi", "we", "wa", "wo", "ya", "ye", "yo", "yu",
        "aa", "ii", "ee", "oo", "uu", "n", "m", "ng", "nn", "nng",
    }
)
_FRICATIVES = frozenset({"s", "sh", "ss", "h", "f", "z", "x", "ts", "tsh", "ty", "sy"})
_STOPS = frozenset({"p", "b", "t", "d", "k", "g", "q", "c", "ch", "j", "kk", "tt", "pp", "cc"})


def parse_htk_lab(path: Path) -> tuple[PhoneInterval, ...]:
    """Read the common HTK 100 ns three-column phone-label format strictly."""
    rows: list[PhoneInterval] = []
    previous_end = 0.0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="strict").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 3:
            raise NeuralEvidenceDatasetError(f"{path}:{line_number}: expected start end phone")
        try:
            start = int(parts[0]) / 10_000.0
            end = int(parts[1]) / 10_000.0
        except ValueError as error:
            raise NeuralEvidenceDatasetError(f"{path}:{line_number}: invalid HTK timestamp") from error
        phone = " ".join(parts[2:]).strip()
        if not phone or start < 0.0 or end <= start or start + 1e-6 < previous_end:
            raise NeuralEvidenceDatasetError(f"{path}:{line_number}: invalid or non-monotonic interval")
        rows.append(PhoneInterval(start_ms=start, end_ms=end, phone=phone))
        previous_end = end
    if not rows:
        raise NeuralEvidenceDatasetError(f"{path}: no phone intervals")
    return tuple(rows)


def phone_roles(phone: str) -> frozenset[str]:
    token = phone.strip().casefold().replace("_", "")
    token = re.sub(r"[0-9]+$", "", token)
    if token in _SILENCE:
        return frozenset({"silence_likelihood"})
    if token in _BREATH:
        return frozenset({"breath_likelihood", "aperiodic_likelihood"})
    if token in _APERIODIC:
        return frozenset({"aperiodic_likelihood", "closure_likelihood"})
    if token in _VOWELS or any(character in "aeiou" for character in token):
        return frozenset({"vowel_stable_likelihood", "periodic_likelihood"})
    if token in _FRICATIVES:
        return frozenset({"frication_likelihood", "consonant_likelihood", "aperiodic_likelihood"})
    if token in _STOPS:
        return frozenset({"closure_likelihood", "burst_likelihood", "consonant_likelihood"})
    return frozenset({"consonant_likelihood"})


def frame_targets(
    frame_times_ms: np.ndarray,
    intervals: Sequence[PhoneInterval],
    *,
    boundary_sigma_ms: float = 15.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create soft event targets and masks from checked phone intervals.

    Transition is intentionally masked: phone-only labels do not define a unique
    acoustic transition interval. All other targets retain an explicit mask so
    future corpus-specific exclusions do not become accidental negative labels.
    """
    times = np.asarray(frame_times_ms, dtype=np.float64)
    targets = {name: np.zeros(len(times), dtype=np.float32) for name in EVENT_TRACKS}
    masks = {name: np.ones(len(times), dtype=np.float32) for name in EVENT_TRACKS}
    masks["transition_likelihood"].fill(0.0)
    if not len(intervals):
        return targets, masks
    for interval in intervals:
        inside = (times >= interval.start_ms) & (times < interval.end_ms)
        roles = phone_roles(interval.phone)
        for role in roles:
            targets[role][inside] = 1.0
        if "silence_likelihood" not in roles:
            targets["active_likelihood"][inside] = 1.0
    for index, interval in enumerate(intervals):
        boundary = interval.start_ms
        if index == 0:
            continue
        heat = np.exp(-0.5 * ((times - boundary) / max(boundary_sigma_ms, 1.0)) ** 2).astype(np.float32)
        targets["boundary_likelihood"] = np.maximum(targets["boundary_likelihood"], heat)
        current_roles = phone_roles(interval.phone)
        previous_roles = phone_roles(intervals[index - 1].phone)
        if "vowel_stable_likelihood" in current_roles:
            targets["vowel_onset_likelihood"] = np.maximum(targets["vowel_onset_likelihood"], heat)
        if "burst_likelihood" in current_roles or "burst_likelihood" in previous_roles:
            targets["burst_likelihood"] = np.maximum(targets["burst_likelihood"], heat)
    return targets, masks


def build_publicdata_manifest(
    data_root: Path,
    output_csv: Path,
    output_json: Path,
    *,
    languages: Sequence[str] = ("ko", "ja"),
    hash_files: bool = False,
    max_recordings: int | None = None,
) -> dict[str, object]:
    if max_recordings is not None and max_recordings < 1:
        raise NeuralEvidenceDatasetError("max_recordings must be positive when supplied")
    selected = tuple(_validate_languages(languages))
    items: list[PublicDataItem] = []
    audit: dict[str, object] = {
        "schema_version": 1,
        "label_schema_version": NEURAL_LABEL_SCHEMA_VERSION,
        "label_source": PUBLICDATA_LABEL_SOURCE,
        "data_root": str(data_root.resolve()),
        "languages": list(selected),
        "hash_files": bool(hash_files),
        "selection_policy": {
            "exclude_incomplete_labels": True,
            "prefer_csd_edited_segment_pairs": True,
            "exclude_csd_unsegmented_wav_and_alternate_htk": True,
        },
        "unmatched_wav": [],
        "unmatched_lab": [],
        "ambiguous_pairs": [],
        "excluded_sources": [],
        "invalid_labs": [],
        "unknown_phone_tokens": Counter(),
        "event_frame_coverage": Counter(),
    }
    paired_source_count = 0
    for language in selected:
        language_root = data_root / SUPPORTED_PUBLICDATA_LANGUAGES[language]
        if not language_root.is_dir():
            raise NeuralEvidenceDatasetError(f"PublicData language directory is missing: {language_root}")
        pairs, pair_audit = _pair_language_files(language_root)
        paired_source_count += len(pairs)
        for key in ("unmatched_wav", "unmatched_lab", "ambiguous_pairs", "excluded_sources"):
            audit[key].extend(pair_audit[key])
        for wav_path, lab_path in pairs:
            if max_recordings is not None and len(items) >= max_recordings:
                break
            try:
                intervals = parse_htk_lab(lab_path)
                duration_ms = float(sf.info(wav_path).duration) * 1000.0
            except (NeuralEvidenceDatasetError, RuntimeError) as error:
                audit["invalid_labs"].append({"wav": str(wav_path), "lab": str(lab_path), "reason": str(error)})
                continue
            if intervals[-1].end_ms > duration_ms + 80.0:
                audit["invalid_labs"].append({"wav": str(wav_path), "lab": str(lab_path), "reason": "label_exceeds_wav_duration"})
                continue
            relative = wav_path.relative_to(language_root)
            item = PublicDataItem(
                language=language,
                wav_path=wav_path,
                lab_path=lab_path,
                corpus_id=_corpus_id(language, relative),
                speaker_id=_speaker_id(language, relative),
                duration_ms=duration_ms,
                intervals=intervals,
            )
            items.append(item)
            for interval in intervals:
                roles = phone_roles(interval.phone)
                if roles == frozenset({"consonant_likelihood"}) and not _looks_consonant(interval.phone):
                    audit["unknown_phone_tokens"][interval.phone] += 1
                for role in roles:
                    audit["event_frame_coverage"][role] += int(round((interval.end_ms - interval.start_ms) / 5.0))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest_csv(items, output_csv, hash_files=hash_files)
    result = {
        "schema_version": 1,
        "manifest_csv": str(output_csv.resolve()),
        "recording_count": len(items),
        "matched_pair_count": len(items),
        "source_matched_pair_count": paired_source_count,
        "excluded_count": len(audit["invalid_labs"]),
        "excluded_source_count": len(audit["excluded_sources"]),
        "unmatched_wav_count": len(audit["unmatched_wav"]),
        "unmatched_lab_count": len(audit["unmatched_lab"]),
        "ambiguous_pair_count": len(audit["ambiguous_pairs"]),
        "label_schema_version": NEURAL_LABEL_SCHEMA_VERSION,
        "languages": list(selected),
        "label_source": PUBLICDATA_LABEL_SOURCE,
        "max_recordings": max_recordings,
    }
    audit["unknown_phone_tokens"] = dict(sorted(audit["unknown_phone_tokens"].items()))
    audit["event_frame_coverage"] = dict(sorted(audit["event_frame_coverage"].items()))
    audit["summary"] = result
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "audit_json": str(output_json.resolve())}


def read_neural_manifest(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple({key: str(value) for key, value in row.items()} for row in csv.DictReader(handle))


def _pair_language_files(language_root: Path) -> tuple[list[tuple[Path, Path]], dict[str, list[object]]]:
    audit: dict[str, list[object]] = {
        "unmatched_wav": [],
        "unmatched_lab": [],
        "ambiguous_pairs": [],
        "excluded_sources": [],
    }
    wavs = _eligible_publicdata_files(language_root.rglob("*.wav"), language_root, "wav", audit)
    labs = _eligible_publicdata_files(language_root.rglob("*.lab"), language_root, "lab", audit)
    labs_by_key: dict[str, list[Path]] = defaultdict(list)
    labs_by_stem: dict[str, list[Path]] = defaultdict(list)
    for lab in labs:
        labs_by_key[_pair_key(lab, language_root)].append(lab)
        labs_by_stem[lab.stem.casefold()].append(lab)
    used_labs: set[Path] = set()
    pairs: list[tuple[Path, Path]] = []
    for wav in wavs:
        candidates = [candidate for candidate in labs_by_key.get(_pair_key(wav, language_root), ()) if candidate not in used_labs]
        if not candidates:
            candidates = [candidate for candidate in labs_by_stem.get(wav.stem.casefold(), ()) if candidate not in used_labs]
        if not candidates:
            audit["unmatched_wav"].append(str(wav))
            continue
        phone_candidates = [candidate for candidate in candidates if "phones" in {part.casefold() for part in candidate.parts}]
        if len(phone_candidates) == 1:
            candidates = phone_candidates
        if len(candidates) != 1:
            audit["ambiguous_pairs"].append({"wav": str(wav), "labs": [str(path) for path in candidates]})
            continue
        pairs.append((wav, candidates[0]))
        used_labs.add(candidates[0])
    audit["unmatched_lab"] = [str(path) for path in labs if path not in used_labs]
    return pairs, audit


def _eligible_publicdata_files(
    paths: Iterable[Path],
    language_root: Path,
    kind: str,
    audit: dict[str, list[object]],
) -> list[Path]:
    accepted: list[Path] = []
    for path in paths:
        reason = _publicdata_exclusion_reason(path, language_root)
        if reason is None:
            accepted.append(path)
        else:
            audit["excluded_sources"].append({"path": str(path), "kind": kind, "reason": reason})
    return sorted(accepted)


def _publicdata_exclusion_reason(path: Path, language_root: Path) -> str | None:
    relative = path.relative_to(language_root)
    parts = tuple(part.casefold() for part in relative.parts)
    if "incomplete" in parts:
        return "incomplete_label_tree"
    if "htklab_of_csd" in parts:
        return "csd_alternate_htk_tree"
    # CSD supplies reviewed, segmented WAV/LAB pairs under CSD_editedlabels.
    # Do not mix them with the unsegmented source WAVs and alternate HTK labels.
    if len(parts) >= 2 and parts[0:2] == ("csd", "csd"):
        return "csd_unsegmented_source_tree"
    return None


def _pair_key(path: Path, language_root: Path) -> str:
    ignored = {"wav", "wavs", "audio", "audios", "lab", "labs", "label", "labels", "htklab", "htklabs"}
    relative = path.relative_to(language_root)
    parts = [part.casefold() for part in relative.parent.parts if part.casefold() not in ignored]
    return "/".join((*parts, path.stem.casefold()))


def _write_manifest_csv(items: Iterable[PublicDataItem], output: Path, *, hash_files: bool) -> None:
    fields = (
        "language", "wav_path", "lab_path", "wav_sha256", "lab_sha256", "corpus_id", "speaker_id", "duration_ms",
        "label_source", "label_schema_version", "interval_count",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "language": item.language,
                    "wav_path": str(item.wav_path.resolve()),
                    "lab_path": str(item.lab_path.resolve()),
                    "wav_sha256": _sha256(item.wav_path) if hash_files else "",
                    "lab_sha256": _sha256(item.lab_path) if hash_files else "",
                    "corpus_id": item.corpus_id,
                    "speaker_id": item.speaker_id,
                    "duration_ms": f"{item.duration_ms:.6f}",
                    "label_source": PUBLICDATA_LABEL_SOURCE,
                    "label_schema_version": NEURAL_LABEL_SCHEMA_VERSION,
                    "interval_count": len(item.intervals),
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_languages(languages: Sequence[str]) -> list[str]:
    result = []
    for language in languages:
        value = str(language).strip().casefold()
        if value not in SUPPORTED_PUBLICDATA_LANGUAGES:
            raise NeuralEvidenceDatasetError(f"unsupported PublicData language: {language}")
        if value not in result:
            result.append(value)
    if not result:
        raise NeuralEvidenceDatasetError("at least one PublicData language is required")
    return result


def _corpus_id(language: str, relative: Path) -> str:
    return f"{language}:{relative.parts[0]}" if relative.parts else language


def _speaker_id(language: str, relative: Path) -> str:
    match = re.search(r"(?:^|[_-])(s\d{2,}|spk\d+|speaker\d+)(?:[_-]|$)", "/".join(relative.parts), flags=re.IGNORECASE)
    if match:
        return f"{language}:{match.group(1).casefold()}"
    return _corpus_id(language, relative)


def _looks_consonant(phone: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", phone.strip()))


__all__ = [
    "EVENT_TRACKS",
    "NEURAL_LABEL_SCHEMA_VERSION",
    "NeuralEvidenceDatasetError",
    "PhoneInterval",
    "PublicDataItem",
    "build_publicdata_manifest",
    "frame_targets",
    "parse_htk_lab",
    "phone_roles",
    "read_neural_manifest",
]

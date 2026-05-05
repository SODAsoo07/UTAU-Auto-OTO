from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class KrProfileSetupResult:
    profile: object
    profile_path: str


@dataclass
class KrTextGridPreparation:
    tg_folder: str
    tg_entries: list[dict]
    tg_exact_map: dict[str, dict]
    tg_norm_map: dict[str, list[dict]]
    normalize_key_fn: Callable[[str], str]
    log_fn: Callable[[str], None]

    def resolve_tg_info(self, fname: str) -> Optional[dict]:
        wav_name = os.path.basename((fname or "").strip())
        base_lower = os.path.splitext(wav_name)[0].lower()
        if base_lower in self.tg_exact_map:
            return self.tg_exact_map[base_lower]

        norm_name = self.normalize_key_fn(wav_name)
        candidates = self.tg_norm_map.get(norm_name, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            exact_name = wav_name.lower()
            same_name = [c for c in candidates if c["real_name"].lower() == exact_name]
            if len(same_name) == 1:
                return same_name[0]
            self.log_fn(
                f"TextGrid ambiguous by normalized key: {wav_name} "
                f"(norm={norm_name}, candidates={len(candidates)})"
            )
            return None
        return None

    def template_match_stats(self, lines) -> tuple[int, int, float]:
        file_names = set()
        for line in lines or []:
            if "=" not in line:
                continue
            file_names.add(line.split("=", 1)[0].strip())
        total = len(file_names)
        if total == 0:
            return 0, 0, 0.0
        matched = sum(1 for fname in file_names if self.resolve_tg_info(fname))
        return matched, total, matched / float(total)


def iter_textgrid_files(tg_folder: str):
    if not tg_folder or not os.path.exists(tg_folder):
        return
    for dirpath, _dirnames, filenames in os.walk(tg_folder):
        for f_name in filenames:
            if f_name.lower().endswith(".textgrid"):
                yield dirpath, f_name


def build_wav_index(wav_root: str, *, normalize_key_fn: Callable[[str], str]) -> dict[str, str]:
    wav_index: dict[str, str] = {}
    if not wav_root or not os.path.isdir(wav_root):
        return wav_index
    try:
        for dirpath, _dirnames, filenames in os.walk(wav_root):
            for fn in filenames:
                if fn.lower().endswith(".wav"):
                    wav_index.setdefault(normalize_key_fn(fn), os.path.join(dirpath, fn))
    except Exception:
        return {}
    return wav_index


def build_kr_textgrid_preparation(
    *,
    tg_folder: str,
    wav_root_for_signal: str,
    wav_index_for_signal: dict[str, str],
    normalize_key_fn: Callable[[str], str],
    find_wav_path_fn: Callable[[str, str, dict[str, str]], str],
    log_fn: Callable[[str], None],
) -> KrTextGridPreparation:
    tg_entries = []
    tg_exact_map: dict[str, dict] = {}
    tg_norm_map: dict[str, list[dict]] = {}
    if os.path.exists(tg_folder):
        for dirpath, f_name in iter_textgrid_files(tg_folder):
            base = os.path.splitext(f_name)[0]
            inferred_name = base + ".wav"
            wav_path = find_wav_path_fn(inferred_name, wav_root_for_signal, wav_index_for_signal)
            output_wav_name = os.path.basename(wav_path) if wav_path else inferred_name
            info = {
                "path": os.path.join(dirpath, f_name),
                "real_name": inferred_name,
                "output_name": output_wav_name,
                "base_lower": base.lower(),
                "norm_key": normalize_key_fn(f_name),
            }
            tg_entries.append(info)
            if info["base_lower"] not in tg_exact_map:
                tg_exact_map[info["base_lower"]] = info
            tg_norm_map.setdefault(info["norm_key"], []).append(info)
    return KrTextGridPreparation(
        tg_folder=tg_folder,
        tg_entries=tg_entries,
        tg_exact_map=tg_exact_map,
        tg_norm_map=tg_norm_map,
        normalize_key_fn=normalize_key_fn,
        log_fn=log_fn,
    )


def prepare_kr_profile_setup(
    *,
    fallback_format: str,
    auto_gen_format: str,
    kr_anchor_profile_path: str,
    log_fn: Callable[[str], None],
    default_cache_path_fn: Callable[[str], str],
    load_profile_fn: Callable[[str], object],
    resolve_reference_dirs_fn: Callable[[], list[str]],
    build_profile_fn: Callable[[list[str]], object],
    save_profile_fn: Callable[[str, object], bool],
    get_preset_fn: Callable[[str], object],
) -> KrProfileSetupResult:
    profile = None
    format_name = fallback_format or auto_gen_format
    profile_path = default_cache_path_fn(format_name)

    try:
        env_profile = os.environ.get("UTOA_KR_PROFILE_JSON", "").strip()
        if env_profile:
            profile = load_profile_fn(env_profile)
            if profile:
                log_fn(f"[KR-Profile] 외부 프로파일 로드: {env_profile}")

        if kr_anchor_profile_path:
            os.environ["UTOA_KR_ANCHOR_PROFILE_PATH"] = str(kr_anchor_profile_path)
            log_fn(f"[KR-Anchor] 외부 앵커 프로파일 사용: {kr_anchor_profile_path}")

        if not profile:
            profile = load_profile_fn(profile_path)

        if not profile:
            ref_dirs = resolve_reference_dirs_fn()
            if ref_dirs:
                trained = build_profile_fn(ref_dirs)
                if trained and save_profile_fn(profile_path, trained):
                    profile = trained
                    log_fn(
                        f"[KR-Profile] 로컬 샘플 기반 프로파일 생성: "
                        f"{profile_path} (rows={trained.get('rows', 0)})"
                    )

        if not profile:
            profile = get_preset_fn(format_name)
            save_profile_fn(profile_path, profile)
            log_fn(f"[KR-Profile] 추상화 프리셋 로드: {profile_path}")

        if profile:
            bucket_count = len((profile.get("buckets") or {}))
            log_fn(f"[KR-Profile] 기준 프로파일 적용 준비: buckets={bucket_count}")
    except Exception as exc:
        log_fn(f"[KR-Profile] 프로파일 로드 실패: {exc}")

    return KrProfileSetupResult(profile=profile, profile_path=profile_path)


__all__ = [
    "KrTextGridPreparation",
    "KrProfileSetupResult",
    "build_kr_textgrid_preparation",
    "build_wav_index",
    "iter_textgrid_files",
    "prepare_kr_profile_setup",
]

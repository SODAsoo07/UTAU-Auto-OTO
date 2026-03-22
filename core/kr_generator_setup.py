from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class KrProfileSetupResult:
    profile: object
    profile_path: str


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
    "KrProfileSetupResult",
    "prepare_kr_profile_setup",
]

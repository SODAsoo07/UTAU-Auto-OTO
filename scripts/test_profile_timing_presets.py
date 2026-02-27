import os
import sys
import struct
import tempfile
import wave

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.oto_generator import _apply_kr_profile_to_oto_file, _parse_oto_line_profile
from core.ja_oto_generator import _apply_ja_style_profile
from core.oto_profile_presets import get_kr_profile_preset, get_ja_profile_preset


def _make_silent_wav(path, sec=1.0, sr=44100):
    n = int(sec * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = (struct.pack("<h", 0) for _ in range(n))
        wf.writeframes(b"".join(frames))


def _ratio(ovl, pre):
    return (ovl / pre) if pre > 0 else 0.0


def test_kr_preset_consonant_timing():
    """
    KR preset 보정 시 자음군별 pre/overlap 경향이 기대대로 나오는지 확인.
    - 치찰(s) pre/ovl 비율 > 파열(k)
    - 유성(n) pre > 무성(h)
    """
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "a.wav")
        oto_path = os.path.join(td, "in.ini")
        _make_silent_wav(wav_path, sec=1.2)

        aliases = ["ka", "sa", "na", "ha"]
        with open(oto_path, "w", encoding="utf-8") as f:
            for a in aliases:
                f.write(f"a.wav={a},0.00,180.00,-320.00,120.00,42.00\n")

        changed = _apply_kr_profile_to_oto_file(
            oto_path=oto_path,
            wav_dir=td,
            profile=get_kr_profile_preset(),
            custom_map=None,
        )
        assert changed >= 1, "KR preset 적용 후 변경 라인이 없습니다."

        rows = {}
        with open(oto_path, "r", encoding="utf-8") as f:
            for raw in f:
                p = _parse_oto_line_profile(raw)
                if p:
                    rows[p["alias"]] = p

        for a in aliases:
            assert a in rows, f"KR 테스트 결과에 alias 누락: {a}"

        pre_k = rows["ka"]["pre"]
        pre_s = rows["sa"]["pre"]
        pre_n = rows["na"]["pre"]
        pre_h = rows["ha"]["pre"]
        ovlr_k = _ratio(rows["ka"]["ovl"], pre_k)
        ovlr_s = _ratio(rows["sa"]["ovl"], pre_s)
        gap_k = rows["ka"]["cons"] - pre_k
        gap_s = rows["sa"]["cons"] - pre_s

        assert pre_s > pre_k, f"KR 치찰 pre 증가 실패: sa={pre_s:.2f}, ka={pre_k:.2f}"
        assert ovlr_s > ovlr_k, f"KR 치찰 ovl/pre 증가 실패: sa={ovlr_s:.3f}, ka={ovlr_k:.3f}"
        assert pre_n > pre_h, f"KR 유성 pre 우위 실패: na={pre_n:.2f}, ha={pre_h:.2f}"
        assert gap_k < gap_s, f"KR 파열 cons_gap 단축 실패: ka={gap_k:.2f}, sa={gap_s:.2f}"

        return {
            "pre_ka": round(pre_k, 2),
            "pre_sa": round(pre_s, 2),
            "pre_na": round(pre_n, 2),
            "pre_ha": round(pre_h, 2),
            "ovlr_ka": round(ovlr_k, 3),
            "ovlr_sa": round(ovlr_s, 3),
        }


def _run_ja_case(preset_format, alias_type, alias_text):
    profile = get_ja_profile_preset(preset_format)
    return _apply_ja_style_profile(
        alias_type=alias_type,
        offset=0.0,
        consonant=220.0,
        cutoff=-420.0,
        pre=120.0,
        ovl=42.0,
        profile=profile,
        alias_text=alias_text,
    )


def test_ja_preset_consonant_timing():
    """
    JA CVVC/VCV preset 보정에서 자음군별 선행발성 경향 확인.
    """
    summary = {}
    for fmt in ("cvvc", "vcv"):
        # CV
        cv_k = _run_ja_case(fmt, "cv", "ka")
        cv_s = _run_ja_case(fmt, "cv", "sa")
        cv_n = _run_ja_case(fmt, "cv", "na")
        cv_h = _run_ja_case(fmt, "cv", "ha")

        pre_k = cv_k[3]
        pre_s = cv_s[3]
        pre_n = cv_n[3]
        pre_h = cv_h[3]
        ovlr_k = _ratio(cv_k[4], pre_k)
        ovlr_s = _ratio(cv_s[4], pre_s)
        gap_k = cv_k[1] - pre_k
        gap_s = cv_s[1] - pre_s

        assert pre_s > pre_k, f"JA({fmt}) 치찰 pre 증가 실패: sa={pre_s:.2f}, ka={pre_k:.2f}"
        assert ovlr_s > ovlr_k, f"JA({fmt}) 치찰 ovl/pre 증가 실패: sa={ovlr_s:.3f}, ka={ovlr_k:.3f}"
        assert pre_n > pre_h, f"JA({fmt}) 유성 pre 우위 실패: na={pre_n:.2f}, ha={pre_h:.2f}"
        assert gap_k < gap_s, f"JA({fmt}) 파열 cons_gap 단축 실패: ka={gap_k:.2f}, sa={gap_s:.2f}"

        # VC bridge
        vc_k = _run_ja_case(fmt, "vc", "a k")
        vc_s = _run_ja_case(fmt, "vc", "a s")
        assert vc_s[3] > vc_k[3], f"JA({fmt}) VC 치찰 pre 증가 실패: as={vc_s[3]:.2f}, ak={vc_k[3]:.2f}"

        summary[fmt] = {
            "cv_pre_ka": round(pre_k, 2),
            "cv_pre_sa": round(pre_s, 2),
            "cv_pre_na": round(pre_n, 2),
            "cv_pre_ha": round(pre_h, 2),
            "cv_ovlr_ka": round(ovlr_k, 3),
            "cv_ovlr_sa": round(ovlr_s, 3),
            "vc_pre_ak": round(vc_k[3], 2),
            "vc_pre_as": round(vc_s[3], 2),
        }
    return summary


def main():
    kr = test_kr_preset_consonant_timing()
    ja = test_ja_preset_consonant_timing()

    print("=== PRESET TIMING TEST: PASS ===")
    print("[KR]", kr)
    print("[JA]", ja)


if __name__ == "__main__":
    main()

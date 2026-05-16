from __future__ import annotations


def build_kr_cv_head_guard_messages(fname, alias, *, offset_reduced=0.0, cutoff_extended=0.0):
    messages = []
    if float(offset_reduced or 0.0) > 1.0:
        messages.append(f"🛡️ {fname}: CV_HEAD 오프셋 과선행 보정(+{float(offset_reduced):.1f}ms) [{alias}]")
    if float(cutoff_extended or 0.0) > 1.0:
        messages.append(f"🛡️ {fname}: CV_HEAD 모음 길이 보정(+{float(cutoff_extended):.1f}ms) [{alias}]")
    return messages


def build_kr_bridge_adjust_message(fname, alias, alias_type, bridge_shift, *, threshold=6.0):
    if str(alias_type or "").strip().lower() not in {"vc", "vv"}:
        return None
    shift = float(bridge_shift or 0.0)
    if abs(shift) <= float(threshold):
        return None
    return f"🧭 {fname}: KR 브리지 앵커 미세보정 ({shift:+.1f}ms) [{alias}]"


__all__ = [
    "build_kr_bridge_adjust_message",
    "build_kr_cv_head_guard_messages",
]

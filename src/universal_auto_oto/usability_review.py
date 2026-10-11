"""Occurrence-local triage and explicit rendering candidates; never auto-clean."""
import csv
import hashlib
import json
import io
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .oto_io import write_lines_atomic


def usage_review(row, timeline):
    selected = set(row["phone_indices"])
    phones = {p.phone_index: p for p in timeline.phone_segments}
    states = timeline.provenance.get("hsmm", {}).get("diagnostics", {}).get("states", [])
    local = [s for s in states if s["phone_index"] in selected]
    repeats = [s for s in local if s.get("repeated_phone_ambiguity", False)]
    competitors = [s for s in local if s.get("global_acoustic_competitor_risk", s.get("wrong_occurrence_risk", False))]
    reasons = []
    anchors = row["final_anchors"]
    role = row["role"].casefold()
    last = phones[max(selected)]
    if role == "vc" and max(selected) + 1 in phones and anchors["cutoff_abs"] > last.end_ms + 0.5:
        reasons.append("tail_extends_into_following_phone")
    if role in {"cv", "cv_head", "v", "vv", "vcv"} and anchors["cutoff_abs"] - anchors["fixed_abs"] < 15:
        reasons.append("short_stretchable_region")
    if row.get("template_repair_log"):
        reasons.append("template_anchor_repaired")
    if repeats:
        reasons.append("repeated_phone_ambiguity")
    if competitors:
        reasons.append("global_acoustic_competitor")
    if any(s.get("selected_vs_best_local_margin", 0) < -0.12 for s in local):
        reasons.append("local_boundary_ambiguity")
    gaps = [b.start_ms - a.end_ms for a, b in zip(
        sorted((phones[i] for i in selected), key=lambda p: p.phone_index),
        sorted((phones[i] for i in selected), key=lambda p: p.phone_index)[1:])]
    if any(g > 80 for g in gaps):
        reasons.append("gap_between_referenced_phones")
    hard = {"tail_extends_into_following_phone", "short_stretchable_region", "template_anchor_repaired", "gap_between_referenced_phones"}
    tier = "inspect_first" if hard.intersection(reasons) else (
        "connection_check" if role in {"vc", "vv", "vcv", "release", "v_end"} or repeats else "spot_check")
    weights = {"tail_extends_into_following_phone": 40, "short_stretchable_region": 40,
        "template_anchor_repaired": 30, "gap_between_referenced_phones": 30,
        "repeated_phone_ambiguity": 15, "local_boundary_ambiguity": 5, "global_acoustic_competitor": 2}
    return {"category": tier, "priority_score": sum(weights[r] for r in reasons),
        "reasons": reasons, "referenced_phone_indices": sorted(selected),
        "state_instance_ids": [s["instance_id"] for s in local],
        "repeat_competitor_indices": sorted({i for s in repeats for i in s["same_phone_competitor_indices"]}),
        "global_warning_is_error_confirmation": False, "usability_confirmed": False,
        "auto_clean_allowed": False}


def write_review_priority(rows, output, *, backup_existing=True):
    counts = Counter()
    occurrences = Counter()
    entries = []
    for row in rows:
        review = row["usage_review"]
        key = (row["recording_wav"], row["alias"])
        occurrence = occurrences[key]
        occurrences[key] += 1
        counts[review["category"]] += 1
        entries.append({"category": review["category"], "priority_score": review["priority_score"],
            "oto": row["output_oto"], "line": row["oto_line"], "wav": key[0], "alias": key[1],
            "occurrence": occurrence, "template": row["template_id"],
            "reasons": ";".join(review["reasons"]),
            "state_instance_ids": ";".join(review["state_instance_ids"]),
            "repeat_competitor_indices": ";".join(map(str, review["repeat_competitor_indices"])),
            "usability": "manual_pending"})
    order = {"inspect_first": 0, "connection_check": 1, "spot_check": 2}
    entries.sort(key=lambda r: (order[r["category"]], -r["priority_score"], r["oto"], r["line"]))
    path = output / "review_priority.csv"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(entries[0]) if entries else ["category", "oto", "line", "alias"])
    writer.writeheader()
    writer.writerows(entries)
    write_lines_atomic(path, buffer.getvalue().splitlines(), encoding="utf-8-sig", newline="\n",
                       backup_existing=backup_existing)
    report = {"rows": len(entries), "category_counts": dict(counts), "csv": str(path.resolve()),
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "automatic_pass_count": 0, "review_workload_reduction_measured": False,
        "reference_oto_timing_used": False, "review_only": True, "promotion_eligible": False}
    write_lines_atomic(output / "review_priority.json", json.dumps(report, indent=2).splitlines(),
                       encoding="utf-8", newline="\n", backup_existing=backup_existing)
    return report


def usability_candidate(plan, recording=None):
    """Review-only overlay, preserving explicit overrides and all alias indices."""
    changes = {}
    if plan.template == "builtin:vc" and len(plan.phone_indices) == 2:
        changes["cutoff_abs"] = "max(fixed_abs, phone[1].end)"
    elif plan.template in {"builtin:vv", "builtin:vv_plain", "builtin:release", "builtin:ending_exhalation"}:
        if (plan.template == "builtin:ending_exhalation" and recording is not None
                and recording.phones[plan.phone_indices[-1]] == "vf"):
            return plan
        # Review hypothesis: retain at most 180 ms of preceding context. This
        # changes expression, not decoded boundaries, and requires listening.
        changes["offset_abs"] = ("max(phone[0].start, phone[0].end - 330)"
            if plan.template == "builtin:release" else "max(phone[0].start, pre_abs - 180)")
    elif plan.template == "builtin:fry_to_sustain" and len(plan.phone_indices) == 2:
        # Periodic consonants were refined using vowel_like, moving the entry
        # later even though the sustained state itself had already been decoded.
        changes["pre_abs"] = "phone[1].start"
    if not changes:
        return plan
    return replace(plan, overrides={**changes, **plan.overrides})


# Where the next consonant hands over to the vowel: a stop's release, the end
# of frication, or the start of a sonorant's transition state.
_CONSONANT_HANDOVER = "fallback(phone[1].release, phone[1].frication_end, phone[1].vowel_transition, phone[1].end)"


def vc_consonant_end_overrides(plan):
    """Fixed/cutoff expressions that keep a VC's stretch span inside the next consonant.

    UTAU stretches the span between fixed and cutoff. Ending a VC at the next
    phone's end stretched its release and vowel transition, which listeners
    heard as an extra vowel. Empty for every other template.
    """
    if not ((plan.template == "builtin:vc" and len(plan.phone_indices) == 2)
            or (plan.template == "builtin:vc_semivowel" and len(plan.phone_indices) == 3)):
        return {}
    return {
        "fixed_abs": f"max(pre_abs, lerp(pre_abs, {_CONSONANT_HANDOVER}, 0.5))",
        "cutoff_abs": f"max(fixed_abs + 1, {_CONSONANT_HANDOVER} - 3)",
    }


def release_vowel_end_overrides(plan):
    """A '-' release with its preutterance at the vowel's end, like a VC into silence.

    The hand-made reference OTO puts it where voicing stops and keeps about
    200 ms after it; the template's end - 150 ms sits inside the vowel.
    Empty for every other template.
    """
    if plan.template != "builtin:release" or len(plan.phone_indices) != 1:
        return {}
    return {
        "pre_abs": "max(offset_abs + 30, phone[0].end)",
        "overlap_abs": "max(offset_abs, pre_abs - 60)",
        "fixed_abs": "pre_abs + 25",
        "cutoff_abs": "pre_abs + 200",
    }


def vc_consonant_end_candidate(plan, recording=None):
    """Review-only overlay: usability_candidate plus vc_consonant_end_overrides."""
    base = usability_candidate(plan, recording)
    changes = vc_consonant_end_overrides(plan)
    if not changes:
        return base
    return replace(base, overrides={**base.overrides, **changes, **plan.overrides})

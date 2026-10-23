from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .alignment import AlignmentError, AlignmentResult, align_recording
from .alias_type_symbolic import (
    DEFAULT_LABELS_FILENAME,
    DEFAULT_REPORT_FILENAME,
    DEFAULT_REVIEW_QUEUE_FILENAME,
    export_symbolic_alias_types,
)
from .oto_io import resolve_output_encoding, with_alias_suffix, write_lines_atomic
from .oto_style import OtoStyle, apply_korean_cvc_window, apply_korean_vcv_window, apply_oto_style
from .stable_window import apply_stable_window
from . import analysis_cache
from .row_risk import load_profile, risk_threshold, row_risks, voicing_onsets_ms
from .audio import file_sha256
from .domain import LandmarkTimeline, RecordingSpec, landmark_timeline_from_dict
from .project import ProjectSpec
from .sonorant_identity import SonorantTemplates, cepstra, has_adjacent_sonorants
from .renderer_calibration import load_renderer_calibration
from .alias_type_confidence import load_alias_type_confidence
from .template_engine import load_template_set, render_project_alias
from .pack_landmarks import add_pack_landmarks
from . import pack_row_risk, style_sample
from .language_pack import language_pack_by_id
from .usability_review import usage_review, write_review_priority


class TimelineReuseError(ValueError):
    """A persisted alignment artifact cannot safely drive OTO generation."""


def align_project(
    project: ProjectSpec,
    *,
    output: Path | None = None,
    force: bool = False,
    max_recordings: int | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    target = (output or (project.root / "output" / "timelines")).resolve()
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(f"timeline 출력 폴더가 비어 있지 않습니다. --force가 필요합니다: {target}")
    target.mkdir(parents=True, exist_ok=True)
    recordings = project.recordings
    if max_recordings is not None:
        recordings = recordings[: max(0, max_recordings)]
    rows: list[dict[str, object]] = []
    backend_counts: dict[str, int] = {}
    for index, recording in enumerate(recordings):
        result = align_recording(project, recording)
        timeline = result.timeline
        safe_stem = _safe_stem(Path(recording.wav).stem)
        filename = f"{index:05d}_{safe_stem}_{timeline.wav_sha256[:10]}.timeline.json"
        timeline_path = target / filename
        _write_json(timeline_path, timeline.to_dict(), backup=timeline_path.exists())
        backend_counts[result.backend_used] = backend_counts.get(result.backend_used, 0) + 1
        rows.append(
            {
                "recording_index": index,
                "wav": recording.wav,
                "wav_sha256": timeline.wav_sha256,
                "timeline": str(timeline_path),
                "backend_requested": result.backend_requested,
                "backend_used": result.backend_used,
                "confidence": timeline.confidence,
                "warnings": list(timeline.warnings),
                "diagnostics": dict(result.diagnostics),
            }
        )
    report = {
        "schema_version": 1,
        "project": project.name,
        "project_sha256": project.project_sha256,
        "inventory_sha256": project.inventory_sha256,
        "recordings_sha256": project.recordings_sha256,
        "requested_backend": project.alignment.backend,
        "feature_backend": project.alignment.feature_backend,
        "evidence_model": project.alignment.evidence_model,
        "scorer_backend": project.alignment.scorer_backend,
        "scorer_model": project.alignment.scorer_model,
        "scorer_model_sha256": project.alignment.scorer_model_sha256,
        "allow_heuristic_fallback": project.alignment.allow_heuristic_fallback,
        "backend_counts": backend_counts,
        "recording_count": len(rows),
        "output": str(target),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    _write_json(target / "alignment_report.json", report, backup=(target / "alignment_report.json").exists())
    return report


def _load_reusable_timelines(
    project: ProjectSpec,
    recordings: Sequence[RecordingSpec],
    timeline_dir: Path,
) -> tuple[list[tuple[LandmarkTimeline, Path, str]], dict[str, object]]:
    root = timeline_dir.resolve()
    report_path = root / "alignment_report.json"
    if not root.is_dir() or not report_path.is_file():
        raise TimelineReuseError(
            "--timeline-dir must contain align-project output and alignment_report.json"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TimelineReuseError(f"cannot read alignment report: {report_path}") from error
    if not isinstance(report, Mapping):
        raise TimelineReuseError("alignment_report.json must be an object")
    for key, expected in {
        "inventory_sha256": project.inventory_sha256,
        "recordings_sha256": project.recordings_sha256,
    }.items():
        if report.get(key) != expected:
            raise TimelineReuseError(f"timeline report {key} does not match the current project")
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list):
        raise TimelineReuseError("alignment_report.json rows must be an array")
    rows_by_index: dict[int, Mapping[str, object]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise TimelineReuseError("alignment report row must be an object")
        index = raw_row.get("recording_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in rows_by_index:
            raise TimelineReuseError("alignment report has an invalid or duplicate recording_index")
        rows_by_index[index] = raw_row
    expected_model_sha256 = _expected_model_sha256(project)
    expected_evidence_model_sha256 = _expected_evidence_model_sha256(project)
    expected_scorer_model_sha256 = _expected_scorer_model_sha256(project)
    reusable: list[tuple[LandmarkTimeline, Path, str]] = []
    for recording_index, recording in enumerate(recordings):
        raw_row = rows_by_index.get(recording_index)
        if raw_row is None:
            raise TimelineReuseError(f"alignment report is missing recording_index={recording_index}")
        raw_path = raw_row.get("timeline")
        if not isinstance(raw_path, str) or not raw_path:
            raise TimelineReuseError(f"recording_index={recording_index} has no timeline path")
        timeline_path = Path(raw_path)
        if not timeline_path.is_absolute():
            timeline_path = root / timeline_path
        timeline_path = timeline_path.resolve()
        if not timeline_path.is_file():
            raise TimelineReuseError(f"timeline does not exist: {timeline_path}")
        try:
            payload = json.loads(timeline_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TimelineReuseError(f"timeline must be an object: {timeline_path}")
            timeline = landmark_timeline_from_dict(payload)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            if isinstance(error, TimelineReuseError):
                raise
            raise TimelineReuseError(f"invalid timeline artifact: {timeline_path}: {error}") from error
        _validate_reusable_timeline(
            project,
            recording,
            timeline,
            expected_model_sha256=expected_model_sha256,
            expected_evidence_model_sha256=expected_evidence_model_sha256,
            expected_scorer_model_sha256=expected_scorer_model_sha256,
            timeline_path=timeline_path,
        )
        expected_wav_sha256 = file_sha256(project.wav_path(recording))
        if timeline.wav_sha256 != expected_wav_sha256:
            raise TimelineReuseError(f"WAV hash changed since alignment: {recording.wav}")
        row_wav_sha256 = raw_row.get("wav_sha256")
        if row_wav_sha256 != timeline.wav_sha256:
            raise TimelineReuseError(f"alignment report WAV hash disagrees with timeline: {timeline_path}")
        reusable.append((timeline, timeline_path, file_sha256(timeline_path)))
    return reusable, {
        "timeline_dir": str(root),
        "alignment_report": str(report_path),
        "alignment_report_sha256": file_sha256(report_path),
    }


def _expected_model_sha256(project: ProjectSpec) -> str | None:
    if project.alignment.model is None:
        return None
    model_path = (project.root / project.alignment.model).resolve()
    if not model_path.is_file():
        raise TimelineReuseError(f"current alignment model does not exist: {model_path}")
    return file_sha256(model_path)


def _expected_evidence_model_sha256(project: ProjectSpec) -> str | None:
    if project.alignment.evidence_model is None:
        return None
    model_path = (project.root / project.alignment.evidence_model).resolve()
    if not model_path.is_file():
        raise TimelineReuseError(f"current neural evidence model does not exist: {model_path}")
    return file_sha256(model_path)


def _expected_scorer_model_sha256(project: ProjectSpec) -> str | None:
    if project.alignment.scorer_backend != "trained_subphone_v1":
        return None
    configured_sha256 = project.alignment.scorer_model_sha256
    model_name = project.alignment.scorer_model
    if configured_sha256 is None or model_name is None:
        raise TimelineReuseError("trained scorer configuration has no model and SHA-256")
    model_path = (project.root / model_name).resolve()
    if not model_path.is_file():
        raise TimelineReuseError(f"current trained scorer model does not exist: {model_path}")
    actual_sha256 = file_sha256(model_path)
    if actual_sha256 != configured_sha256:
        raise TimelineReuseError("current trained scorer model does not match configured SHA-256")
    return actual_sha256


def _validate_reusable_timeline(
    project: ProjectSpec,
    recording: RecordingSpec,
    timeline: LandmarkTimeline,
    *,
    expected_model_sha256: str | None,
    expected_evidence_model_sha256: str | None,
    expected_scorer_model_sha256: str | None,
    timeline_path: Path,
) -> None:
    if timeline.schema_version != 2:
        raise TimelineReuseError(f"unsupported timeline schema_version in {timeline_path}: {timeline.schema_version}")
    if timeline.inventory_sha256 != project.inventory_sha256:
        raise TimelineReuseError(f"inventory changed since alignment: {timeline_path}")
    if timeline.recording_sha256 != _recording_identity_sha256(recording):
        raise TimelineReuseError(f"recording sequence or alias plan changed since alignment: {recording.wav}")
    if timeline.model_sha256 != expected_model_sha256:
        raise TimelineReuseError(f"alignment model changed since alignment: {recording.wav}")
    backend = timeline.provenance.get("backend_requested")
    feature_backend = timeline.provenance.get("feature_backend")
    if backend != project.alignment.backend:
        raise TimelineReuseError(f"alignment backend changed since alignment: {recording.wav}")
    if feature_backend != project.alignment.feature_backend:
        raise TimelineReuseError(f"feature backend changed since alignment: {recording.wav}")
    if timeline.provenance.get("neural_evidence_model_sha256") != expected_evidence_model_sha256:
        raise TimelineReuseError(f"neural evidence model changed since alignment: {recording.wav}")
    if expected_scorer_model_sha256 is not None:
        scorer = timeline.provenance.get("frame_scorer")
        if not isinstance(scorer, Mapping) or scorer.get("configured_scorer_model_sha256") != expected_scorer_model_sha256:
            raise TimelineReuseError(f"trained scorer model changed since alignment: {recording.wav}")
    if len(timeline.phone_segments) != len(recording.phones):
        raise TimelineReuseError(f"phone count differs from current recording: {recording.wav}")
    expected_phones = tuple(enumerate(recording.phones))
    actual_phones = tuple((segment.phone_index, segment.phone_id) for segment in timeline.phone_segments)
    if actual_phones != expected_phones:
        raise TimelineReuseError(f"phone occurrence order differs from current recording: {recording.wav}")
    state_ids = [state.instance_id for phone in timeline.phone_segments for state in phone.state_segments]
    if not state_ids or len(state_ids) != len(set(state_ids)):
        raise TimelineReuseError(f"timeline has missing or duplicate state instance IDs: {timeline_path}")


def _recording_identity_sha256(recording: RecordingSpec) -> str:
    payload = {
        "wav": recording.wav,
        "phones": list(recording.phones),
        "aliases": [
            {
                "alias": alias.alias,
                "phone_indices": list(alias.phone_indices),
                "template": alias.template,
                "role": alias.role,
                "overrides": dict(alias.overrides),
            }
            for alias in recording.aliases
        ],
        "transcript_source": recording.transcript_source,
        "label_source": recording.label_source,
        "metadata": dict(recording.metadata),
        "guide_anchors": [anchor.__dict__ for anchor in recording.guide_anchors],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _with_row_risk(row, risk: float, threshold: float):
    """Split by the fitted row risk (row_risk.py): low-risk rows become clean unless a template
    repair was needed; the confidence becomes 1 - risk so review priority follows the risk."""
    clean = risk <= threshold and "template_constraint_repaired" not in row.reasons
    reasons = tuple(reason for reason in row.reasons if reason != "uncalibrated_confidence")
    marker = "row_risk_low" if clean else "row_risk_high"
    return replace(row, split="clean" if clean else "review", confidence=float(1.0 - risk),
                   reasons=(*reasons, marker))


def _apply_pack_row_risk(row_reports, rows_by_output, pack_recordings, style_sample_info=None) -> list[dict]:
    """Rank language-pack rows by the fitted review risk (pack_row_risk.py), per pack; the riskiest
    share goes to review (template repairs always do) and the confidence becomes 1 - risk.
    Returns the style-sample rows (style_sample.py) picked from the low-risk rows."""
    sample: list[dict] = []
    by_pack: dict[str, list[int]] = defaultdict(list)
    for position, report in enumerate(row_reports):
        pack_id = report["recording_metadata"].get("language_pack")
        if pack_id and int(report["recording_index"]) in pack_recordings:
            by_pack[str(pack_id)].append(position)
    for pack_id, positions in by_pack.items():
        profile = pack_row_risk.load_profile(pack_id)
        if profile is None:
            continue
        reports = [row_reports[p] for p in positions]
        features = pack_row_risk.row_features(
            language_pack_by_id(pack_id), reports,
            {index: timeline for index, (timeline, _wav) in pack_recordings.items()},
            {index: wav for index, (_timeline, wav) in pack_recordings.items()})
        risks = pack_row_risk.predict(profile, features)
        ranks = pack_row_risk.review_ranks(risks)
        pack = language_pack_by_id(pack_id)
        sample += style_sample.sample_rows(pack, reports, style_sample.select_style_sample(pack, reports, risks),
                                           {index: wav for index, (_timeline, wav) in pack_recordings.items()},
                                           style_sample_info)
        for position, item, risk, rank in zip(positions, features, risks, ranks):
            report = row_reports[position]
            target = Path(report["output_oto"])
            row = rows_by_output[target][int(report["oto_line"]) - 1]
            clean = rank == "clean" and "template_constraint_repaired" not in row.reasons
            reasons = tuple(reason for reason in row.reasons if reason != "uncalibrated_confidence")
            marker = "pack_row_risk_low" if clean else ("pack_row_risk_first" if rank == "inspect_first" else "pack_row_risk_high")
            reasons = (*reasons, marker,
                       *(() if clean else pack_row_risk.risk_hints(item)))
            row = replace(row, split="clean" if clean else "review", confidence=float(1.0 - risk), reasons=reasons)
            rows_by_output[target][int(report["oto_line"]) - 1] = row
            # Keep what the risk model read, so a later fit sees the same inputs as this run.
            report["pre_risk"] = {"confidence": report["confidence"], "reasons": list(report["reasons"]),
                                  "usage_review": dict(report["usage_review"])}
            report.update(split=row.split, confidence=row.confidence, reasons=list(row.reasons),
                          pack_row_risk=round(float(risk), 4))
            usage = dict(report["usage_review"])
            usage.update(category="inspect_first" if rank == "inspect_first" else "spot_check",
                         priority_score=int(round(100 * float(risk))))
            report["usage_review"] = usage
    return sample


def _align_or_error(project: ProjectSpec, recording: RecordingSpec, skip_failed: bool,
                    **options: object) -> AlignmentResult | AlignmentError:
    """align_recording, but return the error instead of raising when ``skip_failed``."""
    try:
        return align_recording(project, recording, **options)
    except AlignmentError as exc:
        if not skip_failed:
            raise
        return exc


ProgressCallback = Callable[[str, int, int], None]


def _align_job(job: tuple) -> tuple[AlignmentResult | AlignmentError, object, object]:
    project, recording, skip_failed, options, with_onsets, with_cepstra = job
    result = _align_or_error(project, recording, skip_failed, **options)
    wav = project.wav_path(recording)
    return (result, voicing_onsets_ms(wav) if with_onsets else None,
            cepstra(wav) if with_cepstra and not isinstance(result, AlignmentError) else None)


def _risk_profile(recording: RecordingSpec):
    """The row risk profile of the recording's language/format, or None (rows stay review).

    AUTOOTO_ROW_RISK=off skips it, for generating risk-model training data from the raw rows.
    """
    if os.environ.get("AUTOOTO_ROW_RISK", "").strip().casefold() == "off":
        return None
    return load_profile(recording.metadata.get("language_hint"), recording.metadata.get("format_hint"))


def _needs_row_risk(recording: RecordingSpec) -> bool:
    return _risk_profile(recording) is not None


class _Aligner:
    """Aligns recordings one pass at a time, serially or in worker processes.

    Recordings are independent within a pass, so ``workers`` > 1 changes speed, not results.
    Each recording always goes to the same worker (one single-process pool per worker), so
    that worker's ``analysis_cache`` reuses its acoustic analyses in the identity passes; the
    serial path caches in this process. ``progress(stage, done, total)`` follows every recording.
    """

    def __init__(self, project: ProjectSpec, skip_failed: bool, workers: int = 1,
                 progress: ProgressCallback | None = None) -> None:
        self.project, self.skip_failed = project, skip_failed
        self.workers, self.progress = max(1, int(workers)), progress
        self._pools: list = []
        self._shard: dict[str, int] = {}
        self._load: list[float] = []
        # Voicing onsets (row risk) and cepstra (sonorant templates), computed alongside
        # the first alignment so the serial steps between passes stay short.
        self.onsets: dict[str, object] = {}
        self.cepstra: dict[str, object] = {}
        analysis_cache.enable()
        self._closed = False

    def map(self, stage: str, jobs: Sequence[tuple[RecordingSpec, Mapping[str, object]]],
            with_cepstra: bool = False) -> list[AlignmentResult | AlignmentError]:
        total = len(jobs)
        self._report(stage, 0, total)
        if self.workers == 1 or total < 2:
            results = []
            for done, (recording, options) in enumerate(jobs, start=1):
                results.append(_align_or_error(self.project, recording, self.skip_failed, **options))
                self._report(stage, done, total)
            return results  # serial: onsets are computed at render time
        if not self._pools:
            from concurrent.futures import ProcessPoolExecutor
            count = min(self.workers, total)
            self._pools = [ProcessPoolExecutor(max_workers=1, initializer=analysis_cache.enable) for _ in range(count)]
            self._load = [0.0] * count
        self._assign([recording for recording, _options in jobs])
        futures = {self._pools[self._shard[recording.wav]].submit(
                       _align_job, (self.project, recording, self.skip_failed, dict(options),
                                    _needs_row_risk(recording) and recording.wav not in self.onsets, with_cepstra)): index
                   for index, (recording, options) in enumerate(jobs)}
        from concurrent.futures import as_completed
        results: list[AlignmentResult | AlignmentError | None] = [None] * total
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                result, onsets, cep = future.result()
                results[futures[future]] = result
                recording = jobs[futures[future]][0]
                if onsets is not None:
                    self.onsets[recording.wav] = onsets
                if cep is not None:
                    self.cepstra[str(self.project.wav_path(recording))] = cep
                self._report(stage, done, total)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return results  # type: ignore[return-value]

    def _assign(self, recordings: Sequence[RecordingSpec]) -> None:
        """Longest-first onto the least-loaded worker, by WAV size; kept for later passes."""
        new = [recording for recording in recordings if recording.wav not in self._shard]
        sizes = {}
        for recording in new:
            try:
                sizes[recording.wav] = float(self.project.wav_path(recording).stat().st_size)
            except OSError:
                sizes[recording.wav] = 0.0
        for recording in sorted(new, key=lambda item: -sizes[item.wav]):
            shard = min(range(len(self._load)), key=self._load.__getitem__)
            self._shard[recording.wav] = shard
            self._load[shard] += sizes[recording.wav]

    def _report(self, stage: str, done: int, total: int) -> None:
        if self.progress is not None:
            self.progress(stage, done, total)

    def close(self) -> None:
        for pool in self._pools:
            pool.shutdown(wait=True, cancel_futures=True)
        self._pools = []
        if not self._closed:
            self._closed = True
            analysis_cache.disable()


def _sonorant_identity_two_pass(
    project: ProjectSpec,
    recordings: Sequence[RecordingSpec],
    alignment_options: Mapping[str, object],
    weight: float,
    passes: int = 2,
    scope: str = "sonorant",
    skip_failed: bool = False,
    aligner: _Aligner | None = None,
) -> tuple[list[AlignmentResult | AlignmentError], dict[str, object]]:
    """Align every recording, learn sonorant templates from those timelines, then re-align the
    multi-syllable recordings that have two adjacent sonorants with the templates' identity
    factors. A one-syllable recording has no syllable position to get wrong and keeps pass 1.
    With ``passes`` > 2 the templates are learned again from the previous identity pass and
    the same recordings are re-aligned once more per extra pass."""

    own_aligner = aligner is None
    aligner = aligner or _Aligner(project, skip_failed)
    try:
        return _identity_passes(project, recordings, alignment_options, weight, passes, scope, aligner)
    finally:
        if own_aligner:
            aligner.close()


def _identity_passes(project, recordings, alignment_options, weight, passes, scope, aligner):
    passes = max(2, int(passes))
    refine_japanese = alignment_options.get('japanese_special_refinement', False)
    baseline_options = ({**alignment_options, 'japanese_special_refinement': False}
                        if refine_japanese else alignment_options)
    first = aligner.map(f"align:1/{passes}", [(recording, baseline_options) for recording in recordings],
                        with_cepstra=True)
    targets = [
        index for index, recording in enumerate(recordings)
        if not isinstance(first[index], AlignmentError) and not (isinstance(recording.metadata.get("phone_slots"), Sequence)
                and len(recording.metadata.get("phone_slots")) < 2)
        and has_adjacent_sonorants(recording.phones, project.inventory, scope)
    ]
    final = list(first)
    template_summaries = []
    for _pass in range(2, passes + 1):
        templates = SonorantTemplates.build(
            (
                (project.wav_path(recording), [(seg.phone_id, seg.start_ms, seg.end_ms) for seg in result.timeline.phone_segments])
                for recording, result in zip(recordings, final)
                if not isinstance(result, AlignmentError)
            ),
            project.inventory,
            scope=scope,
            cepstra_cache=aligner.cepstra,
        )
        template_summaries.append(templates.summary())
        current = list(final)
        identity_options = {**baseline_options, "sonorant_identity": templates, "sonorant_identity_weight": weight}
        realigned_all = aligner.map(f"align:{_pass}/{passes}", [(recordings[index], identity_options) for index in targets])
        for index, realigned in zip(targets, realigned_all):
            # A later pass that fails keeps the earlier timeline.
            current[index] = realigned if not isinstance(realigned, AlignmentError) else current[index]
        final = current
    if refine_japanese:
        from .japanese_cvvc import special_slots
        indices = [i for i, recording in enumerate(recordings)
                   if special_slots(recording.wav) and not any(surface in {'s', 'sh'}
                       for _, surface in special_slots(recording.wav))]
        refined = aligner.map('refine:japanese', [(recordings[i],
            {**(identity_options if i in targets else baseline_options), 'japanese_special_refinement': True})
            for i in indices])
        for i, result in zip(indices, refined):
            if not isinstance(result, AlignmentError):
                final[i] = result
    return final, {
        "review_only": True,
        "weight": weight,
        "templates": template_summaries[0],
        "realigned_recordings": len(targets),
        "template_source": "first_pass_runtime_timelines",
        "realign_rule": "two_or_more_syllables_with_adjacent_sonorants",
        **({"passes": int(passes), "later_pass_templates": template_summaries[1:],
            "later_pass_template_source": "previous_identity_pass_timelines"} if int(passes) > 2 else {}),
        **({"scope": scope} if scope != "sonorant" else {}),
    }


def generate_project(
    project: ProjectSpec,
    *,
    output_oto: Path | None = None,
    force: bool = False,
    max_recordings: int | None = None,
    source_encoding: str | None = None,
    output_newline: str = "\r\n",
    provenance_context: Mapping[str, object] | None = None,
    alignment_options: Mapping[str, object] | None = None,
    timeline_dir: Path | None = None,
    write_generation_report: bool = True,
    write_row_report: bool = True,
    write_timelines: bool = True,
    oto_style: OtoStyle | None = None,
    alias_suffix: str = "",
    stable_window: bool = False,
    skip_failed_recordings: bool = False,
    alignment_workers: int = 1,
    progress: ProgressCallback | None = None,
    pack_style_info: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Generate OTO files; the write_* flags skip diagnostic files only, never the OTO files.

    With ``skip_failed_recordings`` a recording whose alignment fails (e.g. a silent take) is
    left out and listed in the report's ``failed_recordings`` instead of stopping the bank.

    The symbolic AliasType export is derived from row_report.jsonl and follows it.
    """
    started = time.perf_counter()
    if output_newline not in {"\n", "\r\n"}:
        raise ValueError("output_newline must be LF or CRLF")
    context = dict(provenance_context or {})
    runtime_alignment_options = dict(alignment_options or {})
    manual_anchors_by_wav = runtime_alignment_options.pop("manual_occurrence_start_anchors_by_wav", None)
    manual_anchor_provenance = runtime_alignment_options.pop("manual_occurrence_anchor_provenance", None)
    if manual_anchors_by_wav is not None and not isinstance(manual_anchors_by_wav, Mapping):
        raise ValueError("manual_occurrence_start_anchors_by_wav must be a mapping")
    if manual_anchor_provenance is not None and not isinstance(manual_anchor_provenance, Mapping):
        raise ValueError("manual_occurrence_anchor_provenance must be a mapping")
    if timeline_dir is not None and runtime_alignment_options:
        raise ValueError("timeline reuse cannot be combined with runtime alignment options")
    sonorant_identity_weight = float(runtime_alignment_options.pop("sonorant_identity_weight", 0.0) or 0.0)
    sonorant_identity_passes = int(runtime_alignment_options.pop("sonorant_identity_passes", 2) or 2)
    sonorant_identity_scope = str(runtime_alignment_options.pop("sonorant_identity_scope", "sonorant"))
    if sonorant_identity_weight > 0.0 and manual_anchors_by_wav is not None:
        raise ValueError("sonorant identity two-pass cannot be combined with manual occurrence anchors")
    runtime_feature_backend = str(
        runtime_alignment_options.get("feature_backend_override", project.alignment.feature_backend)
    )
    static_provenance = {
        "base_oto_timing_reused": False,
        "manual_gold_timing_used_for_generation": False,
        "unreviewed_rows_auto_applied": False,
        "alignment_backend": project.alignment.backend,
        "feature_backend": runtime_feature_backend,
        "alignment_model": project.alignment.model,
        "scorer_backend": project.alignment.scorer_backend,
        "scorer_model_sha256": project.alignment.scorer_model_sha256,
        "allow_heuristic_fallback": project.alignment.allow_heuristic_fallback,
        "alignment_source": "reused_timeline_artifacts" if timeline_dir is not None else "runtime_hsmm",
    }
    for key, expected in static_provenance.items():
        if key in context and context[key] != expected:
            raise ValueError(f"provenance_context cannot override runtime field: {key}")
    for key in {"alignment_profiles", "alignment_model_sha256s", "input_hashes"}:
        if key in context:
            raise ValueError(f"provenance_context cannot override runtime field: {key}")
    if not project.generation.atomic_write:
        raise ValueError("실사용 안전을 위해 generation.atomic_write=true가 필요합니다")
    primary_output = (output_oto or project.output_path()).resolve()
    wav_root = (project.root / "wav").resolve()
    try:
        primary_output.relative_to(wav_root)
    except ValueError:
        pass
    else:
        raise ValueError("원본 보호를 위해 project wav 폴더 안에는 OTO를 생성하지 않습니다")
    template_set = load_template_set(project)
    renderer_calibration_profile = None
    if project.calibration.renderer_enabled:
        if not project.calibration.renderer_profile:
            raise ValueError("renderer calibration profile is required")
        renderer_calibration_profile = load_renderer_calibration(
            project.root / project.calibration.renderer_profile
        )
    alias_type_confidence_profile = None
    if project.calibration.alias_type_enabled:
        if not project.calibration.alias_type_profile:
            raise ValueError("AliasType confidence profile is required")
        alias_type_confidence_profile = load_alias_type_confidence(
            project.root / project.calibration.alias_type_profile
        )
    recordings = project.recordings
    if max_recordings is not None:
        recordings = recordings[: max(0, max_recordings)]
    reused_timelines: list[tuple[LandmarkTimeline, Path, str]] | None = None
    timeline_reuse: dict[str, object] | None = None
    if timeline_dir is not None:
        reused_timelines, timeline_reuse = _load_reusable_timelines(project, recordings, timeline_dir)
    rows_by_output: dict[Path, list] = defaultdict(list)
    row_reports: list[dict[str, object]] = []
    timeline_rows: list[dict[str, object]] = []
    alignment_profiles: dict[str, str] = {}
    model_sha256s: set[str] = set()
    occurrence_ranker_diagnostics: list[Mapping[str, object]] = []
    frame_scorer_diagnostics: list[Mapping[str, object]] = []
    input_wavs: list[dict[str, str]] = []
    generated_timeline_dir = (
        primary_output.parent / "timelines" if reused_timelines is None and write_timelines else None
    )
    expected_outputs = {
        _recording_output_path(project, recording.wav, primary_output)
        for recording in recordings
    }
    for canonical_path in expected_outputs:
        candidates = (
            canonical_path,
            canonical_path.with_name(f"{canonical_path.stem}.clean{canonical_path.suffix}"),
            canonical_path.with_name(f"{canonical_path.stem}.review{canonical_path.suffix}"),
        )
        if not force:
            existing = next((path for path in candidates if path.exists()), None)
            if existing is not None:
                raise FileExistsError(existing)
    if not force:
        report_existing = next(
            (
                path
                for path in (
                    *(
                        (
                            primary_output.parent / "row_report.jsonl",
                            primary_output.parent / DEFAULT_LABELS_FILENAME,
                            primary_output.parent / DEFAULT_REVIEW_QUEUE_FILENAME,
                            primary_output.parent / DEFAULT_REPORT_FILENAME,
                        )
                        if write_row_report
                        else ()
                    ),
                    *((primary_output.parent / "generation_report.json",) if write_generation_report else ()),
                )
                if path.exists()
            ),
            None,
        )
        if report_existing is not None:
            raise FileExistsError(report_existing)
        if generated_timeline_dir is not None and generated_timeline_dir.exists() and any(generated_timeline_dir.iterdir()):
            raise FileExistsError(f"timeline 출력 폴더가 비어 있지 않습니다: {generated_timeline_dir}")
    identity_alignments: list[AlignmentResult | AlignmentError] | None = None
    sonorant_identity_report: dict[str, object] | None = None
    aligner = _Aligner(project, skip_failed_recordings, alignment_workers, progress)
    try:
        if sonorant_identity_weight > 0.0 and reused_timelines is None:
            identity_alignments, sonorant_identity_report = _sonorant_identity_two_pass(
                project, recordings, runtime_alignment_options, sonorant_identity_weight, sonorant_identity_passes,
                sonorant_identity_scope, skip_failed_recordings, aligner)
        elif reused_timelines is None and manual_anchors_by_wav is None:
            identity_alignments = aligner.map("align:1/1", [(recording, runtime_alignment_options) for recording in recordings])
    finally:
        aligner.close()
    failed_recordings: list[dict[str, str]] = []
    pack_recordings: dict[int, tuple[dict[str, object], Path]] = {}
    for recording_index, recording in enumerate(recordings):
        if progress is not None:
            progress("render", recording_index, len(recordings))
        timeline_source_path: Path | None = None
        timeline_source_sha256: str | None = None
        if reused_timelines is None:
            recording_anchors = None if manual_anchors_by_wav is None else manual_anchors_by_wav.get(recording.wav)
            if recording_anchors is not None and not isinstance(recording_anchors, Mapping):
                raise ValueError(f"{recording.wav}: manual occurrence anchors must be a mapping")
            if recording_anchors:
                expected_hashes = dict((manual_anchor_provenance or {}).get("wav_sha256_by_wav", {}))
                expected_hash = expected_hashes.get(recording.wav)
                if not isinstance(expected_hash, str):
                    raise ValueError(f"{recording.wav}: manual occurrence anchor hash is missing")
                actual_hash = file_sha256(project.wav_path(recording))
                if actual_hash != expected_hash:
                    raise ValueError(f"{recording.wav}: manual occurrence anchor WAV hash mismatch")
            aligned = identity_alignments[recording_index] if identity_alignments is not None else _align_or_error(
                project,
                recording,
                skip_failed_recordings,
                **runtime_alignment_options,
                manual_occurrence_start_anchors=recording_anchors,
                manual_occurrence_anchor_provenance=manual_anchor_provenance if recording_anchors else None,
            )
            if isinstance(aligned, AlignmentError):
                failed_recordings.append({"wav": recording.wav, "reason": str(aligned)})
                continue
            timeline = aligned.timeline
            backend_used = aligned.backend_used
        else:
            timeline, timeline_source_path, timeline_source_sha256 = reused_timelines[recording_index]
            backend_used = str(timeline.provenance.get("backend_used", project.alignment.backend))
        if recording.metadata.get("language_pack"):
            timeline = add_pack_landmarks(timeline, project.wav_path(recording), project.inventory)
            pack_recordings[recording_index] = ({"phone_segments": [segment.to_dict() for segment in timeline.phone_segments],
                                                 "landmarks": [landmark.to_dict() for landmark in timeline.landmarks],
                                                 "confidence": timeline.confidence}, project.wav_path(recording))
        profile_id = timeline.provenance.get("alignment_profile")
        profile_sha256 = timeline.provenance.get("alignment_profile_sha256")
        if isinstance(profile_id, str) and isinstance(profile_sha256, str):
            alignment_profiles[profile_sha256] = profile_id
        if timeline.model_sha256:
            model_sha256s.add(timeline.model_sha256)
        ranker_diagnostics = timeline.provenance.get("occurrence_ranker")
        if isinstance(ranker_diagnostics, Mapping):
            occurrence_ranker_diagnostics.append(ranker_diagnostics)
        scorer_diagnostics = timeline.provenance.get("frame_scorer")
        if isinstance(scorer_diagnostics, Mapping):
            frame_scorer_diagnostics.append(scorer_diagnostics)
        input_wavs.append(
            {
                "wav": recording.wav,
                "wav_sha256": timeline.wav_sha256,
                "recording_sha256": timeline.recording_sha256,
            }
        )
        timeline_path: Path | None = None
        if generated_timeline_dir is not None:
            timeline_name = f"{recording_index:05d}_{_safe_stem(Path(recording.wav).stem)}_{timeline.wav_sha256[:10]}.timeline.json"
            timeline_path = generated_timeline_dir / timeline_name
            if timeline_path.exists() and not force:
                raise FileExistsError(timeline_path)
            # Persist before deriving row reports so every rendered OTO row
            # carries the exact runtime timeline identity.
            _write_json(timeline_path, timeline.to_dict(), backup=timeline_path.exists())
            timeline_source_sha256 = file_sha256(timeline_path)
        target_oto = _recording_output_path(project, recording.wav, primary_output)
        rendered = [
            (plan, *render_project_alias(
                project,
                recording,
                timeline,
                plan,
                source_index=alias_index,
                template_set=template_set,
                renderer_calibration_profile=renderer_calibration_profile,
                alias_type_confidence_profile=alias_type_confidence_profile,
            ))
            for alias_index, plan in enumerate(recording.aliases)
        ]
        korean = recording.metadata.get("language_hint") in {"ko", "kr", "korean"}
        format_hint = recording.metadata.get("format_hint")
        segments = {segment.phone_index: segment for segment in timeline.phone_segments}
        risks: list[float] | None = None
        risk_profile = _risk_profile(recording)
        if risk_profile is not None and rendered:
            risks = row_risks(
                [{"alias": row.alias, "role": row.role, "confidence": row.confidence, "reasons": row.reasons,
                  "pre_abs": trace.final_anchors["pre_abs"], "offset_abs": trace.final_anchors["offset_abs"],
                  "fixed_abs": trace.final_anchors["fixed_abs"], "cutoff_abs": trace.final_anchors["cutoff_abs"],
                  "fallback_count": trace.fallback_count, "missing_landmarks": len(trace.missing_landmarks)}
                 for _plan, row, trace in rendered],
                project.wav_path(recording), onsets=aligner.onsets.get(recording.wav), profile=risk_profile)
        for alias_index, (plan, row, trace) in enumerate(rendered):
            if risks is not None:
                row = _with_row_risk(row, risks[alias_index], risk_threshold(risk_profile, format_hint))
            if korean and format_hint == "vcv":
                row = apply_korean_vcv_window(row)
            elif korean and format_hint == "cvc":
                row = apply_korean_cvc_window(row)
            stable_info = None
            # Language-pack rows keep their shape: the pack style and review risk were fitted to it.
            if stable_window and not recording.metadata.get("language_pack"):
                vowel = segments.get(plan.phone_indices[0]) if plan.phone_indices else None
                row, stable_info = apply_stable_window(row, project.wav_path(recording),
                                                       vowel_start_ms=vowel.start_ms if vowel is not None else None)
            row = apply_oto_style(row, oto_style)
            if alias_suffix:  # every output (oto, clean/review, row report) carries the suffixed alias
                row = replace(row, alias=with_alias_suffix(row.alias, alias_suffix))
            rows_by_output[target_oto].append(row)
            row_reports.append(
                {
                    **row.to_report(),
                    "recording_index": recording_index,
                    "recording_wav": recording.wav,
                    "output_oto": str(target_oto),
                    "oto_line": len(rows_by_output[target_oto]),
                    "recording_metadata": dict(recording.metadata),
                    "phone_indices": list(plan.phone_indices),
                    "template_id": trace.template_id,
                    "template_expressions": dict(trace.expressions),
                    "template_source": trace.template_source,
                    "template_source_sha256": trace.template_source_sha256,
                    "raw_anchors": dict(trace.raw_anchors),
                    "calibrated_anchors": dict(trace.calibrated_anchors),
                    "final_anchors": dict(trace.final_anchors),
                    "template_repairs": list(trace.repairs),
                    "template_repair_log": [dict(item) for item in trace.repair_log],
                    "soft_constraint_log": [dict(item) for item in trace.soft_constraint_log],
                    "referenced_landmarks": list(trace.referenced_landmarks),
                    "missing_landmarks": list(trace.missing_landmarks),
                    "fallback_count": trace.fallback_count,
                    "alias_type": dict(trace.alias_type),
                    "alias_type_label_source": "alias_notation_phone_inventory_rules",
                    "alias_type_uses_oto_timing": False,
                    "template_selection_source": trace.template_selection_source,
                    "recommended_template_id": trace.recommended_template_id,
                    "template_selection_warnings": list(trace.selection_warnings),
                    "alias_type_rule_set_sha256": trace.alias_type_rule_set_sha256,
                    "alias_type_calibration": dict(trace.alias_type_calibration),
                    "renderer_calibration": dict(trace.renderer_calibration),
                    "alignment_backend": backend_used,
                    "timeline_source": "reused" if timeline_source_path is not None else "runtime_hsmm",
                    "timeline_source_path": str(timeline_source_path) if timeline_source_path is not None else None,
                    "timeline_source_sha256": timeline_source_sha256,
                }
            )
            row_reports[-1]["usage_review"] = usage_review(row_reports[-1], timeline)
            if stable_info is not None:
                row_reports[-1]["stable_window"] = stable_info
        timeline_rows.append(
            {
                "recording_index": recording_index,
                "wav": recording.wav,
                "timeline": (
                    str(timeline_path or timeline_source_path)
                    if (timeline_path or timeline_source_path) is not None
                    else None
                ),
                "timeline_sha256": timeline_source_sha256,
                "timeline_source": "reused" if timeline_source_path is not None else "runtime_hsmm",
                "backend_used": backend_used,
                "confidence": timeline.confidence,
                "warnings": list(timeline.warnings),
            }
        )

    style_sample_rows: list[dict] = []
    if pack_recordings:
        style_sample_rows = _apply_pack_row_risk(row_reports, rows_by_output, pack_recordings, pack_style_info)
    targets = []
    output_encoding = resolve_output_encoding(project.generation.output_encoding, source_encoding)
    encoding_fallback = None
    if not _encodes_all(rows_by_output.values(), output_encoding):
        # File names or aliases the source encoding cannot hold (a kana bank extracted with the
        # Korean code page has Hangul file names): write UTF-8 rather than fail the bank.
        encoding_fallback = {"requested": output_encoding, "used": "utf-8"}
        output_encoding = "utf-8"
    for canonical_path, rows in sorted(rows_by_output.items(), key=lambda item: str(item[0]).casefold()):
        clean_path = canonical_path.with_name(f"{canonical_path.stem}.clean{canonical_path.suffix}")
        review_path = canonical_path.with_name(f"{canonical_path.stem}.review{canonical_path.suffix}")
        for path in (canonical_path, clean_path, review_path):
            if path.exists() and not force:
                raise FileExistsError(path)
        encoding = output_encoding
        write_lines_atomic(
            canonical_path,
            (row.to_line() for row in rows),
            encoding=encoding,
            newline=output_newline,
            backup_existing=project.generation.backup_existing,
        )
        if project.generation.split_review:
            write_lines_atomic(
                clean_path,
                (row.to_line() for row in rows if row.split == "clean"),
                encoding=encoding,
                newline=output_newline,
                backup_existing=project.generation.backup_existing,
            )
            write_lines_atomic(
                review_path,
                (row.to_line() for row in rows if row.split == "review"),
                encoding=encoding,
                newline=output_newline,
                backup_existing=project.generation.backup_existing,
            )
        targets.append(
            {
                "oto": str(canonical_path),
                "clean": str(clean_path) if project.generation.split_review else None,
                "review": str(review_path) if project.generation.split_review else None,
                "row_count": len(rows),
            }
        )
    report_dir = primary_output.parent
    symbolic_alias_type_export: dict[str, object] | None = None
    if write_row_report:
        write_lines_atomic(
            report_dir / "row_report.jsonl",
            (json.dumps(row, ensure_ascii=False, sort_keys=True) for row in row_reports),
            encoding="utf-8",
            newline="\n",
            backup_existing=project.generation.backup_existing,
        )
        symbolic_alias_type_export = export_symbolic_alias_types(
            report_dir / "row_report.jsonl",
            report_dir / DEFAULT_LABELS_FILENAME,
            report_dir / DEFAULT_REVIEW_QUEUE_FILENAME,
            output_json=report_dir / DEFAULT_REPORT_FILENAME,
            overwrite=force,
            backup_existing=project.generation.backup_existing,
        )
    style_sample_path = None
    if style_sample_rows:
        style_sample_path = report_dir / style_sample.SAMPLE_FILE
        style_sample.write_style_sample(style_sample_path, style_sample_rows)
    split_counts = Counter(str(row["split"]) for row in row_reports)
    usage_review_report = write_review_priority(row_reports, report_dir,
        backup_existing=project.generation.backup_existing)
    reason_counts = Counter(reason for row in row_reports for reason in row["reasons"])
    alias_type_counts = Counter(
        (
            str(row["alias_type"]["structural_role"]),
            str(row["alias_type"]["acoustic_subtype"]),
        )
        for row in row_reports
    )
    selection_warning_counts = Counter(
        warning
        for row in row_reports
        for warning in row["template_selection_warnings"]
    )
    provenance: dict[str, object] = {
        **static_provenance,
        "alignment_profiles": [
            {"profile_id": profile_id, "profile_sha256": profile_sha256}
            for profile_sha256, profile_id in sorted(alignment_profiles.items())
        ],
        "alignment_model_sha256s": sorted(model_sha256s),
        "occurrence_ranker_execution": _summarize_occurrence_ranker_execution(occurrence_ranker_diagnostics),
        "frame_scorer_execution": _summarize_frame_scorer_execution(frame_scorer_diagnostics),
        "input_hashes": {
            "project_sha256": project.project_sha256,
            "inventory_sha256": project.inventory_sha256,
            "recordings_sha256": project.recordings_sha256,
            "template_set_sha256": template_set.sha256,
            "wavs": input_wavs,
            "timeline_reuse": timeline_reuse,
        },
    }
    provenance.update(context)
    report = {
        "schema_version": 1,
        "project": project.name,
        "project_sha256": project.project_sha256,
        "inventory_sha256": project.inventory_sha256,
        "recordings_sha256": project.recordings_sha256,
        "template_set_sha256": template_set.sha256,
        "template_sources": list(template_set.sources),
        "renderer_calibration": (
            {
                "enabled": True,
                "profile_id": renderer_calibration_profile.profile_id,
                "profile_sha256": renderer_calibration_profile.sha256,
                "training_input_sha256": renderer_calibration_profile.training_input_sha256,
                "training_bank_ids": sorted(renderer_calibration_profile.training_bank_ids),
                "training_format_types": sorted(renderer_calibration_profile.training_format_types),
                "gold_timing_used_at_inference": False,
            }
            if renderer_calibration_profile is not None
            else {"enabled": False}
        ),
        "alias_type_confidence_calibration": (
            {
                "enabled": True,
                "profile_id": alias_type_confidence_profile.profile_id,
                "profile_sha256": alias_type_confidence_profile.sha256,
                "training_row_report_sha256": alias_type_confidence_profile.row_report_sha256,
                "training_bank_ids": sorted(alias_type_confidence_profile.training_bank_ids),
                "training_root_speaker_ids": sorted(alias_type_confidence_profile.training_root_speaker_ids),
                "gold_labels_used_at_inference": False,
            }
            if alias_type_confidence_profile is not None
            else {"enabled": False}
        ),
        "alignment_backend": project.alignment.backend,
        "feature_backend": runtime_feature_backend,
        "timeline_reuse": {"used": timeline_reuse is not None, **(timeline_reuse or {})},
        **({"sonorant_identity": sonorant_identity_report} if sonorant_identity_report is not None else {}),
        **({"failed_recordings": failed_recordings} if failed_recordings else {}),
        "recording_count": len(recordings),
        "generated_rows": len(row_reports),
        "clean_rows": split_counts.get("clean", 0),
        "review_rows": split_counts.get("review", 0),
        "usage_review": usage_review_report,
        "style_sample": ({"path": str(style_sample_path), "rows": len(style_sample_rows)}
                         if style_sample_path is not None else None),
        "reason_counts": dict(sorted(reason_counts.items())),
        "alias_type_counts": {
            f"{role}:{subtype}": count
            for (role, subtype), count in sorted(alias_type_counts.items())
        },
        "template_selection_warning_counts": dict(sorted(selection_warning_counts.items())),
        "symbolic_alias_type_export": symbolic_alias_type_export,
        "output_encoding": output_encoding,
        "output_encoding_fallback": encoding_fallback,
        "source_newline": "crlf" if output_newline == "\r\n" else "lf",
        "outputs": targets,
        "timelines": timeline_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "diagnostic_files": {
            "generation_report": write_generation_report,
            "row_report": write_row_report,
            "timelines": generated_timeline_dir is not None,
        },
    }
    if write_generation_report:
        _write_json(report_dir / "generation_report.json", report, backup=(report_dir / "generation_report.json").exists())
    return report


def _summarize_occurrence_ranker_execution(
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    requested = [item for item in diagnostics if item.get("requested") is True]
    applied = [item for item in requested if item.get("applied") is True]
    providers = sorted(
        {
            str(item["execution_provider"])
            for item in applied
            if isinstance(item.get("execution_provider"), str)
        }
    )
    model_sha256s = sorted(
        {
            str(item["model_sha256"])
            for item in applied
            if isinstance(item.get("model_sha256"), str)
        }
    )
    return {
        "recording_count": len(diagnostics),
        "requested_recording_count": len(requested),
        "applied_recording_count": len(applied),
        "applied_boundary_count": sum(
            int(item["applied_boundary_count"])
            for item in applied
            if isinstance(item.get("applied_boundary_count"), int)
        ),
        "execution_providers": providers,
        "model_sha256s": model_sha256s,
    }


def _summarize_frame_scorer_execution(
    diagnostics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    providers = sorted(
        {
            str(item["provider"])
            for item in diagnostics
            if isinstance(item.get("provider"), str)
        }
    )
    neural_enabled = [
        item.get("neural_evidence_shadow")
        for item in diagnostics
        if isinstance(item.get("neural_evidence_shadow"), Mapping)
        and item["neural_evidence_shadow"].get("enabled") is True
    ]
    neural_model_sha256s = sorted(
        {
            str(item["model_sha256"])
            for item in neural_enabled
            if isinstance(item.get("model_sha256"), str)
        }
    )
    transition_scorers = sorted(
        {
            str(item["transition_scorer"])
            for item in neural_enabled
            if isinstance(item.get("transition_scorer"), str)
        }
    )
    fallback_reasons = sorted(
        {
            str(item["fallback_reason"])
            for item in diagnostics
            if item.get("fallback") is True and isinstance(item.get("fallback_reason"), str)
        }
    )
    return {
        "recording_count": len(diagnostics),
        "providers": providers,
        "fallback_recording_count": sum(item.get("fallback") is True for item in diagnostics),
        "fallback_reasons": fallback_reasons,
        "neural_evidence_recording_count": len(neural_enabled),
        "neural_evidence_model_sha256s": neural_model_sha256s,
        "transition_scorers": transition_scorers,
    }


def _safe_stem(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)[:80]


def _recording_output_path(project: ProjectSpec, wav: str, primary_output: Path) -> Path:
    if project.generation.output_layout == "single_root":
        return primary_output
    parent = Path(wav).parent
    parts = parent.parts
    if parts and parts[0].casefold() == "wav":
        parent = Path(*parts[1:]) if len(parts) > 1 else Path()
    return primary_output.parent / parent / primary_output.name


def _encodes_all(row_groups, encoding: str) -> bool:
    try:
        for rows in row_groups:
            for row in rows:
                row.to_line().encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _write_json(path: Path, payload: object, *, backup: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(path, text.splitlines(), encoding="utf-8", newline="\n", backup_existing=backup)

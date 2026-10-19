from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from .acoustic_evidence import extract_acoustic_evidence, write_acoustic_evidence_npz
from .annotations import build_annotation_queue, initialize_annotation, load_annotation
from .alias_type_confidence import evaluate_alias_type_confidence, fit_alias_type_confidence
from .alias_type_symbolic import export_symbolic_alias_types
from .confidence import (
    audit_confidence_calibration_readiness,
    build_confidence_examples,
    build_confidence_examples_from_candidate_items,
    fit_confidence_calibration,
)
from .dataset_evaluation import evaluate_dataset_manifest
from .evaluation import evaluate_oto
from .external_evidence import inspect_external_evidence
from .oto_style import load_oto_style
from .generator import GenerationOptions, generate_voicebank, inspect_voicebank
from .transition_duration_constraint import fit_transition_duration_constraint
from .neural_evidence_dataset import build_publicdata_manifest
from .oto_generator_ui import launch_oto_generator_ui
from .timing_profile import fit_timing_profile
from .temporal_posterior import train_temporal_posterior_from_manifest
from .oto_io import read_text_auto, validate_oto_line_values
from .project import load_project, project_summary
from .project_runtime import align_project, generate_project
from .promotion_gate import audit_manual_gold_manifest, evaluate_release_gate
from .renderer_calibration import (
    build_renderer_calibration_examples,
    evaluate_renderer_calibration_holdout,
    fit_renderer_calibration,
)
from .posterior_model import (
    audit_posterior_training_manifest,
    train_posterior_model,
    train_posterior_model_from_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="universal-auto-oto", description="Acoustic-first UTAU/OpenUtau OTO generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gold_gate_parser = subparsers.add_parser(
        "audit-manual-gold",
        help="Audit Korean VCV/CVVC manual Gold coverage and provenance; never creates Gold.",
    )
    gold_gate_parser.add_argument("--manifest", required=True, type=Path)
    gold_gate_parser.add_argument("--stage", choices=("development", "locked"), default="development")
    gold_gate_parser.add_argument("--output-json", type=Path)

    release_gate_parser = subparsers.add_parser(
        "evaluate-release-promotion",
        help="Fail closed unless locked quality, Gold, packaged runtime, and human blind A/B all pass.",
    )
    release_gate_parser.add_argument("--quality-gate", required=True, type=Path)
    release_gate_parser.add_argument("--gold-audit", required=True, type=Path)
    release_gate_parser.add_argument("--runtime-smoke", required=True, type=Path)
    release_gate_parser.add_argument("--blind-ab", required=True, type=Path)
    release_gate_parser.add_argument("--output-json", type=Path)

    renderer_calibration_parser = subparsers.add_parser(
        "fit-renderer-calibration",
        help="Fit a train-only renderer residual profile from reviewed OTO-anchor examples.",
    )
    renderer_calibration_parser.add_argument("--examples", required=True, type=Path)
    renderer_calibration_parser.add_argument("--output", required=True, type=Path)
    renderer_calibration_parser.add_argument("--minimum-support", type=int, default=20)
    renderer_calibration_parser.add_argument("--shrinkage-strength", type=float, default=20.0)

    renderer_examples_parser = subparsers.add_parser(
        "build-renderer-calibration-examples",
        help="Bind reviewed absolute anchors to row-report renderer inputs and compute residuals.",
    )
    renderer_examples_parser.add_argument("--row-report", required=True, type=Path)
    renderer_examples_parser.add_argument("--reviewed-anchors", required=True, type=Path)
    renderer_examples_parser.add_argument("--output", required=True, type=Path)

    renderer_holdout_parser = subparsers.add_parser(
        "evaluate-renderer-calibration-holdout",
        help="Evaluate renderer calibration on validation or explicitly unlocked locked-test examples.",
    )
    renderer_holdout_parser.add_argument("--examples", required=True, type=Path)
    renderer_holdout_parser.add_argument("--profile", required=True, type=Path)
    renderer_holdout_parser.add_argument("--output", required=True, type=Path)
    renderer_holdout_parser.add_argument("--split", choices=("validation", "locked_test"), default="validation")
    renderer_holdout_parser.add_argument("--correction-threshold-ms", type=float, default=20.0)
    renderer_holdout_parser.add_argument("--minimum-examples", type=int, default=10)
    renderer_holdout_parser.add_argument("--allow-locked-test", action="store_true")

    alias_type_symbolic_parser = subparsers.add_parser(
        "export-symbolic-alias-types",
        help="Export AliasType labels derived from alias notation and phone inventory, plus an uncertainty-only review queue.",
    )
    alias_type_symbolic_parser.add_argument("--row-report", required=True, type=Path)
    alias_type_symbolic_parser.add_argument("--output-labels", required=True, type=Path)
    alias_type_symbolic_parser.add_argument("--review-queue", required=True, type=Path)
    alias_type_symbolic_parser.add_argument("--output-json", type=Path)

    alias_type_confidence_parser = subparsers.add_parser(
        "fit-alias-type-confidence",
        help="Fit train-only empirical AliasType confidence from consensus manual Gold.",
    )
    alias_type_confidence_parser.add_argument("--row-report", required=True, type=Path)
    alias_type_confidence_parser.add_argument("--alias-type-gold", required=True, type=Path)
    alias_type_confidence_parser.add_argument("--output", required=True, type=Path)
    alias_type_confidence_parser.add_argument("--minimum-support", type=int, default=20)
    alias_type_confidence_parser.add_argument("--shrinkage-strength", type=float, default=20.0)

    alias_type_confidence_eval_parser = subparsers.add_parser(
        "evaluate-alias-type-confidence",
        help="Evaluate AliasType accuracy and Brier score on validation or explicitly unlocked locked-test Gold.",
    )
    alias_type_confidence_eval_parser.add_argument("--row-report", required=True, type=Path)
    alias_type_confidence_eval_parser.add_argument("--alias-type-gold", required=True, type=Path)
    alias_type_confidence_eval_parser.add_argument("--profile", required=True, type=Path)
    alias_type_confidence_eval_parser.add_argument("--output", required=True, type=Path)
    alias_type_confidence_eval_parser.add_argument("--split", choices=("validation", "locked_test"), default="validation")
    alias_type_confidence_eval_parser.add_argument("--minimum-examples", type=int, default=20)
    alias_type_confidence_eval_parser.add_argument("--allow-locked-test", action="store_true")

    inspect_parser = subparsers.add_parser("inspect-voicebank", help="WAV와 base OTO 매핑 상태를 검사합니다")
    inspect_parser.add_argument("--bank", required=True, type=Path)
    inspect_parser.add_argument("--base-oto", type=Path)
    inspect_parser.add_argument("--recursive", action="store_true")

    project_parser = subparsers.add_parser(
        "validate-project",
        help="project.yaml과 inventory/recordings 계약을 검사합니다",
    )
    project_parser.add_argument("--project", required=True, type=Path)

    align_project_parser = subparsers.add_parser(
        "align-project",
        help="명시적 음소열을 프로젝트 backend로 정렬하고 LandmarkTimeline v2를 기록합니다",
    )
    align_project_parser.add_argument("--project", required=True, type=Path)
    align_project_parser.add_argument("--output", type=Path)
    align_project_parser.add_argument("--max-recordings", type=int)
    align_project_parser.add_argument("--force", action="store_true")

    evidence_parser = subparsers.add_parser(
        "export-acoustic-evidence",
        help="project recording의 versioned acoustic evidence NPZ를 생성합니다",
    )
    evidence_parser.add_argument("--project", required=True, type=Path)
    evidence_parser.add_argument("--recording-index", required=True, type=int)
    evidence_parser.add_argument("--output", required=True, type=Path)
    evidence_parser.add_argument("--force", action="store_true")

    neural_manifest_parser = subparsers.add_parser(
        "build-neural-evidence-manifest",
        help="Build a recursive Korean/Japanese PublicData WAV/LAB manifest and label audit for CRNN evidence training.",
    )
    neural_manifest_parser.add_argument("--data-root", required=True, type=Path)
    neural_manifest_parser.add_argument("--output-csv", required=True, type=Path)
    neural_manifest_parser.add_argument("--output-json", required=True, type=Path)
    neural_manifest_parser.add_argument("--language", action="append", choices=("ko", "ja"), default=[])
    neural_manifest_parser.add_argument("--hash-files", action="store_true", help="Write SHA-256 for every accepted WAV/LAB pair; required before training.")
    neural_manifest_parser.add_argument("--max-recordings", type=int, help="Audit a bounded deterministic subset for a fast manifest smoke check.")

    external_evidence_parser = subparsers.add_parser(
        "validate-external-evidence",
        help="외부 SSL/음향 frame-head NPZ의 WAV hash, 점수, confidence, provenance 검증",
    )
    external_evidence_parser.add_argument("--artifact", required=True, type=Path)
    external_evidence_parser.add_argument("--wav", required=True, type=Path)

    generate_project_parser = subparsers.add_parser(
        "generate-project",
        help="ProjectSpec·LandmarkTimeline·안전한 template engine으로 OTO를 생성합니다",
    )
    generate_project_parser.add_argument("--project", required=True, type=Path)
    generate_project_parser.add_argument("--output-oto", type=Path)
    generate_project_parser.add_argument(
        "--timeline-dir",
        type=Path,
        help="align-project output directory. Reuses verified timeline artifacts and does not rerun HSMM.",
    )
    generate_project_parser.add_argument("--max-recordings", type=int)
    generate_project_parser.add_argument("--force", action="store_true")

    generate_parser = subparsers.add_parser("generate-voicebank", help="오디오에서 새 OTO timing을 생성합니다")
    generate_parser.add_argument("--bank", required=True, type=Path)
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.add_argument("--base-oto", type=Path, help="alias/WAV 계획만 사용하며 timing은 사용하지 않습니다")
    generate_parser.add_argument(
        "--recordings-manifest",
        type=Path,
        help="Explicit external transcript/alias manifest; takes precedence over filename-derived phones without renaming WAV files.",
    )
    generate_parser.add_argument("--recursive", action="store_true")
    generate_parser.add_argument(
        "--output-encoding",
        default="preserve",
        choices=("preserve", "utf-8", "cp932", "shift_jis", "cp949", "euc-kr"),
    )
    generate_parser.add_argument("--review-threshold", type=float, default=0.45)
    generate_parser.add_argument(
        "--allow-heuristic-clean",
        action="store_true",
        help="보정되지 않은 heuristic score가 threshold를 넘은 행을 clean으로 허용합니다",
    )
    generate_parser.add_argument("--max-files", type=int)
    generate_parser.add_argument("--workers", type=int, default=0,
                                 help="routed_hsmm: parallel alignment processes (0 = automatic, 1 = serial)")
    generate_parser.add_argument(
        "--exclude-wav",
        action="append",
        default=[],
        help="Relative WAV path to exclude explicitly; may be provided more than once.",
    )
    generate_parser.add_argument(
        "--unmapped-alias-policy",
        choices=("fail", "exclude_row"),
        default="fail",
        help="routed_hsmm only: exclude_row leaves unbindable base-OTO rows out of oto.ini and lists them in unmapped_aliases.csv.",
    )
    generate_parser.add_argument(
        "--vc-end-policy",
        choices=("template", "consonant"),
        default="template",
        help="routed_hsmm only: consonant ends VC rows inside the next consonant instead of after it.",
    )
    generate_parser.add_argument(
        "--release-pre-policy",
        choices=("template", "vowel_end"),
        default="template",
        help="routed_hsmm only: vowel_end puts the preutterance of '-' releases at the vowel's end.",
    )
    generate_parser.add_argument(
        "--oto-style",
        type=Path,
        help="JSON preset of advanced OTO style settings (overlap ratios, preutterance shifts, margins).",
    )
    generate_parser.add_argument(
        "--keep-base-values",
        action="store_true",
        help="With --unmapped-alias-policy exclude_row: write rows that could not be generated with their "
             "base oto.ini values (review) instead of leaving them out.",
    )
    generate_parser.add_argument(
        "--alias-suffix",
        default="",
        help="Append _<suffix> to every generated alias (C4 -> ka_C4); aliases that already end with it are kept.",
    )
    generate_parser.add_argument("--bpm", type=float, help="routed_hsmm: recording tempo; syllables follow this beat grid.")
    generate_parser.add_argument(
        "--estimate-bpm",
        action="store_true",
        help="routed_hsmm: estimate the recording tempo from steady list files when --bpm is not given.",
    )
    generate_parser.add_argument(
        "--generate-aliases",
        action="store_true",
        help="routed_hsmm without --base-oto: write the format's alias list (CV, VC, VCV) from the file names.",
    )
    generate_parser.add_argument("--force", action="store_true", help="비어 있지 않은 출력 폴더의 산출물을 교체합니다")
    generate_parser.add_argument("--format-hint", choices=("cv", "cvc", "cvvc", "vcv"))
    generate_parser.add_argument("--language-hint", choices=("korean", "japanese"))
    generate_parser.add_argument("--timing-profile", type=Path)
    generate_parser.add_argument("--no-builtin-profile", action="store_true")
    generate_parser.add_argument(
        "--backend",
        choices=("legacy_heuristic", "routed_hsmm"),
        default="legacy_heuristic",
        help="Use routed_hsmm for the occurrence-aware HSMM path; legacy remains the unpromoted default.",
    )
    generate_parser.add_argument(
        "--development-alignment-profile",
        choices=("occurrence_v1", "isolated_sonorant_v1", "cvvc_slot_rhythm_v1", "cvvc_slot_rhythm_v2", "cvvc_slot_rhythm_v3", "cvvc_slot_rhythm_v4", "cvvc_slot_rhythm_v5", "cvvc_slot_rhythm_v6", "cvvc_slot_rhythm_v7", "cvvc_slot_rhythm_v8", "cvvc_slot_rhythm_v9", "cvvc_slot_rhythm_v10", "cvvc_slot_rhythm_v11", "cvvc_slot_rhythm_v12", "cvvc_slot_rhythm_v13", "cvvc_slot_rhythm_v14", "cvvc_slot_rhythm_v15", "cvvc_slot_rhythm_v16", "pause_aware_v1"),
        help="Explicit development-only HSMM profile; never changes the safe default.",
    )
    generate_parser.add_argument(
        "--occurrence-ranker-model",
        type=Path,
        help="Development-only ONNX occurrence-boundary ranker; requires --backend routed_hsmm and a positive strength.",
    )
    generate_parser.add_argument(
        "--occurrence-ranker-strength",
        type=float,
        default=0.0,
        help="Development-only occurrence ranker weight (0 < value <= 0.35).",
    )
    generate_parser.add_argument(
        "--occurrence-ranker-provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="ONNX provider for the occurrence ranker. auto records a CPU fallback when CUDA is unavailable.",
    )
    generate_parser.add_argument(
        "--development-occurrence-override",
        type=Path,
        help="Exact-WAV review-only occurrence anchors for a development constrained re-decode; never trains or promotes.",
    )

    oto_ui_parser = subparsers.add_parser(
        "oto-ui",
        help="실제 WAV 보이스뱅크 OTO 생성을 위한 AutoOTO v3 UI를 실행합니다.",
    )
    oto_ui_parser.add_argument("--bank", type=Path, help="UI에 미리 넣을 보이스뱅크 폴더")
    oto_ui_parser.add_argument("--output", type=Path, help="UI에 미리 넣을 출력 폴더")

    validate_parser = subparsers.add_parser("validate-oto", help="생성된 OTO의 구문과 시간 순서를 검사합니다")
    validate_parser.add_argument("--oto", required=True, type=Path)
    validate_parser.add_argument("--wav-dir", required=True, type=Path)

    evaluate_parser = subparsers.add_parser("evaluate-oto", help="생성 OTO를 수동 reference OTO와 절대 좌표로 비교합니다")
    evaluate_parser.add_argument("--generated", required=True, type=Path)
    evaluate_parser.add_argument("--reference", required=True, type=Path)
    evaluate_parser.add_argument("--wav-dir", required=True, type=Path)
    evaluate_parser.add_argument("--output-json", type=Path)
    evaluate_parser.add_argument("--output-csv", type=Path, help="행 단위 절대 좌표 비교 CSV (진단/검수용)")

    dataset_parser = subparsers.add_parser(
        "evaluate-dataset",
        help="dataset manifest의 candidate OTO를 manual OTO와 일괄 비교합니다",
    )
    dataset_parser.add_argument("--manifest", required=True, type=Path)
    dataset_parser.add_argument("--candidate-field", default="auto_oto")
    dataset_parser.add_argument("--output-json", type=Path)
    dataset_parser.add_argument("--language")
    dataset_parser.add_argument("--format-type")

    profile_parser = subparsers.add_parser(
        "fit-timing-profile",
        help="candidate timeline과 manual gold로 정규화 timing profile을 학습합니다",
    )
    profile_parser.add_argument("--candidate-manifest", required=True, action="append", type=Path)
    profile_parser.add_argument("--output-json", required=True, type=Path)
    profile_parser.add_argument("--minimum-support", type=int, default=20)

    annotation_queue_parser = subparsers.add_parser(
        "build-annotation-queue",
        help="CV·저신뢰·화자 다양성을 우선한 수동 landmark 작업 큐를 만듭니다",
    )
    annotation_queue_parser.add_argument("--project", required=True, type=Path)
    annotation_queue_parser.add_argument("--alignment-report", required=True, type=Path)
    annotation_queue_parser.add_argument("--output-csv", required=True, type=Path)
    annotation_queue_parser.add_argument("--target-count", type=int, default=500)

    init_annotation_parser = subparsers.add_parser(
        "init-annotation",
        help="Timeline v2를 manual_pending annotation 초안으로 변환합니다",
    )
    init_annotation_parser.add_argument("--project", required=True, type=Path)
    init_annotation_parser.add_argument("--recording-index", required=True, type=int)
    init_annotation_parser.add_argument("--timeline", required=True, type=Path)
    init_annotation_parser.add_argument("--output", required=True, type=Path)

    validate_annotation_parser = subparsers.add_parser(
        "validate-annotation",
        help="annotation schema·WAV hash·inventory·label provenance를 검사합니다",
    )
    validate_annotation_parser.add_argument("--annotation", required=True, type=Path)
    validate_annotation_parser.add_argument("--project", required=True, type=Path)

    confidence_parser = subparsers.add_parser(
        "fit-confidence-calibration",
        help="manual-gold speaker split 예제로 80ms/200ms/slot-shift 위험도를 보정합니다",
    )
    confidence_parser.add_argument("--examples", required=True, type=Path)
    confidence_parser.add_argument("--output-json", required=True, type=Path)
    confidence_parser.add_argument("--minimum-train-examples", type=int, default=100)
    confidence_parser.add_argument("--minimum-validation-examples", type=int, default=20)
    confidence_parser.add_argument("--l2", type=float, default=0.05)

    confidence_examples_parser = subparsers.add_parser(
        "build-confidence-examples",
        help="alignment와 manual Gold marker를 recording-level calibration 예제로 변환합니다",
    )
    confidence_examples_parser.add_argument("--project", required=True, type=Path)
    confidence_examples_parser.add_argument("--alignment-report", required=True, type=Path)
    confidence_examples_parser.add_argument("--annotations-dir", required=True, type=Path)
    confidence_examples_parser.add_argument("--output-jsonl", required=True, type=Path)

    confidence_manifest_parser = subparsers.add_parser(
        "build-confidence-examples-from-candidate-items",
        help="Build one speaker-safe calibration set from a multi-project align-gold-manifest CSV.",
    )
    confidence_manifest_parser.add_argument("--candidate-items-csv", required=True, type=Path)
    confidence_manifest_parser.add_argument("--output-jsonl", required=True, type=Path)

    confidence_readiness_parser = subparsers.add_parser(
        "audit-confidence-calibration-readiness",
        help="Report confidence-fit support and class-balance blockers without creating a profile.",
    )
    confidence_readiness_parser.add_argument("--examples", required=True, type=Path)
    confidence_readiness_parser.add_argument("--output-json", required=True, type=Path)
    confidence_readiness_parser.add_argument("--minimum-train-examples", type=int, default=100)
    confidence_readiness_parser.add_argument("--minimum-validation-examples", type=int, default=20)
    confidence_readiness_parser.add_argument("--minimum-train-class-examples", type=int, default=10)

    posterior_parser = subparsers.add_parser(
        "train-posterior-model",
        help="수동 Gold landmark에서 경량 acoustic posterior를 학습합니다",
    )
    posterior_parser.add_argument("--project", required=True, type=Path)
    posterior_parser.add_argument("--annotations-dir", required=True, type=Path)
    posterior_parser.add_argument("--output-model", required=True, type=Path)
    posterior_parser.add_argument("--minimum-train-recordings", type=int, default=20)
    posterior_parser.add_argument("--minimum-validation-recordings", type=int, default=5)
    posterior_parser.add_argument("--l2", type=float, default=0.05)
    posterior_parser.add_argument("--posterior-blend", type=float, default=0.75)

    posterior_audit_parser = subparsers.add_parser(
        "audit-posterior-training",
        help="Gold/speaker split과 phone-only posterior head 학습 가능 범위를 점검",
    )
    posterior_audit_parser.add_argument("--items-csv", required=True, type=Path)
    posterior_audit_parser.add_argument("--output-json", type=Path)
    posterior_audit_parser.add_argument("--minimum-train-recordings", type=int, default=20)
    posterior_audit_parser.add_argument("--minimum-validation-recordings", type=int, default=5)

    posterior_manifest_parser = subparsers.add_parser(
        "train-posterior-from-manifest",
        help="speaker-locked manual Gold manifest에서 feature backend별 공통 posterior 학습",
    )
    posterior_manifest_parser.add_argument("--items-csv", required=True, type=Path)
    posterior_manifest_parser.add_argument("--output-model", required=True, type=Path)
    posterior_manifest_parser.add_argument("--feature-backend")
    posterior_manifest_parser.add_argument("--minimum-train-recordings", type=int, default=20)
    posterior_manifest_parser.add_argument("--minimum-validation-recordings", type=int, default=5)
    posterior_manifest_parser.add_argument("--l2", type=float, default=0.05)
    posterior_manifest_parser.add_argument("--posterior-blend", type=float, default=0.75)

    temporal_manifest_parser = subparsers.add_parser(
        "train-temporal-posterior",
        help="Train a Conv1D temporal posterior from speaker-locked manual Gold; never promotes it.",
    )
    temporal_manifest_parser.add_argument("--items-csv", required=True, type=Path)
    temporal_manifest_parser.add_argument("--output-model", required=True, type=Path)
    temporal_manifest_parser.add_argument("--feature-backend")
    temporal_manifest_parser.add_argument("--minimum-train-recordings", type=int, default=20)
    temporal_manifest_parser.add_argument("--minimum-validation-recordings", type=int, default=5)
    temporal_manifest_parser.add_argument("--hidden-channels", type=int, default=8)
    temporal_manifest_parser.add_argument("--kernel-size", type=int, default=5)
    temporal_manifest_parser.add_argument("--epochs", type=int, default=300)
    temporal_manifest_parser.add_argument("--learning-rate", type=float, default=0.01)
    temporal_manifest_parser.add_argument("--l2", type=float, default=1e-4)
    temporal_manifest_parser.add_argument("--early-stopping-patience", type=int, default=30)
    temporal_manifest_parser.add_argument("--seed", type=int, default=0)

    duration_constraint_parser = subparsers.add_parser(
        "fit-transition-duration-constraint",
        help="Fit a development-only transition duration constraint from manual train Gold.",
    )
    duration_constraint_parser.add_argument("--items-csv", required=True, type=Path)
    duration_constraint_parser.add_argument("--output-json", required=True, type=Path)
    duration_constraint_parser.add_argument("--target-phone", required=True)
    duration_constraint_parser.add_argument("--target-state", default="transition")
    duration_constraint_parser.add_argument("--minimum-markers", type=int, default=5)
    duration_constraint_parser.add_argument("--minimum-recordings", type=int, default=2)
    duration_constraint_parser.add_argument("--minimum-speakers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()  # parallel alignment workers in a frozen (exe) build
    _configure_console_encoding()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit-manual-gold":
            result = audit_manual_gold_manifest(
                args.manifest,
                stage=args.stage,
                output_json=args.output_json,
            )
        elif args.command == "evaluate-release-promotion":
            result = evaluate_release_gate(
                args.quality_gate,
                args.runtime_smoke,
                args.blind_ab,
                gold_audit=args.gold_audit,
                output_json=args.output_json,
            )
        elif args.command == "fit-renderer-calibration":
            result = fit_renderer_calibration(
                args.examples,
                args.output,
                minimum_support=args.minimum_support,
                shrinkage_strength=args.shrinkage_strength,
            )
        elif args.command == "build-renderer-calibration-examples":
            result = build_renderer_calibration_examples(
                args.row_report,
                args.reviewed_anchors,
                args.output,
            )
        elif args.command == "evaluate-renderer-calibration-holdout":
            result = evaluate_renderer_calibration_holdout(
                args.examples,
                args.profile,
                args.output,
                split=args.split,
                correction_threshold_ms=args.correction_threshold_ms,
                minimum_examples=args.minimum_examples,
                allow_locked_test=args.allow_locked_test,
            )
        elif args.command == "export-symbolic-alias-types":
            result = export_symbolic_alias_types(
                args.row_report,
                args.output_labels,
                args.review_queue,
                output_json=args.output_json,
            )
        elif args.command == "fit-alias-type-confidence":
            result = fit_alias_type_confidence(
                args.row_report,
                args.alias_type_gold,
                args.output,
                minimum_support=args.minimum_support,
                shrinkage_strength=args.shrinkage_strength,
            )
        elif args.command == "evaluate-alias-type-confidence":
            result = evaluate_alias_type_confidence(
                args.row_report,
                args.alias_type_gold,
                args.profile,
                args.output,
                split=args.split,
                minimum_examples=args.minimum_examples,
                allow_locked_test=args.allow_locked_test,
            )
        elif args.command == "inspect-voicebank":
            result = inspect_voicebank(args.bank, args.base_oto, args.recursive)
        elif args.command == "validate-project":
            result = project_summary(load_project(args.project))
        elif args.command == "align-project":
            result = align_project(
                load_project(args.project),
                output=args.output,
                force=args.force,
                max_recordings=args.max_recordings,
            )
        elif args.command == "fit-transition-duration-constraint":
            result = fit_transition_duration_constraint(
                args.items_csv,
                args.output_json,
                target_phone=args.target_phone,
                target_state=args.target_state,
                minimum_markers=args.minimum_markers,
                minimum_recordings=args.minimum_recordings,
                minimum_speakers=args.minimum_speakers,
            )
        elif args.command == "export-acoustic-evidence":
            project = load_project(args.project)
            if not 0 <= args.recording_index < len(project.recordings):
                raise ValueError("--recording-index가 project recording 범위를 벗어났습니다")
            recording = project.recordings[args.recording_index]
            evidence = extract_acoustic_evidence(
                project.wav_path(recording),
                internal_sample_rate=project.audio.internal_sample_rate,
                convert_to_mono=project.audio.convert_to_mono,
                resampler=project.audio.resampler,
                feature_backend=project.alignment.feature_backend,
                evidence_model=(project.root / project.alignment.evidence_model).resolve() if project.alignment.evidence_model else None,
                evidence_provider=project.alignment.evidence_provider,
            )
            artifact = write_acoustic_evidence_npz(evidence, args.output, force=args.force)
            result = {
                "artifact": str(artifact),
                "wav": recording.wav,
                "wav_sha256": evidence.wav_sha256,
                "feature_backend": evidence.metadata["feature_backend"],
                "feature_config_sha256": evidence.feature_config_sha256,
                "frame_count": len(evidence.frame_times_ms),
                "track_count": len(evidence.tracks),
            }
        elif args.command == "build-neural-evidence-manifest":
            result = build_publicdata_manifest(
                args.data_root,
                args.output_csv,
                args.output_json,
                languages=tuple(args.language or ("ko", "ja")),
                hash_files=args.hash_files,
                max_recordings=args.max_recordings,
            )
        elif args.command == "validate-external-evidence":
            result = inspect_external_evidence(args.artifact, args.wav)
        elif args.command == "generate-project":
            result = generate_project(
                load_project(args.project),
                output_oto=args.output_oto,
                force=args.force,
                max_recordings=args.max_recordings,
                timeline_dir=args.timeline_dir,
            )
        elif args.command == "generate-voicebank":
            if not 0.0 <= args.review_threshold <= 1.0:
                raise ValueError("--review-threshold는 0~1 범위여야 합니다")
            result = generate_voicebank(
                GenerationOptions(
                    bank=args.bank,
                    output=args.output,
                    base_oto=args.base_oto,
                    recordings_manifest=args.recordings_manifest,
                    recursive=args.recursive,
                    output_encoding=args.output_encoding,
                    review_threshold=args.review_threshold,
                    allow_heuristic_clean=args.allow_heuristic_clean,
                    max_files=args.max_files,
                    force=args.force,
                    format_hint=args.format_hint,
                    language_hint=args.language_hint,
                    timing_profile=args.timing_profile,
                    use_builtin_profiles=not args.no_builtin_profile,
                    backend=args.backend,
                    development_alignment_profile=args.development_alignment_profile,
                    occurrence_ranker_model=args.occurrence_ranker_model,
                    occurrence_ranker_strength=args.occurrence_ranker_strength,
                    occurrence_ranker_provider=args.occurrence_ranker_provider,
                    development_occurrence_override=args.development_occurrence_override,
                    exclude_wavs=tuple(args.exclude_wav),
                    unmapped_alias_policy=args.unmapped_alias_policy,
                    vc_end_policy=args.vc_end_policy,
                    release_pre_policy=args.release_pre_policy,
                    generate_missing_aliases=args.generate_aliases,
                    bpm=args.bpm,
                    estimate_bpm=args.estimate_bpm,
                    oto_style=load_oto_style(args.oto_style) if args.oto_style else None,
                    alias_suffix=args.alias_suffix,
                    unmapped_row_output="keep_base" if args.keep_base_values else "exclude",
                    alignment_workers=args.workers,
                )
            )
        elif args.command == "oto-ui":
            result = launch_oto_generator_ui(initial_bank=args.bank, initial_output=args.output)
        elif args.command == "validate-oto":
            result = validate_oto(args.oto, args.wav_dir)
        elif args.command == "evaluate-oto":
            result = evaluate_oto(
                args.generated,
                args.reference,
                args.wav_dir,
                output_json=args.output_json,
                output_csv=args.output_csv,
            )
        elif args.command == "evaluate-dataset":
            result = evaluate_dataset_manifest(
                args.manifest,
                candidate_field=args.candidate_field,
                output_json=args.output_json,
                language=args.language,
                format_type=args.format_type,
            )
        elif args.command == "fit-timing-profile":
            result = fit_timing_profile(
                args.candidate_manifest,
                args.output_json,
                minimum_support=args.minimum_support,
            )
        elif args.command == "build-annotation-queue":
            result = build_annotation_queue(
                load_project(args.project),
                args.alignment_report,
                args.output_csv,
                target_count=args.target_count,
            )
        elif args.command == "init-annotation":
            document = initialize_annotation(
                load_project(args.project),
                args.recording_index,
                args.timeline,
                args.output,
            )
            result = {
                "annotation": str(args.output.resolve()),
                "annotation_id": document.annotation_id,
                "revision": document.revision,
                "label_source": document.label_source,
                "marker_count": len(document.markers),
                "automatic_gold_created": False,
            }
        elif args.command == "validate-annotation":
            document = load_annotation(args.annotation, project=load_project(args.project))
            result = {
                "annotation": str(args.annotation.resolve()),
                "annotation_id": document.annotation_id,
                "revision": document.revision,
                "label_source": document.label_source,
                "marker_count": len(document.markers),
                "valid": True,
            }
        elif args.command == "fit-confidence-calibration":
            result = fit_confidence_calibration(
                args.examples,
                args.output_json,
                minimum_train_examples=args.minimum_train_examples,
                minimum_validation_examples=args.minimum_validation_examples,
                l2=args.l2,
            )
        elif args.command == "build-confidence-examples":
            result = build_confidence_examples(
                load_project(args.project),
                args.alignment_report,
                args.annotations_dir,
                args.output_jsonl,
            )
        elif args.command == "build-confidence-examples-from-candidate-items":
            result = build_confidence_examples_from_candidate_items(
                args.candidate_items_csv,
                args.output_jsonl,
            )
        elif args.command == "audit-confidence-calibration-readiness":
            result = audit_confidence_calibration_readiness(
                args.examples,
                args.output_json,
                minimum_train_examples=args.minimum_train_examples,
                minimum_validation_examples=args.minimum_validation_examples,
                minimum_train_class_examples=args.minimum_train_class_examples,
            )
        elif args.command == "train-posterior-model":
            result = train_posterior_model(
                load_project(args.project),
                args.annotations_dir,
                args.output_model,
                minimum_train_recordings=args.minimum_train_recordings,
                minimum_validation_recordings=args.minimum_validation_recordings,
                l2=args.l2,
                posterior_blend=args.posterior_blend,
            )
        elif args.command == "audit-posterior-training":
            result = audit_posterior_training_manifest(
                args.items_csv,
                output_json=args.output_json,
                minimum_train_recordings=args.minimum_train_recordings,
                minimum_validation_recordings=args.minimum_validation_recordings,
            )
        elif args.command == "train-posterior-from-manifest":
            result = train_posterior_model_from_manifest(
                args.items_csv,
                args.output_model,
                feature_backend=args.feature_backend,
                minimum_train_recordings=args.minimum_train_recordings,
                minimum_validation_recordings=args.minimum_validation_recordings,
                l2=args.l2,
                posterior_blend=args.posterior_blend,
            )
        elif args.command == "train-temporal-posterior":
            result = train_temporal_posterior_from_manifest(
                args.items_csv,
                args.output_model,
                feature_backend=args.feature_backend,
                minimum_train_recordings=args.minimum_train_recordings,
                minimum_validation_recordings=args.minimum_validation_recordings,
                hidden_channels=args.hidden_channels,
                kernel_size=args.kernel_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                early_stopping_patience=args.early_stopping_patience,
                seed=args.seed,
            )
        else:
            raise AssertionError(args.command)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command in {"audit-manual-gold", "evaluate-release-promotion"} and not result["passed"]:
        return 3
    if args.command == "evaluate-renderer-calibration-holdout" and not result["passed"]:
        return 3
    if args.command == "evaluate-alias-type-confidence" and not result["passed"]:
        return 3
    return 0


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


def validate_oto(oto_path: Path, wav_dir: Path) -> dict[str, object]:
    text, encoding, newline = read_text_auto(oto_path.resolve())
    errors: list[dict[str, object]] = []
    row_count = 0
    wav_cache: dict[str, float] = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        row_count += 1
        try:
            wav_name, payload = raw.split("=", 1)
            values = payload.split(",")
            if len(values) != 6:
                raise ValueError(f"필드 수 {len(values)}")
            _alias = values[0]
            offset, consonant, cutoff, preutterance, overlap = map(float, values[1:])
            wav_path = wav_dir / wav_name
            if not wav_path.is_file():
                raise FileNotFoundError(wav_path)
            if wav_name not in wav_cache:
                info = sf.info(wav_path)
                wav_cache[wav_name] = 1000.0 * info.frames / info.samplerate
            valid, reasons = validate_oto_line_values(
                duration_ms=wav_cache[wav_name],
                offset=offset,
                consonant=consonant,
                cutoff=cutoff,
                preutterance=preutterance,
                overlap=overlap,
            )
            if not valid:
                errors.append({"line": line_number, "wav": wav_name, "reasons": list(reasons)})
        except Exception as error:
            errors.append({"line": line_number, "error": str(error), "raw": raw[:200]})
    return {
        "oto": str(oto_path.resolve()),
        "wav_dir": str(wav_dir.resolve()),
        "encoding": encoding,
        "newline": "crlf" if newline == "\r\n" else "lf",
        "rows": row_count,
        "wav_files_referenced": len(wav_cache),
        "errors": errors,
        "valid": not errors,
    }


if __name__ == "__main__":
    raise SystemExit(main())

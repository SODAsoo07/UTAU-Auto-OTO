from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .alias_classifier import SUBTYPE_TEMPLATE_IDS, resolve_alias_plan
from .alias_type_confidence import (
    AliasTypeConfidenceProfile,
    load_alias_type_confidence,
    predict_alias_type_confidence,
)
from .domain import AliasPlan, LandmarkTimeline, PhoneSegment, RecordingSpec, RenderedOtoRow
from .oto_io import validate_oto_line_values
from .project import ProjectSpec
from .renderer_calibration import (
    RendererCalibrationProfile,
    apply_renderer_calibration,
    load_renderer_calibration,
    select_renderer_calibration,
)


TEMPLATE_RESOURCE = "profiles/oto_templates_v1.yaml"
REQUIRED_ANCHORS = ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")


class TemplateError(ValueError):
    pass


class MissingAnchorError(TemplateError):
    pass


@dataclass(frozen=True)
class SoftConstraint:
    id: str
    expression: str
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class TemplateDefinition:
    id: str
    role: str
    anchors: Mapping[str, str]
    soft_constraints: tuple[SoftConstraint, ...]
    source: str
    source_sha256: str


@dataclass(frozen=True)
class TemplateSet:
    templates: Mapping[str, TemplateDefinition]
    sha256: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class RenderTrace:
    template_id: str
    expressions: Mapping[str, str]
    raw_anchors: Mapping[str, float]
    final_anchors: Mapping[str, float]
    repairs: tuple[str, ...]
    repair_log: tuple[Mapping[str, object], ...]
    referenced_landmarks: tuple[str, ...]
    missing_landmarks: tuple[str, ...]
    fallback_count: int
    template_source: str
    template_source_sha256: str
    alias_type: Mapping[str, object]
    template_selection_source: str
    recommended_template_id: str | None
    selection_warnings: tuple[str, ...]
    alias_type_rule_set_sha256: str
    renderer_calibration: Mapping[str, object]
    calibrated_anchors: Mapping[str, float]
    soft_constraint_log: tuple[Mapping[str, object], ...]
    alias_type_calibration: Mapping[str, object]


def load_template_set(project: ProjectSpec) -> TemplateSet:
    builtin_resource = files("universal_auto_oto").joinpath(TEMPLATE_RESOURCE)
    builtin_text = builtin_resource.read_text(encoding="utf-8")
    builtin_payload = _load_template_yaml(builtin_text, f"package:{TEMPLATE_RESOURCE}")
    all_templates: dict[str, TemplateDefinition] = {}
    source_hashes: list[str] = []
    sources: list[str] = []
    for requested in project.generation.templates:
        if requested.startswith("builtin:"):
            template_id = requested.split(":", 1)[1]
            try:
                definition = builtin_payload[template_id]
            except KeyError as exc:
                raise TemplateError(f"지원하지 않는 builtin template입니다: {requested}") from exc
            _insert_template(all_templates, requested, definition)
            source_hashes.append(definition.source_sha256)
            sources.append(definition.source)
            continue
        if not requested.startswith("file:"):
            raise TemplateError(f"template source prefix가 유효하지 않습니다: {requested}")
        path = (project.root / requested[5:]).resolve()
        text = path.read_text(encoding="utf-8")
        custom = _load_template_yaml(text, str(path))
        for template_id, definition in custom.items():
            qualified = f"file:{path.as_posix()}#{template_id}"
            _insert_template(all_templates, qualified, definition)
            # The short id is accepted only when unambiguous.
            if template_id not in all_templates:
                all_templates[template_id] = definition
        source_hashes.extend(definition.source_sha256 for definition in custom.values())
        sources.append(str(path))
    digest = hashlib.sha256("\n".join(sorted(source_hashes)).encode("utf-8")).hexdigest()
    return TemplateSet(templates=all_templates, sha256=digest, sources=tuple(dict.fromkeys(sources)))


def render_project_alias(
    project: ProjectSpec,
    recording: RecordingSpec,
    timeline: LandmarkTimeline,
    plan: AliasPlan,
    *,
    source_index: int,
    template_set: TemplateSet | None = None,
    renderer_calibration_profile: RendererCalibrationProfile | None = None,
    alias_type_confidence_profile: AliasTypeConfidenceProfile | None = None,
) -> tuple[RenderedOtoRow, RenderTrace]:
    templates = template_set or load_template_set(project)
    resolved_plan = resolve_alias_plan(project.inventory, recording, plan)
    alias_type_calibration: dict[str, object] = {"enabled": False, "applied": False}
    if project.calibration.alias_type_enabled:
        if not project.calibration.alias_type_profile:
            raise TemplateError("AliasType confidence calibration이 활성화됐지만 profile이 없습니다")
        confidence_profile = alias_type_confidence_profile or load_alias_type_confidence(
            project.root / project.calibration.alias_type_profile
        )
        calibrated_confidence, selection_key, support = predict_alias_type_confidence(
            confidence_profile,
            resolved_plan.alias_type.to_dict(),
        )
        resolved_plan = replace(
            resolved_plan,
            alias_type=replace(resolved_plan.alias_type, confidence=calibrated_confidence),
        )
        alias_type_calibration = {
            "enabled": True,
            "applied": True,
            "profile_id": confidence_profile.profile_id,
            "profile_sha256": confidence_profile.sha256,
            "training_row_report_sha256": confidence_profile.row_report_sha256,
            "training_bank_ids": sorted(confidence_profile.training_bank_ids),
            "training_root_speaker_ids": sorted(confidence_profile.training_root_speaker_ids),
            "selection_key": selection_key,
            "support": support,
            "calibrated_confidence": calibrated_confidence,
            "gold_labels_used_at_inference": False,
        }
    template = _resolve_template(resolved_plan.template_id, templates)
    expressions = dict(template.anchors)
    for name, value in plan.overrides.items():
        if name not in REQUIRED_ANCHORS:
            raise TemplateError(f"지원하지 않는 alias anchor override입니다: {name}")
        expressions[name] = _expression_text(value, f"alias {plan.alias!r} override {name}")
    _validate_anchor_set(expressions, template.id)
    phone_views = _phone_views(project, recording, timeline, plan)
    if not phone_views:
        raise TemplateError(f"AliasPlan phone_indices가 비어 있습니다: {plan.alias}")
    evaluator = phone_views[0].evaluator
    raw_values = _evaluate_anchor_graph(expressions, evaluator)
    calibrated_values = dict(raw_values)
    calibration_trace: dict[str, object] = {"enabled": False, "applied": False}
    calibration_bank_backoff = False
    if project.calibration.renderer_enabled:
        if not project.calibration.renderer_profile:
            raise TemplateError("renderer calibration이 활성화됐지만 profile이 없습니다")
        calibration_profile = renderer_calibration_profile or load_renderer_calibration(
            project.root / project.calibration.renderer_profile
        )
        selection = select_renderer_calibration(
            calibration_profile,
            template_id=template.id,
            bank_id=_optional_metadata(recording.metadata, "bank_id"),
            format_type=_optional_metadata(recording.metadata, "format_type"),
            expression_profile=_optional_metadata(recording.metadata, "expression_profile"),
            pitch_bucket=_optional_metadata(recording.metadata, "pitch_bucket"),
        )
        calibration_bank_backoff = bool(
            _optional_metadata(recording.metadata, "bank_id")
            and selection.scope not in {"bank_template", "expression_template", "pitch_template"}
        )
        calibrated_values = apply_renderer_calibration(raw_values, selection)
        override_fields = tuple(sorted(set(plan.overrides) & set(REQUIRED_ANCHORS)))
        for field in override_fields:
            calibrated_values[field] = raw_values[field]
        calibration_trace = {
            "enabled": True,
            "applied": True,
            "profile_id": calibration_profile.profile_id,
            "profile_sha256": calibration_profile.sha256,
            "training_input_sha256": calibration_profile.training_input_sha256,
            "training_bank_ids": sorted(calibration_profile.training_bank_ids),
            "training_format_types": sorted(calibration_profile.training_format_types),
            "selection_scope": selection.scope,
            "selection_key": selection.key,
            "support": selection.support,
            "shrinkage_weight": selection.shrinkage_weight,
            "residuals_ms": dict(selection.residuals_ms),
            "statistics": {
                name: dict(values)
                for name, values in selection.statistics.items()
            },
            "alias_override_precedence_fields": list(override_fields),
            "gold_timing_used_at_inference": False,
        }
    duration_ms = timeline.duration_ms
    if duration_ms is None or duration_ms <= 0.0:
        raise TemplateError("timeline.duration_ms가 필요합니다")
    final_values, repairs, repair_log = _sanitize_anchors(calibrated_values, duration_ms=duration_ms)
    soft_constraint_log = _evaluate_soft_constraints(template.soft_constraints, evaluator, final_values)
    soft_violations = tuple(item for item in soft_constraint_log if not bool(item["passed"]))
    offset_abs = final_values["offset_abs"]
    overlap_abs = final_values["overlap_abs"]
    pre_abs = final_values["pre_abs"]
    fixed_abs = final_values["fixed_abs"]
    cutoff_abs = final_values["cutoff_abs"]
    offset = offset_abs
    overlap = overlap_abs - offset_abs
    preutterance = pre_abs - offset_abs
    consonant = fixed_abs - offset_abs
    cutoff = -(cutoff_abs - offset_abs)
    valid, validation_reasons = validate_oto_line_values(
        duration_ms=duration_ms,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        preutterance=preutterance,
        overlap=overlap,
    )
    if not valid:
        raise TemplateError(f"template가 invalid OTO를 만들었습니다: {', '.join(validation_reasons)}")
    phone_confidences = [view.segment.confidence for view in phone_views if view.segment is not None]
    confidence = min([timeline.confidence, *phone_confidences]) if phone_confidences else timeline.confidence
    reasons: list[str] = [*timeline.warnings]
    if template.id in SUBTYPE_TEMPLATE_IDS and resolved_plan.warnings:
        reasons.append("alias_type_requires_review")
        reasons.extend(f"alias_type:{warning}" for warning in resolved_plan.warnings)
        confidence *= 0.85
    if template.id in SUBTYPE_TEMPLATE_IDS:
        if not bool(alias_type_calibration.get("applied")):
            reasons.append("alias_type_uncalibrated_confidence")
        elif resolved_plan.alias_type.confidence <= 0.95:
            reasons.append("alias_type_low_calibrated_confidence")
            confidence *= max(0.0, resolved_plan.alias_type.confidence)
    if repairs:
        reasons.append("template_constraint_repaired")
        reasons.extend(f"template_repair:{repair}" for repair in repairs)
        confidence *= 0.82
    if soft_violations:
        reasons.append("soft_constraint_violation")
        reasons.extend(f"soft_constraint:{item['id']}" for item in soft_violations)
        confidence *= 0.90
    if evaluator.fallback_count >= 2:
        reasons.append("fallback_landmark_multiple")
        confidence *= 0.90
    if calibration_bank_backoff:
        reasons.append("renderer_calibration_bank_out_of_range")
        confidence *= 0.90
    if resolved_plan.alias_type.acoustic_subtype.value == "special":
        reasons.append("special_alias")
        confidence *= 0.85
    risk = timeline.provenance.get("risk_probabilities")
    calibrated = bool(timeline.provenance.get("confidence_calibrated")) and isinstance(risk, Mapping)
    if not calibrated:
        reasons.append("uncalibrated_confidence")
    else:
        if float(risk.get("error_gt_80ms", 1.0)) > 0.20:
            reasons.append("calibrated_error_gt80_risk")
        if float(risk.get("error_gt_200ms", 1.0)) > 0.08:
            reasons.append("calibrated_error_gt200_risk")
        if float(risk.get("one_slot_shift", 1.0)) > 0.03:
            reasons.append("calibrated_slot_shift_risk")
    split = "clean" if calibrated and not reasons and not repairs else "review"
    row = RenderedOtoRow(
        wav_name=Path(recording.wav).name,
        alias=plan.alias,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        preutterance=preutterance,
        overlap=overlap,
        role=plan.role or template.role,
        slot_index=plan.phone_indices[0],
        confidence=float(max(0.0, min(1.0, confidence))),
        split=split,
        reasons=tuple(dict.fromkeys(reasons)),
        source_index=source_index,
        absolute={
            "offset_abs": offset_abs,
            "overlap_abs": overlap_abs,
            "pre_abs": pre_abs,
            "fixed_end_abs": fixed_abs,
            "cutoff_abs": cutoff_abs,
        },
    )
    trace = RenderTrace(
        template_id=template.id,
        expressions=expressions,
        raw_anchors=raw_values,
        final_anchors=final_values,
        repairs=repairs,
        repair_log=repair_log,
        referenced_landmarks=tuple(sorted(evaluator.referenced_landmarks)),
        missing_landmarks=tuple(sorted(evaluator.missing_landmarks)),
        fallback_count=evaluator.fallback_count,
        template_source=template.source,
        template_source_sha256=template.source_sha256,
        alias_type=resolved_plan.alias_type.to_dict(),
        template_selection_source=resolved_plan.selection_source,
        recommended_template_id=resolved_plan.recommended_template_id,
        selection_warnings=resolved_plan.warnings,
        alias_type_rule_set_sha256=resolved_plan.rule_set_sha256,
        renderer_calibration=calibration_trace,
        calibrated_anchors=calibrated_values,
        soft_constraint_log=soft_constraint_log,
        alias_type_calibration=alias_type_calibration,
    )
    return row, trace


def _load_template_yaml(text: str, source: str) -> dict[str, TemplateDefinition]:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TemplateError(f"template YAML 구문 오류: {source}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TemplateError(f"template root는 mapping이어야 합니다: {source}")
    _strict_keys(raw, {"schema_version", "template_set_id", "templates"}, {"schema_version", "templates"}, source)
    if raw["schema_version"] != 1:
        raise TemplateError(f"지원하지 않는 template schema_version입니다: {source}")
    templates = raw["templates"]
    if not isinstance(templates, Mapping) or not templates:
        raise TemplateError(f"template 정의가 없습니다: {source}")
    source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    result: dict[str, TemplateDefinition] = {}
    for raw_id, raw_definition in templates.items():
        if not isinstance(raw_id, str) or not raw_id.strip() or not isinstance(raw_definition, Mapping):
            raise TemplateError(f"template id/definition이 유효하지 않습니다: {source}")
        template_id = raw_id.strip()
        path = f"{source}#{template_id}"
        _strict_keys(raw_definition, {"role", "anchors", "soft_constraints"}, {"role", "anchors"}, path)
        role = raw_definition["role"]
        anchors = raw_definition["anchors"]
        if not isinstance(role, str) or not role.strip() or not isinstance(anchors, Mapping):
            raise TemplateError(f"template role/anchors가 유효하지 않습니다: {path}")
        expressions = {str(name): _expression_text(value, f"{path}.{name}") for name, value in anchors.items()}
        _validate_anchor_set(expressions, path)
        # Parse every expression at load time so unsafe syntax never reaches rendering.
        for name, expression in expressions.items():
            _parse_expression(expression, f"{path}.{name}")
        soft_constraints = _parse_soft_constraints(raw_definition.get("soft_constraints", []), path)
        result[template_id] = TemplateDefinition(
            template_id,
            role.strip(),
            expressions,
            soft_constraints,
            source,
            source_sha256,
        )
    return result


def _resolve_template(name: str, templates: TemplateSet) -> TemplateDefinition:
    candidates = [name]
    if name.startswith("builtin:"):
        candidates.append(name.split(":", 1)[1])
    for candidate in candidates:
        definition = templates.templates.get(candidate)
        if definition is not None:
            return definition
    raise TemplateError(f"project template set에 없는 alias template입니다: {name}")


def _parse_soft_constraints(value: object, path: str) -> tuple[SoftConstraint, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TemplateError(f"{path}.soft_constraints는 sequence여야 합니다")
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        item_path = f"{path}.soft_constraints[{index}]"
        if not isinstance(raw, Mapping):
            raise TemplateError(f"{item_path}는 mapping이어야 합니다")
        _strict_keys(raw, {"id", "expression", "min", "max"}, {"id", "expression"}, item_path)
        constraint_id = raw.get("id")
        expression = raw.get("expression")
        if not isinstance(constraint_id, str) or not constraint_id.strip():
            raise TemplateError(f"{item_path}.id가 유효하지 않습니다")
        constraint_id = constraint_id.strip()
        if constraint_id in seen:
            raise TemplateError(f"{item_path}.id가 중복됩니다: {constraint_id}")
        seen.add(constraint_id)
        expression_text = _expression_text(expression, f"{item_path}.expression")
        _parse_expression(expression_text, f"{item_path}.expression")
        minimum = _optional_finite(raw.get("min"), f"{item_path}.min")
        maximum = _optional_finite(raw.get("max"), f"{item_path}.max")
        if minimum is None and maximum is None:
            raise TemplateError(f"{item_path}에는 min 또는 max가 필요합니다")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise TemplateError(f"{item_path} min은 max보다 클 수 없습니다")
        result.append(SoftConstraint(constraint_id, expression_text, minimum, maximum))
    return tuple(result)


def _evaluate_soft_constraints(
    constraints: Sequence[SoftConstraint],
    evaluator: "_ExpressionEvaluator",
    anchors: Mapping[str, float],
) -> tuple[Mapping[str, object], ...]:
    log = []
    for constraint in constraints:
        value = evaluator.evaluate(_parse_expression(constraint.expression, constraint.id), anchors)
        if value is MISSING or not _is_number(value) or not math.isfinite(float(value)):
            raise TemplateError(f"soft constraint 결과가 finite number가 아닙니다: {constraint.id}")
        observed = float(value)
        passed = bool(
            (constraint.minimum is None or observed >= constraint.minimum)
            and (constraint.maximum is None or observed <= constraint.maximum)
        )
        log.append(
            {
                "id": constraint.id,
                "expression": constraint.expression,
                "observed": observed,
                "min": constraint.minimum,
                "max": constraint.maximum,
                "passed": passed,
                "action": "none" if passed else "review_only_no_repair",
            }
        )
    return tuple(log)


def _insert_template(
    target: dict[str, TemplateDefinition],
    name: str,
    definition: TemplateDefinition,
) -> None:
    if name in target and target[name].source_sha256 != definition.source_sha256:
        raise TemplateError(f"template id 충돌입니다: {name}")
    target[name] = definition


def _validate_anchor_set(expressions: Mapping[str, str], path: str) -> None:
    missing = sorted(set(REQUIRED_ANCHORS) - set(expressions))
    unknown = sorted(set(expressions) - set(REQUIRED_ANCHORS))
    if missing or unknown:
        raise TemplateError(f"{path}: anchor set 불일치 missing={missing}, unknown={unknown}")


class _Missing:
    pass


MISSING = _Missing()


@dataclass(frozen=True)
class _PhoneView:
    segment: PhoneSegment | None
    features: Mapping[str, object]
    landmarks: Mapping[str, float]
    evaluator: "_ExpressionEvaluator"

    def get(self, name: str) -> object:
        if self.segment is None:
            return MISSING
        if name == "start":
            return self.segment.start_ms
        if name == "end":
            return self.segment.end_ms
        if name == "features":
            return _FeatureView(self.features)
        aliases = {
            "stable_start": "vowel_stable_start",
            "stable_end": "vowel_stable_end",
        }
        label = aliases.get(name, name)
        value = self.landmarks.get(label, MISSING)
        if value is not MISSING:
            self.evaluator.referenced_landmarks.add(f"phone[{self.segment.phone_index}].{label}")
        else:
            self.evaluator.missing_landmarks.add(f"phone[{self.segment.phone_index}].{label}")
        return value


@dataclass(frozen=True)
class _FeatureView:
    values: Mapping[str, object]

    def get(self, name: str) -> object:
        return self.values.get(name, MISSING)


@dataclass(frozen=True)
class _PhoneAccessor:
    values: Sequence[_PhoneView]

    def get(self, index: int) -> _PhoneView:
        if index < 0 or index >= len(self.values):
            return _PhoneView(None, {}, {}, self.values[0].evaluator if self.values else _ExpressionEvaluator(()))
        return self.values[index]


class _ExpressionEvaluator:
    def __init__(self, phones: Sequence[_PhoneView]):
        self.phones = tuple(phones)
        self.referenced_landmarks: set[str] = set()
        self.missing_landmarks: set[str] = set()
        self.fallback_count = 0

    def evaluate(self, node: ast.AST, anchors: Mapping[str, object]) -> object:
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body, anchors)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str, bool)) and not isinstance(node.value, complex):
                return node.value
            raise TemplateError("지원하지 않는 expression 상수입니다")
        if isinstance(node, ast.Name):
            if node.id == "phone":
                return _PhoneAccessor(self.phones)
            if node.id in anchors:
                return anchors[node.id]
            if node.id in {"true", "True"}:
                return True
            if node.id in {"false", "False"}:
                return False
            raise TemplateError(f"정의되지 않은 expression name입니다: {node.id}")
        if isinstance(node, ast.Subscript):
            value = self.evaluate(node.value, anchors)
            index = self.evaluate(node.slice, anchors)
            if not isinstance(value, _PhoneAccessor) or isinstance(index, bool) or not isinstance(index, int):
                raise TemplateError("phone[정수] subscript만 허용합니다")
            return value.get(index)
        if isinstance(node, ast.Attribute):
            value = self.evaluate(node.value, anchors)
            if value is MISSING:
                return MISSING
            if isinstance(value, (_PhoneView, _FeatureView)):
                return value.get(node.attr)
            raise TemplateError("허용되지 않는 expression attribute입니다")
        if isinstance(node, ast.BinOp):
            left = self.evaluate(node.left, anchors)
            right = self.evaluate(node.right, anchors)
            if left is MISSING or right is MISSING:
                return MISSING
            if not _is_number(left) or not _is_number(right):
                raise TemplateError("산술 연산에는 숫자만 허용합니다")
            if isinstance(node.op, ast.Add):
                return float(left) + float(right)
            if isinstance(node.op, ast.Sub):
                return float(left) - float(right)
            if isinstance(node.op, ast.Mult):
                return float(left) * float(right)
            if isinstance(node.op, ast.Div):
                if float(right) == 0.0:
                    raise TemplateError("사칙 연산에서 0으로 나눌 수 없습니다")
                return float(left) / float(right)
            raise TemplateError("지원하지 않는 산술 연산입니다")
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand, anchors)
            if value is MISSING:
                return MISSING
            if not _is_number(value):
                raise TemplateError("단항 연산에는 숫자만 허용합니다")
            if isinstance(node.op, ast.USub):
                return -float(value)
            if isinstance(node.op, ast.UAdd):
                return float(value)
            raise TemplateError("지원하지 않는 단항 연산입니다")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise TemplateError("허용 함수의 위치 인자 호출만 지원합니다")
            name = node.func.id
            if name == "fallback":
                for index, argument in enumerate(node.args):
                    value = self.evaluate(argument, anchors)
                    if value is not MISSING:
                        if index > 0:
                            self.fallback_count += 1
                        return value
                return MISSING
            values = [self.evaluate(argument, anchors) for argument in node.args]
            if any(value is MISSING for value in values):
                return MISSING
            if not all(_is_number(value) for value in values):
                raise TemplateError(f"{name}에는 숫자만 허용합니다")
            numbers = [float(value) for value in values]
            if name == "min" and numbers:
                return min(numbers)
            if name == "max" and numbers:
                return max(numbers)
            if name == "clamp" and len(numbers) == 3:
                return min(max(numbers[0], numbers[1]), numbers[2])
            if name == "lerp" and len(numbers) == 3:
                return numbers[0] + (numbers[1] - numbers[0]) * numbers[2]
            if name == "duration" and len(numbers) == 2:
                return numbers[1] - numbers[0]
            if name == "ratio" and len(numbers) == 3:
                denominator = numbers[2] - numbers[1]
                if denominator == 0.0:
                    raise TemplateError("ratio 기준 구간의 길이가 0입니다")
                return (numbers[0] - numbers[1]) / denominator
            raise TemplateError(f"지원하지 않는 함수 또는 인자 수입니다: {name}")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise TemplateError("단일 비교만 허용합니다")
            left = self.evaluate(node.left, anchors)
            right = self.evaluate(node.comparators[0], anchors)
            if left is MISSING or right is MISSING:
                return False
            operator = node.ops[0]
            if isinstance(operator, ast.Eq):
                return left == right
            if isinstance(operator, ast.NotEq):
                return left != right
            if isinstance(operator, ast.Lt):
                return left < right
            if isinstance(operator, ast.LtE):
                return left <= right
            if isinstance(operator, ast.Gt):
                return left > right
            if isinstance(operator, ast.GtE):
                return left >= right
            raise TemplateError("지원하지 않는 비교 연산입니다")
        if isinstance(node, ast.IfExp):
            condition = self.evaluate(node.test, anchors)
            return self.evaluate(node.body if bool(condition) else node.orelse, anchors)
        if isinstance(node, ast.BoolOp):
            values = [bool(self.evaluate(value, anchors)) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise TemplateError("지원하지 않는 boolean 연산입니다")
        raise TemplateError(f"지원하지 않는 expression syntax입니다: {type(node).__name__}")


def _phone_views(
    project: ProjectSpec,
    recording: RecordingSpec,
    timeline: LandmarkTimeline,
    plan: AliasPlan,
) -> tuple[_PhoneView, ...]:
    by_index = {segment.phone_index: segment for segment in timeline.phone_segments}
    landmarks: dict[int, dict[str, float]] = {}
    evaluator = _ExpressionEvaluator(())
    for landmark in timeline.landmarks:
        if landmark.phone_index is not None:
            landmarks.setdefault(landmark.phone_index, {})[landmark.label] = landmark.time_ms
    views: list[_PhoneView] = []
    for phone_index in plan.phone_indices:
        if phone_index < 0 or phone_index >= len(recording.phones):
            raise TemplateError(f"AliasPlan phone index 범위를 벗어났습니다: {phone_index}")
        phone_id = recording.phones[phone_index]
        views.append(
            _PhoneView(
                segment=by_index.get(phone_index),
                features=project.inventory[phone_id].features,
                landmarks=landmarks.get(phone_index, {}),
                evaluator=evaluator,
            )
        )
    evaluator.phones = tuple(views)
    return tuple(views)


def _evaluate_anchor_graph(
    expressions: Mapping[str, str], evaluator: _ExpressionEvaluator
) -> dict[str, float]:
    parsed = {name: _parse_expression(expression, name) for name, expression in expressions.items()}
    dependencies = {
        name: _anchor_dependencies(tree, set(expressions))
        for name, tree in parsed.items()
    }
    values: dict[str, object] = {}
    pending = set(expressions)
    while pending:
        ready = sorted(name for name in pending if dependencies[name] <= set(values))
        if not ready:
            raise TemplateError(f"anchor expression 순환 참조입니다: {', '.join(sorted(pending))}")
        for name in ready:
            value = evaluator.evaluate(parsed[name], values)
            if value is MISSING:
                raise MissingAnchorError(f"fallback 없는 필수 landmark 누락: {name} = {expressions[name]}")
            if not _is_number(value) or not math.isfinite(float(value)):
                raise TemplateError(f"anchor expression 결과는 finite number여야 합니다: {name}")
            values[name] = float(value)
            pending.remove(name)
    return {name: float(values[name]) for name in REQUIRED_ANCHORS}


def _anchor_dependencies(tree: ast.Expression, anchor_names: set[str]) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in anchor_names}


def _parse_expression(expression: str, path: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise TemplateError(f"{path}: expression 구문 오류: {exc.msg}") from exc
    forbidden = (
        ast.Lambda,
        ast.Dict,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.NamedExpr,
    )
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        raise TemplateError(f"{path}: 허용되지 않는 expression syntax입니다")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in {"min", "max", "clamp", "lerp", "fallback", "duration", "ratio"}
                or node.keywords
            ):
                raise TemplateError(f"{path}: 허용되지 않는 함수 호출입니다")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise TemplateError(f"{path}: private attribute는 허용하지 않습니다")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise TemplateError(f"{path}: private name은 허용하지 않습니다")
    return tree


def _sanitize_anchors(
    values: Mapping[str, float], *, duration_ms: float
) -> tuple[dict[str, float], tuple[str, ...], tuple[Mapping[str, object], ...]]:
    repairs: list[str] = []
    latest_cutoff = max(6.0, duration_ms - 1.0)
    offset = min(max(0.0, values["offset_abs"]), latest_cutoff - 5.0)
    cutoff = min(max(offset + 5.0, values["cutoff_abs"]), latest_cutoff)
    fixed = min(max(offset, values["fixed_abs"]), cutoff)
    pre = min(max(offset, values["pre_abs"]), fixed)
    overlap = min(max(offset, values["overlap_abs"]), pre)
    final = {
        "offset_abs": offset,
        "overlap_abs": overlap,
        "pre_abs": pre,
        "fixed_abs": fixed,
        "cutoff_abs": cutoff,
    }
    for name in REQUIRED_ANCHORS:
        if abs(final[name] - values[name]) > 0.1:
            repairs.append(name)
    reason_by_field = {
        "offset_abs": "offset_file_range_clamp",
        "overlap_abs": "overlap_order_or_file_range_repair",
        "pre_abs": "preutterance_order_or_file_range_repair",
        "fixed_abs": "fixed_order_or_file_range_repair",
        "cutoff_abs": "cutoff_order_or_file_range_repair",
    }
    repair_log: list[Mapping[str, object]] = []
    for name in repairs:
        before = float(values[name])
        after = float(final[name])
        delta = abs(after - before)
        repair_log.append(
            {
                "field": name,
                "before_ms": before,
                "after_ms": after,
                "delta_ms": delta,
                "reason": reason_by_field[name],
                "severity": "large" if delta > 20.0 else "warning",
            }
        )
    return final, tuple(repairs), tuple(repair_log)


def _expression_text(value: object, path: str) -> str:
    if isinstance(value, bool):
        raise TemplateError(f"{path}: boolean은 anchor expression이 아닙니다")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise TemplateError(f"{path}: finite number가 필요합니다")
        return repr(float(value))
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping) and set(value) == {"expr"}:
        return _expression_text(value["expr"], path)
    raise TemplateError(f"{path}: 문자열 expression 또는 숫자가 필요합니다")


def _optional_finite(value: object, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TemplateError(f"{path}: finite number가 필요합니다")
    return float(value)


def _strict_keys(
    data: Mapping[str, object], allowed: set[str], required: set[str], path: str
) -> None:
    unknown = sorted(set(data) - allowed)
    missing = sorted(required - set(data))
    if unknown or missing:
        raise TemplateError(f"{path}: unknown={unknown}, missing={missing}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _optional_metadata(metadata: Mapping[str, object], key: str) -> str | None:
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None

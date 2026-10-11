from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .alias_classifier import FeatureNormalizationError, normalize_phone_features
from .domain import (
    AliasPlan,
    GuideAnchor,
    PhoneDefinition,
    PhoneStateDefinition,
    RecordingSpec,
)
from .oto_io import write_lines_atomic


PROJECT_FORMAT_VERSION = 1
INVENTORY_FORMAT_VERSION = 1
RECORDINGS_FORMAT_VERSION = 1


class ProjectValidationError(ValueError):
    """Raised when a project violates the versioned on-disk contract."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class AudioConfig:
    recursive: bool
    convert_to_mono: bool
    internal_sample_rate: int
    resampler: str


@dataclass(frozen=True)
class AlignmentConfig:
    backend: str
    device: str
    model: str | None = None
    feature_backend: str = "numpy_multiband_pitch_v2"
    evidence_model: str | None = None
    evidence_provider: str = "cuda"
    scorer_backend: str = "heuristic_fallback"
    scorer_model: str | None = None
    scorer_model_sha256: str | None = None
    allow_heuristic_fallback: bool = True
    use_ctc_prior: bool = False
    use_hsmm: bool = True
    use_prototypes: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    templates: tuple[str, ...]
    output: str
    output_layout: str
    output_encoding: str
    atomic_write: bool
    backup_existing: bool
    write_debug_timeline: bool = True
    write_textgrid: bool = False
    split_review: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool
    profile: str | None = None
    renderer_enabled: bool = False
    renderer_profile: str | None = None
    alias_type_enabled: bool = False
    alias_type_profile: str | None = None


@dataclass(frozen=True)
class ProjectSpec:
    format_version: int
    path: Path
    root: Path
    name: str
    language_hint: str
    inventory_file: str
    recordings_file: str
    inventory: Mapping[str, PhoneDefinition]
    recordings: tuple[RecordingSpec, ...]
    audio: AudioConfig
    alignment: AlignmentConfig
    generation: GenerationConfig
    calibration: CalibrationConfig
    project_sha256: str
    inventory_sha256: str
    recordings_sha256: str

    def wav_path(self, recording: RecordingSpec) -> Path:
        return _resolve_project_relative(self.root, recording.wav, f"recordings[{recording.wav}].wav")

    def output_path(self) -> Path:
        return _resolve_project_relative(self.root, self.generation.output, "generation.output")


def load_project(path: Path) -> ProjectSpec:
    project_path = path.resolve()
    project_data = _load_yaml_mapping(project_path, "project")
    _expect_version(project_data, PROJECT_FORMAT_VERSION, "project.format_version")
    _require_keys(
        project_data,
        {"format_version", "project", "inventory", "recordings", "audio", "alignment", "generation", "calibration"},
        "project",
    )

    project_meta = _mapping(project_data["project"], "project.project")
    _strict_keys(project_meta, {"name", "language_hint"}, {"name"}, "project.project")
    name = _nonempty_string(project_meta["name"], "project.project.name")
    language_hint = _nonempty_string(project_meta.get("language_hint", "mixed"), "project.project.language_hint")

    inventory_file = _nonempty_string(project_data["inventory"], "project.inventory")
    recordings_file = _nonempty_string(project_data["recordings"], "project.recordings")
    root = project_path.parent
    inventory_path = _resolve_project_relative(root, inventory_file, "project.inventory")
    recordings_path = _resolve_project_relative(root, recordings_file, "project.recordings")
    inventory_data = _load_yaml_mapping(inventory_path, "inventory")
    recordings_data = _load_yaml_mapping(recordings_path, "recordings")
    inventory = _parse_inventory(inventory_data)
    recordings = _parse_recordings(recordings_data, inventory)

    audio = _parse_audio(project_data["audio"])
    alignment = _parse_alignment(project_data["alignment"])
    if alignment.model is not None:
        model_path = _resolve_project_relative(root, alignment.model, "project.alignment.model")
        if not model_path.is_file():
            raise ProjectValidationError("project.alignment.model", f"model 파일이 없습니다: {model_path}")
    if alignment.evidence_model is not None:
        evidence_model_path = _resolve_project_relative(root, alignment.evidence_model, "project.alignment.evidence_model")
        if not evidence_model_path.is_file():
            raise ProjectValidationError(
                "project.alignment.evidence_model",
                f"neural evidence model is missing: {evidence_model_path}",
            )
    if alignment.scorer_model is not None:
        scorer_model_path = _resolve_project_relative(root, alignment.scorer_model, "project.alignment.scorer_model")
        if not scorer_model_path.is_file():
            raise ProjectValidationError(
                "project.alignment.scorer_model",
                f"trained scorer model is missing: {scorer_model_path}",
            )
    generation = _parse_generation(project_data["generation"], root)
    calibration = _parse_calibration(project_data["calibration"], root)

    for index, recording in enumerate(recordings):
        wav_path = _resolve_project_relative(root, recording.wav, f"recordings.recordings[{index}].wav")
        if not wav_path.is_file():
            raise ProjectValidationError(f"recordings.recordings[{index}].wav", f"WAV가 없습니다: {wav_path}")

    inventory_sha256 = _semantic_sha256(_inventory_semantics(inventory))
    recordings_sha256 = _semantic_sha256(_recording_semantics(recordings))
    project_sha256 = _semantic_sha256(
        {
            "format_version": PROJECT_FORMAT_VERSION,
            "name": name,
            "language_hint": language_hint,
            "audio": audio,
            "alignment": _alignment_semantics(alignment),
            "generation": generation,
            "calibration": _calibration_semantics(calibration),
            "inventory_sha256": inventory_sha256,
            "recordings_sha256": recordings_sha256,
        }
    )
    return ProjectSpec(
        format_version=PROJECT_FORMAT_VERSION,
        path=project_path,
        root=root,
        name=name,
        language_hint=language_hint,
        inventory_file=inventory_file,
        recordings_file=recordings_file,
        inventory=inventory,
        recordings=recordings,
        audio=audio,
        alignment=alignment,
        generation=generation,
        calibration=calibration,
        project_sha256=project_sha256,
        inventory_sha256=inventory_sha256,
        recordings_sha256=recordings_sha256,
    )


def load_phone_inventory(path: Path) -> tuple[Mapping[str, PhoneDefinition], str]:
    """Load one standalone phones.yaml and return its semantic SHA-256.

    Gold datasets copy phone inventories without the rest of the mutable source
    project.  Keeping this parser here makes their audit use exactly the same
    schema validation and semantic hashing as ``load_project``.
    """

    inventory_path = path.resolve()
    inventory_data = _load_yaml_mapping(inventory_path, "inventory")
    inventory = _parse_inventory(inventory_data)
    return inventory, _semantic_sha256(_inventory_semantics(inventory))


def save_project(project: ProjectSpec, destination: Path, *, force: bool = False) -> Path:
    """Write a normalized, portable project and preserve semantic hashes.

    Hashes are computed over parsed values, so YAML formatting and key ordering do
    not affect identity.  WAV files are deliberately not copied.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    project_path = destination / "project.yaml"
    phones_path = destination / "phones.yaml"
    recordings_path = destination / "recordings.yaml"
    for target in (project_path, phones_path, recordings_path):
        if target.exists() and not force:
            raise FileExistsError(target)

    project_payload = {
        "format_version": PROJECT_FORMAT_VERSION,
        "project": {"name": project.name, "language_hint": project.language_hint},
        "inventory": "phones.yaml",
        "recordings": "recordings.yaml",
        "audio": {
            "recursive": project.audio.recursive,
            "convert_to_mono": project.audio.convert_to_mono,
            "internal_sample_rate": project.audio.internal_sample_rate,
            "resampler": project.audio.resampler,
        },
        "alignment": {
            "model": project.alignment.model,
            "backend": project.alignment.backend,
            "device": project.alignment.device,
            "feature_backend": project.alignment.feature_backend,
            "evidence_model": project.alignment.evidence_model,
            "evidence_provider": project.alignment.evidence_provider,
            "scorer_backend": project.alignment.scorer_backend,
            "scorer_model": project.alignment.scorer_model,
            "scorer_model_sha256": project.alignment.scorer_model_sha256,
            "allow_heuristic_fallback": project.alignment.allow_heuristic_fallback,
            "use_ctc_prior": project.alignment.use_ctc_prior,
            "use_hsmm": project.alignment.use_hsmm,
            "use_prototypes": project.alignment.use_prototypes,
        },
        "generation": {
            "templates": list(project.generation.templates),
            "output": project.generation.output,
            "output_layout": project.generation.output_layout,
            "output_encoding": project.generation.output_encoding,
            "atomic_write": project.generation.atomic_write,
            "backup_existing": project.generation.backup_existing,
            "write_debug_timeline": project.generation.write_debug_timeline,
            "write_textgrid": project.generation.write_textgrid,
            "split_review": project.generation.split_review,
        },
        "calibration": {
            "enabled": project.calibration.enabled,
            **({"profile": project.calibration.profile} if project.calibration.profile else {}),
            "renderer_enabled": project.calibration.renderer_enabled,
            **(
                {"renderer_profile": project.calibration.renderer_profile}
                if project.calibration.renderer_profile
                else {}
            ),
            "alias_type_enabled": project.calibration.alias_type_enabled,
            **(
                {"alias_type_profile": project.calibration.alias_type_profile}
                if project.calibration.alias_type_profile
                else {}
            ),
        },
    }
    inventory_payload = {
        "format_version": INVENTORY_FORMAT_VERSION,
        "phones": {
            phone_id: {
                "display_name": phone.display_name,
                "features": dict(phone.features),
                "states": [
                    {
                        "id": state.id,
                        "acoustic_role": state.acoustic_role,
                        "min_ms": state.min_ms,
                        "max_ms": state.max_ms,
                        **({"mode_ms": state.mode_ms} if state.mode_ms is not None else {}),
                    }
                    for state in phone.states
                ],
                **({"aliases": list(phone.aliases)} if phone.aliases else {}),
                **({"prototype_ids": list(phone.prototype_ids)} if phone.prototype_ids else {}),
            }
            for phone_id, phone in project.inventory.items()
        },
    }
    recordings_payload = {
        "format_version": RECORDINGS_FORMAT_VERSION,
        "recordings": [
            {
                "wav": recording.wav,
                "phones": list(recording.phones),
                "transcript_source": recording.transcript_source,
                "label_source": recording.label_source,
                "metadata": dict(recording.metadata),
                "aliases": [
                    {
                        "alias": alias.alias,
                        "template": alias.template,
                        "phones": list(alias.phone_indices),
                        "role": alias.role,
                        **({"overrides": dict(alias.overrides)} if alias.overrides else {}),
                    }
                    for alias in recording.aliases
                ],
                **(
                    {
                        "guide_anchors": [
                            {
                                "label": anchor.label,
                                "time_ms": anchor.time_ms,
                                "phone_index": anchor.phone_index,
                                "confidence": anchor.confidence,
                                "source": anchor.source,
                            }
                            for anchor in recording.guide_anchors
                        ]
                    }
                    if recording.guide_anchors
                    else {}
                ),
            }
            for recording in project.recordings
        ],
    }
    _write_yaml(project_path, project_payload, backup=project_path.exists())
    _write_yaml(phones_path, inventory_payload, backup=phones_path.exists())
    _write_yaml(recordings_path, recordings_payload, backup=recordings_path.exists())
    return project_path


def project_summary(project: ProjectSpec) -> dict[str, object]:
    return {
        "format_version": project.format_version,
        "project": project.name,
        "root": str(project.root),
        "language_hint": project.language_hint,
        "phone_count": len(project.inventory),
        "recording_count": len(project.recordings),
        "alias_count": sum(len(recording.aliases) for recording in project.recordings),
        "alignment_backend": project.alignment.backend,
        "feature_backend": project.alignment.feature_backend,
        "project_sha256": project.project_sha256,
        "inventory_sha256": project.inventory_sha256,
        "recordings_sha256": project.recordings_sha256,
    }


def _parse_inventory(data: Mapping[str, object]) -> dict[str, PhoneDefinition]:
    _expect_version(data, INVENTORY_FORMAT_VERSION, "inventory.format_version")
    _require_keys(data, {"format_version", "phones"}, "inventory")
    phones_data = _mapping(data["phones"], "inventory.phones")
    if not phones_data:
        raise ProjectValidationError("inventory.phones", "하나 이상의 음소가 필요합니다")
    phones: dict[str, PhoneDefinition] = {}
    alias_owners: dict[str, str] = {}
    for raw_id, raw_phone in phones_data.items():
        phone_id = _nonempty_string(raw_id, "inventory.phones.<id>")
        phone_path = f"inventory.phones.{phone_id}"
        phone_data = _mapping(raw_phone, phone_path)
        _strict_keys(
            phone_data,
            {"display_name", "features", "states", "aliases", "prototype_ids"},
            {"display_name", "features", "states"},
            phone_path,
        )
        display_name = _nonempty_string(phone_data["display_name"], f"{phone_path}.display_name")
        features = dict(_mapping(phone_data["features"], f"{phone_path}.features"))
        raw_states = _sequence(phone_data["states"], f"{phone_path}.states")
        if not raw_states:
            raise ProjectValidationError(f"{phone_path}.states", "선형 상태가 하나 이상 필요합니다")
        states: list[PhoneStateDefinition] = []
        state_ids: set[str] = set()
        for state_index, raw_state in enumerate(raw_states):
            state_path = f"{phone_path}.states[{state_index}]"
            state_data = _mapping(raw_state, state_path)
            _strict_keys(
                state_data,
                {"id", "acoustic_role", "min_ms", "max_ms", "mode_ms"},
                {"id", "acoustic_role", "min_ms", "max_ms"},
                state_path,
            )
            state_id = _nonempty_string(state_data["id"], f"{state_path}.id")
            if state_id in state_ids:
                raise ProjectValidationError(f"{state_path}.id", f"중복 state id입니다: {state_id}")
            state_ids.add(state_id)
            acoustic_role = _nonempty_string(state_data["acoustic_role"], f"{state_path}.acoustic_role")
            min_ms = _finite_number(state_data["min_ms"], f"{state_path}.min_ms", minimum=0.0)
            max_ms = _finite_number(state_data["max_ms"], f"{state_path}.max_ms", minimum=0.001)
            if min_ms > max_ms:
                raise ProjectValidationError(state_path, "min_ms가 max_ms보다 큽니다")
            mode_value = state_data.get("mode_ms")
            mode_ms = None if mode_value is None else _finite_number(mode_value, f"{state_path}.mode_ms", minimum=min_ms)
            if mode_ms is not None and mode_ms > max_ms:
                raise ProjectValidationError(f"{state_path}.mode_ms", "mode_ms가 max_ms보다 큽니다")
            states.append(PhoneStateDefinition(state_id, acoustic_role, min_ms, max_ms, mode_ms))
        aliases = _string_tuple(phone_data.get("aliases", ()), f"{phone_path}.aliases")
        prototype_ids = _string_tuple(phone_data.get("prototype_ids", ()), f"{phone_path}.prototype_ids")
        for alias in aliases:
            owner = alias_owners.get(alias)
            if owner is not None and owner != phone_id:
                raise ProjectValidationError(f"{phone_path}.aliases", f"alias {alias!r}가 {owner!r}와 충돌합니다")
            alias_owners[alias] = phone_id
        phone = PhoneDefinition(
            id=phone_id,
            display_name=display_name,
            features=features,
            states=tuple(states),
            aliases=aliases,
            prototype_ids=prototype_ids,
        )
        try:
            normalize_phone_features(phone)
        except FeatureNormalizationError as exc:
            raise ProjectValidationError(f"{phone_path}.features", str(exc)) from exc
        phones[phone_id] = phone
    return phones


def _parse_recordings(
    data: Mapping[str, object], inventory: Mapping[str, PhoneDefinition]
) -> tuple[RecordingSpec, ...]:
    _expect_version(data, RECORDINGS_FORMAT_VERSION, "recordings.format_version")
    _require_keys(data, {"format_version", "recordings"}, "recordings")
    raw_recordings = _sequence(data["recordings"], "recordings.recordings")
    if not raw_recordings:
        raise ProjectValidationError("recordings.recordings", "하나 이상의 녹음이 필요합니다")
    recordings: list[RecordingSpec] = []
    seen_wavs: set[str] = set()
    for recording_index, raw_recording in enumerate(raw_recordings):
        record_path = f"recordings.recordings[{recording_index}]"
        record = _mapping(raw_recording, record_path)
        _strict_keys(
            record,
            {"wav", "phones", "aliases", "transcript_source", "label_source", "metadata", "guide_anchors"},
            {"wav", "phones", "aliases", "transcript_source", "label_source"},
            record_path,
        )
        wav = _portable_relative_path(record["wav"], f"{record_path}.wav")
        if wav.casefold() in seen_wavs:
            raise ProjectValidationError(f"{record_path}.wav", f"중복 WAV입니다: {wav}")
        seen_wavs.add(wav.casefold())
        phones = _string_tuple(record["phones"], f"{record_path}.phones")
        if not phones:
            raise ProjectValidationError(f"{record_path}.phones", "음소열이 비어 있습니다")
        for phone_index, phone_id in enumerate(phones):
            if phone_id not in inventory:
                raise ProjectValidationError(
                    f"{record_path}.phones[{phone_index}]", f"inventory에 없는 음소입니다: {phone_id}"
                )
        aliases = _parse_alias_plans(record["aliases"], phones, record_path)
        transcript_source = _nonempty_string(record["transcript_source"], f"{record_path}.transcript_source")
        label_source = _nonempty_string(record["label_source"], f"{record_path}.label_source")
        metadata = dict(_mapping(record.get("metadata", {}), f"{record_path}.metadata"))
        guide_anchors = _parse_guide_anchors(record.get("guide_anchors", ()), len(phones), record_path)
        recordings.append(
            RecordingSpec(
                wav=wav,
                phones=phones,
                aliases=aliases,
                transcript_source=transcript_source,
                label_source=label_source,
                metadata=metadata,
                guide_anchors=guide_anchors,
            )
        )
    return tuple(recordings)


def _parse_alias_plans(raw: object, phones: tuple[str, ...], record_path: str) -> tuple[AliasPlan, ...]:
    raw_aliases = _sequence(raw, f"{record_path}.aliases")
    if not raw_aliases:
        raise ProjectValidationError(f"{record_path}.aliases", "하나 이상의 alias plan이 필요합니다")
    plans: list[AliasPlan] = []
    occurrences: dict[str, int] = {}
    for alias_index, raw_alias in enumerate(raw_aliases):
        alias_path = f"{record_path}.aliases[{alias_index}]"
        alias_data = _mapping(raw_alias, alias_path)
        _strict_keys(alias_data, {"alias", "template", "phones", "role", "overrides"}, {"alias", "template", "phones"}, alias_path)
        alias = _nonempty_string(alias_data["alias"], f"{alias_path}.alias")
        template = _nonempty_string(alias_data["template"], f"{alias_path}.template")
        raw_indices = _sequence(alias_data["phones"], f"{alias_path}.phones")
        if not raw_indices:
            raise ProjectValidationError(f"{alias_path}.phones", "phone index가 비어 있습니다")
        indices: list[int] = []
        for position, raw_index in enumerate(raw_indices):
            index = _integer(raw_index, f"{alias_path}.phones[{position}]", minimum=0)
            if index >= len(phones):
                raise ProjectValidationError(f"{alias_path}.phones[{position}]", "recording 음소열 범위를 벗어났습니다")
            indices.append(index)
        role = _nonempty_string(alias_data.get("role", _template_role(template)), f"{alias_path}.role")
        overrides = dict(_mapping(alias_data.get("overrides", {}), f"{alias_path}.overrides"))
        occurrences[alias] = occurrences.get(alias, 0) + 1
        plans.append(AliasPlan(alias, tuple(indices), template, role, overrides))
    return tuple(plans)


def _parse_guide_anchors(raw: object, phone_count: int, record_path: str) -> tuple[GuideAnchor, ...]:
    anchors: list[GuideAnchor] = []
    for anchor_index, raw_anchor in enumerate(_sequence(raw, f"{record_path}.guide_anchors")):
        anchor_path = f"{record_path}.guide_anchors[{anchor_index}]"
        anchor = _mapping(raw_anchor, anchor_path)
        _strict_keys(anchor, {"label", "time_ms", "phone_index", "confidence", "source"}, {"label", "time_ms"}, anchor_path)
        phone_index_value = anchor.get("phone_index")
        phone_index = None if phone_index_value is None else _integer(phone_index_value, f"{anchor_path}.phone_index", minimum=0)
        if phone_index is not None and phone_index >= phone_count:
            raise ProjectValidationError(f"{anchor_path}.phone_index", "recording 음소열 범위를 벗어났습니다")
        anchors.append(
            GuideAnchor(
                label=_nonempty_string(anchor["label"], f"{anchor_path}.label"),
                time_ms=_finite_number(anchor["time_ms"], f"{anchor_path}.time_ms", minimum=0.0),
                phone_index=phone_index,
                confidence=_finite_number(anchor.get("confidence", 1.0), f"{anchor_path}.confidence", minimum=0.0, maximum=1.0),
                source=_nonempty_string(anchor.get("source", "project_manifest"), f"{anchor_path}.source"),
            )
        )
    return tuple(anchors)


def _parse_audio(raw: object) -> AudioConfig:
    path = "project.audio"
    data = _mapping(raw, path)
    _strict_keys(data, {"recursive", "convert_to_mono", "internal_sample_rate", "resampler"}, {"recursive", "convert_to_mono", "internal_sample_rate", "resampler"}, path)
    sample_rate = _integer(data["internal_sample_rate"], f"{path}.internal_sample_rate", minimum=8000)
    if sample_rate > 192000:
        raise ProjectValidationError(f"{path}.internal_sample_rate", "192000 이하이어야 합니다")
    return AudioConfig(
        recursive=_boolean(data["recursive"], f"{path}.recursive"),
        convert_to_mono=_boolean(data["convert_to_mono"], f"{path}.convert_to_mono"),
        internal_sample_rate=sample_rate,
        resampler=_nonempty_string(data["resampler"], f"{path}.resampler"),
    )


def _parse_alignment(raw: object) -> AlignmentConfig:
    path = "project.alignment"
    data = _mapping(raw, path)
    _strict_keys(
        data,
        {
            "model", "backend", "device", "feature_backend", "evidence_model", "evidence_provider",
            "scorer_backend", "scorer_model", "scorer_model_sha256", "allow_heuristic_fallback",
            "use_ctc_prior", "use_hsmm", "use_prototypes",
        },
        {"backend", "device"},
        path,
    )
    backend = _choice(
        data["backend"],
        {"acoustic_heuristic_v1", "hsmm", "hybrid", "posterior_hsmm", "legacy_hsmm_compat"},
        f"{path}.backend",
    )
    device = _choice(data["device"], {"cpu", "cuda", "auto"}, f"{path}.device")
    feature_backend = _choice(
        data.get("feature_backend", "numpy_multiband_pitch_v2"),
        {"numpy_spectral_hsmm_v1", "numpy_multiband_pitch_v2", "numpy_multiband_pitch_v3", "numpy_multiband_pitch_padding_v1", "world_fusion_v2", "onnx_neural_evidence_v1"},
        f"{path}.feature_backend",
    )
    model_value = data.get("model")
    model = None if model_value is None else _nonempty_string(model_value, f"{path}.model")
    evidence_model_value = data.get("evidence_model")
    evidence_model = None if evidence_model_value is None else _nonempty_string(evidence_model_value, f"{path}.evidence_model")
    scorer_backend = _choice(
        data.get("scorer_backend", "heuristic_fallback"),
        {"heuristic_fallback", "trained_subphone_v1"},
        f"{path}.scorer_backend",
    )
    scorer_model_value = data.get("scorer_model")
    scorer_model = None if scorer_model_value is None else _nonempty_string(scorer_model_value, f"{path}.scorer_model")
    scorer_model_sha256_value = data.get("scorer_model_sha256")
    scorer_model_sha256 = (
        None
        if scorer_model_sha256_value is None
        else _sha256_hex(scorer_model_sha256_value, f"{path}.scorer_model_sha256")
    )
    evidence_provider = _choice(data.get("evidence_provider", "cuda"), {"cuda", "cpu", "auto"}, f"{path}.evidence_provider")
    config = AlignmentConfig(
        backend=backend,
        device=device,
        model=model,
        feature_backend=feature_backend,
        evidence_model=evidence_model,
        evidence_provider=evidence_provider,
        scorer_backend=scorer_backend,
        scorer_model=scorer_model,
        scorer_model_sha256=scorer_model_sha256,
        allow_heuristic_fallback=_boolean(data.get("allow_heuristic_fallback", True), f"{path}.allow_heuristic_fallback"),
        use_ctc_prior=_boolean(data.get("use_ctc_prior", False), f"{path}.use_ctc_prior"),
        use_hsmm=_boolean(data.get("use_hsmm", True), f"{path}.use_hsmm"),
        use_prototypes=_boolean(data.get("use_prototypes", False), f"{path}.use_prototypes"),
    )
    if backend in {"hsmm", "hybrid", "posterior_hsmm", "legacy_hsmm_compat"} and not config.use_hsmm:
        raise ProjectValidationError(f"{path}.use_hsmm", f"backend={backend}이면 true여야 합니다")
    if backend == "posterior_hsmm" and model is None:
        raise ProjectValidationError(f"{path}.model", "backend=posterior_hsmm이면 학습된 model 경로가 필요합니다")
    if feature_backend == "onnx_neural_evidence_v1" and evidence_model is None:
        raise ProjectValidationError(f"{path}.evidence_model", "onnx_neural_evidence_v1에는 ONNX evidence_model 경로가 필요합니다")
    if scorer_backend == "trained_subphone_v1" and (scorer_model is None or scorer_model_sha256 is None):
        raise ProjectValidationError(
            f"{path}.scorer_model",
            "trained_subphone_v1 requires scorer_model and scorer_model_sha256",
        )
    if scorer_backend == "heuristic_fallback" and (
        scorer_model is not None or scorer_model_sha256 is not None
    ):
        raise ProjectValidationError(
            f"{path}.scorer_backend",
            "scorer_model fields require scorer_backend=trained_subphone_v1",
        )
    if config.use_prototypes:
        raise ProjectValidationError(
            f"{path}.use_prototypes",
            "prototype backend는 아직 없으므로 false여야 합니다",
        )
    if config.use_ctc_prior and backend == "acoustic_heuristic_v1":
        raise ProjectValidationError(
            f"{path}.use_ctc_prior",
            "CTC prior는 HSMM 계열 backend에서만 사용할 수 있습니다",
        )
    return config


def _parse_generation(raw: object, root: Path) -> GenerationConfig:
    path = "project.generation"
    data = _mapping(raw, path)
    required = {"templates", "output", "output_layout", "output_encoding", "atomic_write", "backup_existing"}
    allowed = required | {"write_debug_timeline", "write_textgrid", "split_review"}
    _strict_keys(data, allowed, required, path)
    templates = _string_tuple(data["templates"], f"{path}.templates")
    if not templates:
        raise ProjectValidationError(f"{path}.templates", "하나 이상의 template이 필요합니다")
    for index, template in enumerate(templates):
        if template.startswith("builtin:"):
            continue
        if not template.startswith("file:"):
            raise ProjectValidationError(f"{path}.templates[{index}]", "builtin: 또는 file: prefix가 필요합니다")
        template_path = _resolve_project_relative(root, template[5:], f"{path}.templates[{index}]")
        if not template_path.is_file():
            raise ProjectValidationError(f"{path}.templates[{index}]", f"template 파일이 없습니다: {template_path}")
    output = _portable_relative_path(data["output"], f"{path}.output")
    _resolve_project_relative(root, output, f"{path}.output")
    return GenerationConfig(
        templates=templates,
        output=output,
        output_layout=_choice(data["output_layout"], {"preserve_source_directories", "single_root"}, f"{path}.output_layout"),
        output_encoding=_choice(data["output_encoding"], {"preserve", "utf-8", "cp932", "shift_jis", "cp949", "euc-kr"}, f"{path}.output_encoding"),
        atomic_write=_boolean(data["atomic_write"], f"{path}.atomic_write"),
        backup_existing=_boolean(data["backup_existing"], f"{path}.backup_existing"),
        write_debug_timeline=_boolean(data.get("write_debug_timeline", True), f"{path}.write_debug_timeline"),
        write_textgrid=_boolean(data.get("write_textgrid", False), f"{path}.write_textgrid"),
        split_review=_boolean(data.get("split_review", True), f"{path}.split_review"),
    )


def _parse_calibration(raw: object, root: Path) -> CalibrationConfig:
    path = "project.calibration"
    data = _mapping(raw, path)
    _strict_keys(
        data,
        {
            "enabled",
            "profile",
            "renderer_enabled",
            "renderer_profile",
            "alias_type_enabled",
            "alias_type_profile",
        },
        {"enabled"},
        path,
    )
    enabled = _boolean(data["enabled"], f"{path}.enabled")
    profile_value = data.get("profile")
    profile = None if profile_value is None else _portable_relative_path(profile_value, f"{path}.profile")
    if enabled and profile is None:
        raise ProjectValidationError(f"{path}.profile", "calibration.enabled=true이면 profile이 필요합니다")
    if profile is not None:
        profile_path = _resolve_project_relative(root, profile, f"{path}.profile")
        if enabled and not profile_path.is_file():
            raise ProjectValidationError(f"{path}.profile", f"calibration profile이 없습니다: {profile_path}")
    renderer_enabled = _boolean(data.get("renderer_enabled", False), f"{path}.renderer_enabled")
    renderer_profile_value = data.get("renderer_profile")
    renderer_profile = (
        None
        if renderer_profile_value is None
        else _portable_relative_path(renderer_profile_value, f"{path}.renderer_profile")
    )
    if renderer_enabled and renderer_profile is None:
        raise ProjectValidationError(
            f"{path}.renderer_profile",
            "calibration.renderer_enabled=true이면 renderer_profile이 필요합니다",
        )
    if renderer_profile is not None:
        renderer_profile_path = _resolve_project_relative(
            root, renderer_profile, f"{path}.renderer_profile"
        )
        if renderer_enabled and not renderer_profile_path.is_file():
            raise ProjectValidationError(
                f"{path}.renderer_profile",
                f"renderer calibration profile이 없습니다: {renderer_profile_path}",
            )
    alias_type_enabled = _boolean(data.get("alias_type_enabled", False), f"{path}.alias_type_enabled")
    alias_type_profile_value = data.get("alias_type_profile")
    alias_type_profile = (
        None
        if alias_type_profile_value is None
        else _portable_relative_path(alias_type_profile_value, f"{path}.alias_type_profile")
    )
    if alias_type_enabled and alias_type_profile is None:
        raise ProjectValidationError(
            f"{path}.alias_type_profile",
            "calibration.alias_type_enabled=true이면 alias_type_profile이 필요합니다",
        )
    if alias_type_profile is not None:
        alias_type_profile_path = _resolve_project_relative(
            root,
            alias_type_profile,
            f"{path}.alias_type_profile",
        )
        if alias_type_enabled and not alias_type_profile_path.is_file():
            raise ProjectValidationError(
                f"{path}.alias_type_profile",
                f"AliasType confidence profile이 없습니다: {alias_type_profile_path}",
            )
    return CalibrationConfig(
        enabled=enabled,
        profile=profile,
        renderer_enabled=renderer_enabled,
        renderer_profile=renderer_profile,
        alias_type_enabled=alias_type_enabled,
        alias_type_profile=alias_type_profile,
    )


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} YAML이 없습니다: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ProjectValidationError(label, "프로젝트 YAML은 UTF-8이어야 합니다") from exc
    except yaml.YAMLError as exc:
        raise ProjectValidationError(label, f"YAML 구문 오류: {exc}") from exc
    return _mapping(payload, label)


def _write_yaml(path: Path, payload: Mapping[str, object], *, backup: bool) -> None:
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
    write_lines_atomic(path, text.splitlines(), encoding="utf-8", newline="\n", backup_existing=backup)


def _semantic_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.__dict__,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _alignment_semantics(alignment: AlignmentConfig) -> Mapping[str, object]:
    """Return the hash-bound alignment contract without rewriting legacy IDs.

    Neural evidence and ``scorer_*`` were added after v1 review bundles had
    already sealed their source-project digest.  Their unconfigured defaults
    are exactly the prior runtime behaviour, so they are deliberately omitted
    from semantic identity.  Any opt-in model setting remains hash-bound.
    """
    payload: dict[str, object] = {
        "backend": alignment.backend,
        "device": alignment.device,
        "model": alignment.model,
        "feature_backend": alignment.feature_backend,
        "use_ctc_prior": alignment.use_ctc_prior,
        "use_hsmm": alignment.use_hsmm,
        "use_prototypes": alignment.use_prototypes,
    }
    if (
        alignment.scorer_backend != "heuristic_fallback"
        or alignment.scorer_model is not None
        or alignment.scorer_model_sha256 is not None
        or not alignment.allow_heuristic_fallback
    ):
        payload.update(
            {
                "scorer_backend": alignment.scorer_backend,
                "scorer_model": alignment.scorer_model,
                "scorer_model_sha256": alignment.scorer_model_sha256,
                "allow_heuristic_fallback": alignment.allow_heuristic_fallback,
            }
        )
    if (
        alignment.feature_backend == "onnx_neural_evidence_v1"
        or alignment.evidence_model is not None
        or alignment.evidence_provider != "cuda"
    ):
        payload.update(
            {
                "evidence_model": alignment.evidence_model,
                "evidence_provider": alignment.evidence_provider,
            }
        )
    return payload


def _calibration_semantics(calibration: CalibrationConfig) -> Mapping[str, object]:
    """Keep pre-renderer-calibration projects on their sealed v1 identity."""
    payload: dict[str, object] = {
        "enabled": calibration.enabled,
        "profile": calibration.profile,
    }
    if (
        calibration.renderer_enabled
        or calibration.renderer_profile is not None
        or calibration.alias_type_enabled
        or calibration.alias_type_profile is not None
    ):
        payload.update(
            {
                "renderer_enabled": calibration.renderer_enabled,
                "renderer_profile": calibration.renderer_profile,
                "alias_type_enabled": calibration.alias_type_enabled,
                "alias_type_profile": calibration.alias_type_profile,
            }
        )
    return payload


def _inventory_semantics(inventory: Mapping[str, PhoneDefinition]) -> dict[str, object]:
    return {
        phone_id: {
            "display_name": phone.display_name,
            "features": dict(phone.features),
            "aliases": list(phone.aliases),
            "prototype_ids": list(phone.prototype_ids),
            "states": [
                {
                    "id": state.id,
                    "acoustic_role": state.acoustic_role,
                    "min_ms": state.min_ms,
                    "max_ms": state.max_ms,
                    "mode_ms": state.mode_ms,
                }
                for state in phone.states
            ],
        }
        for phone_id, phone in inventory.items()
    }


def _recording_semantics(recordings: Sequence[RecordingSpec]) -> list[object]:
    return [
        {
            "wav": recording.wav,
            "phones": list(recording.phones),
            "transcript_source": recording.transcript_source,
            "label_source": recording.label_source,
            "metadata": dict(recording.metadata),
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
            "guide_anchors": [anchor.__dict__ for anchor in recording.guide_anchors],
        }
        for recording in recordings
    ]


def _template_role(template: str) -> str:
    value = template.rsplit(":", 1)[-1].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    value = value.rsplit(".", 1)[0].strip().casefold()
    structural_roles = {
        "cv_hard": "cv",
        "cv_soft": "cv",
        "cv_softer_hard": "cv",
        "cv_glide": "cv",
        "cv_cluster": "cv",
        "vcv_cluster": "vcv",
        "vc_hard": "vc",
        "vc_soft_voiced": "vc",
        "vc_soft_unvoiced": "vc",
    }
    return structural_roles.get(value, value or "custom")


def _resolve_project_relative(root: Path, value: str, path: str) -> Path:
    relative = Path(_portable_relative_path(value, path))
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ProjectValidationError(path, "프로젝트 루트 밖의 경로는 허용하지 않습니다") from exc
    return resolved


def _portable_relative_path(value: object, path: str) -> str:
    text = _nonempty_string(value, path).replace("\\", "/")
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ProjectValidationError(path, "프로젝트 내부 상대 경로만 허용합니다")
    return candidate.as_posix()


def _expect_version(data: Mapping[str, object], expected: int, path: str) -> None:
    if "format_version" not in data:
        raise ProjectValidationError(path, "필수 필드가 없습니다")
    version = _integer(data["format_version"], path, minimum=1)
    if version != expected:
        raise ProjectValidationError(path, f"지원 버전은 {expected}입니다: {version}")


def _require_keys(data: Mapping[str, object], exact: set[str], path: str) -> None:
    _strict_keys(data, exact, exact, path)


def _strict_keys(data: Mapping[str, object], allowed: set[str], required: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    missing = sorted(required - set(data))
    if unknown:
        raise ProjectValidationError(path, f"지원하지 않는 필드: {', '.join(unknown)}")
    if missing:
        raise ProjectValidationError(path, f"필수 필드 누락: {', '.join(missing)}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProjectValidationError(path, "mapping이어야 합니다")
    if any(not isinstance(key, str) for key in value):
        raise ProjectValidationError(path, "모든 key는 문자열이어야 합니다")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProjectValidationError(path, "목록이어야 합니다")
    return value


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    return tuple(_nonempty_string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path)))


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectValidationError(path, "비어 있지 않은 문자열이어야 합니다")
    return value.strip()


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ProjectValidationError(path, "boolean이어야 합니다")
    return value


def _sha256_hex(value: object, path: str) -> str:
    digest = _nonempty_string(value, path).casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ProjectValidationError(path, "must be a SHA-256 hex digest")
    return digest


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectValidationError(path, "정수여야 합니다")
    if value < minimum:
        raise ProjectValidationError(path, f"{minimum} 이상이어야 합니다")
    return value


def _finite_number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectValidationError(path, "숫자여야 합니다")
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise ProjectValidationError(path, "finite number여야 합니다")
    if minimum is not None and number < minimum:
        raise ProjectValidationError(path, f"{minimum} 이상이어야 합니다")
    if maximum is not None and number > maximum:
        raise ProjectValidationError(path, f"{maximum} 이하이어야 합니다")
    return number


def _choice(value: object, choices: set[str], path: str) -> str:
    text = _nonempty_string(value, path)
    if text not in choices:
        raise ProjectValidationError(path, f"지원값: {', '.join(sorted(choices))}")
    return text

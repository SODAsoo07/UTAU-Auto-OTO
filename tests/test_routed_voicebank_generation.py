from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.domain import SourceOtoRow
from universal_auto_oto.audio import file_sha256
from universal_auto_oto.generator import GenerationOptions, generate_voicebank
from universal_auto_oto.voicebank_project_adapter import (
    VoicebankReviewRequired,
    build_routed_voicebank_project,
    parse_filename_phone_slots,
)


def _write_cv(path: Path, *, sample_rate: int = 16000) -> None:
    silence = np.zeros(int(0.10 * sample_rate), dtype=np.float32)
    consonant = (0.08 * np.random.default_rng(19).standard_normal(int(0.08 * sample_rate))).astype(np.float32)
    times = np.arange(int(0.32 * sample_rate), dtype=np.float64) / sample_rate
    vowel = (0.24 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    tail = np.zeros(int(0.06 * sample_rate), dtype=np.float32)
    sf.write(path, np.concatenate((silence, consonant, vowel, tail)), sample_rate, subtype="PCM_16")


def _base_oto(path: Path, timing: tuple[float, float, float, float, float], *, alias: str = "ka") -> None:
    offset, consonant, cutoff, preutterance, overlap = timing
    line = f"ka.wav={alias},{offset},{consonant},{cutoff},{preutterance},{overlap}\r\n"
    path.write_bytes(line.encode("cp932" if alias == "か" else "utf-8"))


def test_routed_hsmm_uses_common_runtime_and_preserves_source_encoding(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")
    base = bank / "oto.ini"
    _base_oto(base, (900.0, 1.0, -2.0, 3.0, 4.0), alias="か")

    output = tmp_path / "routed"
    report = generate_voicebank(
        GenerationOptions(
            bank=bank,
            output=output,
            base_oto=base,
            backend="routed_hsmm",
            format_hint="cv",
            language_hint="mixed",
        )
    )

    assert report["generation_backend"] == "routed_hsmm"
    assert report["alignment_backend"] == "legacy_hsmm_compat"
    assert report["generated_rows"] == 1
    assert (output / "oto.ini").read_text(encoding="cp932").startswith("ka.wav=か,")
    persisted = json.loads((output / "generation_report.json").read_text(encoding="utf-8"))
    provenance = persisted["provenance"]
    assert provenance["generation_backend"] == "routed_hsmm"
    assert provenance["alignment_profiles"][0]["profile_sha256"]
    assert provenance["alignment_model"] is None
    assert provenance["alignment_model_sha256s"] == []
    assert provenance["input_hashes"]["wavs"][0]["wav_sha256"]
    assert provenance["voicebank_input_hashes"]["base_oto_sha256"]
    assert provenance["base_oto_timing_reused"] is False
    assert len(report["timelines"][0]["timeline_sha256"]) == 64
    row = json.loads((output / "row_report.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(row["timeline_source_sha256"]) == 64


def test_routed_hsmm_can_replace_bank_oto_with_explicit_toggle_and_backup(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")
    oto = bank / "oto.ini"
    _base_oto(oto, (900.0, 1.0, -2.0, 3.0, 4.0))
    original = oto.read_bytes()

    report = generate_voicebank(
        GenerationOptions(
            bank=bank,
            output=bank,
            base_oto=oto,
            backend="routed_hsmm",
            format_hint="cv",
            language_hint="mixed",
            force=True,
            allow_in_place_output=True,
        )
    )

    backups = list(bank.glob("oto.ini.*.bak"))
    assert report["in_place_output"] is True
    assert oto.read_bytes() != original
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_routed_development_alignment_profile_is_explicit_and_recorded(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")
    output = tmp_path / "profiled"

    report = generate_voicebank(
        GenerationOptions(
            bank=bank,
            output=output,
            backend="routed_hsmm",
            format_hint="cv",
            development_alignment_profile="occurrence_v1",
        )
    )

    assert report["provenance"]["development_alignment_profile"] == "occurrence_v1"
    timeline_path = Path(report["timelines"][0]["timeline"])
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert report["feature_backend"] == "numpy_multiband_pitch_v2"
    assert report["provenance"]["feature_backend"] == "numpy_multiband_pitch_v2"
    assert timeline["provenance"]["feature_backend"] == "numpy_multiband_pitch_v2"
    assert timeline["provenance"]["occurrence_prior_weight_scale"] == 0.75
    assert timeline["provenance"]["occurrence_rhythm_tracking"] == "adaptive_interval"
    assert timeline["provenance"]["internal_gap_policy"] == "prior_strict_30ms"
    assert report["provenance"]["occurrence_ranker_execution"] == {
        "recording_count": 1,
        "requested_recording_count": 0,
        "applied_recording_count": 0,
        "applied_boundary_count": 0,
        "execution_providers": [],
        "model_sha256s": [],
    }
    assert report["provenance"]["frame_scorer_execution"] == {
        "recording_count": 1,
        "providers": ["heuristic_frame_score_v1"],
        "fallback_recording_count": 1,
        "fallback_reasons": ["no_trained_frame_score_provider_configured"],
        "neural_evidence_recording_count": 0,
        "neural_evidence_model_sha256s": [],
        "transition_scorers": [],
    }

    with pytest.raises(ValueError, match="require backend=routed_hsmm"):
        generate_voicebank(
            GenerationOptions(
                bank=bank,
                output=tmp_path / "invalid",
                development_alignment_profile="occurrence_v1",
            )
        )


def test_routed_occurrence_ranker_is_forwarded_and_hash_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")
    model = tmp_path / "ranker.onnx"
    model.write_bytes(b"test occurrence ranker")
    captured: dict[str, object] = {}

    def fake_generate_project(project, **kwargs):
        captured["project"] = project
        captured["alignment_options"] = kwargs["alignment_options"]
        captured["provenance_context"] = kwargs["provenance_context"]
        return {"provenance": dict(kwargs["provenance_context"]), "timelines": []}

    monkeypatch.setattr("universal_auto_oto.generator.generate_project", fake_generate_project)
    report = generate_voicebank(
        GenerationOptions(
            bank=bank,
            output=tmp_path / "ranked",
            backend="routed_hsmm",
            format_hint="cv",
            occurrence_ranker_model=model,
            occurrence_ranker_strength=0.06,
        )
    )

    forwarded = captured["alignment_options"]
    assert isinstance(forwarded, dict)
    assert forwarded["occurrence_ranker_model"] == model.resolve()
    assert forwarded["occurrence_ranker_strength"] == 0.06
    assert captured["project"].alignment.evidence_provider == "auto"
    ranker = report["provenance"]["occurrence_ranker"]
    assert ranker == {
        "requested": True,
        "model_path": str(model.resolve()),
        "model_sha256": file_sha256(model),
        "strength": 0.06,
        "provider": "auto",
    }


def test_occurrence_ranker_requires_routed_backend_model_and_bounded_strength(tmp_path: Path) -> None:
    model = tmp_path / "ranker.onnx"
    model.write_bytes(b"test occurrence ranker")

    with pytest.raises(ValueError, match="requires backend=routed_hsmm"):
        generate_voicebank(
            GenerationOptions(bank=tmp_path, output=tmp_path / "legacy", occurrence_ranker_model=model, occurrence_ranker_strength=0.06)
        )
    with pytest.raises(ValueError, match="model is required"):
        generate_voicebank(
            GenerationOptions(bank=tmp_path, output=tmp_path / "ranked", backend="routed_hsmm", occurrence_ranker_strength=0.06)
        )
    with pytest.raises(ValueError, match="between 0 and 0.35"):
        generate_voicebank(
            GenerationOptions(bank=tmp_path, output=tmp_path / "strong", backend="routed_hsmm", occurrence_ranker_model=model, occurrence_ranker_strength=0.36)
        )
    with pytest.raises(ValueError, match="cuda, cpu, or auto"):
        generate_voicebank(
            GenerationOptions(
                bank=tmp_path,
                output=tmp_path / "provider",
                backend="routed_hsmm",
                occurrence_ranker_model=model,
                occurrence_ranker_strength=0.06,
                occurrence_ranker_provider="tpu",
            )
        )


def test_routed_parser_tags_repeated_phone_occurrences_deterministically(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    wav = bank / "ka_ka.wav"
    _write_cv(wav)

    first = build_routed_voicebank_project(
        bank=bank,
        wavs=[wav],
        base_rows_by_wav={},
        format_hint="cv",
        language_hint="mixed",
        recursive=False,
        output_encoding="utf-8",
    )
    second = build_routed_voicebank_project(
        bank=bank,
        wavs=[wav],
        base_rows_by_wav={},
        format_hint="cv",
        language_hint="mixed",
        recursive=False,
        output_encoding="utf-8",
    )
    tokens = first.project.recordings[0].metadata["phone_tokens"]

    assert [token["occurrence_id"] for token in tokens] == ["k#0001", "a#0001", "k#0002", "a#0002"]
    assert first.project.recordings[0].metadata["bank_id"] == "bank"
    assert first.project.recordings[0].metadata["split"] == "train"
    assert first.project.recordings[0].metadata["root_speaker_id"] == ""
    assert first.project.recordings_sha256 == second.project.recordings_sha256
    assert first.project.project_sha256 == second.project.project_sha256


@pytest.mark.parametrize(
    ("wav_name", "expected_terminal_slot"),
    (
        ("aNGH.wav", ("a", "ng", "h")),
        ("geuK.wav", ("g", "eu", "k")),
        ("niN.wav", ("n", "i", "n")),
        ("ruL.wav", ("r", "u", "l")),
    ),
)
def test_routed_parser_accepts_only_documented_korean_vcv_terminal_codas(
    wav_name: str,
    expected_terminal_slot: tuple[str, ...],
) -> None:
    assert parse_filename_phone_slots(
        wav_name,
        format_hint="vcv",
        language_hint="korean",
    ) == (expected_terminal_slot,)

    with pytest.raises(VoicebankReviewRequired, match="review_required:unparsed_filename_token"):
        parse_filename_phone_slots(
            f"{wav_name.removesuffix('.wav')}_a.wav",
            format_hint="vcv",
            language_hint="korean",
        )


@pytest.mark.parametrize("variant", ("br1", "br2"))
def test_routed_parser_normalizes_numbered_breath_variants(variant: str) -> None:
    assert parse_filename_phone_slots(
        f"{variant}_a.wav",
        format_hint="cvvc",
        language_hint="korean",
    ) == (("br",), ("a",))


def test_routed_parser_combines_explicit_korean_cvvc_leading_n_coda() -> None:
    assert parse_filename_phone_slots(
        "n_ba_bi_bu.wav",
        format_hint="cvvc",
        language_hint="korean",
    ) == (("n", "b", "a"), ("b", "i"), ("b", "u"))


def test_routed_leading_n_coda_alias_binds_to_the_initial_n_b_boundary(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    wav_name = "n_ba_bi_bu.wav"
    wav = bank / wav_name
    _write_cv(wav)
    rows = [
        SourceOtoRow(wav_name, alias, index)
        for index, alias in enumerate(("baE4", "biE4", "i bE4", "a bE4", "n bE4"))
    ]

    build = build_routed_voicebank_project(
        bank=bank,
        wavs=[wav],
        base_rows_by_wav={wav_name: rows},
        format_hint="cvvc",
        language_hint="korean",
        recursive=False,
        output_encoding="utf-8",
    )
    aliases = {alias.alias: alias for alias in build.project.recordings[0].aliases}

    assert aliases["baE4"].phone_indices == (1, 2)
    assert aliases["n bE4"].phone_indices == (0, 1)
    assert aliases["n bE4"].role == "vc"


def test_routed_parser_expands_korean_compact_duplicate_cvvc_occurrences() -> None:
    assert parse_filename_phone_slots(
        "ddadda_ddiddi.wav",
        format_hint="cvvc",
        language_hint="korean",
    ) == (("dd", "a"), ("dd", "a"), ("dd", "i"), ("dd", "i"))


def test_routed_compact_korean_cvvc_binds_cv_and_vc_to_separate_occurrences(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    wav_name = "ddadda_ddiddi.wav"
    wav = bank / wav_name
    _write_cv(wav)
    source_rows = [
        SourceOtoRow(wav_name, alias, index)
        for index, alias in enumerate(("dda", "at", "ddi", "it"))
    ]

    build = build_routed_voicebank_project(
        bank=bank,
        wavs=[wav],
        base_rows_by_wav={wav_name: source_rows},
        format_hint="cvvc",
        language_hint="korean",
        recursive=False,
        output_encoding="utf-8",
    )
    recording = build.project.recordings[0]

    assert recording.phones == ("dd", "a", "dd", "a", "dd", "i", "dd", "i")
    assert [(alias.alias, alias.phone_indices, alias.role) for alias in recording.aliases] == [
        ("dda", (0, 1), "cv"),
        ("at", (1, 2), "vc"),
        ("ddi", (4, 5), "cv"),
        ("it", (5, 6), "vc"),
    ]


def test_routed_parser_accepts_korean_weo_vowel_surface() -> None:
    assert parse_filename_phone_slots(
        "bweo.wav",
        format_hint="cvvc",
        language_hint="korean",
    ) == (("b", "weo"),)


def test_routed_parser_accepts_korean_eui_vowel_surface() -> None:
    assert parse_filename_phone_slots(
        "beui.wav",
        format_hint="cvvc",
        language_hint="korean",
    ) == (("b", "eui"),)


@pytest.mark.parametrize(
    ("wav_name", "expected_slots"),
    (
        ("_a(vf).wav", (("vf", "a", "vf"),)),
        ("_eo(vf).wav", (("vf", "eo", "vf"),)),
        ("_L(vf).wav", (("vf", "l", "vf"),)),
        ("_NG(vf).wav", (("vf", "ng", "vf"),)),
        ("_eu'a'i'o'e'u'eo'eu(cl).wav", (("q", "eu"), ("q", "a"), ("q", "i"), ("q", "o"), ("q", "e"), ("q", "u"), ("q", "eo"), ("q", "eu"))),
    ),
)
def test_routed_parser_accepts_only_documented_korean_cvvc_vocal_fry_and_closed_variants(
    wav_name: str,
    expected_slots: tuple[tuple[str, ...], ...],
) -> None:
    assert parse_filename_phone_slots(
        wav_name,
        format_hint="cvvc",
        language_hint="korean",
    ) == expected_slots

    with pytest.raises(VoicebankReviewRequired, match="review_required:unparsed_filename_token"):
        parse_filename_phone_slots(
            wav_name.replace("(vf)", "(noise)").replace("(cl)", "(noise)"),
            format_hint="cvvc",
            language_hint="korean",
        )


def test_routed_korean_vcv_terminal_coda_preserves_exact_phone_occurrences(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    wav_name = "a'a'eo'a'e'a'o'aNGH.wav"
    wav = bank / wav_name
    _write_cv(wav)
    aliases = ("a a", "a eo", "eo a", "a e", "e a", "a o", "o a", "a ng", "ng h")
    source_rows = [SourceOtoRow(wav_name, alias, index) for index, alias in enumerate(aliases)]

    build = build_routed_voicebank_project(
        bank=bank,
        wavs=[wav],
        base_rows_by_wav={wav_name: source_rows},
        format_hint="vcv",
        language_hint="korean",
        recursive=False,
        output_encoding="utf-8",
    )
    recording = build.project.recordings[0]

    assert recording.phones == ("a", "a", "eo", "a", "e", "a", "o", "a", "ng", "h")
    assert [token["occurrence_id"] for token in recording.metadata["phone_tokens"]] == [
        "a#0001", "a#0002", "eo#0001", "a#0003", "e#0001",
        "a#0004", "o#0001", "a#0005", "ng#0001", "h#0001",
    ]
    assert [(plan.alias, plan.phone_indices, plan.role) for plan in recording.aliases] == [
        ("a a", (0, 1), "vv"),
        ("a eo", (1, 2), "vv"),
        ("eo a", (2, 3), "vv"),
        ("a e", (3, 4), "vv"),
        ("e a", (4, 5), "vv"),
        ("a o", (5, 6), "vv"),
        ("o a", (6, 7), "vv"),
        ("a ng", (7, 8), "vc"),
        ("ng h", (8, 9), "vc"),
    ]


def test_routed_inference_marks_korean_nasals_and_liquids_acoustically(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    wavs = [bank / "niN.wav", bank / "ruL.wav"]
    for wav in wavs:
        _write_cv(wav)

    build = build_routed_voicebank_project(
        bank=bank,
        wavs=wavs,
        base_rows_by_wav={},
        format_hint="vcv",
        language_hint="korean",
        recursive=False,
        output_encoding="utf-8",
    )

    inventory = build.project.inventory
    assert inventory["n"].features["manner"] == "nasal"
    assert inventory["n"].features["sonorant"] is True
    assert inventory["n"].states[0].acoustic_role == "consonant"
    assert inventory["r"].features["manner"] == "liquid"
    assert inventory["l"].features["manner"] == "liquid"
    assert inventory["r"].states[0].acoustic_role == "consonant"


def test_routed_adapter_prefers_inventory_manifest_then_reclist_over_filename(tmp_path: Path) -> None:
    manifest_bank = tmp_path / "manifest_bank"
    manifest_bank.mkdir()
    manifest_wav = manifest_bank / "opaque01.wav"
    _write_cv(manifest_wav)
    (manifest_bank / "phones.yaml").write_text(
        """format_version: 1
phones:
  k:
    display_name: k
    features: {manner: stop, voiced: false, continuant: false}
    states:
      - {id: closure, acoustic_role: closure, min_ms: 8, mode_ms: 45, max_ms: 220}
      - {id: release, acoustic_role: burst, min_ms: 5, mode_ms: 14, max_ms: 80}
  a:
    display_name: a
    features: {syllabic: true, voiced: true, vowel_like: true}
    states:
      - {id: transition, acoustic_role: vowel_transition, min_ms: 15, mode_ms: 55, max_ms: 220}
      - {id: stable, acoustic_role: vowel_stable, min_ms: 45, mode_ms: 360, max_ms: 1400}
""",
        encoding="utf-8",
    )
    (manifest_bank / "recordings.yaml").write_text(
        """format_version: 1
recordings:
  - wav: opaque01.wav
    phones: [k, a]
    transcript_source: voicebank_manifest
    label_source: none
    aliases:
      - {alias: ka, template: builtin:cv, phones: [0, 1], role: cv}
""",
        encoding="utf-8",
    )
    manifest_build = build_routed_voicebank_project(
        bank=manifest_bank,
        wavs=[manifest_wav],
        base_rows_by_wav={},
        format_hint="cv",
        language_hint="mixed",
        recursive=False,
        output_encoding="utf-8",
    )
    assert manifest_build.project.recordings[0].transcript_source == "voicebank_recording_manifest"
    assert manifest_build.project.recordings[0].aliases[0].alias == "ka"
    assert set(manifest_build.project.inventory) == {"k", "a"}

    reclist_bank = tmp_path / "reclist_bank"
    reclist_bank.mkdir()
    reclist_wav = reclist_bank / "ka.wav"
    _write_cv(reclist_wav)
    (reclist_bank / "_reclist.txt").write_bytes("_ka\r\n".encode("cp932"))
    reclist_build = build_routed_voicebank_project(
        bank=reclist_bank,
        wavs=[reclist_wav],
        base_rows_by_wav={},
        format_hint="cv",
        language_hint="mixed",
        recursive=False,
        output_encoding="utf-8",
    )
    assert reclist_build.project.recordings[0].transcript_source == "voicebank_reclist"
    assert "reclist" in reclist_build.input_hashes["contract_artifacts"]


def test_routed_unknown_format_requires_review_without_legacy_fallback(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")

    with pytest.raises(VoicebankReviewRequired, match="review_required:unknown_parse_format") as caught:
        generate_voicebank(
            GenerationOptions(
                bank=bank,
                output=tmp_path / "unknown",
                backend="routed_hsmm",
                format_hint="unregistered",
            )
        )

    assert caught.value.status == "review_required"
    assert caught.value.reason == "unknown_parse_format"


def test_routed_base_oto_timing_cannot_change_generated_timing(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")
    base = bank / "oto.ini"
    _base_oto(base, (0.0, 0.0, 0.0, 0.0, 0.0))
    first = generate_voicebank(
        GenerationOptions(
            bank=bank,
            output=tmp_path / "first",
            base_oto=base,
            backend="routed_hsmm",
            format_hint="cv",
        )
    )
    first_oto = (tmp_path / "first" / "oto.ini").read_bytes()

    _base_oto(base, (9999.0, 8888.0, -7777.0, 6666.0, 5555.0))
    second = generate_voicebank(
        GenerationOptions(
            bank=bank,
            output=tmp_path / "second",
            base_oto=base,
            backend="routed_hsmm",
            format_hint="cv",
        )
    )

    assert first_oto == (tmp_path / "second" / "oto.ini").read_bytes()
    assert first["provenance"]["base_oto_sha256"] != second["provenance"]["base_oto_sha256"]
    assert first["project_sha256"] == second["project_sha256"]
    assert first["provenance"]["base_oto_timing_reused"] is False
    assert second["provenance"]["base_oto_timing_reused"] is False


def test_legacy_default_matches_explicit_legacy_backend(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")

    default_report = generate_voicebank(GenerationOptions(bank=bank, output=tmp_path / "default"))
    explicit_report = generate_voicebank(
        GenerationOptions(bank=bank, output=tmp_path / "explicit", backend="legacy_heuristic")
    )

    assert (tmp_path / "default" / "oto.ini").read_bytes() == (tmp_path / "explicit" / "oto.ini").read_bytes()
    assert default_report["generation_backend"] == "legacy_heuristic"
    assert explicit_report["generation_backend"] == "legacy_heuristic"


def test_exclude_row_policy_lists_unbindable_aliases_instead_of_stopping(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")
    (bank / "oto.ini").write_text("ka.wav=ka,0,0,0,0,0\r\nka.wav=zzz,0,0,0,0,0\r\n", encoding="utf-8")
    common = dict(bank=bank, base_oto=bank / "oto.ini", backend="routed_hsmm", format_hint="cv", language_hint="japanese")
    with pytest.raises(VoicebankReviewRequired, match="unmapped_alias"):
        generate_voicebank(GenerationOptions(output=tmp_path / "strict", **common))
    report = generate_voicebank(GenerationOptions(output=tmp_path / "loose", unmapped_alias_policy="exclude_row", **common))
    assert report["excluded_alias_rows"] == 1
    lines = (tmp_path / "loose" / "oto.ini").read_text(encoding="utf-8").splitlines()
    assert [line.split("=", 1)[1].split(",")[0] for line in lines] == ["ka"]
    listed = (tmp_path / "loose" / "unmapped_aliases.csv").read_text(encoding="utf-8-sig")
    assert "zzz" in listed and "unmapped_alias" in listed
    with pytest.raises(ValueError, match="requires backend=routed_hsmm"):
        generate_voicebank(GenerationOptions(bank=bank, output=tmp_path / "legacy", unmapped_alias_policy="exclude_row"))


def _write_cv_list(path: Path, count: int, *, period_s: float = 0.5, sample_rate: int = 16000) -> None:
    rng = np.random.default_rng(23)
    parts = [np.zeros(int(0.15 * sample_rate), dtype=np.float32)]
    for _ in range(count):
        consonant = (0.08 * rng.standard_normal(int(0.07 * sample_rate))).astype(np.float32)
        times = np.arange(int((period_s - 0.07) * sample_rate), dtype=np.float64) / sample_rate
        vowel = (0.24 * np.sin(2.0 * np.pi * 220.0 * times) * np.minimum(1.0, 20 * times)).astype(np.float32)
        parts.extend((consonant, vowel))
    parts.append(np.zeros(int(0.3 * sample_rate), dtype=np.float32))
    sf.write(path, np.concatenate(parts), sample_rate, subtype="PCM_16")


def test_routed_generates_format_aliases_without_base_oto(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv_list(bank / "ka_ki_ku.wav", 3)
    _write_cv_list(bank / "kaki.wav", 2)
    output = tmp_path / "out"
    report = generate_voicebank(
        GenerationOptions(
            bank=bank, output=output, backend="routed_hsmm", format_hint="cvvc", language_hint="korean",
            generate_missing_aliases=True,
        )
    )
    aliases = [line.split("=", 1)[1].split(",", 1)[0]
               for line in (output / "oto.ini").read_text(encoding="utf-8").splitlines() if line]
    assert aliases == ["- ka", "ka", "a k", "ki", "i k", "ku", "u -"]
    assert report["generated_alias_wavs"] == 1
    assert [item["wav"] for item in report["alias_generation_skipped"]] == ["kaki.wav"]


def test_routed_bpm_and_oto_style_are_applied_and_recorded(tmp_path: Path) -> None:
    from universal_auto_oto.oto_style import OtoStyle

    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv_list(bank / "ka_ki_ku_ke.wav", 4)
    output = tmp_path / "out"
    report = generate_voicebank(
        GenerationOptions(
            bank=bank, output=output, backend="routed_hsmm", format_hint="cvvc", language_hint="korean",
            generate_missing_aliases=True, bpm=120.0, oto_style=OtoStyle(cv_overlap_ratio=0.5),
        )
    )
    tempo = report["recording_tempo"]
    assert tempo["source"] == "user_bpm"
    assert tempo["syllable_period_ms"] == pytest.approx(500.0)
    persisted = json.loads((output / "generation_report.json").read_text(encoding="utf-8"))
    assert persisted["provenance"]["oto_style"]["cv_overlap_ratio"] == 0.5
    for line in (output / "oto.ini").read_text(encoding="utf-8").splitlines():
        alias, offset, consonant, cutoff, pre, overlap = line.split("=", 1)[1].split(",")
        if alias in {"ka", "ki", "ku", "ke"}:
            assert float(overlap) == pytest.approx(float(pre) / 2.0, abs=0.01)


def test_bpm_out_of_range_is_rejected(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv_list(bank / "ka_ki.wav", 2)
    with pytest.raises(ValueError, match="BPM"):
        generate_voicebank(
            GenerationOptions(bank=bank, output=tmp_path / "out", backend="routed_hsmm", format_hint="cvvc",
                              language_hint="korean", generate_missing_aliases=True, bpm=20.0)
        )


def test_exclude_row_policy_lists_a_recording_that_cannot_be_aligned(tmp_path: Path, monkeypatch) -> None:
    import csv
    from universal_auto_oto import project_runtime
    from universal_auto_oto.alignment import AlignmentError

    bank = tmp_path / "bank"
    bank.mkdir()
    _write_cv(bank / "ka.wav")
    _write_cv(bank / "ki.wav")
    (bank / "oto.ini").write_text("ka.wav=ka,0,1,-2,3,4\r\nki.wav=ki,0,1,-2,3,4\r\n", encoding="utf-8")
    real = project_runtime.align_recording

    def silent_ki(project, recording, **options):
        if recording.wav == "ki.wav":
            raise AlignmentError("HSMM decode 실패: timeline_decode_failed (ki.wav)")
        return real(project, recording, **options)

    monkeypatch.setattr(project_runtime, "align_recording", silent_ki)
    options = dict(bank=bank, base_oto=bank / "oto.ini", backend="routed_hsmm", format_hint="cv", language_hint="mixed")
    with pytest.raises(AlignmentError):
        generate_voicebank(GenerationOptions(output=tmp_path / "strict", **options))
    report = generate_voicebank(GenerationOptions(output=tmp_path / "out", unmapped_alias_policy="exclude_row", **options))
    assert report["generated_rows"] == 1 and report["failed_recordings"][0]["wav"] == "ki.wav"
    assert (tmp_path / "out" / "oto.ini").read_text(encoding="utf-8").startswith("ka.wav=ka,")
    with (tmp_path / "out" / "unmapped_aliases.csv").open(encoding="utf-8-sig") as stream:
        listed = list(csv.DictReader(stream))
    assert [(row["wav"], row["alias"], row["reason"]) for row in listed] == [("ki.wav", "ki", "alignment_failed")]


def test_parallel_alignment_matches_serial_and_reports_progress(tmp_path: Path) -> None:
    from universal_auto_oto.generator import GenerationCancelled

    bank = tmp_path / "bank"
    bank.mkdir()
    for name in ("ka", "ki", "ku"):
        _write_cv(bank / f"{name}.wav")
    (bank / "oto.ini").write_text("".join(f"{n}.wav={n},0,1,-2,3,4\r\n" for n in ("ka", "ki", "ku")), encoding="utf-8")
    seen: list[tuple[str, int, int]] = []
    common = dict(bank=bank, base_oto=bank / "oto.ini", backend="routed_hsmm", format_hint="cv", language_hint="mixed")
    generate_voicebank(GenerationOptions(output=tmp_path / "serial", alignment_workers=1, **common))
    generate_voicebank(GenerationOptions(output=tmp_path / "parallel", alignment_workers=2,
                                         progress=lambda *event: seen.append(event), **common))
    assert (tmp_path / "serial" / "oto.ini").read_bytes() == (tmp_path / "parallel" / "oto.ini").read_bytes()
    assert ("align:1/1", 3, 3) in seen and seen[0] == ("align:1/1", 0, 3) and ("render", 2, 3) in seen

    def cancel(stage, done, total):
        if done:
            raise GenerationCancelled()

    with pytest.raises(GenerationCancelled):
        generate_voicebank(GenerationOptions(output=tmp_path / "cancelled", alignment_workers=2, progress=cancel, **common))
    assert not (tmp_path / "cancelled" / "oto.ini").exists()


def test_batch_stops_on_cancel_but_not_on_folder_errors(tmp_path: Path) -> None:
    from universal_auto_oto.batch import BankFolder, run_batch
    from universal_auto_oto.generator import GenerationCancelled

    folders = [BankFolder(path=tmp_path / name, relative=name, wav_count=1, has_oto=False, detection=None)
               for name in ("a", "b", "c")]
    calls: list[str] = []

    def generate_one(folder, output):
        calls.append(folder.relative)
        if folder.relative == "a":
            raise ValueError("broken folder")
        raise GenerationCancelled()

    with pytest.raises(GenerationCancelled):
        run_batch(folders, root=tmp_path, output_root=tmp_path / "out", in_place=False, generate_one=generate_one)
    assert calls == ["a", "b"]

"""Language packs: English CZ VCCV read from YAML instead of hand-written rules."""
import json

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.bank_detection import detect_format
from universal_auto_oto.domain import Landmark, LandmarkTimeline, PhoneSegment, SourceOtoRow
from universal_auto_oto.generic_alias_binding import bind_pack_alias, parse_pack_alias
from universal_auto_oto.language_pack import find_language_pack, load_language_pack_text, LanguagePackError
from universal_auto_oto.pack_landmarks import add_pack_landmarks
from universal_auto_oto.template_engine import _pack_template
from universal_auto_oto.voicebank_project_adapter import VoicebankReviewRequired, build_routed_voicebank_project


@pytest.fixture(scope="module")
def pack():
    found = find_language_pack("english", "vccv")
    assert found is not None and found.pack_id == "en_cz_vccv"
    return found


def phones_of(pack, name):
    return [phone for chunk in pack.parse_filename(name) for phone in chunk]


def bind(pack, name, alias):
    plan, reason = bind_pack_alias(pack, alias, phones_of(pack, name))
    assert plan is not None, reason
    return plan


def test_case_sensitive_symbols_and_longest_match(pack):
    assert pack.tokenize("A") == ("ey",) and pack.tokenize("a") == ("aa",)
    assert pack.tokenize("shAng") == ("sh", "ey", "ng")
    assert pack.tokenize("ll") == ("el",)
    assert pack.tokenize("xyz?") is None


def test_filename_tags_markers_and_case_dashes_are_not_recorded(pack):
    # l_ groups the list, trailing dashes keep kAl- apart from kal on Windows.
    assert pack.parse_filename("l_kAl_kAl-.wav") == (("k", "ey", "l"), ("k", "ey", "l"))
    assert pack.parse_filename("b__kebd_kebd.wav") == (("k", "eh", "b", "d"), ("k", "eh", "b", "d"))
    # A marked first chunk is recorded even without a vowel-less tag shape.
    assert pack.parse_filename("-ll__kxl.wav") == (("el",), ("k", "ax", "l"))
    assert pack.parse_filename("_8&n.wav") == (("aw", "ae_n", "n"),)


def test_head_mid_coda_and_final_vc_occurrences(pack):
    name = "-b@_b@b_k@b.wav"  # b ae | b ae b | k ae b
    assert bind(pack, name, "-b@").phone_indices == (0, 1)
    assert bind(pack, name, "b@").phone_indices == (2, 3)          # unmarked CV is the mid take
    assert bind(pack, name, "@ b").phone_indices == (1, 2)         # spaced: b starts a syllable
    assert bind(pack, name, "@b").phone_indices == (3, 4)          # glued: coda before k
    assert bind(pack, name, "@b-").phone_indices == (6, 7)         # end marker: phrase end
    assert bind(pack, name, "@ b").group_starts == (0, 1)


def test_rhyme_units_after_consonant_and_equivalents(pack):
    # b8n is a CV whose vowel unit carries the n: mid take, not the coda rule.
    assert bind(pack, "-b8n_bx_b8n.wav", "b8n").phone_indices == (5, 6, 7)
    assert bind(pack, "-@__k@.wav", "_@").phone_indices == (2,)
    assert bind(pack, "-@__k@.wav", "@").phone_indices == (0,)
    assert bind(pack, "-@__k@.wav", "@-").phone_indices == (2,)
    # Syllabic l after a consonant is recorded as x l.
    assert bind(pack, "-ll__kxl.wav", "_ll").phone_indices == (2, 3)
    assert bind(pack, "-ll__kxl.wav", "-ll").phone_indices == (0,)
    # n before k is [ng]; aliases spell it ng.
    assert bind(pack, "-bAnk_bx_bAnk.wav", "bAng").phone_indices == (6, 7, 8)


def test_shapes_select_templates(pack):
    assert parse_pack_alias(pack, "-hhy@").shape == "-CCV"
    assert pack.rule_for("-CCV").template == "cv_head"
    assert pack.rule_for("VC-").template == "vc_end"
    assert pack.rule_for("V CC").template == "vc_onset_cluster"
    assert bind(pack, "-b@_b@b_k@b.wav", "@b-").template == "pack:en_cz_vccv:vc_end"
    assert bind_pack_alias(pack, "-m&_m&", phones_of(pack, "-m&_m&.wav"))[1] == "alias_not_tokenized"


def test_every_pack_template_loads(pack):
    for template_id in pack.templates:
        definition = _pack_template(pack.template_ref(template_id))
        assert definition.source_sha256 == pack.sha256


def test_pack_validation_rejects_unknown_phones():
    with pytest.raises(LanguagePackError):
        load_language_pack_text("schema_version: 1\npack_id: x\nphones: {a: {class: vowel}}\nsymbols: {b: b}\n", "t")


def test_routed_project_uses_pack_inventory_and_bindings(tmp_path, pack):
    bank = tmp_path / "bank"
    bank.mkdir()
    sf.write(bank / "-b@_b@b_k@b.wav", np.zeros(4410), 44100)
    sf.write(bank / "-dd@_dd@.wav", np.zeros(4410), 44100)
    rows = {"-b@_b@b_k@b.wav": [SourceOtoRow("-b@_b@b_k@b.wav", alias, i)
                                for i, alias in enumerate(["-b@", "@ b", "@b-", "-b@_b@b_k@b"])],
            "-dd@_dd@.wav": []}
    build = build_routed_voicebank_project(
        bank=bank, wavs=sorted(bank.glob("*.wav")), base_rows_by_wav=rows, format_hint="vccv",
        language_hint="english", recursive=False, output_encoding="utf-8", unmapped_alias_policy="exclude_row")
    project = build.project
    assert [recording.wav for recording in project.recordings] == ["-b@_b@b_k@b.wav"]  # no rows -> skipped
    recording = project.recordings[0]
    assert recording.phones == ("b", "ae", "b", "ae", "b", "k", "ae", "b")
    assert recording.metadata["language_pack"] == "en_cz_vccv"
    assert [slot["vowel"] for slot in recording.metadata["phone_slots"]] == [1, 3, 6]
    assert project.inventory["ae"].features["vowel_like"] is True
    assert project.inventory["b"].features["manner"] == "stop"
    assert [plan.alias for plan in recording.aliases] == ["-b@", "@ b", "@b-"]
    assert [row["alias"] for row in build.excluded_rows] == ["-b@_b@b_k@b"]
    assert build.manifest["language_pack"]["id"] == "en_cz_vccv"
    assert "pack:en_cz_vccv:vc_end" in project.generation.templates


def test_unknown_format_without_pack_needs_review(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    sf.write(bank / "-ba.wav", np.zeros(441), 44100)
    with pytest.raises(VoicebankReviewRequired, match="no_language_pack"):
        build_routed_voicebank_project(bank=bank, wavs=[bank / "-ba.wav"], base_rows_by_wav={}, format_hint="vccv",
                                       language_hint="korean", recursive=False, output_encoding="utf-8")


def test_vowel_offset_lands_on_the_energy_fall_not_the_aligned_end(tmp_path, pack):
    sample_rate = 44100
    t = np.arange(int(0.6 * sample_rate)) / sample_rate
    voice = 0.5 * np.sin(2 * np.pi * 220 * t)
    voice[int(0.40 * sample_rate):] *= 0.001  # the vowel stops at 400 ms; a stop closure follows
    wav = tmp_path / "v.wav"
    sf.write(wav, voice, sample_rate)
    inventory = {phone: pack.phone_definition(phone) for phone in ("ae", "b")}
    timeline = LandmarkTimeline(
        schema_version=2, wav="v.wav", wav_sha256="x", recording_sha256="x", inventory_sha256="x",
        feature_config_sha256="x", aligner_config_sha256="x", model_sha256=None,
        phone_segments=(PhoneSegment(0, "ae", 100.0, 470.0, 0.9, ()), PhoneSegment(1, "b", 470.0, 560.0, 0.9, ())),
        landmarks=(Landmark("phone_boundary", 100.0, 0, 0.9, "hsmm"),), usable_start_ms=100.0, usable_end_ms=560.0,
        confidence=0.9, warnings=(), provenance={}, sample_rate=sample_rate, duration_ms=600.0)
    refined = add_pack_landmarks(timeline, wav, inventory)
    offsets = [item for item in refined.landmarks if item.label == "vowel_offset"]
    assert len(offsets) == 1 and abs(offsets[0].time_ms - 400.0) <= 10.0
    assert add_pack_landmarks(refined, wav, inventory) is refined


def test_english_bank_detection_prefers_the_pack(tmp_path):
    names = ["-b@_b@b_k@b.wav", "-dd@_dd@.wav", "l_k@l_k@l.wav"]
    detection = detect_format(tmp_path, names, base_aliases=["-b@", "@ b", "@l-"], language="english")
    assert detection.format == "vccv" and detection.source == "language_pack"
    assert detect_format(tmp_path, names, language="korean").source != "language_pack"


def test_template_expressions_reach_last_phone_nucleus_and_groups():
    from universal_auto_oto.template_engine import _ExpressionEvaluator, _PhoneView, _parse_expression

    evaluator = _ExpressionEvaluator(())
    segments = [PhoneSegment(i, phone, 100.0 * i, 100.0 * i + 90.0, 0.9, ()) for i, phone in enumerate(("s", "k", "ae"))]
    features = [{"manner": "fricative"}, {"manner": "stop"}, {"vowel_like": True, "syllabic": True}]
    evaluator.phones = tuple(_PhoneView(segment, feature, {}, evaluator) for segment, feature in zip(segments, features))
    evaluator.group_starts = (0, 1)

    def value(expression):
        return evaluator.evaluate(_parse_expression(expression, "t"), {})

    assert value("phone[-1].start") == 200.0
    assert value("nucleus.start") == 200.0
    assert value("group_start[1].start") == 100.0
    assert value("fallback(group_start[2].start, 7)") == 7
    assert value("1 if phone[2].features.vowel_like == True else 0") == 1


def test_style_shifts_are_fitted_per_shape_and_attached_to_plans(tmp_path, pack):
    from universal_auto_oto.pack_style import StyleSample, fit_style, load_style_profile, write_style_profile

    samples = [StyleSample(("CV",), {"pre": 100.0, "offset": 50.0}, {"pre": 110.0 + i, "offset": 40.0}) for i in range(5)]
    samples.append(StyleSample(("VV",), {"pre": 0.0}, {"pre": 500.0}))  # too few rows: left out
    shapes = fit_style(samples)
    assert shapes == {"CV": {"rows": 5, "offset": -10.0, "pre": 12.0}}
    path = tmp_path / "style.yaml"
    write_style_profile(path, pack.pack_id, shapes, {"test": True})
    style = load_style_profile(path)
    assert style.shifts_for("CV") == {"offset_abs": -10.0, "pre_abs": 12.0}
    plan, _ = bind_pack_alias(pack, "b@", phones_of(pack, "-b@_b@b_k@b.wav"), style=style)
    assert plan.anchor_shifts == {"offset_abs": -10.0, "pre_abs": 12.0}
    unstyled, _ = bind_pack_alias(pack, "b@", phones_of(pack, "-b@_b@b_k@b.wav"))
    assert unstyled.anchor_shifts == {}


def test_pack_default_style_and_profile_selection(tmp_path, pack):
    from universal_auto_oto.pack_style import PackStyle

    assert pack.style is not None and pack.style.shifts_for("CV")
    bank = tmp_path / "bank"
    bank.mkdir()
    sf.write(bank / "-b@_b@b_k@b.wav", np.zeros(4410), 44100)
    rows = {"-b@_b@b_k@b.wav": [SourceOtoRow("-b@_b@b_k@b.wav", "b@", 0)]}
    common = dict(bank=bank, wavs=[bank / "-b@_b@b_k@b.wav"], base_rows_by_wav=rows, format_hint="vccv",
                  language_hint="english", recursive=False, output_encoding="utf-8")
    default = build_routed_voicebank_project(**common).project.recordings[0].aliases[0]
    assert default.anchor_shifts == pack.style.shifts_for("CV|stop", "CV")
    plain = build_routed_voicebank_project(**common, style_profile="none").project.recordings[0].aliases[0]
    assert plain.anchor_shifts == {}
    with pytest.raises(VoicebankReviewRequired, match="style_profile_pack_mismatch"):
        build_routed_voicebank_project(**common, style_profile=PackStyle("other", {}, "t", "x"))


def test_vowel_onset_follows_the_sonorant_release_not_the_aligned_boundary(tmp_path, pack):
    sample_rate = 44100
    t = np.arange(int(0.9 * sample_rate)) / sample_rate
    murmur = 0.3 * np.sin(2 * np.pi * 150 * t)                      # nasal murmur: low band only
    vowel = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    signal = np.where(t < 0.2, 0.0, np.where(t < 0.5, murmur, vowel))  # m 200-500 ms, vowel after
    wav = tmp_path / "ma.wav"
    sf.write(wav, signal, sample_rate)
    inventory = {phone: pack.phone_definition(phone) for phone in ("m", "aa")}
    timeline = LandmarkTimeline(
        schema_version=2, wav="ma.wav", wav_sha256="x", recording_sha256="x", inventory_sha256="x",
        feature_config_sha256="x", aligner_config_sha256="x", model_sha256=None,
        # The aligner put the vowel start 120 ms early, inside the murmur.
        phone_segments=(PhoneSegment(0, "m", 200.0, 380.0, 0.9, ()), PhoneSegment(1, "aa", 380.0, 880.0, 0.9, ())),
        landmarks=(), usable_start_ms=200.0, usable_end_ms=880.0,
        confidence=0.9, warnings=(), provenance={}, sample_rate=sample_rate, duration_ms=900.0)
    onsets = [item for item in add_pack_landmarks(timeline, wav, inventory).landmarks if item.label == "vowel_onset"]
    assert len(onsets) == 1 and onsets[0].phone_index == 1 and abs(onsets[0].time_ms - 500.0) <= 15.0


def test_before_and_after_reach_the_phones_outside_the_alias():
    from universal_auto_oto.template_engine import _ExpressionEvaluator, _PhoneView, _parse_expression

    evaluator = _ExpressionEvaluator(())
    segment = lambda i, phone: PhoneSegment(i, phone, 100.0 * i, 100.0 * i + 90.0, 0.9, ())
    evaluator.phones = (_PhoneView(segment(1, "b"), {"manner": "stop"}, {}, evaluator),)
    evaluator.context = {"after": _PhoneView(segment(2, "aa"), {"vowel_like": True}, {"vowel_onset": 230.0}, evaluator)}

    def value(expression):
        return evaluator.evaluate(_parse_expression(expression, "t"), {})

    assert value("after.vowel_onset") == 230.0
    assert value("fallback(before.start, 5)") == 5
    assert value("1 if after.features.vowel_like == True else 0") == 1


def test_style_lookup_prefers_the_class_key(pack):
    from universal_auto_oto.generic_alias_binding import style_keys
    from universal_auto_oto.pack_style import parse_style

    style = parse_style({"VC-": {"pre": 5}, "VC-|son": {"pre": -20}}, pack_id=pack.pack_id, source="t")
    assert style_keys(pack, parse_pack_alias(pack, "@l-")) == ("VC-|son", "VC-")
    assert style.shifts_for(*style_keys(pack, parse_pack_alias(pack, "@l-"))) == {"pre_abs": -20.0}
    assert style.shifts_for(*style_keys(pack, parse_pack_alias(pack, "@b-"))) == {"pre_abs": 5.0}


def test_gui_keeps_english_vccv_on_the_pack_alignment_profile():
    from universal_auto_oto.oto_generator_ui import RECOMMENDED_DEVELOPMENT_PROFILE, recommended_alignment_profile

    assert recommended_alignment_profile("english", "vccv") == "cvvc_slot_rhythm_v16"
    assert recommended_alignment_profile("korean", "cvvc") == RECOMMENDED_DEVELOPMENT_PROFILE
    assert recommended_alignment_profile("japanese", "cv") == RECOMMENDED_DEVELOPMENT_PROFILE


def test_format_menu_hides_formats_a_language_does_not_have():
    from universal_auto_oto.oto_generator_ui import AUTO_FORMAT_CHOICE, format_choices_for

    assert format_choices_for("한국어") == (AUTO_FORMAT_CHOICE, "CV", "CVC", "CVVC", "VCV")
    assert format_choices_for("일본어") == (AUTO_FORMAT_CHOICE, "CV", "CVVC", "VCV")
    assert format_choices_for("영어") == (AUTO_FORMAT_CHOICE, "VCCV")

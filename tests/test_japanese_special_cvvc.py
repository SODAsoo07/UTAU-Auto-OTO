from pathlib import Path

import pytest

from universal_auto_oto.alias_generation import generate_alias_rows, generate_aliases
from universal_auto_oto.domain import LandmarkTimeline, PhoneSegment, SourceOtoRow
from universal_auto_oto.japanese_cvvc import special_slots
from universal_auto_oto.template_engine import render_project_alias
from universal_auto_oto.voicebank_project_adapter import (
    VoicebankReviewRequired, _flatten_slots, _planned_aliases,
    build_routed_voicebank_project, parse_filename_phone_slots,
)


NAMES = [
    '_lらlりlるlれlろlらんlら.wav', '_lり_lる_lれ_lろ.wav',
    '_xらxりxるxれxろxらんxら.wav', '_xり_xる_xれ_xろ.wav',
    '_あshかshきshくshけshこ.wav', '_あshたshてぃshとぅshてshと.wav',
    '_あsかsきsくsけsこ.wav', '_あsたsてぃsとぅsてsと.wav',
    *[f'_{v}{v}hh.wav' for v in 'あいうえおん'],
    '_おんmぽ_うんmぷ.wav', '_おんngこ_うんngく.wav', '_おんnと_うんnとぅ.wav',
    '_えんmぺ_いんmぴ.wav', '_えんngけ_いんngき.wav', '_えんnて_いんnてぃ.wav',
]


@pytest.mark.parametrize('name', NAMES)
def test_all_supported_families_generate_bind_and_render(tmp_path: Path, name: str):
    # Build consumes aliases only; no reference OTO timings enter the contract.
    wav = tmp_path / name
    wav.write_bytes(b'filename-contract-test')
    rows = generate_alias_rows(name, format_hint='cvvc', language_hint='japanese')
    project = build_routed_voicebank_project(bank=tmp_path, wavs=[wav],
        base_rows_by_wav={name: rows}, format_hint='cvvc', language_hint='japanese',
        recursive=False, output_encoding='utf-8').project
    rec = project.recordings[0]
    phones = tuple(PhoneSegment(i, p, i*1000., (i+1)*1000., .9, ())
                   for i, p in enumerate(rec.phones))
    duration = len(phones)*1000.
    timeline = LandmarkTimeline(1, name, *(['0'*64]*5), None, phones, (), 0., duration,
                                .9, (), {}, 1000, duration)
    for i, plan in enumerate(rec.aliases):
        row, trace = render_project_alias(project, rec, timeline, plan, source_index=i)
        assert not trace.repairs
        if plan.template in {'builtin:cv_trill', 'builtin:vc_trill'}:
            # Preutterance belongs to r, strictly after the irregular trill.
            r_index = plan.phone_indices[1 if plan.role != 'vc' else 2]
            assert rec.phones[r_index] == 'r'
            assert trace.final_anchors['pre_abs'] == phones[r_index].start_ms
        if plan.template == 'builtin:long_vowel':
            assert trace.final_anchors['cutoff_abs'] - trace.final_anchors['fixed_abs'] >= 600
        if plan.alias.endswith(' H'):
            assert plan.phone_indices == (2, 3)
            assert row.absolute['pre_abs'] == 3000


def test_nasal_internal_transition_and_explicit_occurrences():
    name = '_おんmぽ_うんmぷ.wav'
    _, slots = _flatten_slots(parse_filename_phone_slots(name, format_hint='cvvc', language_hint='ja'))
    rows = [SourceOtoRow(name, a, i) for i, a in enumerate(['o んm_D4', 'n んm_D4', 'n んm#2_D4'])]
    plans = _planned_aliases(rows, slots, 'cvvc', name, language_hint='ja')
    assert [p.phone_indices for p in plans] == [(0, 1, 2), (1, 2), (6, 7)]


def test_devoiced_names_have_no_phantom_vowels_and_keep_aliases():
    name = '_あshかshきshくshけshこ.wav'
    assert [surface for _, surface in special_slots(name)] == ['あ', 'sh', 'か', 'sh', 'き', 'sh', 'く', 'sh', 'け', 'sh', 'こ']
    assert generate_aliases(name, format_hint='cvvc', language_hint='ja') == [f'sh {k}' for k in 'かきくけこ']


def test_long_and_variant_names_do_not_change_ordinary_or_unknown_parsing():
    assert parse_filename_phone_slots('_らり.wav', format_hint='cvvc', language_hint='ja') == (('r', 'a'), ('r', 'i'))
    assert generate_aliases('_lらlり.wav', format_hint='cvvc', language_hint='ja') == ['- lら', 'lら', 'lり', 'a l']
    assert generate_aliases('_lり_lる.wav', format_hint='cvvc', language_hint='ja')[:3] == ['- lり', 'lり', '- lる']
    for language in ('ja', 'korean'):
        with pytest.raises(VoicebankReviewRequired):
            parse_filename_phone_slots('_あ・い・う・え・お・ん・あvf.wav', format_hint='cvvc', language_hint=language)
    with pytest.raises(VoicebankReviewRequired):
        parse_filename_phone_slots('_xら.wav', format_hint='cvvc', language_hint='korean')

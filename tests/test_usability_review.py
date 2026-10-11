from types import SimpleNamespace as NS
import csv

import numpy as np

from universal_auto_oto.alignment import _build_hsmm_diagnostics
from universal_auto_oto.domain import AliasPlan, CompiledState
from universal_auto_oto.hsmm_decoder import HSMMStateSpec
from universal_auto_oto.usability_review import usage_review, usability_candidate, write_review_priority


def test_candidate_expressions_render_without_cycles_and_preserve_fry_tail(tmp_path):
    from dataclasses import replace
    import soundfile as sf
    from universal_auto_oto.domain import Landmark, LandmarkTimeline, PhoneSegment, SourceOtoRow, PhoneDefinition
    from universal_auto_oto.template_engine import render_project_alias
    from universal_auto_oto.voicebank_project_adapter import build_routed_voicebank_project

    wav = "_a'a'i'a'u'a'e'a.wav"
    sf.write(tmp_path / wav, np.zeros(10000), 1000)
    project = build_routed_voicebank_project(bank=tmp_path, wavs=[tmp_path / wav],
        base_rows_by_wav={wav: [SourceOtoRow(wav, alias, i) for i, alias in enumerate(('a a', 'a -', 'a R'))]},
        format_hint='cvvc', language_hint='korean', recursive=False, output_encoding='utf-8').project
    rec = project.recordings[0]
    phones = tuple(PhoneSegment(i, p, i*1000., (i+1)*1000., .8, ()) for i, p in enumerate(rec.phones))
    timeline = LandmarkTimeline(1, wav, '0'*64, '0'*64, '0'*64, '0'*64, '0'*64, None,
        phones, (), 0., 9000., .8, (), {}, 1000, 10000.)
    for i, plan in enumerate(rec.aliases):
        row, trace = render_project_alias(project, rec, timeline, usability_candidate(plan, rec), source_index=i)
        assert trace.repair_log == ()
        assert row.absolute['pre_abs'] - row.offset == 180
    # A sustained consonant must enter at its decoded boundary even if the
    # vowel-like refinement moved a landmark 80 ms later. Its tail is isolated.
    rec = replace(rec, phones=('vf', 'a', 'vf'))
    phones = tuple(PhoneSegment(i, p, i*1000., (i+1)*1000., .8, ()) for i, p in enumerate(rec.phones))
    timeline = replace(timeline, phone_segments=phones,
        landmarks=(Landmark('vowel_stable_start', 1080., 1, .8, 'test'),))
    head = AliasPlan('*A', (0, 1), 'builtin:fry_to_sustain', 'cv')
    project = replace(project, inventory={**project.inventory, 'vf': PhoneDefinition('vf', 'fry', {}, ())},
        generation=replace(project.generation,
        templates=project.generation.templates + ('builtin:fry_to_sustain',)))
    row, _ = render_project_alias(project, rec, timeline, usability_candidate(head, rec), source_index=0)
    assert row.absolute['pre_abs'] == 1000
    tail = AliasPlan('A *', (1, 2), 'builtin:ending_exhalation', 'vc')
    assert usability_candidate(tail, rec) is tail


def test_global_peak_is_not_a_confirmed_wrong_occurrence():
    states = [CompiledState('a', 0, 'a', 'stable', 'periodic', 5, 100),
              CompiledState('b', 1, 'i', 'stable', 'periodic', 5, 100)]
    specs = [HSMMStateSpec(s.instance_id, 'periodic', 5, 100, 20, 40) for s in states]
    decoded = [NS(start_ms=0, end_ms=20), NS(start_ms=400, end_ms=420)]
    times = np.arange(0, 450, 5.)
    scores = {s.instance_id: np.where(times >= 400, 1., 0.) for s in states}
    d = _build_hsmm_diagnostics(states, specs, decoded, scores, times)
    assert d['states'][0]['wrong_occurrence_risk']
    assert not d['states'][0]['repeated_phone_ambiguity']
    assert not d['states'][0]['occurrence_error_confirmed']
    states[1] = CompiledState('b', 1, 'a', 'stable', 'periodic', 5, 100)
    d = _build_hsmm_diagnostics(states, specs, decoded, scores, times)
    assert d['states'][0]['same_phone_competitor_indices'] == [1]


def test_review_scopes_risk_to_alias_and_keeps_all_rows_pending(tmp_path):
    phones = [NS(phone_index=i, start_ms=i*100., end_ms=(i+1)*100.) for i in range(4)]
    states = [{'phone_index':3, 'instance_id':'other', 'repeated_phone_ambiguity':True,
        'same_phone_competitor_indices':[2], 'global_acoustic_competitor_risk':True}]
    timeline = NS(phone_segments=phones, provenance={'hsmm':{'diagnostics':{'states':states}}})
    row = {'phone_indices':[0,1], 'role':'vc', 'final_anchors':{'cutoff_abs':230.,'fixed_abs':190.},
        'template_id':'vc', 'recording_wav':'a.wav', 'alias':'a k', 'output_oto':str(tmp_path/'oto.ini'), 'oto_line':7}
    review = usage_review(row, timeline)
    assert review['category'] == 'inspect_first'
    assert review['repeat_competitor_indices'] == []
    assert review['reasons'] == ['tail_extends_into_following_phone']
    row['usage_review'] = review
    report = write_review_priority([row], tmp_path)
    exported = list(csv.DictReader((tmp_path/'review_priority.csv').open(encoding='utf-8-sig')))
    assert exported[0]['line'] == '7' and exported[0]['usability'] == 'manual_pending'
    assert report['automatic_pass_count'] == 0


def test_explicit_overrides_win_and_candidates_preserve_alias_occurrence():
    for template, indices in [('builtin:vc',(0,1)), ('builtin:vv',(0,1)),
                              ('builtin:release',(0,)), ('builtin:fry_to_sustain',(0,1))]:
        original = AliasPlan('a k#2',indices,template,'vc', {'cutoff_abs':999})
        candidate = usability_candidate(original)
        assert candidate.alias == original.alias and candidate.phone_indices == indices
        assert candidate.overrides['cutoff_abs'] == 999
    ordinary = AliasPlan('ka',(0,1),'builtin:cv','cv')
    assert usability_candidate(ordinary) is ordinary


def test_vc_consonant_end_candidate_keeps_the_stretch_span_inside_the_consonant(tmp_path):
    import soundfile as sf
    from universal_auto_oto.domain import Landmark, LandmarkTimeline, PhoneSegment, SourceOtoRow
    from universal_auto_oto.template_engine import render_project_alias
    from universal_auto_oto.usability_review import vc_consonant_end_candidate
    from universal_auto_oto.voicebank_project_adapter import build_routed_voicebank_project

    wav = "_ga'ga.wav"
    sf.write(tmp_path / wav, np.zeros(3000), 1000)
    project = build_routed_voicebank_project(bank=tmp_path, wavs=[tmp_path / wav],
        base_rows_by_wav={wav: [SourceOtoRow(wav, 'a g', 0), SourceOtoRow(wav, 'ga', 1)]},
        format_hint='cvvc', language_hint='korean', recursive=False, output_encoding='utf-8').project
    rec = project.recordings[0]
    spans = ((0, 100), (100, 900), (900, 1000), (1000, 1800))
    phones = tuple(PhoneSegment(i, p, float(a), float(b), .8, ()) for i, (p, (a, b)) in enumerate(zip(rec.phones, spans)))
    landmarks = (Landmark('closure_start', 920., 2, .8, 'test'), Landmark('release', 980., 2, .8, 'test'))
    timeline = LandmarkTimeline(1, wav, '0'*64, '0'*64, '0'*64, '0'*64, '0'*64, None,
        phones, landmarks, 0., 3000., .8, (), {}, 1000, 3000.)
    vc = next(plan for plan in rec.aliases if plan.alias == 'a g')
    row, trace = render_project_alias(project, rec, timeline, vc_consonant_end_candidate(vc, rec), source_index=0)
    assert trace.repair_log == ()
    assert row.absolute['pre_abs'] == 920 and row.absolute['fixed_end_abs'] == 950 and row.absolute['cutoff_abs'] == 977
    cv = next(plan for plan in rec.aliases if plan.alias == 'ga')
    assert vc_consonant_end_candidate(cv, rec) == usability_candidate(cv, rec)


def test_vc_end_policy_consonant_applies_the_vc_end_expressions_at_build_time(tmp_path):
    import soundfile as sf
    from universal_auto_oto.domain import SourceOtoRow
    from universal_auto_oto.usability_review import vc_consonant_end_overrides
    from universal_auto_oto.voicebank_project_adapter import build_routed_voicebank_project

    wav = "_ga'ga.wav"
    sf.write(tmp_path / wav, np.zeros(3000), 1000)
    build = lambda policy: build_routed_voicebank_project(bank=tmp_path, wavs=[tmp_path / wav],
        base_rows_by_wav={wav: [SourceOtoRow(wav, 'a g', 0), SourceOtoRow(wav, 'ga', 1)]},
        format_hint='cvvc', language_hint='korean', recursive=False, output_encoding='utf-8',
        vc_end_policy=policy)
    plain, consonant = build("template"), build("consonant")
    vc = {p.alias: p for p in consonant.project.recordings[0].aliases}
    base = {p.alias: p for p in plain.project.recordings[0].aliases}
    assert dict(vc['a g'].overrides) == vc_consonant_end_overrides(base['a g']) != {}
    assert vc['ga'] == base['ga']
    assert consonant.manifest["vc_end_policy"] == "consonant" and "vc_end_policy" not in plain.manifest
    assert consonant.project.recordings_sha256 != plain.project.recordings_sha256


def test_release_vowel_end_overrides_only_touch_dash_releases():
    from universal_auto_oto.domain import AliasPlan
    from universal_auto_oto.usability_review import release_vowel_end_overrides
    release = AliasPlan("a -", (5,), "builtin:release", "release")
    assert release_vowel_end_overrides(release)["pre_abs"] == "max(offset_abs + 30, phone[0].end)"
    assert release_vowel_end_overrides(AliasPlan("a R", (5, 6), "builtin:ending_exhalation", "release")) == {}
    assert release_vowel_end_overrides(AliasPlan("ka", (0, 1), "builtin:cv", "cv")) == {}

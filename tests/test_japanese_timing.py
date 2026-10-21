from types import SimpleNamespace as NS

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.domain import StateSegment, SourceOtoRow
from universal_auto_oto.japanese_timing import _features, _detect_breath_start, refine_special_timing
from universal_auto_oto.voicebank_project_adapter import _flatten_slots, _planned_aliases, parse_filename_phone_slots


def _state(phone, start, end):
    return StateSegment(f'p{phone}.state', phone, 'stable', 'periodic', start, end, .8,
                        raw_start_ms=start, raw_end_ms=end)


def test_long_split_and_breath_use_audio_without_reference_timings(tmp_path):
    rate = 16000
    tone = lambda seconds: .3*np.sin(2*np.pi*220*np.arange(round(rate*seconds))/rate)
    audio = np.r_[np.zeros(round(rate*.2)), tone(1), np.zeros(round(rate*.25)), tone(.7),
                  np.random.default_rng(5).normal(0, .025, round(rate*.25)), np.zeros(round(rate*.2))]
    path = tmp_path/'_ああhh.wav'
    sf.write(path, audio, rate)
    evidence = NS(wav_path=path, frame_times_ms=np.arange(15., len(audio)/rate*1000-15, 5.))
    rec = NS(wav=path.name, phones=('a', 'sil', 'a', 'exh'), metadata={'language_hint':'ja', 'format_hint':'cvvc'})
    original = [_state(0, 200, 1600), _state(1, 1600, 1800), _state(2, 1800, 2300), _state(3, 2300, 2600)]
    refined, policy = refine_special_timing(rec, evidence, original)
    assert refined[0].end_ms < 1220
    assert 2100 < refined[3].start_ms < 2200
    assert refined[3].end_ms < 2450
    assert refined[0].raw_end_ms == 1600  # raw decoder evidence survives the overlay
    assert {x['family'] for x in policy['applied']} == {'long', 'long_breath'}
    assert policy['base_oto_timing_reused'] is False
    assert len(policy['landmarks']) == 2


def test_no_clear_two_phrase_structure_stays_unresolved(tmp_path):
    rate = 8000
    path = tmp_path/'_いいhh.wav'
    sf.write(path, .3*np.sin(2*np.pi*250*np.arange(rate*2)/rate), rate)
    rec = NS(wav=path.name, phones=('i', 'sil', 'i', 'exh'), metadata={'language_hint':'ja', 'format_hint':'cvvc'})
    original = [_state(i, i*400, (i+1)*400) for i in range(4)]
    result, policy = refine_special_timing(rec, NS(wav_path=path, frame_times_ms=np.arange(15., 1985, 5.)), original)
    assert result is original
    assert policy['applied'] == [] and policy['unresolved']


def test_head_internal_take_and_explicit_ordinals_are_distinct():
    name = '_lらlりlるlれlろlらんlら.wav'
    _, slots = _flatten_slots(parse_filename_phone_slots(name, format_hint='cvvc', language_hint='ja'))
    plans = _planned_aliases([SourceOtoRow(name, a, i) for i, a in enumerate(
        ['- lら', 'lら', 'lら#1', 'lら#2'])], slots, 'cvvc', name, language_hint='ja')
    assert [p.phone_indices for p in plans] == [(0, 1), (10, 11), (0, 1), (10, 11)]


def test_continuous_long_recording_has_no_invented_second_vowel_or_silence(tmp_path):
    from universal_auto_oto.alias_generation import generate_alias_rows
    from universal_auto_oto.voicebank_project_adapter import build_routed_voicebank_project
    rate = 8000
    name = '_ああhh.wav'
    audio = np.r_[np.zeros(800), .3*np.sin(2*np.pi*250*np.arange(rate*2)/rate),
                  np.random.default_rng(7).normal(0, .02, 1600), np.zeros(800)]
    sf.write(tmp_path/name, audio, rate)
    build = build_routed_voicebank_project(bank=tmp_path, wavs=[tmp_path/name],
        base_rows_by_wav={name:generate_alias_rows(name, format_hint='cvvc', language_hint='ja')},
        format_hint='cvvc', language_hint='ja', recursive=False, output_encoding='utf-8')
    rec = build.project.recordings[0]
    assert rec.phones == ('a', 'exh')
    assert rec.transcript_source == 'japanese_continuous_long_acoustic_layout_v1'
    assert [p.phone_indices for p in rec.aliases] == [(0,), (0, 1)]
    result, policy = refine_special_timing(rec, NS(wav_path=tmp_path/name,
        frame_times_ms=np.arange(15., len(audio)/rate*1000-15, 5.)), [_state(0, 100, 2100), _state(1, 2100, 2300)])
    assert len(result) == 2
    assert policy['applied'][-1]['layout'] == 'continuous'


def test_short_japanese_release_renders_without_anchor_repairs(tmp_path):
    from universal_auto_oto.domain import LandmarkTimeline, PhoneSegment
    from universal_auto_oto.template_engine import render_project_alias
    from universal_auto_oto.voicebank_project_adapter import build_routed_voicebank_project
    name = '_ああhh.wav'
    (tmp_path/name).write_bytes(b'contract-only')
    project = build_routed_voicebank_project(bank=tmp_path, wavs=[tmp_path/name],
        base_rows_by_wav={name:[SourceOtoRow(name, 'a H', 0)]}, format_hint='cvvc',
        language_hint='ja', recursive=False, output_encoding='utf-8').project
    rec = project.recordings[0]
    times = [(100., 1000.), (1000., 1200.), (1200., 2000.), (2000., 2030.)]
    phones = tuple(PhoneSegment(i, p, *times[i], .8, ()) for i, p in enumerate(rec.phones))
    timeline = LandmarkTimeline(1, name, *(['0'*64]*5), None, phones, (), 100., 2030., .8, (), {}, 1000, 2500.)
    # The general template used to put fixed at 2025 and cutoff at 2015.
    _, trace = render_project_alias(project, rec, timeline, rec.aliases[0], source_index=0)
    assert not trace.repairs
    assert trace.final_anchors['pre_abs'] <= trace.final_anchors['fixed_abs'] < trace.final_anchors['cutoff_abs'] <= 2030


def test_refinement_never_contaminates_identity_template_passes(monkeypatch):
    import universal_auto_oto.project_runtime as runtime
    monkeypatch.setattr(runtime, 'has_adjacent_sonorants', lambda *args: True)
    monkeypatch.setattr(runtime.SonorantTemplates, 'build', lambda *args, **kwargs: NS(summary=lambda: {}))
    calls = []
    class Aligner:
        cepstra = {}
        def map(self, label, work, **kwargs):
            calls.append((label, work))
            return [NS(timeline=NS(phone_segments=[])) for _ in work]
    recordings = [NS(wav='_lらlり.wav', phones=('l','a','l','i'), metadata={}),
                  NS(wav='_あsかsき.wav', phones=('a','s','k','a','s','k','i'), metadata={})]
    project = NS(inventory={}, wav_path=lambda r: r.wav)
    runtime._identity_passes(project, recordings, {'japanese_special_refinement':True}, .7, 3, 'speech', Aligner())
    assert all(not options['japanese_special_refinement'] for _, work in calls[:3] for _, options in work)
    assert calls[-1][0] == 'refine:japanese'
    assert [r.wav for r, _ in calls[-1][1]] == ['_lらlり.wav']


def _release_cues(tmp_path, variant, pitch=250):
    rate = 16000
    times = np.arange(round(2.8*rate))/rate
    active = (times >= .2) & (times < 2.5)
    tone = .2*(np.sin(2*np.pi*pitch*times)+.18*np.sin(2*np.pi*4*pitch*times)+.08*np.sin(2*np.pi*8*pitch*times))
    rng = np.random.default_rng(41)
    background = rng.normal(0, .0003, len(times))
    if variant == 'voiced_breath':
        release = np.clip((times-1.9)/.05, 0, 1)
        envelope = 1-.25*release
        audio = active*(tone*envelope + rng.normal(0, .045, len(times))*release)+background
    elif variant == 'fade':
        envelope = 10**(-25*np.clip((times-1.9)/.6, 0, 1)/20)
        audio = active*tone*envelope+background
    elif variant == 'tremolo':
        audio = active*tone*(1+.2*np.sin(2*np.pi*6*times))+background
    elif variant == 'pitch_glide':
        instantaneous = pitch+170*np.clip((times-1.9)/.6, 0, 1)
        phase = 2*np.pi*np.cumsum(instantaneous)/rate
        changing_tone = .2*(np.sin(phase)+.18*np.sin(4*phase)+.08*np.sin(8*phase))
        envelope = 10**(-25*np.clip((times-1.9)/.6, 0, 1)/20)
        audio = active*changing_tone*envelope+background
    elif variant == 'short_noise':
        spike = (times > 2.1) & (times < 2.12)
        audio = active*tone+rng.normal(0, .1, len(times))*spike+background
    else:
        raise AssertionError(variant)
    path = tmp_path/f'{variant}.wav'
    sf.write(path, audio, rate)
    return _features(NS(wav_path=path, frame_times_ms=np.arange(15., 2785., 5.)))


@pytest.mark.parametrize('pitch', [220, 330, 520])
def test_voiced_breath_is_detected_before_periodicity_disappears(tmp_path, pitch):
    cues = _release_cues(tmp_path, 'voiced_breath', pitch)
    start, tail = cues['phrases'][-1]
    onset, evidence = _detect_breath_start(cues, start, tail)
    assert onset is not None and 1880 <= onset <= 2060
    index = np.argmin(abs(cues['times']-onset))
    assert cues['harmonicity'][index] > .65  # the old voicing-end test still sees a vowel
    assert evidence['reason'] == 'sustained_spectral_release'


def test_fade_tremolo_and_short_noise_are_not_breath_labels(tmp_path):
    for variant in ['fade', 'tremolo', 'pitch_glide', 'short_noise']:
        cues = _release_cues(tmp_path, variant)
        onset, evidence = _detect_breath_start(cues, *cues['phrases'][-1])
        assert onset is None, (variant, onset, evidence)

from dataclasses import replace
import json

import numpy as np
import soundfile as sf

from universal_auto_oto.alignment import _isolated_sonorant_states
from universal_auto_oto.acoustic_evidence import extract_acoustic_evidence
from universal_auto_oto.domain import GuideAnchor, SourceOtoRow
from universal_auto_oto.generator import GenerationOptions, generate_voicebank
from universal_auto_oto.sequence_compiler import compile_recording_states
from universal_auto_oto.voicebank_project_adapter import build_routed_voicebank_project


def test_runtime_profile_renders_hold_without_collapsed_cv_anchors(tmp_path):
    bank = tmp_path / 'bank'
    bank.mkdir()
    rate = 16000
    t = np.arange(rate) / rate
    wave = np.concatenate((np.zeros(1600), .15*np.sin(2*np.pi*220*t), np.zeros(1600)))
    sf.write(bank / '_N(vf).wav', wave, rate)
    base = bank / 'oto.ini'
    base.write_text('_N(vf).wav=*N,900,1,-2,3,4\n_N(vf).wav=N *,900,1,-2,3,4\n')
    original = base.read_bytes()
    output = tmp_path / 'generated'
    report = generate_voicebank(GenerationOptions(bank=bank, base_oto=base, output=output,
        backend='routed_hsmm', format_hint='cvvc', language_hint='korean',
        development_alignment_profile='isolated_sonorant_v1'))
    timeline = json.loads(next((output/'timelines').glob('*.json')).read_text())
    policy = timeline['provenance']['isolated_sonorant_policy']
    assert policy['review_only'] and not policy['promotion_eligible']
    assert timeline['provenance']['hsmm']['meta']['trailing_gap_weighted']
    assert policy['original_state']['acoustic_role'] == 'consonant'
    assert policy['compiled_state']['acoustic_role'] == 'periodic'
    rows = [json.loads(line) for line in (output/'row_report.jsonl').read_text().splitlines()]
    assert rows[0]['template_id'] == 'fry_to_sustain'
    assert rows[0]['template_repair_log'] == []
    anchors = rows[0]['final_anchors']
    assert abs(anchors['pre_abs']-anchors['overlap_abs']-45) < 1e-6
    assert abs(anchors['fixed_abs']-anchors['pre_abs']-55) < 1e-6
    assert not report['provenance']['base_oto_timing_reused']
    assert base.read_bytes() == original

    project = build_routed_voicebank_project(bank=bank, wavs=[bank/'_N(vf).wav'],
        base_rows_by_wav={'_N(vf).wav': [SourceOtoRow('_N(vf).wav', '*N', 0)]},
        format_hint='cvvc', language_hint='korean', recursive=False, output_encoding='utf-8').project
    recording = project.recordings[0]
    evidence = extract_acoustic_evidence(bank/'_N(vf).wav', feature_backend='numpy_multiband_pitch_v2',
        internal_sample_rate=44100, convert_to_mono=True, resampler='soxr_hq')
    compiled = compile_recording_states(recording, project.inventory)
    # Explicit transcript/label routes must retain their authored topology.
    for changed in (replace(recording, transcript_source='recordings_manifest'),
                    replace(recording, label_source='manual_gold'),
                    replace(recording, guide_anchors=(GuideAnchor('start', 100, 1),)),
                    replace(recording, phones=('vf', 'n', 'vf', 'n'))):
        states, applied = _isolated_sonorant_states(project, changed, evidence, compiled)
        assert states is compiled and applied is None


def test_profile_leaves_ordinary_cv_alignment_and_oto_identical(tmp_path):
    bank = tmp_path/'bank'
    bank.mkdir()
    sf.write(bank/'ka.wav', np.sin(2*np.pi*220*np.arange(12000)/16000)*.15, 16000)
    timelines = []
    outputs = []
    for profile in (None, 'isolated_sonorant_v1', 'cvvc_slot_rhythm_v1', 'cvvc_slot_rhythm_v2', 'cvvc_slot_rhythm_v3', 'cvvc_slot_rhythm_v4', 'cvvc_slot_rhythm_v5', 'cvvc_slot_rhythm_v6', 'cvvc_slot_rhythm_v7', 'cvvc_slot_rhythm_v8', 'cvvc_slot_rhythm_v9'):
        output = tmp_path/str(profile)
        generate_voicebank(GenerationOptions(bank=bank, output=output, backend='routed_hsmm',
            format_hint='cv', language_hint='korean', development_alignment_profile=profile))
        outputs.append((output/'oto.ini').read_bytes())
        timelines.append(next((output/'timelines').glob('*.json')).read_bytes())
    assert all(output == outputs[0] for output in outputs)
    assert all(timeline == timelines[0] for timeline in timelines)


def test_v11_and_v12_leave_a_one_syllable_cv_identical_to_v10(tmp_path):
    bank = tmp_path/'bank'
    bank.mkdir()
    sf.write(bank/'ka.wav', np.sin(2*np.pi*220*np.arange(12000)/16000)*.15, 16000)
    outputs, timelines = [], []
    for profile in ('cvvc_slot_rhythm_v10', 'cvvc_slot_rhythm_v11', 'cvvc_slot_rhythm_v12'):
        output = tmp_path/profile
        generate_voicebank(GenerationOptions(bank=bank, output=output, backend='routed_hsmm',
            format_hint='cv', language_hint='korean', development_alignment_profile=profile))
        outputs.append((output/'oto.ini').read_bytes())
        timelines.append(next((output/'timelines').glob('*.json')).read_bytes())
    assert all(output == outputs[0] for output in outputs) and all(t == timelines[0] for t in timelines)


def test_pause_aware_gaps_do_not_skip_sounding_frames(tmp_path):
    import json
    bank = tmp_path/'bank'
    bank.mkdir()
    sf.write(bank/'ka.wav', np.sin(2*np.pi*220*np.arange(12000)/16000)*.15, 16000)
    phones = {}
    for profile in (None, 'pause_aware_v1'):
        output = tmp_path/str(profile)
        generate_voicebank(GenerationOptions(bank=bank, output=output, backend='routed_hsmm',
            format_hint='cv', language_hint='korean', development_alignment_profile=profile))
        timeline = json.loads(next((output/'timelines').glob('*.json')).read_text(encoding='utf-8'))
        phones[profile] = [(p['start_ms'], p['end_ms']) for p in timeline['phone_segments']]
    # The plain decode leaves a gap inside the continuous tone; a pause-aware decode does not.
    assert phones[None][1][0] > phones[None][0][1]
    assert phones['pause_aware_v1'][1][0] - phones['pause_aware_v1'][0][1] <= 5.0 + 1e-6

"""Review-only acoustic refinement for explicit Japanese CVVC recording families.

No reference OTO timings are read. Weak/missing acoustic cues retain the HSMM
result and are recorded as unresolved rather than replaced by a guessed boundary.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import soundfile as sf

from .japanese_cvvc import special_slots


def long_phrase_layout(path):
    """Only count substantial audible phrases; never use existing OTO boundaries."""
    audio, rate = sf.read(path, dtype='float64', always_2d=True)
    mono = audio.mean(axis=1)
    window, hop = max(16, round(.03*rate)), max(1, round(.005*rate))
    starts = np.arange(0, max(1, len(mono)-window), hop)
    frames = np.pad(mono, (0, window))[starts[:, None]+np.arange(window)]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20*np.log10(np.maximum(rms, 1e-9)/max(float(rms.max()), 1e-9))
    active = _fill_short_gaps(db > -32, round(.065*rate/hop))
    count = sum((b-a)*hop/rate >= .35 for a, b in _runs(active))
    return 'continuous' if count == 1 else 'separated' if count == 2 else 'unresolved'


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _fill_short_gaps(mask, maximum):
    result = mask.copy()
    for a, b in _runs(~mask):
        if a and b < len(mask) and b-a <= maximum:
            result[a:b] = True
    return result


def _features(evidence):
    audio, rate = sf.read(evidence.wav_path, dtype='float64', always_2d=True)
    mono = audio.mean(axis=1)
    times = evidence.frame_times_ms
    window = max(16, round(.03*rate))
    starts = np.round(times*rate/1000).astype(int)
    padded = np.pad(mono, (window//2, window))
    frames = padded[starts[:, None]+np.arange(window)]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20*np.log10(np.maximum(rms, 1e-9)/max(float(rms.max()), 1e-9))
    spectrum = np.abs(np.fft.rfft(frames*np.hanning(window), axis=1))**2
    frequencies = np.fft.rfftfreq(window, 1/rate)
    total = spectrum.sum(axis=1)+1e-12
    low = spectrum[:, frequencies < 500].sum(axis=1)/total
    mid = spectrum[:, (frequencies >= 500) & (frequencies < 3000)].sum(axis=1)/total
    balance = low-mid
    bands = np.stack([spectrum[:, (frequencies >= a) & (frequencies < b)].sum(axis=1)/total
                      for a, b in [(0, 500), (500, 1500), (1500, 3000), (3000, 8000)]], axis=1)
    step = float(np.median(np.diff(times)))
    active = _fill_short_gaps(db > -32, max(1, round(65/step)))
    phrases = [(float(times[a]), float(times[b-1])) for a, b in _runs(active)
               if (b-a)*step >= 350]
    centered = (frames-frames.mean(axis=1, keepdims=True))*np.hanning(window)
    nfft = 1 << (2*window-1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(centered, n=nfft, axis=1))**2, n=nfft, axis=1)
    harmonicity = ac[:, max(1, int(rate/700)):min(window, int(rate/70))].max(axis=1)/np.maximum(ac[:, 0], 1e-12)
    lags = max(1, int(rate/700))+ac[:, max(1, int(rate/700)):min(window, int(rate/70))].argmax(axis=1)
    f0 = rate/lags
    distance = np.abs((frequencies[None, :]+f0[:, None]/2) % f0[:, None]-f0[:, None]/2)
    noise_band = (frequencies >= 1500) & (frequencies <= 8000)
    interharmonic = noise_band[None, :] & (distance > np.maximum(1.5*rate/window, .12*f0[:, None]))
    noise_power = (spectrum*interharmonic).sum(axis=1)/np.maximum(interharmonic.sum(axis=1), 1)
    band_power = spectrum[:, noise_band].mean(axis=1)+1e-15
    flatness_db = 10*np.log10(np.exp(np.log(np.maximum(spectrum[:, noise_band], 1e-15)).mean(axis=1))/band_power)
    noise_ratio_db = 10*np.log10(np.maximum(noise_power, 1e-15)/band_power)
    noise_db = 10*np.log10(np.maximum(noise_power, 1e-15))
    voiced = (harmonicity >= .65) & (db > -32)
    return dict(times=times, db=db, balance=balance, step=step, phrases=phrases,
                voiced=voiced, rms=rms, bands=bands, harmonicity=harmonicity,
                noise_ratio_db=noise_ratio_db, noise_db=noise_db, flatness_db=flatness_db)


def _retime(states, phone, start, end):
    selected = [s for s in states if s.phone_index == phone]
    if not selected or end-start < len(selected)*5:
        return states
    lengths = np.array([max(1., s.end_ms-s.start_ms) for s in selected])
    cuts = start+np.r_[0, np.cumsum(lengths/lengths.sum())]*(end-start)
    replacements = {s.instance_id: replace(s, start_ms=float(cuts[i]), end_ms=float(cuts[i+1]))
                    for i, s in enumerate(selected)}
    return [replacements.get(s.instance_id, s) for s in states]


def _detect_breath_start(cues, start, tail):
    """Find a sustained noisy release, including one that is still periodic.

    Inter-harmonic noise is compared with this vowel's own stable tail and
    with the recording's quiet noise floor. Energy fade alone is insufficient.
    These acoustic thresholds are review heuristics, not calibrated labels.
    """
    times, step = cues['times'], cues['step']
    radius = max(1, round(15/step))
    def smooth(values):
        return np.array([np.median(values[max(0, i-radius):i+radius+1]) for i in range(len(values))])
    energy, noise = smooth(cues['db']), smooth(cues['noise_ratio_db'])
    flatness = smooth(cues['flatness_db'])
    # Select a clean, stable reference by sound, rather than assuming the
    # last 400 ms before the tail cannot already contain a long breath.
    windows = []
    for end in np.arange(start+350, tail-100, 40):
        selected = (times >= end-200) & (times <= end)
        head = (times >= end-200) & (times <= end-150)
        last = (times >= end-50) & (times <= end)
        level_shift = abs(float(np.median(energy[head])-np.median(energy[last])))
        if (selected.sum()*step >= 150 and cues['voiced'][selected].mean() >= .6
                and np.std(energy[selected]) <= 3.5 and level_shift <= 2):
            windows.append((float(np.median(noise[selected])), end, selected))
    if not windows:
        return None, {'reason':'insufficient_stable_release_reference'}
    clean_limit = float(np.percentile([w[0] for w in windows], 25))+3
    reference = [w for w in windows if w[0] <= clean_limit][-1][2]
    reference_end = float(times[np.flatnonzero(reference)[-1]])
    reference_db = float(np.median(energy[reference]))
    reference_noise = float(np.median(noise[reference]))
    reference_harmonicity = float(np.median(cues['harmonicity'][reference]))
    reference_flatness = float(np.median(flatness[reference]))
    quiet = cues['db'] < -45
    floor = float(np.percentile(cues['noise_db'][quiet], 90)) if quiet.any() else float(np.percentile(cues['noise_db'], 10))
    decay = reference_db-energy
    noise_rise = noise-reference_noise
    harmonicity_drop = reference_harmonicity-smooth(cues['harmonicity'])
    flatness_rise = flatness-reference_flatness
    supported = ((noise > -6) & (cues['noise_db'] > floor+6) & (energy > -38)
                 & (times > reference_end) & (times <= tail-45))
    change = (((decay >= 5.5) & (noise_rise >= 4))
              | ((noise_rise >= 8) & (harmonicity_drop >= .04))
              | ((decay >= 6) & (noise_rise >= 2) & (flatness_rise >= 3)))
    mask = _fill_short_gaps(supported & change, max(1, round(15/step)))
    runs = [(a, b) for a, b in _runs(mask) if (b-a)*step >= 50]
    diagnostic = dict(detector='voiced_breath_spectral_change_v1', reference_db=reference_db,
                      reference_noise_ratio_db=reference_noise, noise_floor_db=floor,
                      reference_interval_ms=[float(times[np.flatnonzero(reference)[0]]),
                                             float(times[np.flatnonzero(reference)[-1]])])
    if not runs:
        return None, dict(diagnostic, reason='no_sustained_noise_and_release_change')
    index = runs[0][0]
    return float(times[index]), dict(diagnostic, reason='sustained_spectral_release',
        energy_decay_db=float(decay[index]), noise_ratio_rise_db=float(noise_rise[index]),
        harmonicity=float(cues['harmonicity'][index]), flatness_rise_db=float(flatness_rise[index]),
        sustained_ms=float((runs[0][1]-index)*step), boundary_resolution_ms=30.)


def refine_special_timing(recording, evidence, states):
    if (recording.metadata.get('language_hint') not in {'ja', 'jp', 'japanese'}
            or recording.metadata.get('format_hint') != 'cvvc'):
        return states, None
    units = special_slots(recording.wav)
    if not units:
        return states, None
    cues = _features(evidence)
    policy = dict(version='japanese_special_timing_v2', review_only=True,
                  base_oto_timing_reused=False, applied=[], unresolved=[], landmarks=[])
    if units[0][1].endswith('L'):
        phrases = cues['phrases']
        continuous = len(phrases) == 1 and len(recording.phones) == 2
        if (not continuous and len(phrases) != 2) or min((b-a for a, b in phrases), default=0) < 500:
            policy['unresolved'].append('long_vowel_requires_two_clear_phrases')
            return states, policy
        (start, end) = phrases[0]
        (second, tail) = phrases[-1]
        if not continuous:
            states = _retime(states, 0, start+30, end-25)
            states = _retime(states, 1, end, second)
        times = cues['times']
        radius = max(2, round(30/cues['step']))
        variability = np.array([np.std(cues['db'][max(0, i-radius):i+radius+1]) for i in range(len(times))])
        stable = (cues['voiced'] & (variability < 1.5) & (times > start+40) & (times < end-200))
        stable_runs = [(a, b) for a, b in _runs(_fill_short_gaps(stable, 2)) if (b-a)*cues['step'] >= 80]
        if stable_runs:
            pre = float(times[stable_runs[0][0]])
            policy['landmarks'].extend([dict(phone_index=0, label='japanese_long_pre', time_ms=pre),
                                       dict(phone_index=0, label='japanese_long_fixed', time_ms=pre+40)])
        else:
            policy['unresolved'].append('no_stable_long_vowel_attack')
        policy['applied'].append(dict(family='long', first_phrase=[start, end]))
        times = cues['times']
        mask = cues['voiced'] & (times >= second) & (times <= tail)
        runs = [(a, b) for a, b in _runs(_fill_short_gaps(mask, round(25/cues['step'])))
                if (b-a)*cues['step'] >= 80]
        if not runs:
            policy['unresolved'].append('no_sustained_voicing_in_breath_phrase')
            return states, policy
        breath, breath_evidence = _detect_breath_start(cues, second, tail)
        policy['breath_detection'] = breath_evidence
        if breath is None:
            breath = float(times[runs[-1][1]-1])
            if tail-breath < 25:
                policy['unresolved'].append('breath_too_short_to_separate')
                if not continuous:
                    return states, policy
                # No independent breath cue: retain a clipped decoder boundary.
                breath = float(np.clip(next(s.start_ms for s in states if s.phone_index == 1), start+400, tail-30))
            else:
                policy['breath_detection']['fallback'] = 'sustained_voicing_end'
        if continuous:
            states = _retime(states, 0, start+30, breath)
            states = _retime(states, 1, breath, max(tail, breath+30))
        else:
            states = _retime(states, 2, second, breath)
            states = _retime(states, 3, breath, tail)
        policy['applied'].append(dict(family='long_breath', first_phrase=[start, end],
                                      second_phrase=[second, tail], breath_start_ms=breath,
                                      layout='continuous' if continuous else 'separated'))
    elif any(surface.startswith('ん') and len(surface) > 1 for _, surface in units):
        if len(cues['phrases']) != 2:
            policy['unresolved'].append('nasal_requires_two_clear_phrases')
            return states, policy
        times, balance = cues['times'], cues['balance']
        for phrase, (start, end) in enumerate(cues['phrases']):
            base = phrase*5  # vowel, n, nasal target, stop, vowel
            quiet = (cues['db'] < -28) & (times > start+350) & (times < end-220)
            closures = [(a, b) for a, b in _runs(quiet) if (b-a)*cues['step'] >= 10]
            if not closures:
                policy['unresolved'].append(f'no_stop_closure:{phrase}')
                continue
            closure = float(times[closures[0][0]])
            candidates = np.flatnonzero((times > start+100) & (times < closure-100))
            scores = []
            for i in candidates:
                t = times[i]
                before = balance[(times >= t-100) & (times <= t-20)]
                after = balance[(times >= t+20) & (times <= min(t+100, closure-25))]
                scores.append(float(np.median(after)-np.median(before)) if len(before) and len(after) else -1.)
            if not scores or max(scores) < .25:
                policy['unresolved'].append(f'weak_vowel_nasal_contrast:{phrase}')
                continue
            onset = float(times[candidates[int(np.argmax(scores))]])
            internal = next(s.start_ms for s in states if s.phone_index == base+2)
            # Only keep a decoded internal transition inside the observed nasal.
            # Without an independent contrast it stays explicitly unconfirmed.
            internal = float(np.clip(internal, onset+60, closure-35))
            policy['unresolved'].append(f'nasal_internal_transition_unconfirmed:{phrase}')
            states = _retime(states, base, start, onset)
            states = _retime(states, base+1, onset, internal)
            states = _retime(states, base+2, internal, closure)
            following_vowel = next(s.start_ms for s in states if s.phone_index == base+4)
            following_vowel = max(closure+30, min(following_vowel, end-100))
            states = _retime(states, base+3, closure, following_vowel)
            states = _retime(states, base+4, following_vowel, end)
            policy['applied'].append(dict(family='vowel_nasal', phrase=phrase,
                onset_ms=onset, closure_ms=closure, contrast=max(scores), internal_gold=False))
    elif any(phones[0] == 'trill' for phones, _ in units):
        times = cues['times']
        voiced_runs = [(a, b) for a, b in _runs(_fill_short_gaps(cues['voiced'], round(10/cues['step'])))
                       if (b-a)*cues['step'] >= 20]
        for slot in recording.metadata.get('phone_slots', ()):
            first, last = int(slot['start']), int(slot['end'])
            if recording.phones[first] != 'trill':
                continue
            original = {p: (min(s.start_ms for s in states if s.phone_index == p),
                            max(s.end_ms for s in states if s.phone_index == p))
                        for p in (first, first+1, last)}
            r_hint = original[first+1][0]
            candidates = [a for a, b in voiced_runs if times[a] > original[first][0]-30
                          and abs(times[a]-r_hint) <= 200]
            if not candidates:
                policy['unresolved'].append(f'no_trill_recovery:{first}')
                continue
            recovery = float(times[min(candidates, key=lambda i: abs(times[i]-r_hint))])
            active = np.flatnonzero((times >= original[first][0]) & (times < recovery) & (cues['db'] > -32))
            onset = float(times[active[0]]) if len(active) else recovery-10
            onset = min(onset, recovery-10)
            vowel = max(recovery+15, original[last][0])
            end = original[last][1]
            phrase = next(((a, b) for a, b in cues['phrases'] if a-30 <= recovery <= b), None)
            if phrase:
                end = min(end, phrase[1]-15)
            if end-vowel < 70:
                policy['unresolved'].append(f'trill_recovery_leaves_no_vowel:{first}')
                continue
            states = _retime(states, first, onset, recovery)
            states = _retime(states, first+1, recovery, vowel)
            states = _retime(states, last, vowel, end)
            policy['applied'].append(dict(family='trill', phone=first, recovery_ms=recovery,
                                          vowel_ms=vowel, end_ms=end, cue='sustained_periodicity_after_irregular_onset'))
    elif any(phones[0] == 'l' for phones, _ in units):
        times = cues['times']
        for slot in recording.metadata.get('phone_slots', ()):
            first, last = int(slot['start']), int(slot['end'])
            if recording.phones[first] != 'l':
                continue
            start = min(s.start_ms for s in states if s.phone_index == first)
            hint = min(s.start_ms for s in states if s.phone_index == last)
            end = max(s.end_ms for s in states if s.phone_index == last)
            candidates = np.flatnonzero((times >= max(start+15, hint-100)) & (times <= min(hint+120, end-100))
                                        & (cues['db'] > -30))
            scores = []
            for i in candidates:
                t = times[i]
                left = cues['bands'][(times >= t-45) & (times <= t-10)]
                right = cues['bands'][(times >= t+10) & (times <= t+45)]
                scores.append(float(np.abs(left.mean(axis=0)-right.mean(axis=0)).sum()) if len(left) and len(right) else 0.)
            if not scores or max(scores) < .15:
                policy['unresolved'].append(f'weak_lateral_vowel_contrast:{first}')
                continue
            boundary = float(times[candidates[int(np.argmax(scores))]])
            phrase = next(((a, b) for a, b in cues['phrases'] if a-30 <= boundary <= b), None)
            if phrase:
                start, end = max(start, phrase[0]), min(end, phrase[1]-15)
            states = _retime(states, first, start, boundary)
            states = _retime(states, last, boundary, end)
            policy['applied'].append(dict(family='lateral', phone=first, vowel_ms=boundary, spectral_contrast=max(scores)))
    return states, policy

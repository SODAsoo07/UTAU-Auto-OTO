"""Generate a self-contained HTML labeling page for A/B listening.

For each sampled case, the page shows the alias / voicebank / role plus three
"Play" buttons (one per version) that jump to that note's start time in the
voicebank's full USTx render. Defect categories are toggled with buttons or
keyboard shortcuts. Progress auto-saves to `localStorage` and a "Save CSV"
button downloads a CSV in the same schema as the original sample CSV.

Inputs
------
- Manifest (provides voicebank ids, languages, USTx paths, versions).
- Sample CSV (which case_ids to listen to; produced by `sample_ab_labels.py`).
- Run directory (where this script expects rendered wavs at the convention
  path `<run-dir>/<version>/<voicebank>/render.wav`).

Output
------
A single `.html` file. Open in Chrome/Edge/Firefox. No server needed.

Render wav convention
---------------------
After installing each version's OTO into the voicebank (see
`install_voicebank_oto.py`), open the USTx in OpenUtau and export the full
mix as wav. Save to::

    <run-dir>/<version>/<voicebank>/render.wav

(Alternatively, set `audio_path` per voicebank-version pair in the manifest;
not yet supported.) Missing wavs make the play buttons inert but everything
else still works.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from typing import Any

from core.coarse_crnn.alias_role import classify_alias_role, normalize_role
from core.coarse_crnn.boundary_targets import _infer_alias_type as _boundary_alias_type
from core.coarse_crnn.boundary_targets import _infer_transition_type as _boundary_transition_type
from core.coarse_crnn.lang import normalize_language
from ml.scripts.coarse_crnn.ustx_timing import (
    collect_notes_with_timing,
    index_notes_by_lyric,
    parse_ustx_file,
    total_duration_ms,
)


DEFECT_KEYS: tuple[str, ...] = (
    "click",
    "consonant_clip",
    "vc_misalign",
    "overlap_unnatural",
    "cutoff_overreach",
)
DEFECT_LABELS_KO: dict[str, str] = {
    "click": "클릭",
    "consonant_clip": "자음 잘림",
    "vc_misalign": "VC 어긋남",
    "overlap_unnatural": "Overlap",
    "cutoff_overreach": "Cutoff 과욕",
}
DEFECT_SHORTCUTS: dict[str, str] = {
    "click": "1",
    "consonant_clip": "2",
    "vc_misalign": "3",
    "overlap_unnatural": "4",
    "cutoff_overreach": "5",
}


def _read_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "run_id" not in data or "voicebanks" not in data:
        raise SystemExit("manifest missing run_id or voicebanks")
    return data


def _read_sample_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{k: (v or "").strip() if isinstance(v, str) else v for k, v in row.items()} for row in reader]
    return fieldnames, rows


def _role_for(alias: str, language: str) -> str:
    lang = normalize_language(language) or "korean"
    alias_type = _boundary_alias_type(alias, language=lang)
    transition_type = _boundary_transition_type(alias, language=lang)
    return normalize_role(
        classify_alias_role(
            lang, alias,
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )


def _resolve_render_path(*, run_dir: str, version_id: str, vb_id: str) -> str:
    return os.path.normpath(os.path.join(run_dir, version_id, vb_id, "render.wav"))


def _relative_path(*, from_file: str, to_file: str) -> str:
    rel = os.path.relpath(to_file, start=os.path.dirname(from_file))
    # Use forward slashes so the browser handles paths consistently.
    return rel.replace("\\", "/")


def build_case_records(
    *,
    manifest: dict[str, Any],
    sample_rows: list[dict[str, str]],
    run_dir: str,
    out_html_path: str,
    max_matches_per_case: int = 5,
    case_offset_pad_ms: float = 80.0,
    case_post_pad_ms: float = 120.0,
) -> dict[str, Any]:
    """Build the JSON payload that the HTML page consumes."""
    voicebanks_by_id = {str(vb["id"]): vb for vb in manifest["voicebanks"]}
    version_ids = [str(v["id"]) for v in manifest.get("versions", [])]

    # Parse USTx once per voicebank
    notes_by_vb: dict[str, list] = {}
    ustx_warnings: list[str] = []
    for vb_id, vb in voicebanks_by_id.items():
        ustx_path = str(vb.get("ustx", "") or "")
        if not ustx_path or not os.path.isfile(ustx_path) or ustx_path.endswith(")"):
            ustx_warnings.append(f"voicebank {vb_id}: ustx path missing or invalid: {ustx_path!r}")
            notes_by_vb[vb_id] = []
            continue
        try:
            ustx = parse_ustx_file(ustx_path)
            notes = collect_notes_with_timing(ustx)
        except Exception as exc:
            ustx_warnings.append(f"voicebank {vb_id}: USTx parse failed: {exc}")
            notes = []
        notes_by_vb[vb_id] = notes

    # Audio source map per (voicebank, version)
    audio_sources: dict[str, str] = {}
    audio_status: dict[str, dict[str, Any]] = {}
    for vb_id in voicebanks_by_id:
        for ver in version_ids:
            wav_path = _resolve_render_path(run_dir=run_dir, version_id=ver, vb_id=vb_id)
            key = f"{vb_id}|{ver}"
            audio_sources[key] = _relative_path(from_file=out_html_path, to_file=wav_path)
            audio_status[key] = {
                "exists": os.path.isfile(wav_path),
                "abs_path": wav_path,
            }

    # Build per-case records (deduplicate by case_id)
    seen_case_ids: set[str] = set()
    cases: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    for row in sample_rows:
        case_id = str(row.get("case_id", "") or "")
        if not case_id or case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)
        vb_id = str(row.get("voicebank_id", "") or "")
        alias = str(row.get("alias", "") or "")
        language = str(row.get("language", "") or "")
        role = _role_for(alias, language)
        role_counts[role] = role_counts.get(role, 0) + 1
        # Find matches in USTx
        lyric_index = index_notes_by_lyric(notes_by_vb.get(vb_id, []))
        candidates = lyric_index.get(alias, [])
        match_records = [
            {
                "start_ms": float(max(0.0, n.start_ms - case_offset_pad_ms)),
                "end_ms": float(n.end_ms + case_post_pad_ms),
                "raw_start_ms": float(n.start_ms),
                "raw_end_ms": float(n.end_ms),
                "label": _fmt_timestamp(n.start_ms),
                "part_index": int(n.part_index),
                "note_index": int(n.note_index),
            }
            for n in candidates[:max_matches_per_case]
        ]
        cases.append(
            {
                "case_id": case_id,
                "voicebank_id": vb_id,
                "language": language,
                "alias": alias,
                "wav_name": str(row.get("wav_name", "") or ""),
                "role": role,
                "match_count": len(candidates),
                "matches": match_records,
            }
        )

    summary = {
        "run_id": str(manifest["run_id"]),
        "case_count": len(cases),
        "version_count": len(version_ids),
        "voicebank_count": len(voicebanks_by_id),
        "role_counts": role_counts,
        "ustx_warnings": ustx_warnings,
        "audio_status": audio_status,
    }
    payload = {
        "run_id": str(manifest["run_id"]),
        "versions": version_ids,
        "voicebanks": [
            {
                "id": vb_id,
                "language": str(voicebanks_by_id[vb_id].get("language", "")),
                "ustx_total_ms": float(total_duration_ms(notes_by_vb.get(vb_id, []))),
            }
            for vb_id in voicebanks_by_id
        ],
        "audio_sources": audio_sources,
        "audio_status": audio_status,
        "cases": cases,
        "defect_keys": list(DEFECT_KEYS),
        "defect_labels_ko": dict(DEFECT_LABELS_KO),
        "defect_shortcuts": dict(DEFECT_SHORTCUTS),
        "ustx_warnings": ustx_warnings,
    }
    return {"payload": payload, "summary": summary}


def _fmt_timestamp(ms: float) -> str:
    total_seconds = float(ms) / 1000.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds - 60 * minutes
    return f"{minutes}:{seconds:05.2f}"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>A/B Listening — __TITLE__</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 0; background: #fafafa; color: #222; }
  header { position: sticky; top: 0; z-index: 10; background: #fff;
           border-bottom: 1px solid #ddd; padding: 12px 20px;
           display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 18px; margin: 0; }
  header .stat { font-size: 13px; color: #555; }
  header button, header select { padding: 6px 12px; font-size: 13px; cursor: pointer; }
  header button.primary { background: #2a6fd6; color: white; border: none; }
  header button.primary:hover { background: #1a5fc6; }
  main { padding: 16px 20px; max-width: 1100px; margin: 0 auto; }
  .case { background: #fff; border: 1px solid #ddd; border-radius: 6px;
          margin-bottom: 14px; padding: 12px 16px; }
  .case.done { background: #f4faf4; border-color: #b6d8b6; }
  .case-header { display: flex; gap: 12px; align-items: baseline; margin-bottom: 8px; flex-wrap: wrap; }
  .case-id { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; color: #777; }
  .alias { font-size: 16px; font-weight: 600; }
  .meta { font-size: 12px; color: #555; }
  .role-vc { color: #c2410c; }
  .role-vcv, .role-v-cv { color: #b45309; }
  .role-cv, .role--cv { color: #047857; }
  .role-v, .role-vv { color: #1d4ed8; }
  .matches { font-size: 12px; color: #888; margin-bottom: 6px; }
  .version-row { display: flex; gap: 10px; align-items: center;
                 padding: 8px 0; border-top: 1px solid #f0f0f0; flex-wrap: wrap; }
  .version-row:first-of-type { border-top: none; }
  .version-name { font-weight: 600; min-width: 140px; font-family: ui-monospace, monospace; font-size: 13px; }
  .play { background: #eef; border: 1px solid #ccd; border-radius: 4px;
          padding: 4px 10px; cursor: pointer; font-size: 12px; }
  .play:hover { background: #dde; }
  .play:disabled { opacity: 0.4; cursor: not-allowed; }
  .defect-toggle { display: inline-flex; align-items: center; gap: 4px;
                   padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px;
                   background: #f8f8f8; cursor: pointer; font-size: 12px; }
  .defect-toggle.active { background: #fde2e2; border-color: #d97070; color: #b41818; }
  .defect-toggle input { display: none; }
  .notes-input { flex: 1; min-width: 180px; padding: 4px 6px; font-size: 12px;
                 border: 1px solid #ccc; border-radius: 4px; }
  .help { font-size: 12px; color: #555; line-height: 1.5;
          background: #fffbe6; border: 1px solid #f0d97f; border-radius: 4px;
          padding: 10px 14px; margin-bottom: 14px; }
  .warning { font-size: 12px; color: #b41818; background: #fdecec;
             border: 1px solid #e8a8a8; border-radius: 4px;
             padding: 8px 12px; margin-bottom: 10px; }
  .filter-bar { display: flex; gap: 10px; padding: 10px 20px; background: #fff;
                border-bottom: 1px solid #eee; flex-wrap: wrap; align-items: center; }
  .filter-bar label { font-size: 12px; color: #555; display: inline-flex; gap: 4px; align-items: center; }
  .audio-pool { display: none; }
</style>
</head>
<body>
<header>
  <h1>A/B Listening — __TITLE__</h1>
  <span class="stat">Cases: <strong id="case-count">0</strong></span>
  <span class="stat">Labeled: <strong id="labeled-count">0</strong> / <strong id="case-total">0</strong></span>
  <button id="save-csv" class="primary">Save CSV</button>
  <button id="load-csv">Load CSV</button>
  <button id="reset">Reset</button>
  <button id="help-toggle">Help</button>
  <input type="file" id="csv-import" accept=".csv" style="display:none">
</header>

<div class="filter-bar">
  <label>Voicebank: <select id="filter-vb"><option value="">all</option></select></label>
  <label>Role: <select id="filter-role"><option value="">all</option></select></label>
  <label>Status: <select id="filter-status">
    <option value="">all</option>
    <option value="todo">unlabeled</option>
    <option value="done">labeled</option>
  </select></label>
</div>

<main>
  <div id="help-panel" class="help" style="display:none">
    <strong>키보드 단축키 (case에 포커스된 상태)</strong>:
    1=클릭, 2=자음잘림, 3=VC어긋남, 4=Overlap, 5=Cutoff 과욕 — 현재 포커스된 버전 행에 적용.
    Tab으로 다음 버튼/필드 이동. Esc로 재생 정지.
    <br><strong>저장</strong>: 변경 즉시 브라우저에 저장(localStorage). 백업·공유는 "Save CSV"로 파일 다운로드.
    <br><strong>오디오 누락</strong>: 각 버전의 render.wav가 없으면 Play 버튼이 비활성화됨. 매니페스트 경로 규약대로 wav를 두면 자동으로 활성화.
  </div>
  <div id="audio-warnings"></div>
  <div id="case-list"></div>
</main>

<div class="audio-pool" id="audio-pool"></div>

<script id="payload" type="application/json">__PAYLOAD_JSON__</script>
<script>
(function() {
  const PAYLOAD = JSON.parse(document.getElementById('payload').textContent);
  const RUN_ID = PAYLOAD.run_id;
  const VERSIONS = PAYLOAD.versions;
  const VOICEBANKS = PAYLOAD.voicebanks;
  const AUDIO_SRCS = PAYLOAD.audio_sources;
  const AUDIO_STATUS = PAYLOAD.audio_status;
  const CASES = PAYLOAD.cases;
  const DEFECT_KEYS = PAYLOAD.defect_keys;
  const DEFECT_LABELS = PAYLOAD.defect_labels_ko;
  const DEFECT_SHORTCUTS = PAYLOAD.defect_shortcuts;
  const STORAGE_KEY = 'ablisten/' + RUN_ID;

  let state = loadState();
  let activeAudio = null;
  let stopHandler = null;
  let focusedCaseId = null;
  let focusedVersionId = null;

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return JSON.parse(raw);
    } catch (e) { console.warn('loadState failed', e); }
    return {};
  }
  function saveState() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
    catch (e) { console.warn('saveState failed', e); }
  }
  function getCaseState(caseId, versionId) {
    state[caseId] = state[caseId] || {};
    state[caseId][versionId] = state[caseId][versionId] || { defects: {}, notes: '' };
    return state[caseId][versionId];
  }
  function isCaseDone(caseId) {
    const s = state[caseId];
    if (!s) return false;
    return VERSIONS.every(v => {
      const cs = s[v];
      if (!cs) return false;
      // A version is "considered touched" if any defect was toggled OR notes typed
      const anyDefect = Object.values(cs.defects || {}).some(Boolean);
      const hasNotes = (cs.notes || '').trim().length > 0;
      return cs.touched || anyDefect || hasNotes;
    });
  }

  function buildAudioPool() {
    const pool = document.getElementById('audio-pool');
    pool.innerHTML = '';
    Object.entries(AUDIO_SRCS).forEach(([key, src]) => {
      const audio = document.createElement('audio');
      audio.id = 'audio-' + key;
      audio.src = src;
      audio.preload = 'metadata';
      pool.appendChild(audio);
    });
  }

  function pauseAll() {
    if (activeAudio) {
      activeAudio.pause();
      if (stopHandler) {
        activeAudio.removeEventListener('timeupdate', stopHandler);
        stopHandler = null;
      }
    }
    activeAudio = null;
  }

  function playFragment(vbId, verId, startMs, endMs) {
    pauseAll();
    const key = vbId + '|' + verId;
    const audio = document.getElementById('audio-' + key);
    if (!audio) return;
    const status = AUDIO_STATUS[key];
    if (status && status.exists === false) {
      alert('오디오 파일 없음:\\n' + (status.abs_path || src));
      return;
    }
    activeAudio = audio;
    audio.currentTime = Math.max(0, startMs / 1000);
    const stopAt = endMs / 1000;
    stopHandler = () => {
      if (audio.currentTime >= stopAt) {
        audio.pause();
        audio.removeEventListener('timeupdate', stopHandler);
        stopHandler = null;
        activeAudio = null;
      }
    };
    audio.addEventListener('timeupdate', stopHandler);
    audio.play().catch(err => {
      console.warn('play failed', err);
      pauseAll();
    });
  }

  function renderCases() {
    const list = document.getElementById('case-list');
    list.innerHTML = '';
    const fVb = document.getElementById('filter-vb').value;
    const fRole = document.getElementById('filter-role').value;
    const fStatus = document.getElementById('filter-status').value;
    let shown = 0, done = 0;
    CASES.forEach(c => {
      if (fVb && c.voicebank_id !== fVb) return;
      if (fRole && c.role !== fRole) return;
      const caseDone = isCaseDone(c.case_id);
      if (fStatus === 'todo' && caseDone) return;
      if (fStatus === 'done' && !caseDone) return;
      shown++;
      if (caseDone) done++;
      list.appendChild(renderCase(c, caseDone));
    });
    document.getElementById('case-count').textContent = shown;
    let totalDone = 0;
    CASES.forEach(c => { if (isCaseDone(c.case_id)) totalDone++; });
    document.getElementById('labeled-count').textContent = totalDone;
    document.getElementById('case-total').textContent = CASES.length;
  }

  function renderCase(c, caseDone) {
    const wrap = document.createElement('div');
    wrap.className = 'case' + (caseDone ? ' done' : '');
    wrap.dataset.caseId = c.case_id;

    const hdr = document.createElement('div');
    hdr.className = 'case-header';
    hdr.innerHTML = (
      '<span class="alias">' + escapeHtml(c.alias) + '</span>' +
      '<span class="meta">vb: <strong>' + escapeHtml(c.voicebank_id) + '</strong></span>' +
      '<span class="meta role-' + escapeHtml(c.role) + '">role: ' + escapeHtml(c.role) + '</span>' +
      '<span class="meta">lang: ' + escapeHtml(c.language) + '</span>' +
      '<span class="case-id">' + escapeHtml(c.case_id) + '</span>'
    );
    wrap.appendChild(hdr);

    const matches = document.createElement('div');
    matches.className = 'matches';
    if (c.matches.length === 0) {
      matches.innerHTML = '<em>USTx에서 alias "' + escapeHtml(c.alias) + '" 매칭 노트 없음 — Play 버튼이 0:00으로 시작</em>';
    } else if (c.match_count > c.matches.length) {
      matches.textContent = 'Matches: ' + c.matches.map(m => m.label).join(', ') +
        ' (showing first ' + c.matches.length + ' of ' + c.match_count + ')';
    } else {
      matches.textContent = 'Matches: ' + c.matches.map(m => m.label).join(', ');
    }
    wrap.appendChild(matches);

    VERSIONS.forEach(verId => {
      const row = document.createElement('div');
      row.className = 'version-row';
      row.tabIndex = 0;
      row.dataset.versionId = verId;
      row.addEventListener('focus', () => {
        focusedCaseId = c.case_id;
        focusedVersionId = verId;
      });

      const name = document.createElement('span');
      name.className = 'version-name';
      name.textContent = verId;
      row.appendChild(name);

      // Play buttons (one per match; fallback button at 0:00 if no matches)
      const playGroup = document.createElement('span');
      if (c.matches.length === 0) {
        const btn = document.createElement('button');
        btn.className = 'play';
        btn.textContent = '▶ 0:00 (no match)';
        const status = AUDIO_STATUS[c.voicebank_id + '|' + verId];
        if (status && status.exists === false) btn.disabled = true;
        btn.addEventListener('click', () => playFragment(c.voicebank_id, verId, 0, 30000));
        playGroup.appendChild(btn);
      } else {
        c.matches.forEach(m => {
          const btn = document.createElement('button');
          btn.className = 'play';
          btn.textContent = '▶ ' + m.label;
          const status = AUDIO_STATUS[c.voicebank_id + '|' + verId];
          if (status && status.exists === false) btn.disabled = true;
          btn.addEventListener('click', () => playFragment(c.voicebank_id, verId, m.start_ms, m.end_ms));
          playGroup.appendChild(btn);
          playGroup.appendChild(document.createTextNode(' '));
        });
      }
      row.appendChild(playGroup);

      // Defect toggles
      DEFECT_KEYS.forEach(key => {
        const cs = getCaseState(c.case_id, verId);
        const label = document.createElement('label');
        label.className = 'defect-toggle' + (cs.defects[key] ? ' active' : '');
        label.title = 'shortcut: ' + DEFECT_SHORTCUTS[key];
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = !!cs.defects[key];
        cb.addEventListener('change', () => {
          cs.defects[key] = cb.checked;
          cs.touched = true;
          label.classList.toggle('active', cb.checked);
          saveState();
          wrap.classList.toggle('done', isCaseDone(c.case_id));
          updateLabeledCounter();
        });
        label.appendChild(cb);
        label.appendChild(document.createTextNode(DEFECT_LABELS[key] + ' (' + DEFECT_SHORTCUTS[key] + ')'));
        row.appendChild(label);
      });

      const notes = document.createElement('input');
      notes.type = 'text';
      notes.className = 'notes-input';
      notes.placeholder = 'notes…';
      const cs = getCaseState(c.case_id, verId);
      notes.value = cs.notes || '';
      notes.addEventListener('input', () => {
        cs.notes = notes.value;
        cs.touched = true;
        saveState();
        wrap.classList.toggle('done', isCaseDone(c.case_id));
        updateLabeledCounter();
      });
      row.appendChild(notes);

      wrap.appendChild(row);
    });

    return wrap;
  }

  function updateLabeledCounter() {
    let done = 0;
    CASES.forEach(c => { if (isCaseDone(c.case_id)) done++; });
    document.getElementById('labeled-count').textContent = done;
  }

  function escapeHtml(text) {
    return String(text || '').replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function buildFilters() {
    const vbSel = document.getElementById('filter-vb');
    new Set(CASES.map(c => c.voicebank_id)).forEach(id => {
      const o = document.createElement('option');
      o.value = id; o.textContent = id;
      vbSel.appendChild(o);
    });
    const roleSel = document.getElementById('filter-role');
    new Set(CASES.map(c => c.role)).forEach(r => {
      const o = document.createElement('option');
      o.value = r; o.textContent = r;
      roleSel.appendChild(o);
    });
    [vbSel, roleSel, document.getElementById('filter-status')].forEach(el => {
      el.addEventListener('change', renderCases);
    });
  }

  function exportCSV() {
    const headers = ['case_id','voicebank_id','language','wav_name','alias','version_id',
                     ...DEFECT_KEYS, 'notes'];
    const lines = [headers.join(',')];
    CASES.forEach(c => {
      VERSIONS.forEach(ver => {
        const cs = (state[c.case_id] && state[c.case_id][ver]) || { defects: {}, notes: '' };
        const cells = [
          c.case_id, c.voicebank_id, c.language, c.wav_name, c.alias, ver,
          ...DEFECT_KEYS.map(k => cs.defects[k] ? '1' : '0'),
          (cs.notes || '').replace(/[\\r\\n]/g, ' ')
        ];
        lines.push(cells.map(cell => csvEscape(cell)).join(','));
      });
    });
    const blob = new Blob([lines.join('\\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = RUN_ID + '.labeled.csv';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
  function csvEscape(value) {
    const s = String(value == null ? '' : value);
    if (/[",\\n\\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function importCSV(file) {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || '');
      const rows = parseCSV(text);
      if (!rows.length) return;
      const headers = rows[0];
      const idx = name => headers.indexOf(name);
      let imported = 0;
      for (let i = 1; i < rows.length; i++) {
        const row = rows[i];
        if (!row || row.length < headers.length) continue;
        const caseId = row[idx('case_id')];
        const verId = row[idx('version_id')];
        if (!caseId || !verId) continue;
        const cs = getCaseState(caseId, verId);
        DEFECT_KEYS.forEach(k => {
          const v = row[idx(k)];
          if (v !== undefined && v !== '') cs.defects[k] = (v === '1' || v === 'true');
        });
        const notesIdx = idx('notes');
        if (notesIdx >= 0) cs.notes = row[notesIdx] || '';
        cs.touched = true;
        imported++;
      }
      saveState();
      renderCases();
      alert('Imported ' + imported + ' rows');
    };
    reader.readAsText(file, 'utf-8');
  }
  function parseCSV(text) {
    // Minimal RFC4180-ish CSV parser (handles quoted fields with "").
    const rows = []; let row = []; let cell = ''; let inQuote = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQuote) {
        if (ch === '"') {
          if (text[i+1] === '"') { cell += '"'; i++; } else { inQuote = false; }
        } else { cell += ch; }
      } else {
        if (ch === '"') inQuote = true;
        else if (ch === ',') { row.push(cell); cell = ''; }
        else if (ch === '\\r') { /* skip */ }
        else if (ch === '\\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
        else { cell += ch; }
      }
    }
    if (cell.length || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }

  function bindButtons() {
    document.getElementById('save-csv').addEventListener('click', exportCSV);
    document.getElementById('load-csv').addEventListener('click', () => {
      document.getElementById('csv-import').click();
    });
    document.getElementById('csv-import').addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) importCSV(file);
      e.target.value = '';
    });
    document.getElementById('reset').addEventListener('click', () => {
      if (!confirm('현재 라벨링 진행 상황을 모두 지울까요?')) return;
      state = {}; saveState(); renderCases();
    });
    document.getElementById('help-toggle').addEventListener('click', () => {
      const p = document.getElementById('help-panel');
      p.style.display = (p.style.display === 'none' ? 'block' : 'none');
    });
  }

  function bindShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
      if (e.key === 'Escape') { pauseAll(); return; }
      const matchKey = Object.entries(DEFECT_SHORTCUTS).find(([k, sc]) => sc === e.key);
      if (matchKey && focusedCaseId && focusedVersionId) {
        const [defectKey] = matchKey;
        const cs = getCaseState(focusedCaseId, focusedVersionId);
        cs.defects[defectKey] = !cs.defects[defectKey];
        cs.touched = true;
        saveState();
        renderCases();
        e.preventDefault();
      }
    });
  }

  function renderWarnings() {
    const target = document.getElementById('audio-warnings');
    target.innerHTML = '';
    const missing = Object.entries(AUDIO_STATUS).filter(([_, s]) => !s.exists);
    if (missing.length) {
      const div = document.createElement('div');
      div.className = 'warning';
      div.innerHTML = '<strong>오디오 파일 누락 ' + missing.length + '건</strong> — Play 버튼이 비활성됩니다:<br>' +
        missing.map(([k, s]) => '<code>' + escapeHtml(k) + '</code> → ' + escapeHtml(s.abs_path)).join('<br>');
      target.appendChild(div);
    }
  }

  // Boot
  buildAudioPool();
  buildFilters();
  renderWarnings();
  renderCases();
  bindButtons();
  bindShortcuts();
})();
</script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    title = html.escape(str(payload.get("run_id", "")))
    payload_json = json.dumps(payload, ensure_ascii=False)
    # `</script>` inside the JSON would break HTML parsing.
    payload_json = payload_json.replace("</", "<\\/")
    return (
        HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a self-contained HTML labeling page for A/B listening.")
    ap.add_argument("--manifest", required=True, help="Run manifest JSON (with voicebank ustx paths).")
    ap.add_argument("--sample-csv", required=True, help="Sampled label CSV (output of sample_ab_labels.py).")
    ap.add_argument("--run-dir", default="", help="Run directory (default: ml/eval/ab_listening/runs/<run_id>).")
    ap.add_argument("--out", default="", help="Output HTML path (default: ml/eval/ab_listening/reports/<run_id>.labeling.html).")
    args = ap.parse_args()

    manifest = _read_manifest(os.path.abspath(args.manifest))
    run_id = str(manifest["run_id"])
    run_dir = os.path.abspath(args.run_dir or os.path.join("ml", "eval", "ab_listening", "runs", run_id))
    out_path = os.path.abspath(args.out or os.path.join("ml", "eval", "ab_listening", "reports", f"{run_id}.labeling.html"))

    _, sample_rows = _read_sample_rows(os.path.abspath(args.sample_csv))

    result = build_case_records(
        manifest=manifest,
        sample_rows=sample_rows,
        run_dir=run_dir,
        out_html_path=out_path,
    )
    payload = result["payload"]
    summary = result["summary"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(render_html(payload))
    print(f"[listening-html] wrote {out_path}")
    print(f"  cases={summary['case_count']} versions={summary['version_count']} voicebanks={summary['voicebank_count']}")
    if summary["ustx_warnings"]:
        for warn in summary["ustx_warnings"]:
            print(f"  WARN: {warn}")
    missing_audio = [k for k, v in summary["audio_status"].items() if not v["exists"]]
    if missing_audio:
        print(f"  AUDIO MISSING ({len(missing_audio)}): {missing_audio}")
        print(f"  → render each (voicebank, version) USTx in OpenUtau and save as run-dir/<version>/<voicebank>/render.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import csv

import yaml

from universal_auto_oto.language_pack import find_language_pack
from universal_auto_oto.pack_style import load_style_profile, write_style_profile
from universal_auto_oto.style_sample import FIELDS, fit_style_from_sample, select_style_sample

PACK = find_language_pack("english", "vccv")


def _report(wav: str, alias: str) -> dict[str, object]:
    return {"wav": wav, "alias": alias}


def test_sample_takes_low_risk_rows_from_different_wavs():
    reports = [_report(f"w{k // 2}.wav", alias) for k, alias in enumerate(["a k"] * 12)]
    risks = [0.9 if k >= 6 else 0.1 * k for k in range(12)]
    chosen = select_style_sample(PACK, reports, risks, per_shape=3)
    assert len(chosen) == 3
    assert all(risks[p] < 0.9 for p in chosen)
    assert len({reports[p]["wav"] for p in chosen}) == 3


def test_sample_skips_rare_shapes():
    reports = [_report("a.wav", "a k"), _report("b.wav", "a k")]
    assert select_style_sample(PACK, reports, [0.1, 0.2]) == []


def test_fit_adds_shrunk_median_to_the_generation_style(tmp_path):
    base = tmp_path / "base.yaml"
    write_style_profile(base, PACK.pack_id, {"V C": {"pre": 10.0}, "V C|stop": {"pre": 4.0}, "CV": {"pre": 1.0}}, {})
    style = load_style_profile(base)
    generated = tmp_path / "oto.ini"
    lines, sample = [], []
    for k in range(5):
        wav = f"w{k}.wav"
        # generated: offset 1000, pre 1100; the user moved pre 40 ms later on four of five rows.
        moved = 40 if k < 4 else 0
        lines.append(f"{wav}=a k,1000,120,-400,{100 + moved},50")
        sample.append({"oto": str(generated), "line": k + 1, "wav": wav, "alias": "a k", "shape": "V C",
                       "wav_path": str(tmp_path / wav), "language_pack": PACK.pack_id,
                       "style_source": str(base), "style_sha256": style.sha256,
                       "offset": 1000, "overlap": 1050, "pre": 1100, "fixed": 1120, "cutoff": 1400})
    generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sample_path = tmp_path / "style_sample.csv"
    with open(sample_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(sample)
    result = fit_style_from_sample(sample_path, tmp_path / "mine.yaml")
    assert result["rows_checked"] == 5 and result["rows_changed"] == 4
    shapes = yaml.safe_load((tmp_path / "mine.yaml").read_text(encoding="utf-8"))["shapes"]
    # median 40 ms x 5/8 = 25 ms, added to both keys of the shape; other shapes untouched.
    assert shapes["V C"]["pre"] == 35.0
    assert shapes["V C|stop"]["pre"] == 29.0
    assert shapes["CV"]["pre"] == 1.0
    assert shapes["V C"]["offset"] == 0.0


def test_fit_weights_each_file_by_its_rows_of_the_shape(tmp_path):
    big, small = tmp_path / "big", tmp_path / "small"
    big.mkdir()
    small.mkdir()
    sample = []
    # big file: 30 "a k" rows, sampled rows moved +40 ms; small file: 10 rows, sampled rows moved -40 ms.
    for folder, rows, moved in ((big, 30, 40), (small, 10, -40)):
        lines = []
        for k in range(rows):
            pre = 100 + (moved if k < 5 else 0)
            lines.append(f"w{k}.wav=a k,1000,120,-400,{pre},50")
            if k < 5:
                sample.append({"oto": str(folder / "oto.ini"), "line": k + 1, "wav": f"w{k}.wav", "alias": "a k",
                               "shape": "V C", "wav_path": str(folder / f"w{k}.wav"), "language_pack": PACK.pack_id,
                               "style_source": "", "style_sha256": "",
                               "offset": 1000, "overlap": 1050, "pre": 1100, "fixed": 1120, "cutoff": 1400})
        (folder / "oto.ini").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sample_path = tmp_path / "style_sample.csv"
    with open(sample_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(sample)
    result = fit_style_from_sample(sample_path, tmp_path / "mine.yaml")
    # per file 40 x 5/8 = 25 and -25; weighted 30:10 -> 12.5
    assert result["corrections"]["V C"]["pre"] == 12.5

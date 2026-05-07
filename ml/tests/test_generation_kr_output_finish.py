from dataclasses import dataclass

from core.generation.kr_output_finish import persist_kr_generation_output


@dataclass
class FakePostFileContext:
    out_path: str
    tg_folder: str
    kr_profile: object
    custom_map: object
    custom_phonemes_path: str
    enable_ml_correction: bool
    auto_gen_format: str
    ml_policy: object
    runtime_report: object
    log_fn: object
    validate_fn: object
    normalize_key_fn: object
    ml_route: str


def test_persist_kr_generation_output_runs_write_post_pipeline_and_wav_mapping(monkeypatch):
    monkeypatch.delenv("UTOA_ML_ROUTE", raising=False)
    events = []
    logs = []
    errors = []

    def write_oto(path, lines):
        events.append(("write", path, tuple(lines)))

    def run_post(context):
        events.append(("post", context.out_path, context.ml_route, context.auto_gen_format))

    def apply_wav_map(path, wav_name_map):
        events.append(("map", path, dict(wav_name_map)))
        return 2

    renamed = persist_kr_generation_output(
        out_path="out.ini",
        final_lines=["a.wav=a,1,2,3,4,5"],
        tg_folder="tg",
        kr_profile={"p": 1},
        custom_map={"a": "b"},
        custom_phonemes_path="custom.txt",
        enable_ml_correction=True,
        auto_gen_format="cvvc",
        ml_policy="auto",
        runtime_report={},
        log_fn=logs.append,
        validate_fn=lambda *args: args,
        normalize_key_fn=lambda value: value,
        wav_name_map={"real.wav": "out.wav"},
        apply_output_wav_name_map_fn=apply_wav_map,
        errors=errors,
        write_oto_lines_fn=write_oto,
        run_kr_post_file_pipeline_fn=run_post,
        post_file_context_factory=FakePostFileContext,
    )

    assert renamed == 2
    assert events == [
        ("write", "out.ini", ("a.wav=a,1,2,3,4,5",)),
        ("post", "out.ini", "legacy", "cvvc"),
        ("map", "out.ini", {"real.wav": "out.wav"}),
    ]
    assert errors == []
    assert any("1차 생성 완료" in line for line in logs)
    assert any("WAV 이름 매핑 적용: 2개" in line for line in logs)


def test_persist_kr_generation_output_uses_ml_route_environment(monkeypatch):
    monkeypatch.setenv("UTOA_ML_ROUTE", "v2")
    events = []
    errors = []

    persist_kr_generation_output(
        out_path="out.ini",
        final_lines=[],
        tg_folder="tg",
        kr_profile=None,
        custom_map=None,
        custom_phonemes_path="",
        enable_ml_correction=False,
        auto_gen_format="cv",
        ml_policy="",
        runtime_report=None,
        log_fn=lambda _msg: None,
        validate_fn=lambda *args: args,
        normalize_key_fn=lambda value: value,
        wav_name_map={},
        apply_output_wav_name_map_fn=lambda _path, _map: 0,
        errors=errors,
        write_oto_lines_fn=lambda _path, _lines: None,
        run_kr_post_file_pipeline_fn=lambda context: events.append(context.ml_route),
        post_file_context_factory=FakePostFileContext,
    )

    assert events == ["v2"]
    assert errors == []


def test_persist_kr_generation_output_records_errors_without_raising():
    errors = []
    logs = []

    renamed = persist_kr_generation_output(
        out_path="out.ini",
        final_lines=[],
        tg_folder="tg",
        kr_profile=None,
        custom_map=None,
        custom_phonemes_path="",
        enable_ml_correction=False,
        auto_gen_format="cv",
        ml_policy="",
        runtime_report=None,
        log_fn=logs.append,
        validate_fn=lambda *args: args,
        normalize_key_fn=lambda value: value,
        wav_name_map={},
        apply_output_wav_name_map_fn=lambda _path, _map: 0,
        errors=errors,
        write_oto_lines_fn=lambda _path, _lines: (_ for _ in ()).throw(RuntimeError("boom")),
        run_kr_post_file_pipeline_fn=lambda _context: None,
        post_file_context_factory=FakePostFileContext,
    )

    assert renamed == 0
    assert errors == ["OTO 저장 실패: boom"]
    assert logs == []

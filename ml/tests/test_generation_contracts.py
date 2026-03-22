from core.generation.contracts import (
    GenerationRequest,
    attach_request_metadata,
    finalize_runtime_report,
    initialize_runtime_report,
)


def test_runtime_report_schema_lifecycle():
    report = {}
    initialize_runtime_report(report, language="korean", stage="generate")
    attach_request_metadata(
        report,
        GenerationRequest(
            language="korean",
            tg_folder="tg",
            tpl_path="tpl",
            out_path="out",
            fallback_format="cvvc",
            auto_format="auto",
            alias_suffix="C4",
            options={"auto_gen_format": "cvvc"},
        ),
    )
    finalize_runtime_report(report, processed=2, total=3, errors=["e1"])

    assert report["schema_version"] == "1.0"
    assert report["language"] == "korean"
    assert report["stage"] == "generate"
    assert report["code"] == "GENERATOR_ERROR"
    assert report["processed"] == 2
    assert report["total"] == 3
    assert report["error_count"] == 1
    assert "request" in report
    assert report["request"]["fallback_format"] == "cvvc"

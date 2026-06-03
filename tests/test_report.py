"""Unit tests for LLM report generation (no network; API call is faked)."""

import json

from pipeline.report import (
    build_report_prompt,
    compact_results,
    generate_report,
)


def _results() -> dict:
    return {
        "effect": {
            "effect": 0.10, "ci_low": 0.05, "ci_high": 0.15, "p_value": 0.001,
            "significant": True, "method": "welch_t",
        },
        "assumptions": {
            "srm": {"status": "pass"}, "sutva": {"status": "pass"},
            "novelty": {"status": "pass"}, "overall": "pass",
        },
        "sequential": {
            "decision": "reject_h0", "final_p_value": 0.001,
            # Bulky series that must be dropped from the prompt:
            "sample_sizes": list(range(2000)),
            "p_values": [0.5] * 2000,
            "log_lambda": [0.1] * 2000,
            "effect": [0.1] * 2000,
        },
    }


class _FakeBlock:
    type = "text"
    text = "REPORT BODY"


class _FakeResponse:
    content = [_FakeBlock()]


class _RecordingClient:
    def __init__(self) -> None:
        self.kwargs = None

        class _Messages:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.kwargs = kwargs
                return _FakeResponse()

        self.messages = _Messages(self)


class TestCompaction:
    def test_drops_bulky_series(self) -> None:
        compact = compact_results(_results())
        seq = compact["sequential"]
        assert "sample_sizes" not in seq
        assert "p_values" not in seq
        assert seq["decision"] == "reject_h0"  # scalars kept

    def test_prompt_is_valid_and_small(self) -> None:
        prompt = build_report_prompt(_results())
        assert "REPORT" not in prompt  # no model output
        assert "reject_h0" in prompt
        # The 2000-length arrays must not be embedded.
        assert prompt.count("0.5") < 50


class TestGenerate:
    def test_skips_without_key(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        res = generate_report(_results())
        assert res["status"] == "skip"
        assert res["report"] is None
        assert res["prompt"]  # prompt still returned

    def test_uses_injected_client(self) -> None:
        client = _RecordingClient()
        res = generate_report(_results(), client=client, model="test-model")
        assert res["status"] == "ok"
        assert res["report"] == "REPORT BODY"
        assert client.kwargs["model"] == "test-model"
        assert client.kwargs["system"]
        assert client.kwargs["messages"][0]["role"] == "user"

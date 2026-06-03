"""
LLM report generation.

Takes the structured outputs of the earlier stages (ingestion, EDA, assumption
tests, effect estimate, CUPED, sequential test, HTE) and asks Claude to write a
plain-English report: what happened, whether the result is trustworthy, and the
recommended decision - explicitly flagging any assumption violations.

The deterministic prompt-building (``compact_results`` / ``build_report_prompt``)
is separated from the network call (``generate_report``) so it can be unit
tested without an API key.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are a senior experimentation analyst. Given structured outputs from an "
    "A/B test analysis pipeline, write a concise, plain-English report for a "
    "product decision-maker. Be rigorous and honest about uncertainty. Your "
    "report MUST cover, in order:\n"
    "1. Headline result: the estimated effect, its confidence interval and "
    "p-value, and the relative lift.\n"
    "2. Trustworthiness: explicitly state whether each assumption check "
    "(SRM, SUTVA, novelty) passed or failed and what any failure means for "
    "interpreting the result. If a check failed, the headline result should be "
    "treated with caution or discarded - say so.\n"
    "3. Supporting detail: variance reduction from CUPED, the always-valid "
    "(sequential) conclusion, and any notable heterogeneity (who responds "
    "differently), if available.\n"
    "4. Recommendation: ship / do not ship / keep running / investigate, with a "
    "one-line justification.\n"
    "Never claim significance if the pipeline did not find it. Never describe an "
    "SRM or SUTVA failure as a real treatment effect."
)

# Time-series / per-user arrays that should not be dumped into the prompt.
_DROP_KEYS = {"sample_sizes", "effect", "log_lambda", "p_values"}


def compact_results(results: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce raw stage outputs to a compact, prompt-friendly summary.

    Drops long arrays (sequential time series) and per-user vectors while
    keeping all scalar statistics and explanations.
    """
    compact: dict[str, Any] = {}
    for stage, payload in results.items():
        if isinstance(payload, dict):
            compact[stage] = {
                k: v for k, v in payload.items()
                if k not in _DROP_KEYS and not _is_bulky(v)
            }
        else:
            compact[stage] = payload
    return compact


def _is_bulky(value: Any) -> bool:
    """True for long list/array-like values not worth sending to the model."""
    if isinstance(value, (list, tuple)):
        return len(value) > 30
    return False


def build_report_prompt(results: dict[str, Any]) -> str:
    """Build the user prompt: a compact JSON dump of the pipeline outputs."""
    compact = compact_results(results)
    payload = json.dumps(compact, indent=2, default=str)
    return (
        "Here are the structured outputs from the A/B test pipeline. Write the "
        "report as specified.\n\n```json\n" + payload + "\n```"
    )


def generate_report(
    results: dict[str, Any],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
    client: Any | None = None,
) -> dict[str, Any]:
    """
    Generate a plain-English report from the pipeline outputs via the Anthropic API.

    Parameters
    ----------
    results : dict
        Mapping of stage name -> that stage's report dict (e.g.
        ``{"effect": ..., "assumptions": ..., "cuped": ..., ...}``).
    api_key : str, optional
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY`` env var.
    model : str
        Model id (default ``claude-sonnet-4-20250514``).
    max_tokens : int
        Response length cap.
    client : optional
        A pre-built Anthropic-compatible client (used for testing). If given,
        ``api_key`` is ignored.

    Returns
    -------
    dict
        ``{"status", "report", "model", "prompt"}``. On a missing key/SDK the
        status is ``"skip"`` and the prompt is still returned so it can be run
        manually.
    """
    prompt = build_report_prompt(results)

    if client is None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return {
                "status": "skip",
                "report": None,
                "model": model,
                "prompt": prompt,
                "message": (
                    "No Anthropic API key (set ANTHROPIC_API_KEY or pass api_key). "
                    "The prompt is returned so you can run it manually."
                ),
            }
        try:
            import anthropic
        except ImportError:
            return {
                "status": "skip",
                "report": None,
                "model": model,
                "prompt": prompt,
                "message": "The 'anthropic' package is not installed (pip install anthropic).",
            }
        client = anthropic.Anthropic(api_key=key)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    return {"status": "ok", "report": text, "model": model, "prompt": prompt}

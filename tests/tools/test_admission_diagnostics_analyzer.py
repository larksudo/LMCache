# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the admission-diagnostics analyzer's validity gates.

The analyzer prints a VERDICT per question, and those verdicts get quoted. It
must therefore refuse to judge when the data cannot answer the question -- a
constant column, a tiny sample, or a topology that structurally cannot exhibit
the effect. These tests pin that behaviour using the shapes that actually
occurred in the 2026-11-11/12 runs.
"""

# Standard
from pathlib import Path
import importlib.util
import sys

# Third Party
import pytest

_ANALYZER = (
    Path(__file__).resolve().parents[2] / "tools" / "analyze_admission_diagnostics.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("adm_analyzer", _ANALYZER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adm_analyzer"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def analyzer():
    return _load()


def _rec(**kw):
    base = {
        "rid": "r",
        "rank": 0,
        "tail": 3072,
        "hit": 12288,
        "prompt": 15360,
        "d_ms": 100.0,
        "hold_ms": 0.0,
        "total_ms": 100.0,
        "reason": "NO_SIBLING",
        "bucket": 12,
        "inflight": 0,
        "ready": 1,
        "kv_free": -1,
        "waiting": -1,
        "running": -1,
        "ewma_ms": -1.0,
        "eta_ms": -1.0,
        "deadline_left_ms": -1.0,
    }
    base.update(kw)
    return base


def test_constant_hit_column_refuses_to_rank(analyzer):
    """Fixed-shape benches give one hit size; a rank correlation is undefined."""
    adm = [_rec(rid=f"r{i}", d_ms=50.0 + i) for i in range(100)]
    out = analyzer.q1_hit_vs_tail(adm)
    assert out["decided"] is False


def test_small_sample_refuses_even_when_columns_vary(analyzer):
    adm = [
        _rec(rid=f"r{i}", hit=4096 + i * 4096, tail=3072 + i, d_ms=100.0 + i)
        for i in range(10)
    ]
    out = analyzer.q1_hit_vs_tail(adm)
    assert out["decided"] is False


def test_varying_large_sample_still_decides(analyzer):
    adm = [
        _rec(
            rid=f"r{i}",
            hit=4096 + i * 512,
            tail=1024 + i * 128,
            d_ms=100.0 + i * 2.0,
            bucket=10,
        )
        for i in range(60)
    ]
    out = analyzer.q1_hit_vs_tail(adm)
    assert "decided" not in out or out.get("decided") is not False


def test_tp1_constant_inflight_cannot_answer_serialization(analyzer):
    """On TP1 every retrieve reports inflight=0; the question is unanswerable."""
    adm = [_rec(rid=f"r{i}", d_ms=100.0 + i) for i in range(100)]
    out = analyzer.q2_serialization(adm)
    assert out["decided"] is False


def test_small_reason_sample_does_not_emit_opportunity_verdict(analyzer, capsys):
    adm = [_rec(rid=f"r{i}", reason="NO_SIBLING") for i in range(20)]
    analyzer.q4_reasons(adm)
    text = capsys.readouterr().out
    assert "CANNOT CONCLUDE" in text


def test_c0_not_estimated_without_prefill_tokens(analyzer, capsys):
    """prefill_tok stays -1 until scheduler-side data exists; no fake c0."""
    steps = [
        {
            "step": i,
            "rank": 0,
            "dur_ms": 50.0 + (i % 3) * 40,
            "admitted": i % 3,
            "admitted_tails": (i % 3) * 3072,
            "prefill_tok": -1,
            "decode_tok": -1,
            "running": -1,
            "waiting": -1,
            "budget": 8192,
            "budget_left": -1,
        }
        for i in range(90)
    ]
    analyzer.q5_steps(steps)
    text = capsys.readouterr().out
    assert "CANNOT ESTIMATE c0" in text
    assert "NOT c0" in text


def test_holds_are_counted_from_hold_ms_not_reason(analyzer, capsys):
    """A request that was held then released reads K_REACHED, not HELD.

    Counting reason==HELD reported zero holds for a gate that demonstrably
    held 413 of 1224 requests in the 2026-11-12 TP8 forced arm.
    """
    adm = [
        _rec(rid=f"h{i}", reason="K_REACHED", hold_ms=50.0 + i, d_ms=100.0)
        for i in range(10)
    ] + [
        _rec(rid=f"n{i}", reason="NO_SIBLING", hold_ms=0.0, d_ms=100.0)
        for i in range(90)
    ]
    analyzer.q4_reasons(adm)
    text = capsys.readouterr().out
    assert "held(hold_ms>0.5ms)=10" in text
    assert "hold_ms: mean=" in text

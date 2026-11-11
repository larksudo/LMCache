# SPDX-License-Identifier: Apache-2.0
"""Tests for the admission-coalescing diagnostics recorder.

These cover the properties that make the measurements trustworthy, not just
that lines are emitted: a held retrieve is polled once per step, so a naive
recorder would silently fold the hold into ``d_ms`` and destroy the very
quantity being measured.
"""

# Standard
import time

# First Party
from lmcache.integration.vllm.admission_diagnostics import (
    REASON_HELD,
    REASON_NO_SIBLING,
    AdmissionDiagnostics,
    StepRecord,
)


class _Collector:
    """Sink that keeps every flushed line, so tests read real emitted output
    rather than inspecting the recorder's internal buffer (which flush
    empties)."""

    def __init__(self):
        self.lines = []

    def __call__(self, batch):
        self.lines.extend(batch)


def _mk(rank=0, flush_every=1000, enabled=True):
    col = _Collector()
    d = AdmissionDiagnostics(
        enabled=enabled, rank=rank, flush_every=flush_every, sink=col
    )
    return d, col


def _records(diag, col):
    diag.flush()
    return list(col.lines)


def _fields(line):
    """Parse `key=value` pairs out of a diagnostic line."""
    out = {}
    for kv in line.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def test_disabled_retains_no_state():
    d, col = _mk(enabled=False)
    d.on_submit("a", 100, 4096, 4196, bucket=10)
    d.on_complete("a", REASON_NO_SIBLING)
    d.on_report("a")
    assert d.n_records == 0
    assert d._open == {}
    assert _records(d, col) == []


def test_records_one_line_per_retrieve():
    d, col = _mk(rank=3)
    d.on_submit("a", tail_tokens=1536, hit_tokens=13824, prompt_tokens=15360, bucket=10)
    d.on_complete("a", REASON_NO_SIBLING, n_inflight=0, n_ready=1)
    d.on_report("a")
    lines = _records(d, col)
    assert len(lines) == 1
    ln = lines[0]
    # the fields that answer E1 (is d driven by hit or by tail?)
    assert "rid=a" in ln
    assert "rank=3" in ln
    assert "tail=1536" in ln
    assert "hit=13824" in ln
    assert "prompt=15360" in ln
    assert "reason=NO_SIBLING" in ln
    assert "bucket=10" in ln


def test_hold_time_is_separate_from_retrieve_time():
    """d_ms must exclude the hold; hold_ms must capture it."""
    d, col = _mk()
    d.on_submit("a", 100, 4096, 4196, bucket=8)
    d.on_complete("a", REASON_HELD, n_inflight=1, n_ready=1)
    # simulate the gate holding for ~120 ms before release
    time.sleep(0.12)
    d.on_report("a", reason=REASON_NO_SIBLING)
    ln = _records(d, col)[0]
    fields = _fields(ln)
    d_ms = float(fields["d_ms"])
    hold_ms = float(fields["hold_ms"])
    total_ms = float(fields["total_ms"])
    assert d_ms < 100.0, f"d_ms must exclude the hold, got {d_ms}"
    assert hold_ms >= 100.0, f"hold_ms should be ~120, got {hold_ms}"
    assert total_ms >= d_ms + hold_ms - 1.0


def test_repeated_complete_does_not_inflate_d():
    """A held retrieve is polled every step; only the first stamp counts.

    This is the regression that would otherwise make d_ms grow with the hold
    and silently invalidate the E1 measurement.
    """
    d, col = _mk()
    d.on_submit("a", 100, 4096, 4196, bucket=8)
    d.on_complete("a", REASON_HELD, n_inflight=2, n_ready=1)
    first_complete = d._open["a"].complete_ts
    assert first_complete > 0.0
    time.sleep(0.15)
    # polled again on later steps while held
    for _ in range(5):
        d.on_complete("a", REASON_HELD, n_inflight=1, n_ready=1)
    assert d._open["a"].complete_ts == first_complete, (
        "complete_ts must be first-write-wins"
    )
    d.on_report("a", reason=REASON_NO_SIBLING)
    ln = _records(d, col)[0]
    fields = _fields(ln)
    assert float(fields["d_ms"]) < 100.0
    assert float(fields["hold_ms"]) >= 140.0


def test_final_reason_overrides_provisional():
    """The releasing step's reason wins over the provisional one."""
    d, col = _mk()
    d.on_submit("a", 100, 4096, 4196, bucket=8)
    d.on_complete("a", REASON_HELD)
    d.on_report("a", reason="COHORT_TOO_SMALL")
    assert "reason=COHORT_TOO_SMALL" in _records(d, col)[0]


def test_drop_prevents_unbounded_growth():
    """Preempted/cancelled retrieves must not leak open records."""
    d, col = _mk()
    for i in range(50):
        d.on_submit(f"r{i}", 100, 4096, 4196, bucket=8)
    assert len(d._open) == 50
    d.on_drop([f"r{i}" for i in range(50)])
    assert d._open == {}
    assert d.n_records == 0


def test_clear_drops_open_records():
    d, col = _mk()
    d.on_submit("a", 100, 4096, 4196, bucket=8)
    d.clear()
    assert d._open == {}


def test_flush_batches():
    d, col = _mk(flush_every=4)
    for i in range(4):
        d.on_submit(f"r{i}", 100, 4096, 4196, bucket=8)
        d.on_complete(f"r{i}", REASON_NO_SIBLING)
        d.on_report(f"r{i}")
    # flush_every=4 -> one automatic flush happened, buffer emptied
    assert d.n_flushes >= 1
    assert d.n_records == 4


def test_step_record_fields():
    d, col = _mk(rank=1)
    t0 = time.monotonic()
    rec = StepRecord(
        step_id=7,
        rank=0,
        step_start=t0,
        step_end=t0 + 0.05,
        admitted_now=2,
        admitted_tails=3072,
        prefill_tokens=3072,
        decode_tokens=512,
        num_running=16,
        num_waiting=4,
        token_budget=8192,
        budget_left=4608,
    )
    d.on_step(rec)
    ln = _records(d, col)[0]
    assert "step=7" in ln
    assert "rank=1" in ln, "on_step must stamp the recorder's rank"
    assert "admitted=2" in ln
    assert "admitted_tails=3072" in ln
    assert float(ln.split("dur_ms=")[1].split()[0]) >= 45.0


def test_unknown_request_is_ignored():
    """Reporting an id the recorder never saw must not raise."""
    d, col = _mk()
    d.on_complete("ghost", REASON_NO_SIBLING)
    d.on_report("ghost")
    assert _records(d, col) == []


def test_diagnostics_never_raise(caplog):
    """A malformed call must be swallowed, never propagated to the engine."""
    d, col = _mk()
    # non-numeric where a number is expected
    d.on_submit("a", "not-an-int", 4096, 4196, bucket=8)  # type: ignore[arg-type]
    assert True  # reaching here at all is the assertion: nothing raised

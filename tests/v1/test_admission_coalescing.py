# SPDX-License-Identifier: Apache-2.0
"""Tests for the admission-coalescing gate (AdmissionCoalescer).

The gate is a pure decision function over (ready, pending) sets with
injectable timing, so every release condition is testable without MP
adapter mocks.
"""

# Standard
import time

# First Party
from lmcache.integration.vllm.admission_coalescing import (
    AdmissionCoalescer,
    _bucket_for,
)

BUDGET = 8192  # alpha=0.75 -> fill_floor = 6144
TWO_TAILS = 3072  # 2 * 3072 = 6144 == fill_floor


def _make(enabled: bool = True, budget: int = BUDGET) -> AdmissionCoalescer:
    return AdmissionCoalescer(enabled=enabled, budget_tokens=budget)


def _submit(c, rid: str, tail: int, age_s: float = 0.0) -> None:
    """Submit a retrieve whose submission timestamp is age_s in the past."""
    c.on_submit(rid, tail)
    c._submit_ts[rid] = time.monotonic() - age_s


def test_disabled_is_passthrough():
    c = _make(enabled=False)
    c.on_submit("a", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a"}) == {"a"}


def test_k_reached_releases():
    c = _make()
    for r in ("a", "b", "d"):
        c.on_submit(r, TWO_TAILS)
    out = c.select_to_report({"a", "b", "d"}, {"a", "b", "d"})
    assert out == {"a", "b", "d"}


def test_no_sibling_releases():
    c = _make()
    c.on_submit("a", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a"}) == {"a"}


def test_tail_fills_step_releases():
    c = _make()
    c.on_submit("a", 6144)
    c.on_submit("b", 100)
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_sum_cannot_fill_releases():
    c = _make()
    c.on_submit("a", 500)
    c.on_submit("b", 500)
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_deadline_expired_releases():
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c._submit_ts["a"] = time.monotonic() - 4.0
    c._submit_ts["b"] = time.monotonic() - 3.5
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_cold_start_releases():
    # No duration observations at all: release rather than hold blind.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_hold_taken_then_sibling_arrives():
    # With a duration estimate and a sibling inside the window: hold,
    # then report both when the sibling arrives.
    c = _make()
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})  # releases, EWMA seeded
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    out = c.select_to_report({"a"}, {"a", "b"})
    assert out == set()  # held
    out = c.select_to_report({"a", "b"}, {"a", "b"})
    assert out == {"a", "b"}  # K reached next step


def test_reported_exactly_once():
    c = _make()
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a", "b"}) == set()
    assert c.select_to_report({"a", "b"}, {"a", "b"}) == {"a", "b"}
    assert "a" not in c._submit_ts
    assert c.select_to_report(set(), set()) == set()


def test_drain_clears_state():
    c = _make()
    c.on_submit("x", 100)
    c.drain()
    assert not c._submit_ts
    assert not c._tail
    assert not c._first_hold


def test_hold_time_excluded_from_ewma():
    # A request that was held contributes (first_hold - submit), not
    # (report - submit), so the EWMA tracks the retrieve pipeline.
    c = _make()
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c._duration_ewma = 0.3
    assert c.select_to_report({"a"}, {"a", "b"}) == set()  # held
    c._first_hold["a"] = time.monotonic() - 0.5  # held for 0.5s
    c._submit_ts["a"] = time.monotonic() - 1.0
    c.select_to_report({"a", "b"}, set())
    assert c._duration_ewma is not None
    assert c._duration_ewma < 0.8 * 0.3 + 0.2 * 1.0


# ============ Preemption tests ============


def test_forget_drops_state_without_reporting():
    # Preempted/cancelled retrieves leave retrieve_futures without reporting.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c.forget({"a"})
    assert "a" not in c._submit_ts
    assert "a" not in c._tail
    assert "a" not in c._first_hold
    assert "b" in c._submit_ts  # sibling untouched


def test_preemption_gc_clears_stale():
    # Simulates the adapter-level GC: entries in _submit_ts that are no
    # longer in retrieve_futures should be forgotten.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c.on_submit("c", TWO_TAILS)
    # "c" gets preempted: leaves retrieve_futures but coalescer state remains
    # (adapter GC calls forget)
    stale = set(c._submit_ts) - {"a", "b"}  # simulate retrieve_futures = {a, b}
    if stale:
        c.forget(stale)
    assert "c" not in c._submit_ts
    assert "a" in c._submit_ts
    assert "b" in c._submit_ts


def test_resubmit_after_preemption_clean():
    # A preempted request that is resumed with the same request_id should
    # not see a stale _first_hold from its previous life.
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c._first_hold["a"] = time.monotonic() - 5.0  # old hold timestamp
    c.forget({"a"})  # preemption cleanup
    c.on_submit("a", TWO_TAILS)  # resume with same id
    assert c._first_hold.get("a") is None  # stale timestamp gone


# ============ Bucketed EWMA tests ============


def test_bucket_for_correct_boundaries():
    assert _bucket_for(0) == 8
    assert _bucket_for(100) == 8  # closest to 256
    assert _bucket_for(256) == 8
    assert _bucket_for(1024) == 10
    assert _bucket_for(4096) == 12
    assert _bucket_for(16384) == 14
    assert _bucket_for(65536) == 16
    assert _bucket_for(500) == 8  # log2(500)~8.97, tie between 8 and 10, min picks 8
    assert _bucket_for(3000) == 12  # log2(3000)~11.5, closer to 12


def test_bucket_ewma_independent():
    # Short and long retrieves maintain independent duration estimates.
    c = _make()
    # Submit a short retrieve (tail ~500, bucket 10)
    c.on_submit("short", 500)
    c._first_hold["short"] = time.monotonic()
    c._submit_ts["short"] = time.monotonic() - 0.1  # 100ms duration
    c._tail["short"] = 500
    c.select_to_report({"short"}, set())

    # Submit a long retrieve (tail ~20000, bucket 14)
    c.on_submit("long", 20000)
    c._first_hold["long"] = time.monotonic()
    c._submit_ts["long"] = time.monotonic() - 2.0  # 2000ms duration
    c._tail["long"] = 20000
    c.select_to_report({"long"}, set())

    # Verify independent estimates
    short_est = c._estimate_duration(500)
    long_est = c._estimate_duration(20000)
    assert short_est is not None and short_est < 0.5
    assert long_est is not None and long_est > 1.0
    assert long_est > short_est * 2  # clearly separated


def test_eta_uses_sibling_bucket():
    # A short-tailed sibling uses the short bucket's estimate; a bucket
    # with no observations means no estimate, which releases (fail-open).
    c = _make()
    # Seed the long bucket with a slow duration
    c.on_submit("long", 20000)
    c._submit_ts["long"] = time.monotonic() - 2.0
    c._first_hold["long"] = time.monotonic()
    c._tail["long"] = 20000
    c.select_to_report({"long"}, set())
    assert c._bucket_ewma[14] > 1.0  # long bucket is slow

    # Now a short sibling arrives; the short bucket has no data -> no
    # estimate -> release (fail-open, no blind hold)
    c.on_submit("a", TWO_TAILS)
    c.on_submit("short_b", 500)
    c._submit_ts["a"] = time.monotonic() - 0.1
    c._submit_ts["short_b"] = time.monotonic()
    out = c.select_to_report({"a"}, {"a", "short_b"})
    assert out == {"a"}  # short bucket empty -> released


def test_bucket_cold_start_releases():
    # A bucket with no observations falls back to global; if global is
    # also None, releases (no blind hold).
    c = _make()
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", 30000)  # different bucket
    # No duration observations at all
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_two_ready_releases_without_waiting_for_a_third():
    """M2: two ready tails already realise a merge; a third adds no saving.

    Regression guard. The gate used ``0 < len(ready) < _K`` with ``_K = 3``, so
    two ready retrieves kept holding for a third or for the deadline. Both
    tails already land on one step boundary at two, so the extra wait was pure
    TTFT cost with no throughput benefit -- and on a saturated agentic workload
    that cost is what shows up as the TTFT regression.
    """
    c = _make(enabled=True)
    # Seed the duration EWMA so the ETA condition can hold rather than
    # releasing for lack of an estimate.
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})

    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c.on_submit("c", TWO_TAILS)  # a third sibling is in flight
    # Both a and b are ready: release now, do not wait for c.
    assert c.select_to_report({"a", "b"}, {"a", "b", "c"}) == {"a", "b"}


def test_one_ready_still_holds_when_a_sibling_can_arrive():
    """The M2 change must not disable holding entirely: one ready still waits."""
    c = _make(enabled=True)
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})

    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    # Only "a" is ready and "b" is still in flight -> holding can pay off.
    assert c.select_to_report({"a"}, {"a", "b"}) == set()
    assert c.select_to_report({"a", "b"}, {"a", "b"}) == {"a", "b"}


def test_k_reached_still_releases():
    """_K remains an upper bound: reaching it releases immediately."""
    c = _make(enabled=True)
    c.on_submit("seed", TWO_TAILS)
    c.select_to_report({"seed"}, {"seed"})
    for rid in ("a", "b", "c"):
        c.on_submit(rid, TWO_TAILS)
    assert c.select_to_report({"a", "b", "c"}, {"a", "b", "c"}) == {"a", "b", "c"}


# --- Price-knob regression tests: defaults must reproduce the shipped
# --- constants; non-defaults enable the hold-cost upper-bound experiment.


def _make_custom(**kwargs) -> AdmissionCoalescer:
    return AdmissionCoalescer(enabled=True, budget_tokens=BUDGET, **kwargs)


def test_default_knobs_reproduce_shipped_constants():
    c = _make()
    assert c._deadline_s == 3.0
    assert c._fill_alpha == 0.75
    assert c._coalesce_min == 2
    assert c._require_eta is True


def test_fill_alpha_zero_disables_fill_floor_and_holds_big_tail():
    """fill_alpha <= 0 must disable the fill checks, not make them always true.

    With defaults a 30000-token tail is >= fill_floor and releases via
    TAIL_FILLS_STEP; with the check off and require_eta off, the same shape
    holds -- that is the forced-merge upper-bound configuration.
    """
    c_default = _make()
    c_ub = _make_custom(fill_alpha=0.0, require_eta=False)
    for c in (c_default, c_ub):
        c.on_submit("a", 30000)
        c.on_submit("b", 30000)
    assert c_default.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert c_ub.select_to_report({"a"}, {"a", "b"}) == set()


def test_fill_alpha_zero_holds_cohort_that_cannot_fill_a_step():
    c_ub = _make_custom(fill_alpha=0.0, require_eta=False)
    c_default = _make()
    for c in (c_default, c_ub):
        c.on_submit("a", 100)
        c.on_submit("b", 100)
    # Defaults: 200 total << fill_floor -> COHORT_TOO_SMALL release.
    assert c_default.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert c_ub.select_to_report({"a"}, {"a", "b"}) == set()


def test_require_eta_false_holds_without_observations():
    """No EWMA observations: defaults release with NO_ETA; require_eta off holds."""
    c_ub = _make_custom(require_eta=False)
    c_default = _make()
    for c in (c_default, c_ub):
        c.on_submit("a", TWO_TAILS)
        c.on_submit("b", TWO_TAILS)
    assert c_default.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert c_ub.select_to_report({"a"}, {"a", "b"}) == set()


def test_deadline_knob_shortens_the_hold_window():
    c = _make_custom(require_eta=False, deadline_s=0.5)
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    c._submit_ts["a"] = time.monotonic() - 1.0
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}


def test_coalesce_min_one_releases_a_single_ready():
    c = _make_custom(coalesce_min=1, require_eta=False)
    c.on_submit("a", TWO_TAILS)
    c.on_submit("b", TWO_TAILS)
    assert c.select_to_report({"a"}, {"a", "b"}) == {"a"}

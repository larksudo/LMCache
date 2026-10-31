# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the MP admission coalescing gate.

The gate is exercised through its public interface only: ``on_submit``,
``select_to_report``, ``release_reasons``, ``stats`` and ``drain``. Time is
injected by replacing the module's ``time`` binding with a fake clock, so the
measured series (step duration, completion gap) are deterministic.
"""

# Standard
from dataclasses import dataclass

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import admission_coalescing

# The gate is pure bookkeeping over ids and timestamps; it never touches an
# accelerator, so the shared-allocator patch is not needed.
pytestmark = pytest.mark.no_shared_allocator


@dataclass
class _Clock:
    """Monotonic clock under test control."""

    now: float = 1000.0

    def monotonic(self) -> float:
        """Return the current fake time."""
        return self.now

    def advance_ms(self, ms: float) -> None:
        """Move the clock forward by *ms* milliseconds."""
        self.now += ms / 1000.0


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Install a controllable clock inside the gate module."""
    fake = _Clock()
    monkeypatch.setattr(admission_coalescing, "time", fake)
    return fake


def _primed_gate(
    clock: _Clock,
    *,
    budget: int = 8192,
    max_hold_steps: int = 2,
    latency_weight: float = 1.0,
    tail: int = 1536,
    step_ms: float = 50.0,
    calls: int = 8,
    enabled: bool = True,
) -> admission_coalescing.AdmissionCoalescer:
    """Build a gate whose step and completion-gap series are established.

    Each priming call completes one retrieve with no sibling in flight, which
    is released immediately but teaches the gate the step duration and the
    completion gap.
    """
    gate = admission_coalescing.AdmissionCoalescer(
        enabled=enabled,
        budget_tokens=budget,
        max_hold_steps=max_hold_steps,
        latency_weight=latency_weight,
    )
    for i in range(calls):
        request_id = f"prime{i}"
        gate.on_submit(request_id, tail)
        clock.advance_ms(step_ms)
        gate.select_to_report({request_id}, {request_id})
    return gate


def test_disabled_gate_reports_everything(clock: _Clock) -> None:
    """A disabled gate reports every ready retrieve and keeps no state."""
    gate = _primed_gate(clock, enabled=False)

    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.stats()["calls"] == 0.0
    assert gate.release_reasons() == {}


def test_cold_start_reports_instead_of_holding(clock: _Clock) -> None:
    """Without a measured step or gap the gate cannot price a hold."""
    gate = admission_coalescing.AdmissionCoalescer(enabled=True, budget_tokens=8192)
    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)
    clock.advance_ms(50)

    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.release_reasons().get("cold") == 1


def test_no_sibling_in_flight_is_never_held(clock: _Clock) -> None:
    """With nothing left to wait for, holding is pure latency."""
    gate = _primed_gate(clock)
    before = gate.release_reasons().get("nosib", 0)

    assert gate.select_to_report({"a"}, {"a"}) == {"a"}
    assert gate.release_reasons().get("nosib", 0) == before + 1


def test_tails_that_cannot_share_a_step_are_not_held(clock: _Clock) -> None:
    """Two 4096-token tails against an 8192 budget cannot share a step.

    The decode reserve leaves a 6554-token head, so a sibling does not fit
    beside a 4096-token tail and holding it removes no step.
    """
    gate = _primed_gate(clock, tail=4096)
    gate.on_submit("a", 4096)
    gate.on_submit("b", 4096)
    clock.advance_ms(50)

    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.release_reasons().get("fit") == 1


def test_holds_when_a_sibling_is_expected_within_one_step(clock: _Clock) -> None:
    """A 50 ms gap against a 50 ms step makes one join certain."""
    gate = _primed_gate(clock)
    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)
    clock.advance_ms(50)

    assert gate.select_to_report({"a"}, {"a", "b"}) == set()
    assert gate.stats()["hold_events"] == 1.0


def test_releases_when_the_next_completion_is_far_away(clock: _Clock) -> None:
    """A completion gap far beyond the hold window cannot pay for a hold."""
    gate = admission_coalescing.AdmissionCoalescer(enabled=True, budget_tokens=8192)
    gate.on_submit("p0", 1536)
    clock.advance_ms(50)
    gate.select_to_report({"p0"}, {"p0"})
    for _ in range(19):
        clock.advance_ms(50)
        gate.select_to_report(set(), set())

    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)
    clock.advance_ms(50)

    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.release_reasons().get("cost") == 1


def test_hold_escalation_requires_more_joins(clock: _Clock) -> None:
    """Waiting a second step for a single sibling is not worth it by default."""
    gate = _primed_gate(clock, latency_weight=1.0)
    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)
    clock.advance_ms(50)
    assert gate.select_to_report({"a"}, {"a", "b"}) == set()

    clock.advance_ms(50)
    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.release_reasons().get("cost") == 1


def test_hold_is_bounded_by_max_hold_steps(clock: _Clock) -> None:
    """A permissive weight still cannot exceed the step bound."""
    gate = _primed_gate(clock, max_hold_steps=2, latency_weight=0.5)
    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)

    clock.advance_ms(50)
    assert gate.select_to_report({"a"}, {"a", "b"}) == set()
    clock.advance_ms(50)
    assert gate.select_to_report({"a"}, {"a", "b"}) == set()

    clock.advance_ms(50)
    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.release_reasons().get("hold") == 1


def test_merge_is_counted_when_a_sibling_joins(clock: _Clock) -> None:
    """A hold that ends with two tails reported together saved one step."""
    gate = _primed_gate(clock)
    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)
    clock.advance_ms(50)
    assert gate.select_to_report({"a"}, {"a", "b"}) == set()

    clock.advance_ms(50)
    assert gate.select_to_report({"a", "b"}, {"a", "b"}) == {"a", "b"}

    stats = gate.stats()
    assert stats["merges"] == 1.0
    assert stats["saved_steps"] == 1.0
    assert stats["merge_rate"] == 1.0
    assert stats["hold_n"] == 1.0
    assert stats["hold_ms_mean"] == pytest.approx(50.0, rel=0.1)


def test_reported_request_is_reported_once(clock: _Clock) -> None:
    """Reported ids drop their state, so a repeat report counts no merge."""
    gate = _primed_gate(clock)
    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)
    clock.advance_ms(50)
    gate.select_to_report({"a"}, {"a", "b"})
    clock.advance_ms(50)

    assert gate.select_to_report({"a", "b"}, {"a", "b"}) == {"a", "b"}
    assert gate.select_to_report({"a", "b"}, {"a", "b"}) == {"a", "b"}
    assert gate.stats()["merges"] == 1.0


def test_drain_drops_held_state(clock: _Clock) -> None:
    """The unhealthy path drains state and reports everything afterwards."""
    gate = _primed_gate(clock)
    gate.on_submit("a", 1536)
    gate.on_submit("b", 1536)
    clock.advance_ms(50)
    gate.select_to_report({"a"}, {"a", "b"})

    gate.drain()
    clock.advance_ms(50)

    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.stats()["hold_n"] == 0.0


def test_request_without_prefill_is_not_held(clock: _Clock) -> None:
    """A fully cached request has no prefill step to coalesce."""
    gate = _primed_gate(clock)
    gate.on_submit("a", 0)
    gate.on_submit("b", 0)
    clock.advance_ms(50)

    assert gate.select_to_report({"a"}, {"a", "b"}) == {"a"}
    assert gate.release_reasons().get("noprefill") == 1


def test_extra_config_keys_coerce_to_their_default_types() -> None:
    """The two new keys exist and coerce the way the connector parses them."""
    pytest.importorskip("vllm")

    # Local
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        _coerce_extra_config_value,
    )

    steps = ExtraConfigDefault.admit_coalesce_max_hold_steps
    weight = ExtraConfigDefault.admit_coalesce_latency_weight

    assert steps.default == 2
    assert weight.default == 1.0
    assert _coerce_extra_config_value(steps.default, "3") == 3
    assert _coerce_extra_config_value(weight.default, "0.5") == 0.5

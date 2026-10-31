# SPDX-License-Identifier: Apache-2.0
"""Worker-side admission coalescing for the MP adapter.

Holds a completed KV retrieve for a bounded number of engine steps so that
several prefill tails are admitted on the same step boundary. See
``docs/design/integration/vllm/admission_coalescing.md``.

The gate compares cost against benefit in the unit both sides share: engine
steps. Holding a ready retrieve for ``w`` steps charges the held request
``w`` steps of added TTFT; every sibling that joins the coalesced admission
removes one prefill-carrying step from the run. The gate holds only when the
expected number of joins pays for the steps it charges.

Three properties decide whether the mechanism helps or hurts on a given
workload:

- A hold only removes a step when the coalesced tails fit inside one, so the
  fit check is a hard precondition rather than a heuristic. Two 4096-token
  tails against a 8192-token budget never share a step, and holding them is
  added latency with no step removed.
- The number of joins is estimated from the measured gap between retrieve
  completions: that is the quantity answering "how long until the next
  sibling is ready". The retrieve duration answers a different question.
- The hold is bounded in steps, not seconds, because one step is the
  smallest delay the engine can express.
"""

# Standard
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Fallback step budget in tokens; the connector supplies vLLM's real
# max_num_batched_tokens, so this only applies to direct construction.
DEFAULT_BUDGET_TOKENS: int = 8192
# Steps a single retrieve may be held for. Two steps is where the expected
# joins still cover the added TTFT on the workloads measured so far.
DEFAULT_MAX_HOLD_STEPS: int = 2
# Expected joins required per step of hold. This is the only tradeoff knob:
# raise it (e.g. 2.0) to protect TTFT, lower it (0.5) to favour TPOT.
DEFAULT_LATENCY_WEIGHT: float = 1.0

# Share of the step budget reserved for decode tokens. The scheduler hands out
# at most max_num_batched_tokens per step and running decode requests consume
# part of it, so tails that fit the raw budget may still not fit the step.
_FIT_RESERVE_RATIO: float = 0.2
# EWMA factor for the measured series (step duration, completion gap, tail).
_EWMA_ALPHA: float = 0.3
# A completion gap above this many seconds is an idle stretch rather than an
# arrival rate; sampling it would make the gate refuse to hold for minutes
# after any lull.
_MAX_GAP_S: float = 5.0
# Log the decision summary every N gate calls (0 = never). One line per 2000
# engine steps is the mechanism's only production observability.
_LOG_EVERY: int = 2000


class AdmissionCoalescer:
    """Per-step gate deciding which completed retrieves to report.

    Knows nothing about futures or the MP adapter: each engine step the caller
    passes the ready and pending request-id sets. Per-request state (submission
    timestamp, prefill tail, steps held) lives here and is dropped when the
    request is reported or drained.

    Disabled gate: every method returns immediately and no per-request or
    measured state is kept.
    """

    def __init__(
        self,
        enabled: bool,
        budget_tokens: int,
        max_hold_steps: int = DEFAULT_MAX_HOLD_STEPS,
        latency_weight: float = DEFAULT_LATENCY_WEIGHT,
    ) -> None:
        """Initialize the gate.

        Args:
            enabled: Whether the gate may hold completed retrieves.
            budget_tokens: Per-step token budget (vLLM's
                ``max_num_batched_tokens``).
            max_hold_steps: Upper bound on the steps one retrieve may be held.
            latency_weight: Expected joins required per step of hold; the
                single TTFT/TPOT tradeoff knob.
        """
        self._enabled = enabled
        self._budget = budget_tokens
        self._max_hold_steps = max_hold_steps
        self._latency_weight = latency_weight

        # request_id -> retrieve submission time.
        self._submit_ts: dict[str, float] = {}
        # request_id -> prefill tail in tokens (prompt length minus cache hit).
        self._tail: dict[str, int] = {}
        # request_id -> timestamp of the first hold.
        self._hold_ts: dict[str, float] = {}
        # request_id -> engine steps this retrieve has been held for.
        self._held_steps: dict[str, int] = {}

        # Measured series; None until the first observation.
        self._step_ewma: float | None = None
        self._gap_ewma: float | None = None
        self._tail_ewma: float | None = None
        self._last_call_ts: float | None = None
        self._last_ready_ts: float | None = None

        # Decision summary counters.
        self._calls = 0
        self._hold_events = 0
        self._held_requests = 0
        self._merges = 0
        self._saved_steps = 0
        self._hold_s_sum = 0.0
        self._hold_s_max = 0.0
        self._hold_s_n = 0
        self._reasons: dict[str, int] = {}

    def on_submit(self, request_id: str, tail_tokens: int) -> None:
        """Record submission time and prefill tail for a retrieve.

        The tail is computed by the adapter from the op itself (packed token
        count minus the cached range) because new metadata fields do not
        survive scheduler-to-worker serialization.

        Args:
            request_id: The request identifier.
            tail_tokens: Prefill tail in tokens.
        """
        if not self._enabled:
            return
        self._submit_ts[request_id] = time.monotonic()
        self._tail[request_id] = tail_tokens
        self._tail_ewma = (
            float(tail_tokens)
            if self._tail_ewma is None
            else _ewma(self._tail_ewma, float(tail_tokens))
        )

    def drain(self) -> None:
        """Drop all per-request state (unhealthy path reports everything)."""
        self._submit_ts.clear()
        self._tail.clear()
        self._hold_ts.clear()
        self._held_steps.clear()

    def select_to_report(self, ready: set[str], pending: set[str]) -> set[str]:
        """Return the subset of *ready* to report this engine step.

        Held ids keep their state and are re-evaluated on the next call, so a
        completed retrieve is reported exactly once. Reported requests update
        the hold statistics and drop their state.

        Args:
            ready: Request ids whose retrieve completed this call.
            pending: All in-flight retrieve ids (superset of *ready*).

        Returns:
            The subset of *ready* to report this engine step; an empty set
            means hold everything for one more step.
        """
        if not self._enabled:
            return ready

        now = time.monotonic()
        self._note_step(now)
        self._calls += 1
        if not ready:
            return ready
        self._note_ready(now)

        held = self._select_held(ready, pending)
        if held:
            self._hold_events += 1
            self._held_requests += len(held)
            for request_id in held:
                self._hold_ts.setdefault(request_id, now)
                self._held_steps[request_id] = (
                    self._held_steps.get(request_id, 0) + 1
                )

        reported = ready - held
        self._record_reported(reported, now)
        self._maybe_log()
        return reported

    def stats(self) -> dict[str, float]:
        """Return the decision summary for logs, metrics, or tests.

        Returns:
            Counters and measured series. ``merge_rate`` is ``0.0`` before the
            first hold is released; every ratio guards its denominator.
        """
        merge_rate = self._merges / self._hold_events if self._hold_events else 0.0
        return {
            "calls": float(self._calls),
            "hold_events": float(self._hold_events),
            "held_requests": float(self._held_requests),
            "merges": float(self._merges),
            "saved_steps": float(self._saved_steps),
            "merge_rate": merge_rate,
            "hold_ms_mean": (
                self._hold_s_sum / self._hold_s_n * 1000.0 if self._hold_s_n else 0.0
            ),
            "hold_ms_max": self._hold_s_max * 1000.0,
            "hold_n": float(self._hold_s_n),
            "step_ms": (self._step_ewma or 0.0) * 1000.0,
            "gap_ms": (self._gap_ewma or 0.0) * 1000.0,
            "tail_tokens": self._tail_ewma or 0.0,
        }

    def release_reasons(self) -> dict[str, int]:
        """Return how often each release condition rejected a hold.

        Returns:
            Copy of the per-reason counters, e.g. ``{"fit": 12, "cost": 3}``.
            This is what tells an operator why the gate is idle: ``fit`` means
            the tails cannot share a step, ``cost`` means no sibling is
            expected in time.
        """
        return dict(self._reasons)

    def _select_held(self, ready: set[str], pending: set[str]) -> set[str]:
        """Decide whether to hold this step's ready retrieves.

        Args:
            ready: Request ids whose retrieve completed this call.
            pending: All in-flight retrieve ids.

        Returns:
            ``ready`` to hold it for another step, or an empty set to report.
        """
        inflight = pending - ready
        if not inflight:
            return self._release("nosib")

        head = self._budget * (1.0 - _FIT_RESERVE_RATIO)
        ready_tail = sum(self._tail.get(request_id, 0) for request_id in ready)
        if ready_tail <= 0:
            return self._release("noprefill")
        if ready_tail > head:
            return self._release("fit")
        if (
            self._step_ewma is None
            or self._gap_ewma is None
            or self._tail_ewma is None
        ):
            return self._release("cold")

        # Tails that fit beside the ready ones: no room means a sibling could
        # never share their step, so holding cannot remove a step.
        fits = int((head - ready_tail) // max(self._tail_ewma, 1.0))
        if fits < 1:
            return self._release("fit")

        waited = max(self._held_steps.get(request_id, 0) for request_id in ready) + 1
        if waited > self._max_hold_steps:
            return self._release("hold")

        # Expected joins if we hold for ``waited`` steps: the completion gap
        # says how long the next sibling needs, the fit count caps how many can
        # join the same step.
        joins = min(fits, len(inflight))
        window = waited * self._step_ewma
        p_join = min(1.0, window / max(self._gap_ewma, self._step_ewma))
        if p_join * joins < self._latency_weight * waited:
            return self._release("cost")
        return ready

    def _release(self, reason: str) -> set[str]:
        """Record a release reason and return the empty held set."""
        self._reasons[reason] = self._reasons.get(reason, 0) + 1
        return set()

    def _record_reported(self, reported: set[str], now: float) -> None:
        """Update hold statistics and drop state for reported requests."""
        held_with_merge = [
            request_id
            for request_id in reported
            if self._held_steps.get(request_id, 0) > 0
        ]
        if held_with_merge and len(reported) >= 2:
            # The fit check admitted this cohort into one step, so it cost one
            # step instead of len(reported).
            self._merges += 1
            self._saved_steps += len(reported) - 1

        for request_id in reported:
            hold_ts = self._hold_ts.pop(request_id, None)
            if hold_ts is not None:
                waited = now - hold_ts
                self._hold_s_sum += waited
                self._hold_s_max = max(self._hold_s_max, waited)
                self._hold_s_n += 1
            self._submit_ts.pop(request_id, None)
            self._tail.pop(request_id, None)
            self._held_steps.pop(request_id, None)

    def _note_step(self, now: float) -> None:
        """Sample the engine step duration from consecutive gate calls."""
        last = self._last_call_ts
        self._last_call_ts = now
        if last is None:
            return
        elapsed = now - last
        if elapsed <= 0.0 or elapsed > _MAX_GAP_S:
            return
        self._step_ewma = (
            elapsed if self._step_ewma is None else _ewma(self._step_ewma, elapsed)
        )

    def _note_ready(self, now: float) -> None:
        """Sample the gap between consecutive retrieve completions."""
        last = self._last_ready_ts
        self._last_ready_ts = now
        if last is None:
            return
        gap = now - last
        if gap <= 0.0 or gap > _MAX_GAP_S:
            return
        self._gap_ewma = gap if self._gap_ewma is None else _ewma(self._gap_ewma, gap)

    def _maybe_log(self) -> None:
        """Emit the decision summary every ``_LOG_EVERY`` gate calls."""
        if not _LOG_EVERY or self._calls % _LOG_EVERY:
            return
        stats = self.stats()
        logger.info(
            "admit_coalesce: calls=%d holds=%d held=%d merges=%d "
            "merge_rate=%.2f saved_steps=%d hold_ms{mean=%.0f,max=%.0f,n=%d} "
            "step_ms=%.0f gap_ms=%.0f tail=%d reasons=%s",
            int(stats["calls"]),
            int(stats["hold_events"]),
            int(stats["held_requests"]),
            int(stats["merges"]),
            stats["merge_rate"],
            int(stats["saved_steps"]),
            stats["hold_ms_mean"],
            stats["hold_ms_max"],
            int(stats["hold_n"]),
            stats["step_ms"],
            stats["gap_ms"],
            int(stats["tail_tokens"]),
            self.release_reasons(),
        )


def _ewma(current: float, sample: float) -> float:
    """Blend *sample* into *current* with :data:`_EWMA_ALPHA`.

    Args:
        current: Current moving average.
        sample: New observation.

    Returns:
        The updated moving average.
    """
    return (1.0 - _EWMA_ALPHA) * current + _EWMA_ALPHA * sample

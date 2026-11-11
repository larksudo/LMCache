# SPDX-License-Identifier: Apache-2.0
"""Worker-side admission coalescing for the MP adapter.

Holds completed KV retrieves for one engine step so that several
prefill tails are admitted on the same step boundary and the per-step
token budget is filled. See docs/design/integration/vllm/admission_coalescing.md.
"""

# Standard
import math
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Fallback step budget in tokens; the connector supplies vLLM's real
# max_num_batched_tokens, so this only applies to direct construction.
DEFAULT_BUDGET_TOKENS: int = 8192

# Hold deadline in seconds, measured from retrieve submission. This is
# deliberately generous: it must comfortably exceed the typical sibling
# retrieve duration (d), otherwise a sibling submitted shortly after the
# ready request can never arrive inside the window and every hold is
# wasted latency.
_DEADLINE_S: float = 3.0
# Fraction of the step budget that counts as a full step.
_FILL_ALPHA: float = 0.75
# Minimum ready retrieves that make a hold worthwhile: two tails on one step
# boundary already realise the merge, so a third adds cost without benefit.
_COALESCE_MIN: int = 2
# Upper bound on how many ready retrieves a single step can absorb. Reaching it
# releases immediately rather than waiting for more.
_K: int = 3
# Log a decision summary every N gate calls (0 = never).
_LOG_EVERY: int = 200
# Duration buckets in log2(tail_tokens): ~256, ~1K, ~4K, ~16K, ~64K.
_DURATION_BUCKETS: tuple[int, ...] = (8, 10, 12, 14, 16)
# EWMA decay rate for per-bucket duration estimates.
_EWMA_ALPHA: float = 0.2


# Release reasons. Recorded per request so the diagnostics recorder can report
# *which* condition let a retrieve through. Without this, "the mechanism did
# nothing" is indistinguishable from "the mechanism did the wrong thing".
REASON_K_REACHED = "K_REACHED"
REASON_NOT_HELD_SHAPE = "NOT_HELD_SHAPE"
REASON_NO_SIBLING = "NO_SIBLING"
REASON_TAIL_FILLS_STEP = "TAIL_FILLS_STEP"
REASON_COHORT_TOO_SMALL = "COHORT_TOO_SMALL"
REASON_DEADLINE_EXPIRED = "DEADLINE_EXPIRED"
REASON_NO_ETA = "NO_ETA"
REASON_ETA_TOO_LATE = "ETA_TOO_LATE"
REASON_HELD = "HELD"
REASON_EMPTY_PENDING = "EMPTY_PENDING"


def _bucket_for(tail_tokens: int) -> int:
    """Return the nearest bucket key in log2(tail_tokens) space."""
    if tail_tokens <= 0:
        return _DURATION_BUCKETS[0]
    log2 = math.log2(tail_tokens)
    return min(_DURATION_BUCKETS, key=lambda b: abs(log2 - b))


class AdmissionCoalescer:
    """Per-step gate deciding which completed retrieves to report.

    Knows nothing about futures or the MP adapter: each engine step the
    caller passes the ready and pending request-id sets. Per-request
    state (submission timestamp, prefill tail) lives here and is dropped
    when the request is reported or drained.
    """

    def __init__(
        self,
        enabled: bool,
        budget_tokens: int,
        record_decisions: bool = False,
        *,
        deadline_s: float = _DEADLINE_S,
        fill_alpha: float = _FILL_ALPHA,
        coalesce_min: int = _COALESCE_MIN,
        require_eta: bool = True,
    ) -> None:
        self._enabled = enabled
        self._budget = budget_tokens
        # Gate price knobs. The defaults reproduce the module constants
        # exactly; non-default values exist so the hold-cost upper-bound
        # experiment (hold aggressively, ignore the price checks) runs from
        # configuration alone. fill_alpha <= 0 disables the fill-floor
        # conditions rather than making them always true, and require_eta
        # off skips the ETA estimate entirely.
        self._deadline_s = float(deadline_s)
        self._fill_alpha = float(fill_alpha)
        self._coalesce_min = int(coalesce_min)
        self._require_eta = bool(require_eta)
        # Retain per-request release reasons for the diagnostics recorder.
        # Off by default so the non-instrumented path keeps no extra state:
        # an unreported request would otherwise retain its decision forever.
        self._record_decisions = bool(record_decisions)
        # request_id -> retrieve submission timestamp.
        self._submit_ts: dict[str, float] = {}
        # request_id -> prefill tail in tokens.
        self._tail: dict[str, int] = {}
        # Per-bucket EWMA of retrieve durations; None until observed.
        self._bucket_ewma: dict[int, float | None] = dict.fromkeys(_DURATION_BUCKETS)
        self._log_calls = 0
        self._log_holds = 0
        # request_id -> (release reason, decision context), set when the id
        # becomes ready and cleared when it is reported. Bounded by the
        # number of in-flight retrieves.
        self._decisions: dict[str, tuple[str, dict[str, object]]] = {}
        self._reason_counts: dict[str, int] = {}
        self._last_reason: str = REASON_NOT_HELD_SHAPE
        self._last_ctx: dict[str, object] = {}
        # request_id -> timestamp of the first hold decision, used to
        # subtract hold time from duration samples so the EWMA tracks
        # the retrieve pipeline rather than our own waiting.
        self._first_hold: dict[str, float] = {}

    def on_submit(self, request_id: str, tail_tokens: int) -> None:
        """Record submission time and prefill tail for a retrieve.

        No-op when the gate is disabled: the off path keeps no
        per-request state, so it cannot grow without bound.

        Args:
            request_id: The request identifier.
            tail_tokens: Prefill tail in tokens.
        """
        if not self._enabled:
            return
        self._submit_ts[request_id] = time.monotonic()
        self._tail[request_id] = tail_tokens

    def forget(self, request_ids: set[str]) -> None:
        """Drop per-request state without reporting.

        Used for requests that leave ``retrieve_futures`` without going
        through the gate (preempted or cancelled retrieves).

        Args:
            request_ids: Request ids whose state should be dropped.
        """
        for request_id in request_ids:
            self._submit_ts.pop(request_id, None)
            self._tail.pop(request_id, None)
            self._first_hold.pop(request_id, None)
            self._decisions.pop(request_id, None)

    def drain(self) -> None:
        """Drop all per-request state (unhealthy path reports everything)."""
        self._submit_ts.clear()
        self._tail.clear()
        self._first_hold.clear()
        self._decisions.clear()

    def select_to_report(self, ready: set[str], pending: set[str]) -> set[str]:
        """Return the subset of *ready* to report this engine step.

        Held ids keep their state and are re-evaluated on the next call,
        so a completed retrieve is reported exactly once.

        Args:
            ready: Request ids whose retrieve completed this call.
            pending: All in-flight retrieve ids (superset of *ready*).

        Returns:
            The subset of *ready* to report this engine step; an empty
            set means hold everything for one more step.
        """
        if not self._enabled:
            return ready
        reported, reason, ctx = self._hold(ready, pending)
        now = time.monotonic()
        if len(ready) > len(reported):
            self._log_holds += len(ready) - len(reported)
        self._last_reason = reason
        self._last_ctx = ctx
        self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1
        # Remember why each newly-ready request was released (or that it is
        # being held), so the diagnostics recorder can attribute the decision
        # at report time rather than guessing from counters. Skipped entirely
        # when nothing will read them.
        if self._record_decisions:
            for request_id in ready:
                self._decisions[request_id] = (reason, dict(ctx))
        if _LOG_EVERY and self._enabled:
            self._log_calls += 1
            if self._log_calls % _LOG_EVERY == 0:
                ewma = (
                    ", ".join(
                        f"b{k}:{int(v * 1000)}ms"
                        for k, v in sorted(self._bucket_ewma.items())
                        if v is not None
                    )
                    or "n/a"
                )
                reasons = " ".join(
                    f"{k}={v}" for k, v in sorted(self._reason_counts.items())
                )
                logger.info(
                    "admit_coalesce: calls=%d holds=%d ewma[%s] deadline_ms=%d "
                    "reasons[%s]",
                    self._log_calls,
                    self._log_holds,
                    ewma,
                    int(self._deadline_s * 1000),
                    reasons,
                )
        for request_id in ready - reported:
            self._first_hold.setdefault(request_id, now)
        self._note_durations(reported)
        for request_id in reported:
            self._submit_ts.pop(request_id, None)
            self._tail.pop(request_id, None)
            self._first_hold.pop(request_id, None)
            # NOTE: _decisions is deliberately NOT popped here. The adapter
            # reads the reason after this call returns (take_decision), and a
            # pop here would leave it with nothing to attribute. Entries are
            # reclaimed by take_decision, forget, or drain.
        return reported

    def take_decision(self, request_id: str) -> tuple[str, dict[str, object]]:
        """Return and forget the recorded release reason for *request_id*.

        Called by the diagnostics recorder at report time. Missing entries
        yield a benign default so a caller can never fail on a request the
        gate never saw (for example one dropped while unhealthy, or when
        ``record_decisions`` is off).
        """
        if not self._decisions:
            return REASON_NOT_HELD_SHAPE, {}
        reason, ctx = self._decisions.pop(request_id, (REASON_NOT_HELD_SHAPE, {}))
        return reason, ctx

    def peek_decision(self, request_id: str) -> tuple[str, dict[str, object]]:
        """Return the recorded release reason without forgetting it."""
        return self._decisions.get(request_id, (REASON_NOT_HELD_SHAPE, {}))

    def _estimate_duration(self, tail_tokens: int) -> float | None:
        """Return the bucket EWMA for this tail length, or None.

        None means the bucket has no observations yet; the caller
        releases rather than holding with no estimate.
        """
        return self._bucket_ewma.get(_bucket_for(tail_tokens))

    def _hold(
        self, ready: set[str], pending: set[str]
    ) -> tuple[set[str], str, dict[str, object]]:
        """Hold completed retrieves for one step so that several prefill
        tails are admitted on the same step boundary.

        Release is implicit when any condition stops holding: K or more
        ready, no sibling retrieve in flight, a ready tail alone fills a
        step, the deadline (from submit) expired, or the earliest
        in-flight sibling is estimated to arrive after it.

        Returns:
            A 3-tuple ``(to_report, reason, context)``. ``reason`` names the
            first condition that stopped holding; ``context`` carries the
            numbers that condition compared. Together they let an offline
            analysis attribute every release instead of inferring it, which
            is what makes "the gate never fired" distinguishable from "the
            gate fired on the wrong requests".
        """
        ctx: dict[str, object] = {
            "n_ready": len(ready),
            "n_pending": len(pending),
            "n_inflight": 0,
            "bucket_ewma_ms": -1.0,
            "eta_min_ms": -1.0,
            "deadline_left_ms": -1.0,
            "fill_floor": self._fill_alpha * self._budget,
        }
        if not self._enabled:
            return ready, REASON_NOT_HELD_SHAPE, ctx
        if not pending:
            return ready, REASON_EMPTY_PENDING, ctx
        # Release as soon as two are ready. A merge needs exactly two prefill
        # tails on one step boundary, so waiting for a third buys no extra
        # saving while costing its whole wait in TTFT. _K stays as an upper
        # bound on how many a single step can absorb, not as a target to reach.
        if not 0 < len(ready) < self._coalesce_min:
            reason = REASON_NOT_HELD_SHAPE if not ready else REASON_K_REACHED
            return ready, reason, ctx
        inflight = pending - ready
        ctx["n_inflight"] = len(inflight)
        if not inflight:
            return ready, REASON_NO_SIBLING, ctx

        # A ready request whose tail alone fills a step gains nothing
        # from waiting: a sibling cannot join an already-full step. And
        # if even all ready + in-flight tails together cannot fill a
        # step, waiting cannot produce a full step either. fill_alpha
        # <= 0 disables both checks (upper-bound experiment).
        fill_floor = self._fill_alpha * self._budget
        tails = self._tail
        if fill_floor > 0:
            if any(tails.get(r, 0) >= fill_floor for r in ready):
                ctx["max_ready_tail"] = max(tails.get(r, 0) for r in ready)
                return ready, REASON_TAIL_FILLS_STEP, ctx
            cohort = ready | inflight
            cohort_sum = sum(tails.get(r, 0) for r in cohort)
            ctx["cohort_tail_sum"] = cohort_sum
            if cohort_sum < fill_floor:
                return ready, REASON_COHORT_TOO_SMALL, ctx

        now = time.monotonic()
        oldest_submit = min(self._submit_ts.get(r, 0.0) for r in ready)
        deadline_left = self._deadline_s - (now - oldest_submit)
        ctx["deadline_left_ms"] = deadline_left * 1000.0
        if deadline_left <= 0:
            return ready, REASON_DEADLINE_EXPIRED, ctx

        if not self._require_eta:
            return set(), REASON_HELD, ctx

        # Hold only while the earliest in-flight sibling is estimated to
        # finish before the deadline: its remaining time is the typical
        # retrieve duration for ITS tail length minus how long it has
        # already waited. Without an estimate for that length, release.
        eta_min = None
        ewma_used = None
        for r in inflight:
            est = self._estimate_duration(tails.get(r, 0))
            if est is None:
                continue
            waited = now - self._submit_ts.get(r, now)
            eta = max(0.0, est - waited)
            if eta_min is None or eta < eta_min:
                eta_min = eta
                ewma_used = est
        if ewma_used is not None:
            ctx["bucket_ewma_ms"] = ewma_used * 1000.0
        if eta_min is not None:
            ctx["eta_min_ms"] = eta_min * 1000.0
        if eta_min is None or eta_min > deadline_left:
            reason = REASON_NO_ETA if eta_min is None else REASON_ETA_TOO_LATE
            return ready, reason, ctx

        return set(), REASON_HELD, ctx

    def _note_durations(self, reported: set[str]) -> None:
        """Update the per-bucket duration EWMA from reported requests.

        A request that was held contributes its duration up to the first
        hold, so the estimate tracks the retrieve pipeline rather than
        the gate's own waiting.
        """
        now = time.monotonic()
        for request_id in reported:
            t0 = self._submit_ts.get(request_id)
            if t0 is None:
                continue
            duration = now - t0
            first_hold = self._first_hold.get(request_id)
            if first_hold is not None:
                duration = max(0.0, first_hold - t0)
            bucket = _bucket_for(self._tail.get(request_id, 0))
            prev = self._bucket_ewma[bucket]
            self._bucket_ewma[bucket] = (
                duration
                if prev is None
                else (1 - _EWMA_ALPHA) * prev + _EWMA_ALPHA * duration
            )

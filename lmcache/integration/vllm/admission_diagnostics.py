# SPDX-License-Identifier: Apache-2.0
"""Observability for MP retrieves and the admission-coalescing gate.

The gate in :mod:`lmcache.integration.vllm.admission_coalescing` decides
whether to hold a completed retrieve based on *estimates*. This module
records the **ground truth** next to each estimate so the two can be
compared offline, without changing any gate behaviour.

It exists to answer five questions that end-to-end A/B runs cannot, because
their effect sizes sit inside run-to-run variance:

  E1  Is retrieve duration ``d`` driven by the HIT size or by the TAIL size?
      The gate buckets its duration EWMA on ``log2(tail_tokens)``, but
      ``GetRetrieveMetadata`` retrieves the range
      ``[num_vllm_hit_tokens, num_lmcache_hit_tokens)`` -- i.e. the *hit*.
      If ``d`` tracks the hit, every ETA the gate computes is systematically
      wrong, and the AgentX corpus (hit 140k..676k vs tail 1.4k..6.8k) is the
      worst case for it.
  E2  Does the server serialize concurrent retrieves?  With
      ``--max-gpu-workers 1`` the ``AffinityThreadPool`` routes every client
      key to slot ``next_slot % 1 == 0``, so all ranks share one thread.
      ``d`` then grows with concurrency rather than staying flat.
  E3  What is the per-rank spread of ``d``?  ``get_finished_count()`` returns
      ``None``, so vLLM's ``KVOutputAggregator`` waits for *all* ranks
      (AND). The request becomes visible only at ``max(d_1..d_R)``, so cost
      is paid on the max while the gate's benefit needs agreement.
  E4  What is ``c0``, the wall-clock actually saved by admitting one more
      request on an existing step?  Everything downstream is priced by it.
  E5  Is the engine saturated while we hold?  vLLM allocates KV blocks
      before the retrieve completes (``WAITING_FOR_REMOTE_KVS`` holds
      blocks), so a hold pins blocks and can cascade into queueing delay.

Every record is one INFO line with a stable ``key=value`` shape so it can be
grepped and parsed offline (see ``tools/analyze_admission_diagnostics.py``).
Records are buffered and flushed in batches: emitting one line per retrieve
per rank would dominate the log at high concurrency.

This module never raises into the engine path. A diagnostics failure must
not cost a request, so every public entry point swallows its own errors.
"""

# Standard
from collections import deque
from dataclasses import dataclass
import os
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Flush the buffer after this many records. Kept small so a short run (a smoke
# test with a handful of requests) still emits its records before the process
# exits; close()/shutdown() flush the remainder in any case.
_FLUSH_EVERY = 32
# Hard cap on buffered records; oldest are dropped when exceeded. This bounds
# memory if the logger itself blocks.
_MAX_BUFFER = 8192

# Release reasons, mirroring the return points of AdmissionCoalescer._hold.
# These are the single most valuable output: they say *which* condition let a
# retrieve through, which turns "the mechanism did nothing" from a guess into
# a fact.
REASON_DISABLED = "DISABLED"
REASON_EMPTY_PENDING = "EMPTY_PENDING"
REASON_K_REACHED = "K_REACHED"
REASON_NOT_HELD_SHAPE = "NOT_HELD_SHAPE"
REASON_NO_SIBLING = "NO_SIBLING"
REASON_TAIL_FILLS_STEP = "TAIL_FILLS_STEP"
REASON_COHORT_TOO_SMALL = "COHORT_TOO_SMALL"
REASON_DEADLINE_EXPIRED = "DEADLINE_EXPIRED"
REASON_NO_ETA = "NO_ETA"
REASON_ETA_TOO_LATE = "ETA_TOO_LATE"
REASON_HELD = "HELD"


@dataclass
class RetrieveRecord:
    """Ground truth for one retrieve, on one rank."""

    request_id: str
    rank: int
    tail_tokens: int
    hit_tokens: int
    prompt_tokens: int
    submit_ts: float
    complete_ts: float = 0.0
    report_ts: float = 0.0
    reason: str = REASON_HELD
    bucket: int = -1
    bucket_ewma_ms: float = -1.0
    eta_min_ms: float = -1.0
    deadline_left_ms: float = -1.0
    n_inflight_at_complete: int = -1
    n_ready_at_complete: int = -1
    # Snapshot of engine-side pressure, if the caller can supply it (E5).
    kv_free_blocks: int = -1
    num_waiting: int = -1
    num_running: int = -1

    @property
    def d_ms(self) -> float:
        """Retrieve duration: submission to future completion."""
        return (self.complete_ts - self.submit_ts) * 1000.0

    @property
    def hold_ms(self) -> float:
        """Extra latency the gate added: completion to report."""
        if self.report_ts <= 0.0 or self.complete_ts <= 0.0:
            return 0.0
        return (self.report_ts - self.complete_ts) * 1000.0

    @property
    def total_ms(self) -> float:
        """Submission to report -- what the request actually waited."""
        if self.report_ts <= 0.0:
            return 0.0
        return (self.report_ts - self.submit_ts) * 1000.0

    def to_line(self) -> str:
        return (
            "ADM_DIAG rid={rid} rank={rank} tail={tail} hit={hit} prompt={prompt} "
            "d_ms={d:.1f} hold_ms={h:.1f} total_ms={t:.1f} reason={reason} "
            "bucket={bucket} ewma_ms={ewma:.1f} eta_ms={eta:.1f} "
            "deadline_left_ms={dl:.1f} inflight={inf} ready={rdy} "
            "kv_free={kvf} waiting={w} running={run}"
        ).format(
            rid=self.request_id,
            rank=self.rank,
            tail=self.tail_tokens,
            hit=self.hit_tokens,
            prompt=self.prompt_tokens,
            d=self.d_ms,
            h=self.hold_ms,
            t=self.total_ms,
            reason=self.reason,
            bucket=self.bucket,
            ewma=self.bucket_ewma_ms,
            eta=self.eta_min_ms,
            dl=self.deadline_left_ms,
            inf=self.n_inflight_at_complete,
            rdy=self.n_ready_at_complete,
            kvf=self.kv_free_blocks,
            w=self.num_waiting,
            run=self.num_running,
        )


@dataclass
class StepRecord:
    """Per-engine-step observation, for pricing ``c0`` (E4) and saturation (E5).

    ``c0`` is the coefficient of ``admitted_now`` in a regression of step
    duration on (prefill tokens, decode requests, requests newly admitted).
    Recording it per step lets that regression run offline on real data
    instead of relying on the "~600 ms fixed overhead" assumption in the PR
    description.
    """

    step_id: int
    rank: int
    step_start: float
    step_end: float = 0.0
    admitted_now: int = 0
    admitted_tails: int = 0
    prefill_tokens: int = -1
    decode_tokens: int = -1
    num_running: int = -1
    num_waiting: int = -1
    token_budget: int = -1
    budget_left: int = -1

    def to_line(self) -> str:
        dur = (self.step_end - self.step_start) * 1000.0 if self.step_end else -1.0
        return (
            "STEP_DIAG step={s} rank={r} dur_ms={d:.2f} admitted={a} "
            "admitted_tails={at} prefill_tok={pt} decode_tok={dt} "
            "running={run} waiting={w} budget={b} budget_left={bl}"
        ).format(
            s=self.step_id,
            r=self.rank,
            d=dur,
            a=self.admitted_now,
            at=self.admitted_tails,
            pt=self.prefill_tokens,
            dt=self.decode_tokens,
            run=self.num_running,
            w=self.num_waiting,
            b=self.token_budget,
            bl=self.budget_left,
        )


class AdmissionDiagnostics:
    """Buffered, rank-aware recorder for retrieve and step observations.

    Args:
        enabled: Master switch. When ``False`` every method is a no-op and no
            state is retained, so the off path cannot grow.
        rank: This worker's identity. **Must** be the vLLM worker id, not the
            kv rank: the point of E3 is to see all ranks of one scheduler
            group, and only the worker id distinguishes them.
        flush_every: Emit a batch after this many records.
    """

    def __init__(
        self,
        enabled: bool,
        rank: int = -1,
        flush_every: int = _FLUSH_EVERY,
        sink=None,
    ) -> None:
        """
        Args:
            sink: Optional ``callable(list[str])`` receiving each flushed
                batch. When given, records go to the sink *instead of* the
                logger. This exists for two reasons: a long profiling run can
                emit far more records than should share the engine log, and
                it makes the recorder directly testable. A sink that raises
                is disabled, not propagated.
        """
        self._enabled = bool(enabled)
        self._rank = int(rank)
        self._flush_every = max(1, int(flush_every))
        self._sink = sink
        self._sink_failed = False
        self._buf: deque[str] = deque(maxlen=_MAX_BUFFER)
        # request_id -> record under construction. Bounded by the number of
        # in-flight retrieves, and cleared on report/drop.
        self._open: dict[str, RetrieveRecord] = {}
        self._lock = threading.Lock()
        # Counters for the periodic summary; cheap enough to always keep.
        self.n_records = 0
        self.n_steps = 0
        self.n_flushes = 0
        self.n_emitted = 0
        self.reason_counts: dict[str, int] = {}
        self._step_id = 0
        if self._enabled:
            logger.info(
                "ADM_DIAG enabled rank=%d flush_every=%d pid=%d "
                "(per-rank recorder: compare across ranks to measure the "
                "AND-aggregation spread)",
                self._rank,
                self._flush_every,
                os.getpid(),
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- retrieve lifecycle ------------------------------------------------

    def on_submit(
        self,
        request_id: str,
        tail_tokens: int,
        hit_tokens: int,
        prompt_tokens: int,
        bucket: int = -1,
    ) -> None:
        """Open a record when a retrieve is submitted."""
        if not self._enabled:
            return
        try:
            with self._lock:
                self._open[request_id] = RetrieveRecord(
                    request_id=request_id,
                    rank=self._rank,
                    tail_tokens=int(tail_tokens),
                    hit_tokens=int(hit_tokens),
                    prompt_tokens=int(prompt_tokens),
                    submit_ts=time.monotonic(),
                    bucket=int(bucket),
                )
        except Exception:  # pragma: no cover - diagnostics must never break
            logger.debug("ADM_DIAG on_submit failed for %s", request_id, exc_info=True)

    def on_complete(
        self,
        request_id: str,
        reason: str,
        bucket_ewma_ms: float = -1.0,
        eta_min_ms: float = -1.0,
        deadline_left_ms: float = -1.0,
        n_inflight: int = -1,
        n_ready: int = -1,
    ) -> None:
        """Stamp completion plus the gate's decision context.

        Called when the future resolves, i.e. *before* any hold, so ``d_ms``
        measures the retrieve pipeline alone.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                rec = self._open.get(request_id)
                if rec is None:
                    return
                # A held retrieve stays in ``retrieve_futures`` and is polled
                # again on every later step, so it reaches this method once
                # per step. Stamping again would overwrite the true completion
                # time with the release time and inflate d_ms by exactly the
                # hold, destroying the very measurement being taken. The first
                # stamp wins.
                if rec.complete_ts > 0.0:
                    # Still refresh the decision context: the reason and ETA the
                    # gate finally acted on belong to the releasing step.
                    rec.reason = reason
                    rec.bucket_ewma_ms = float(bucket_ewma_ms)
                    rec.eta_min_ms = float(eta_min_ms)
                    rec.deadline_left_ms = float(deadline_left_ms)
                    rec.n_inflight_at_complete = int(n_inflight)
                    rec.n_ready_at_complete = int(n_ready)
                    return
                rec.complete_ts = time.monotonic()
                rec.reason = reason
                rec.bucket_ewma_ms = float(bucket_ewma_ms)
                rec.eta_min_ms = float(eta_min_ms)
                rec.deadline_left_ms = float(deadline_left_ms)
                rec.n_inflight_at_complete = int(n_inflight)
                rec.n_ready_at_complete = int(n_ready)
        except Exception:  # pragma: no cover
            logger.debug(
                "ADM_DIAG on_complete failed for %s", request_id, exc_info=True
            )

    def on_report(self, request_id: str, reason: str | None = None) -> None:
        """Close and buffer a record when the retrieve is reported to vLLM.

        ``hold_ms`` is the difference between this stamp and completion, so it
        isolates exactly what the gate added on top of the transfer.

        Args:
            reason: The *final* release reason, known only after the gate has
                run. It overrides the provisional reason stamped at completion
                time, because the decision that actually released this request
                is made in ``select_to_report``, after ``on_complete``.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                rec = self._open.pop(request_id, None)
                if rec is None:
                    return
                if reason:
                    rec.reason = reason
                rec.report_ts = time.monotonic()
                self._emit_locked(rec.to_line())
                self.n_records += 1
                self.reason_counts[rec.reason] = (
                    self.reason_counts.get(rec.reason, 0) + 1
                )
        except Exception:  # pragma: no cover
            logger.debug("ADM_DIAG on_report failed for %s", request_id, exc_info=True)

    def on_drop(self, request_ids) -> None:
        """Discard records for requests that left tracking without a report.

        Preempted, cancelled, or failed retrieves never reach ``on_report``;
        without this the open dict would grow for the process lifetime.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                for rid in request_ids:
                    self._open.pop(rid, None)
        except Exception:  # pragma: no cover
            logger.debug("ADM_DIAG on_drop failed", exc_info=True)

    def clear(self) -> None:
        """Drop all open records (unhealthy/drain path)."""
        if not self._enabled:
            return
        with self._lock:
            self._open.clear()

    # -- step lifecycle (E4/E5) -------------------------------------------

    def next_step_id(self) -> int:
        with self._lock:
            self._step_id += 1
            return self._step_id

    def on_step(self, rec: StepRecord) -> None:
        """Buffer one step observation."""
        if not self._enabled:
            return
        try:
            with self._lock:
                rec.rank = self._rank
                self._emit_locked(rec.to_line())
                self.n_steps += 1
        except Exception:  # pragma: no cover
            logger.debug("ADM_DIAG on_step failed", exc_info=True)

    # -- summary / flush ---------------------------------------------------

    def summary(self, tag: str = "") -> None:
        """Emit aggregate counters. Call periodically, not per request."""
        if not self._enabled:
            return
        try:
            with self._lock:
                reasons = " ".join(
                    f"{k}={v}" for k, v in sorted(self.reason_counts.items())
                )
                logger.info(
                    "ADM_DIAG_SUMMARY rank=%d%s records=%d steps=%d flushes=%d "
                    "open=%d reasons[%s]",
                    self._rank,
                    f" {tag}" if tag else "",
                    self.n_records,
                    self.n_steps,
                    self.n_flushes,
                    len(self._open),
                    reasons,
                )
        except Exception:  # pragma: no cover
            logger.debug("ADM_DIAG summary failed", exc_info=True)

    def flush(self) -> None:
        """Force-emit whatever is buffered (call at shutdown)."""
        if not self._enabled:
            return
        with self._lock:
            self._flush_locked()

    def _emit_locked(self, line: str) -> None:
        """Append and flush when the batch is full. Caller holds the lock."""
        self._buf.append(line)
        if len(self._buf) >= self._flush_every:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        batch = list(self._buf)
        self._buf.clear()
        self.n_flushes += 1
        self.n_emitted += len(batch)
        if self._sink is not None and not self._sink_failed:
            try:
                self._sink(batch)
                return
            except Exception:  # pragma: no cover
                # A broken sink must not lose the engine path; fall back to
                # the logger once and stop retrying the sink.
                self._sink_failed = True
                logger.warning(
                    "ADM_DIAG sink failed; falling back to the logger",
                    exc_info=True,
                )
        try:
            # One log call per batch keeps the line count manageable while
            # keeping each record grep-able on its own line.
            logger.info(
                "ADM_DIAG_BATCH rank=%d n=%d\n%s",
                self._rank,
                len(batch),
                "\n".join(batch),
            )
        except Exception:  # pragma: no cover
            logger.debug("ADM_DIAG flush failed", exc_info=True)


def diagnostics_enabled_from_env() -> bool:
    """Read the env-var override, so the switch can be flipped without
    touching vLLM's kv_connector_extra_config (useful when the deployment is
    already running and only a restart of the worker is possible)."""
    return os.environ.get("LMCACHE_MP_ADM_DIAG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

# SPDX-License-Identifier: Apache-2.0
"""Worker-side admission coalescing for the MP adapter.

Holds completed KV retrieves for one engine step so that several
prefill tails are admitted on the same step boundary and the per-step
token budget is filled. See docs/design/v1/multiprocess/admission_coalescing.md.

Tuning build: parameters (alpha, K, deadline, sum condition, ETA) are
re-read from a control file on mtime change, so A/B sweeps can flip
configurations between benchmark passes without restarting vLLM. The
file also toggles the switch itself, so off/on arms alternate without a
restart too. See _TUNING_FILE below for the format.
"""

# Standard
import json
import os
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Fallback step budget in tokens; the connector supplies vLLM's real
# max_num_batched_tokens, so this only applies to direct construction.
DEFAULT_BUDGET_TOKENS: int = 8192

# Tuning control file. When present, its JSON content overrides the
# compile-time defaults AND the startup switch on the fly (mtime-cached,
# one stat per gate call). Absent file = startup config governs.
# Format: {"on": true|false, "alpha": 0.75, "k": 3,
#          "deadline_s": 3.0, "sum": true, "eta": true}
_TUNING_FILE = "/tmp/lmc_ac_tuning.json"

# Compile-time defaults (used until the tuning file says otherwise).
_DEADLINE_S: float = 3.0
_FILL_ALPHA: float = 0.75
_K: int = 3
_SUM_GUARD: bool = True
_SIBLING_ETA: bool = True
# Log a decision summary every N gate calls (0 = never).
_LOG_EVERY: int = 2000


class _Tuning:
    """Mtime-cached view of _TUNING_FILE; falls back to defaults."""

    def __init__(self) -> None:
        self._cache: tuple[float, dict] | None = None

    def get(self) -> dict:
        try:
            mtime = os.stat(_TUNING_FILE).st_mtime
        except OSError:
            return {}
        if self._cache is None or self._cache[0] != mtime:
            try:
                with open(_TUNING_FILE) as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    self._cache = (mtime, cfg)
                    logger.info("admit_coalesce tuning: %s", cfg)
                    return cfg
            except (OSError, ValueError):
                pass
        return self._cache[1] if self._cache else {}


_TUNING = _Tuning()


class AdmissionCoalescer:
    """Per-step gate deciding which completed retrieves to report.

    Knows nothing about futures or the MP adapter: each engine step the
    caller passes the ready and pending request-id sets. Per-request
    state (submission timestamp, prefill tail) lives here and is dropped
    when the request is reported or drained.
    """

    def __init__(self, enabled: bool, budget_tokens: int) -> None:
        self._enabled = enabled
        self._budget = budget_tokens
        # request_id -> retrieve submission timestamp.
        self._submit_ts: dict[str, float] = {}
        # request_id -> prefill tail in tokens (prompt length minus cache hit).
        self._tail: dict[str, int] = {}
        # EWMA of retrieve durations (submission to report); None until
        # the first observation.
        self._duration_ewma: float | None = None
        # Periodic decision summary counters.
        self._log_calls = 0
        self._log_holds = 0
        # Per-branch release counters for the summary line.
        self._rel = {
            "k": 0, "nosib": 0, "tail": 0, "sum": 0,
            "dl": 0, "eta": 0, "cold": 0,
        }
        # request_id -> timestamp of the first hold decision, used to
        # subtract hold time from duration samples.
        self._first_hold: dict[str, float] = {}

    def _cfg(self, key: str, default):
        return type(default)(_TUNING.get().get(key, default))

    def on_submit(self, request_id: str, tail_tokens: int) -> None:
        """Record submission time and prefill tail for a retrieve."""
        self._submit_ts[request_id] = time.monotonic()
        self._tail[request_id] = tail_tokens

    def drain(self) -> None:
        """Drop all per-request state (unhealthy path reports everything)."""
        self._submit_ts.clear()
        self._tail.clear()
        self._first_hold.clear()

    def select_to_report(self, ready: set[str], pending: set[str]) -> set[str]:
        """Return the subset of *ready* to report this engine step.

        Held ids keep their state and are re-evaluated on the next call,
        so a completed retrieve is reported exactly once. Reported
        requests update the duration EWMA and drop their state.
        """
        reported = self._hold(ready, pending)
        now = time.monotonic()
        if len(ready) > len(reported):
            self._log_holds += len(ready) - len(reported)
        if _LOG_EVERY and self._enabled:
            self._log_calls += 1
            if self._log_calls % _LOG_EVERY == 0:
                rel = self._rel
                tot = sum(rel.values()) or 1
                logger.info(
                    "admit_coalesce: calls=%d holds=%d(%.1f%%) "
                    "rel{k=%d(%.0f%%),nosib=%d(%.0f%%),tail=%d,sum=%d,"
                    "dl=%d,eta=%d,cold=%d} ewma_ms=%s "
                    "cfg{a=%.2f,k=%d,T=%.1f,sum=%d,eta=%d}",
                    self._log_calls,
                    self._log_holds,
                    self._log_holds / self._log_calls * 100,
                    rel["k"], rel["k"] / tot * 100,
                    rel["nosib"], rel["nosib"] / tot * 100,
                    rel["tail"], rel["sum"],
                    rel["dl"], rel["eta"], rel["cold"],
                    f"{self._duration_ewma * 1000:.0f}"
                    if self._duration_ewma is not None else "n/a",
                    self._cfg("alpha", _FILL_ALPHA),
                    self._cfg("k", _K),
                    self._cfg("deadline_s", _DEADLINE_S),
                    self._cfg("sum", _SUM_GUARD),
                    self._cfg("eta", _SIBLING_ETA),
                )
                for k in self._rel:
                    self._rel[k] = 0
        for request_id in ready - reported:
            self._first_hold.setdefault(request_id, now)
        self._note_durations(reported)
        for request_id in reported:
            self._submit_ts.pop(request_id, None)
            self._tail.pop(request_id, None)
            self._first_hold.pop(request_id, None)
        return reported

    def _hold(self, ready: set[str], pending: set[str]) -> set[str]:
        """Hold completed retrieves for one step so that several prefill
        tails are admitted on the same step boundary.

        Release is implicit when any condition stops holding. Each
        release path bumps its counter so the periodic summary shows
        which branch dominates.
        """
        cfg = _TUNING.get()
        if not self._enabled:
            return ready
        if cfg.get("on") is False:
            return ready  # tuning file forces the arm off
        if not pending:
            return ready
        k = int(cfg.get("k", _K))
        if not 0 < len(ready) < k:
            self._rel["k"] += 1
            return ready
        inflight = pending - ready
        if not inflight:
            self._rel["nosib"] += 1
            return ready

        alpha = float(cfg.get("alpha", _FILL_ALPHA))
        fill_floor = alpha * self._budget
        tails = self._tail
        if any(tails.get(r, 0) >= fill_floor for r in ready):
            self._rel["tail"] += 1
            return ready
        if cfg.get("sum", _SUM_GUARD):
            cohort = ready | inflight
            if sum(tails.get(r, 0) for r in cohort) < fill_floor:
                self._rel["sum"] += 1
                return ready

        deadline_s = float(cfg.get("deadline_s", _DEADLINE_S))
        now = time.monotonic()
        oldest_submit = min(self._submit_ts.get(r, 0.0) for r in ready)
        deadline_left = deadline_s - (now - oldest_submit)
        if deadline_left <= 0:
            self._rel["dl"] += 1
            return ready

        if cfg.get("eta", _SIBLING_ETA):
            if self._duration_ewma is None:
                self._rel["cold"] += 1
                return ready
            eta_min = min(
                max(0.0, self._duration_ewma - (now - self._submit_ts.get(r, now)))
                for r in inflight
            )
            if eta_min > deadline_left:
                self._rel["eta"] += 1
                return ready

        return set()

    def _note_durations(self, reported: set[str]) -> None:
        """Update the retrieve-duration EWMA (submission to report).

        Sampling happens at report time. A request that was held across
        steps contributes a duration that includes its hold time; hold
        time is subtracted so the EWMA tracks the retrieve pipeline
        rather than our own waiting.
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
            self._duration_ewma = (
                duration
                if self._duration_ewma is None
                else 0.8 * self._duration_ewma + 0.2 * duration
            )

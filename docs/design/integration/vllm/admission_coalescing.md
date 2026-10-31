# Admission Coalescing

## Problem

In multiprocess mode, KV retrieves complete asynchronously at staggered
times. The scheduler can only admit a request whose retrieve has
completed, and a step boundary typically sees at most one ready request,
so prefill tails are admitted one per step. Every prefill step carries a
large fixed cost, so spreading N tails over N steps wastes engine time
that admitting them together on fewer, fuller steps would save.

## Mechanism

When a retrieve completes, the worker adapter may hold it for a bounded
number of engine steps so that it is reported together with a sibling and
both prefill tails land on the same step boundary. One step is removed
from the run per sibling that joins.

Holding happens in `get_finished()`; the user enables it with:

```
"lmcache.mp.admit_coalesce": true
```

Holding is only worth its latency when the coalesced tails **fit one
step**. The gate therefore prices a hold in the unit cost and benefit
share — the engine step:

- cost: the held request waits `w` steps longer, so its TTFT grows by `w`
  steps;
- benefit: every sibling that joins the coalesced admission removes one
  prefill-carrying step from the run.

The gate holds only when the expected joins pay for the steps it charges.
No stall time in milliseconds has to be calibrated, and no workload-specific
token target has to be picked: both sides count steps.

## Parameters

| Symbol | Meaning | Source | Default |
|---|---|---|---|
| switch | enable the mechanism | user config | false |
| B | per-step token budget | vLLM `max_num_batched_tokens`, passed by the connector | — |
| weight | expected joins required per step of hold | user config | 1.0 |
| max_hold_steps | upper bound on steps one retrieve is held | user config | 2 |
| step | engine step duration | measured: gap between consecutive gate calls | learned |
| gap | retrieve completion interval | measured: gap between consecutive completions | learned |
| tail | prefill tail of a request | read from the op at submit time | learned |
| head | tokens a step can give to prefill | `B * (1 - 0.2)`, the reserve leaves room for decode | derived |
| fits | further tails that fit beside the ready ones | `(head - ready_tail) / tail` | derived |

Per-request state: submission timestamp, prefill tail in tokens, and the
number of steps already held. New metadata fields do not survive
scheduler-to-worker serialization, so the tail is read from the op
(packed token count minus the cached range).

`weight` is the single tradeoff knob. Raise it (2.0) to protect TTFT, lower
it (0.5) to favour TPOT. `max_hold_steps` bounds the worst-case added TTFT
and is a safety limit rather than a tuning value.

## Gate

```
on each get_finished() call (once per engine step):

    ready    = retrieves whose future completed this call
    failed   = ready retrieves whose result is an error
    inflight = submitted retrieves still pending        # retrieve_futures - ready

    report failed immediately; they never take part in the hold decision

    report everything (no hold) if any of:
        no inflight sibling                              # nothing to wait for
        ready tail is zero                               # no prefill step to save
        sum(ready tails) > head                          # cannot fit one step
        (head - sum(ready tails)) / tail < 1             # no room for a sibling
        step or gap not measured yet                     # cold start: no price
        steps_held + 1 > max_hold_steps                  # bound reached
        p_join * min(fits, len(inflight)) < weight * (steps_held + 1)
                                                         # joins cannot pay

    otherwise: hold the ready retrieves for one step
               (they stay in retrieve_futures and are re-evaluated next call)

    p_join = min(1, w * step / max(gap, step)),  w = steps_held + 1

after reporting: update the hold statistics from each reported request
                 (first hold -> report), and count a merge when a held
                 request is released together with a sibling.
```

## Why each release condition

- **No sibling in flight** — with nothing left to wait for, holding is
  pure latency.
- **Zero ready tail** — a fully cached request has no prefill step of its
  own to remove, so merging it saves nothing.
- **The cohort cannot fit one step** — the whole point of a hold is to
  replace N prefill steps with fewer. Two 4096-token tails against an 8192
  budget never share a step (the decode reserve leaves a 6554-token head),
  so holding them adds latency and removes nothing. This condition is what
  keeps the mechanism from regressing on long-tail workloads, where a
  fill-ratio guard such as `sum(tails) >= alpha * B` is satisfied while the
  merge is impossible.
- **No room for a sibling** — the same arithmetic one step further: if
  `head - ready_tail` cannot hold another tail, a sibling cannot join the
  step even if it arrives.
- **Cold start** — until the step duration and the completion gap have
  been observed, a hold cannot be priced. Reporting keeps the status quo.
- **Bound reached** — `max_hold_steps` caps the added TTFT; a hold that
  outlives its bound is a latency leak, not coalescing.
- **Joins cannot pay** — the only economic condition. The completion gap
  says how long the next sibling needs; the fit count caps how many can
  share the step. If the expected joins do not cover `weight` steps of
  hold, the request is reported and admitted immediately.

## Telemetry

`AdmissionCoalescer.stats()` and `release_reasons()` are the mechanism's
acceptance criteria, and a summary line is logged every 2000 gate calls:

```
admit_coalesce: calls=… holds=… held=… merges=… merge_rate=… saved_steps=…
                hold_ms{mean=…,max=…,n=…} step_ms=… gap_ms=… tail=…
                reasons={nosib=…, fit=…, cost=…, hold=…, cold=…}
```

- `merge_rate` (merges / hold events) is the gate's own hit rate. A gate
  that holds often but merges rarely is charging latency for nothing.
- `saved_steps` is the mechanism's benefit in the unit the gate prices.
- `reasons` says why the gate is idle: `fit` means the workload's tails can
  never share a step (the mechanism does not apply), `cost` means completions
  are too sparse for a sibling to arrive in time.

## Guarantees

- A completed retrieve is reported exactly once: held ids stay in
  `retrieve_futures` and are re-reported only when released.
- Failed retrieves bypass the gate and are reported in the same
  `get_finished()` call, so the error/recovery path keeps its latency.
  Nothing is dropped on failure paths: unhealthy drains report everything
  immediately.
- The added latency of any held request is bounded by
  `max_hold_steps * step`, and a request is never held when the coalesced
  tail cannot fit one step.
- Disabled (default): the gate returns immediately and no per-request or
  measured state is kept or updated.

## Known limits

- The decision is per worker process, while admission is global: vLLM
  recovers a request only after every rank reports `finished_recving`. With
  TP > 1 the effective added latency is the maximum over ranks and the
  effective hold rate is the union over ranks, so the cost side is larger
  than a single rank's counters suggest. A single decision per engine step
  (scheduler-side, delivered through connector metadata) would remove that
  gap.
- The gate cannot see the decode cohort or the free budget of the next
  step, so the decode reserve is a fixed 20% of `B` rather than the real
  occupancy.
- Coalescing removes prefill steps but does not change how many prefill
  steps the scheduler spends per tail. A workload whose tails are already
  chunked across many steps has little for this mechanism to recover.

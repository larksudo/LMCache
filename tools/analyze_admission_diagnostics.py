#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline analysis of admission-coalescing diagnostics.

Consumes ``ADM_DIAG`` / ``STEP_DIAG`` records (see
``lmcache/integration/vllm/admission_diagnostics.py``) and answers the five
questions that end-to-end A/B runs cannot, because their effect sizes sit
inside run-to-run variance.

Run it against worker logs::

    python tools/analyze_admission_diagnostics.py vllm_worker_*.log
    python tools/analyze_admission_diagnostics.py run.log --csv out.csv

Each question maps to a section below. The sections print a verdict, not just
numbers, so the output can be pasted into a PR comment.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter, defaultdict
import argparse
import glob
import math
import re
import statistics
import sys

ADM_RE = re.compile(
    r"ADM_DIAG (?P<kv>rid=.*?)(?=\s+ADM_DIAG |\s+STEP_DIAG |$)", re.MULTILINE
)
STEP_RE = re.compile(
    r"STEP_DIAG (?P<kv>step=.*?)(?=\s+ADM_DIAG |\s+STEP_DIAG |$)", re.MULTILINE
)
KV_RE = re.compile(r"(\w+)=(-?[\w.\-]+)")

# vLLM's own scheduler-side iteration dump (--enable-logging-iteration-details).
# This is the ONLY place prefill/decode token counts and KV pressure are visible:
# the worker-side STEP_DIAG records them as -1 sentinels. Sample line:
#   Engine 000: Iteration(4): 1 context requests, 8191 context tokens, 1
#   generation requests, 1 generation tokens, iteration elapsed time:
#   1001.96 ms, NPU KV cache usage: 47.2%
ITER_RE = re.compile(
    r"Iteration\((?P<idx>\d+)\): "
    r"(?P<ctx_req>\d+) context requests, (?P<ctx_tok>\d+) context tokens, "
    r"(?P<gen_req>\d+) generation requests, (?P<gen_tok>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed>[\d.]+) ms, "
    r"NPU KV cache usage: (?P<kv>[\d.]+)%"
)

# ADM_DIAG fields
INT_FIELDS = {
    "rank",
    "tail",
    "hit",
    "prompt",
    "bucket",
    "inflight",
    "ready",
    "kv_free",
    "waiting",
    "running",
}
FLOAT_FIELDS = {
    "d_ms",
    "hold_ms",
    "total_ms",
    "ewma_ms",
    "eta_ms",
    "deadline_left_ms",
}
# STEP_DIAG fields. Without these the values stay strings and every numeric
# comparison in q5_steps raises TypeError instead of reporting.
INT_FIELDS |= {
    "step",
    "admitted",
    "admitted_tails",
    "prefill_tok",
    "decode_tok",
    "budget",
    "budget_left",
}
FLOAT_FIELDS |= {"dur_ms"}


def parse_kv(text: str) -> dict:
    out = {}
    for k, v in KV_RE.findall(text):
        if k in INT_FIELDS:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
        elif k in FLOAT_FIELDS:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def load(paths):
    adm, step, it = [], [], []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError as exc:
            print(f"  skip {p}: {exc}", file=sys.stderr)
            continue
        for m in ADM_RE.finditer(text):
            rec = parse_kv(m.group("kv"))
            if "rid" in rec:
                rec["_file"] = p
                adm.append(rec)
        for m in STEP_RE.finditer(text):
            rec = parse_kv(m.group("kv"))
            rec["_file"] = p
            step.append(rec)
        for m in ITER_RE.finditer(text):
            it.append(
                {
                    "idx": int(m.group("idx")),
                    "ctx_req": int(m.group("ctx_req")),
                    "ctx_tok": int(m.group("ctx_tok")),
                    "gen_req": int(m.group("gen_req")),
                    "gen_tok": int(m.group("gen_tok")),
                    "elapsed_ms": float(m.group("elapsed")),
                    "kv_pct": float(m.group("kv")),
                }
            )
    return adm, step, it


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
    return s[idx]


def _spread(values):
    """Return (max-min)/median, or nan when the column is constant."""
    if not values:
        return float("nan")
    med = statistics.median(values)
    if med <= 0:
        return float("nan")
    return (max(values) - min(values)) / med


def _can_rank(values, min_distinct=3):
    """A rank correlation needs enough distinct values to mean anything.

    A constant column (every request same hit size, as in the fixed-shape
    benchmark) yields a perfectly flat rank vector, so Spearman is undefined.
    Reporting a verdict from it is worse than reporting nothing: it looks like
    evidence while carrying none.
    """
    return len(set(values)) >= min_distinct


def describe(values):
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={statistics.mean(values):.1f} "
        f"p50={pct(values, 0.5):.1f} p90={pct(values, 0.9):.1f} "
        f"max={max(values):.1f}"
    )


def spearman(xs, ys):
    """Rank correlation without scipy. n<3 -> nan."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def q1_hit_vs_tail(adm):
    """E1: is d driven by HIT (bytes moved) or by TAIL (recompute work)?"""
    section("E1  Is retrieve duration driven by HIT size or by TAIL size?")
    good = [r for r in adm if r.get("d_ms", -1) >= 0 and r.get("hit", 0) > 0]
    if len(good) < 8:
        print(f"  insufficient data (n={len(good)}; need >=8). Cannot answer.")
        return None
    hits = [r["hit"] for r in good]
    tails = [r["tail"] for r in good]
    ds = [r["d_ms"] for r in good]
    rho_hit = spearman(hits, ds)
    rho_tail = spearman(tails, ds)
    print(f"  {describe(ds)}  (retrieve duration)")
    print(f"  hit   range {min(hits)}..{max(hits)} tokens")
    print(f"  tail  range {min(tails)}..{max(tails)} tokens")
    print(f"  Spearman(d, hit ) = {rho_hit:+.3f}")
    print(f"  Spearman(d, tail) = {rho_tail:+.3f}")

    # Validity gates. Without them a constant hit/tail column (the fixed-shape
    # benchmark) or a tiny sample still produced a confident VERDICT.
    need = []
    if not _can_rank(hits):
        need.append(f"hit is constant ({len(set(hits))} distinct value(s))")
    if not _can_rank(tails):
        need.append(f"tail is constant ({len(set(tails))} distinct value(s))")
    if len(good) < 30:
        need.append(f"only n={len(good)} records (<30)")
    if need:
        print("  CANNOT DECIDE:")
        for n in need:
            print(f"    - {n}")
        if not _can_rank(hits) or not _can_rank(tails):
            print("    E1 needs a workload where hit or tail actually varies;")
            print("    a fixed prefix/suffix shape cannot answer this question.")
        else:
            print("    The columns do vary, but the sample is too small to rank.")
        return {"rho_hit": rho_hit, "rho_tail": rho_tail, "decided": False}
    # The gate buckets on tail. Show the damage directly.
    by_bucket = defaultdict(list)
    for r in good:
        by_bucket[r.get("bucket", -1)].append(r["d_ms"])
    print("  per-bucket d spread (the gate uses ONE ewma scalar per bucket):")
    worst = 0.0
    for b in sorted(by_bucket):
        v = by_bucket[b]
        if len(v) < 2:
            continue
        spread = max(v) / max(1e-9, min(v))
        worst = max(worst, spread)
        print(
            f"    bucket {b:>3}: n={len(v):<5} min={min(v):8.1f} "
            f"max={max(v):9.1f} spread={spread:6.1f}x"
        )
    driver = "HIT" if rho_hit > rho_tail else "TAIL"
    print()
    print(f"  VERDICT: d is driven by {driver}.")
    if rho_hit > rho_tail + 0.15 and worst > 3.0:
        print("    -> The gate buckets its ETA on log2(tail), but d tracks the")
        print(f"       hit. Within-bucket spread reaches {worst:.1f}x, so every")
        print("       ETA the gate computes is systematically wrong.")
        print("    -> FIX: bucket on hit_tokens (or retrieved bytes), not tail.")
    elif rho_tail >= rho_hit:
        print("    -> Tail bucketing is defensible on this data.")
    return {"rho_hit": rho_hit, "rho_tail": rho_tail, "worst_spread": worst}


def q2_serialization(adm):
    """E2: does d grow with concurrency (server-side serialization)?"""
    section("E2  Does retrieve duration grow with concurrency (serialization)?")
    good = [r for r in adm if r.get("d_ms", -1) >= 0 and r.get("inflight", -1) >= 0]
    if len(good) < 8:
        print(f"  insufficient data (n={len(good)}). Cannot answer.")
        return None
    by_inflight = defaultdict(list)
    for r in good:
        by_inflight[r["inflight"]].append(r["d_ms"])
    print("  concurrent in-flight retrieves at completion -> d")
    for k in sorted(by_inflight):
        v = by_inflight[k]
        print(
            f"    inflight={k:<3} n={len(v):<5} d mean={statistics.mean(v):9.1f} "
            f"p90={pct(v, 0.9):9.1f}"
        )
    xs = [r["inflight"] for r in good]
    ys = [r["d_ms"] for r in good]
    rho = spearman(xs, ys)
    print(f"  Spearman(d, inflight) = {rho:+.3f}")
    if len(set(xs)) < 3 or len(good) < 30:
        print("  CANNOT DECIDE:")
        print(f"    inflight has {len(set(xs))} distinct value(s), n={len(good)}")
        print("    On TP1 the server has a single GPU client, so retrieve")
        print("    concurrency is structurally ~1 and this question is")
        print("    unanswerable. Measure it on TP8 with --max-gpu-workers 1/4/8.")
        return {"rho_inflight": rho, "decided": False}
    if rho > 0.4:
        keys = sorted(by_inflight)
        if len(keys) >= 2:
            lo = statistics.mean(by_inflight[keys[0]])
            hi = statistics.mean(by_inflight[keys[-1]])
            print()
            print("  VERDICT: d grows with concurrency -> retrieves are QUEUING,")
            print(f"    not running in parallel (d {lo:.0f}ms -> {hi:.0f}ms).")
            print("    Suspect --max-gpu-workers 1: AffinityThreadPool routes")
            print("    every client key to slot (next_slot % 1 == 0), so all")
            print("    ranks share one thread and their tasks are serialized.")
            print("    -> FIX: raise --max-gpu-workers to >= number of ranks.")
    else:
        print()
        print("  VERDICT: d is flat in concurrency -> transfers are parallel.")
    return {"rho_inflight": rho}


def q3_rank_spread(adm):
    """E3: per-rank spread of d (AND-aggregation makes cost the max)."""
    section("E3  Per-rank spread of d (AND-aggregation => cost is the max)")
    if not adm:
        print("  no data.")
        return None
    by_rank = defaultdict(list)
    for r in adm:
        if r.get("d_ms", -1) >= 0:
            by_rank[r.get("rank", -1)].append(r["d_ms"])
    if len(by_rank) < 2:
        print(f"  only {len(by_rank)} rank(s) present in the log.")
        print("  To measure the spread you must collect logs from ALL ranks")
        print("  (each worker process logs its own ADM_DIAG lines).")
        return None
    print("  rank -> d distribution")
    means = []
    for rk in sorted(by_rank):
        v = by_rank[rk]
        means.append(statistics.mean(v))
        print(f"    rank {rk:>2}: {describe(v)}")
    print(
        f"  cross-rank mean spread: {min(means):.0f}..{max(means):.0f} ms "
        f"({max(means) / max(1e-9, min(means)):.2f}x)"
    )

    # Pair requests seen on multiple ranks and measure the real max penalty.
    per_rid = defaultdict(dict)
    for r in adm:
        if r.get("d_ms", -1) >= 0:
            per_rid[r.get("rid")][r.get("rank")] = r["d_ms"]
    multi = {k: v for k, v in per_rid.items() if len(v) >= 2}
    if multi:
        deltas = [max(v.values()) - statistics.mean(v.values()) for v in multi.values()]
        maxes = [max(v.values()) for v in multi.values()]
        avg_of_means = [statistics.mean(v.values()) for v in multi.values()]
        print(f"  requests seen on >=2 ranks: n={len(multi)}")
        print(
            f"    mean(max d)      = {statistics.mean(maxes):.0f} ms "
            f"<- what vLLM waits for"
        )
        print(f"    mean(avg d)      = {statistics.mean(avg_of_means):.0f} ms")
        print(
            f"    max-minus-mean   = {statistics.mean(deltas):.0f} ms "
            f"of pure AND penalty"
        )
        print()
        print("  VERDICT: get_finished_count() returns None, so vLLM's")
        print("    KVOutputAggregator waits for ALL ranks (AND). Visibility is")
        print("    therefore gated on max(d_i), not the mean. The gate's cost is")
        print("    paid on the slowest rank while its benefit needs agreement.")
        print("    -> FIX: one decision point (scheduler-side), or rank-0 decides.")
    else:
        print("  no request appeared on 2+ ranks in this log set.")
        print("  (expected: each worker logs only its own rank.)")
    return {"n_ranks": len(by_rank)}


def q4_reasons(adm):
    """Release-reason attribution: which condition let requests through."""
    section("Release-reason attribution (why the gate did what it did)")
    if not adm:
        print("  no data.")
        return None
    cnt = Counter(r.get("reason", "?") for r in adm)
    total = sum(cnt.values())
    for reason, n in cnt.most_common():
        print(f"  {reason:<20} {n:>7}  {100.0 * n / total:5.1f}%")
    held = cnt.get("HELD", 0)
    print(
        f"\n  total records={total}  held={held} ({100.0 * held / max(1, total):.1f}%)"
    )
    no_sib = cnt.get("NO_SIBLING", 0)
    if total < 30:
        print(
            "  CANNOT CONCLUDE: only "
            f"{total} records. Reason shares are descriptive only;"
        )
        print("    a cache that evicted mid-run also produces NO_SIBLING, so")
        print("    check the server log for 'triggering eviction' first.")
    elif total and no_sib / total > 0.4:
        print(f"  VERDICT: {100.0 * no_sib / total:.0f}% of releases were NO_SIBLING")
        print("    (nothing in flight to coalesce with). The gate cannot act:")
        print("    the opportunity does not exist at this arrival rate.")
        print("    -> This is a workload property, not a tuning problem.")
    small = cnt.get("COHORT_TOO_SMALL", 0)
    big = cnt.get("TAIL_FILLS_STEP", 0)
    if total and (small + big) / total > 0.3:
        print(f"  NOTE: {100.0 * (small + big) / total:.0f}% released by the fill")
        print("    conditions (COHORT_TOO_SMALL / TAIL_FILLS_STEP). AgentX tails")
        print("    are bimodal - a few very long cold prefills and many short")
        print("    tool-call increments - and both ends fall outside the window")
        print("    [alpha*B/K, alpha*B) where holding can help.")
    eta = cnt.get("ETA_TOO_LATE", 0) + cnt.get("NO_ETA", 0)
    if eta:
        print(f"  NOTE: {eta} releases came from the ETA conditions, which are")
        print("    computed from the tail-bucketed EWMA. If E1 shows d tracks the")
        print("    hit, these decisions were made on a wrong estimate.")
    return {"counts": cnt}


def _q5_from_iterations(it):
    """c0 and saturation from vLLM's own iteration dump.

    Each iteration reports how many context (prefill) tokens it carried, how
    many generation requests it advanced, and how long it took. That is exactly
    the covariate the worker-side STEP_DIAG lacks, so the prefill coefficient
    and the fixed per-iteration intercept become estimable here.

    A least-squares fit of elapsed_ms ~ 1 + ctx_tok + gen_req is small enough to
    do without numpy: three normal equations in three unknowns.
    """
    durs = [r["elapsed_ms"] for r in it]
    ctx = [r["ctx_tok"] for r in it]
    gen = [r["gen_req"] for r in it]
    kv = [r["kv_pct"] for r in it]
    print(f"  iterations: {describe(durs)} ms")
    print(f"  context tokens/iter: {describe([float(c) for c in ctx])}")
    print(f"  prefill-carrying iterations: {sum(1 for c in ctx if c > 0)}/{len(it)}")
    print(f"  KV usage: {describe(kv)} %")

    if len({c for c in ctx}) < 3 or len(it) < 30:
        print("  CANNOT FIT: need >=3 distinct context-token counts and n>=30.")
        return

    # d ~ b0 + b1*ctx + b2*gen  (normal equations, 3x3 solve)
    n = len(it)
    pts = [
        (1.0, float(c), float(g), float(d))
        for c, g, d in zip(ctx, gen, durs, strict=True)
    ]
    a = [[sum(p[i] * p[j] for p in pts) for j in range(3)] for i in range(3)]
    b = [sum(p[i] * p[3] for p in pts) for i in range(3)]

    # Gaussian elimination with partial pivoting.
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            print("  CANNOT FIT: singular design (context/gen collinear).")
            return
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, 3):
            f = a[r][col] / a[col][col]
            for c2 in range(col, 3):
                a[r][c2] -= f * a[col][c2]
            b[r] -= f * b[col]
    coef = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        acc = b[r] - sum(a[r][c] * coef[c] for c in range(r + 1, 3))
        coef[r] = acc / a[r][r]

    ybar = sum(durs) / n
    ss_tot = sum((d - ybar) ** 2 for d in durs)
    ss_res = sum(
        (d - (coef[0] + coef[1] * c + coef[2] * g)) ** 2
        for c, g, d in zip(ctx, gen, durs, strict=True)
    )
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    print()
    print("  least-squares: elapsed_ms ~ b0 + b1*context_tokens + b2*gen_reqs")
    print(f"    b0 (fixed per-iteration cost) = {coef[0]:9.2f} ms")
    print(f"    b1 (per context token)        = {coef[1]:9.4f} ms/token")
    print(f"    b2 (per generation request)   = {coef[2]:9.2f} ms/req")
    print(f"    R^2 = {r2:.3f}   n = {n}")
    print()
    print("  NOTE: b0 is the fixed cost of carrying ANY prefill on a step; it is")
    print("    the closest measurable analogue of c0, but it is not identical --")
    print("    c0 is the marginal cost of admitting one more request on an")
    print("    existing step. Use b0 as the order-of-magnitude bound and say so.")

    busy = sum(d for d, c in zip(durs, ctx, strict=True) if c > 0)
    print()
    print(
        f"  prefill-carrying iteration time: {busy:.1f} ms total, "
        f"{busy / sum(durs) * 100:.1f}% of all iteration time"
    )
    print(
        f"  median KV usage during prefill iters: "
        f"{pct([k for k, c in zip(kv, ctx, strict=True) if c > 0] or kv, 0.5):.1f}%"
    )
    if pct([k for k, c in zip(kv, ctx, strict=True) if c > 0] or kv, 0.5) > 85.0:
        print("  VERDICT: prefill iterations run near the KV watermark -> the")
        print("    engine is saturated while admitting prefill; holding there")
        print("    converts directly into queueing delay.")
    else:
        print("  VERDICT: KV headroom exists during prefill iterations; the")
        print("    engine is not KV-saturated, so hold cost is TTFT-only.")


def q5_steps(step, iterations=None):
    """E4/E5: price c0 and check saturation."""
    section("E4/E5  Step pricing (c0) and saturation")

    if iterations:
        _q5_from_iterations(iterations)
        return {"source": "vllm_iterations", "n": len(iterations)}

    if not step:
        print("  no STEP_DIAG records. Steps are only recorded when the")
        print("  gate path runs with diagnostics on.")
        return None

    def _num(v, default=-1.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    good = [s for s in step if _num(s.get("dur_ms")) >= 0]
    durs = [_num(s["dur_ms"]) for s in good]
    adm_counts = [int(_num(s.get("admitted"), 0)) for s in good]
    print(f"  step duration: {describe(durs)} ms")
    print(f"  admitted/step: {describe([float(a) for a in adm_counts])}")

    # c0 needs the token volume as a covariate. STEP_DIAG records it as -1
    # until the gate has scheduler-side visibility, and without it the
    # admitted groups differ in prefill work as well as in admission count --
    # so the difference between groups is not c0. Say so instead of printing
    # a number that will be quoted.
    have_prefill = any(_num(s.get("prefill_tok"), -1) >= 0 for s in good)
    if not have_prefill:
        print("  CANNOT ESTIMATE c0:")
        print("    prefill_tok is -1 on every step (worker-side signal only).")
        print("    Group means below confound admission count with token volume;")
        print("    do not use the difference as c0. Needs scheduler-side")
        print("    prefill/decode tokens (C2), or vLLM iteration_details.")

    # Group steps by how many requests they admitted: the delta between the
    # groups at similar token volume is c0.
    by_adm = defaultdict(list)
    for s in good:
        by_adm[int(_num(s.get("admitted"), 0))].append(_num(s["dur_ms"]))
    print("  admitted -> step duration (this difference IS c0)")
    base = None
    for k in sorted(by_adm):
        v = by_adm[k]
        m = statistics.mean(v)
        if base is None:
            base = m
        print(
            f"    admitted={k:<3} n={len(v):<6} dur mean={m:9.2f} "
            f"delta_vs_first={m - base:+9.2f}"
        )
    if len(by_adm) >= 2 and base is not None:
        keys = sorted(by_adm)
        c0 = statistics.mean(by_adm[keys[-1]]) - base
        if not have_prefill:
            print(f"\n  confounded admitted-group delta = {c0:.1f} ms (NOT c0)")
        else:
            print(f"\n  rough c0 estimate = {c0:.1f} ms per admission")
        print("    (confounded by token volume; fit dur ~ 1 + prefill_tokens")
        print("     + decode_reqs + admitted on STEP_DIAG records for a clean")
        print("     coefficient and CI)")
    else:
        print("\n  not enough variation in admitted/step to estimate c0.")

    sat = [s for s in good if _num(s.get("waiting")) >= 0]
    if sat:
        w = [_num(s["waiting"]) for s in sat]
        print(f"\n  saturation: num_waiting {describe([float(x) for x in w])}")
        if statistics.mean(w) > 0:
            print("    VERDICT: the queue is non-empty on average -> the engine")
            print("    is SATURATED. Steps are already full, so merging tails")
            print("    saves no budget; the hold converts 1:1 into TTFT.")
            print("    -> FIX: never hold while saturated (needs scheduler-side")
            print("       visibility, which the worker does not have).")
        else:
            print("    queue is empty on average -> headroom exists.")
    else:
        print("\n  num_waiting not recorded (scheduler-side signal).")
        print("  Until the gate moves scheduler-side, saturation is invisible")
        print("  to it - this is the core design gap.")
    return {"n_steps": len(good)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+", help="log files or globs")
    ap.add_argument("--csv", help="also write parsed ADM_DIAG records to CSV")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])
    files = sorted(set(files))
    if not files:
        print("no input files matched", file=sys.stderr)
        return 2
    print(f"reading {len(files)} file(s)")
    adm, step, it = load(files)
    print(
        f"parsed {len(adm)} ADM_DIAG records, {len(step)} STEP_DIAG records, "
        f"{len(it)} vLLM iterations"
    )
    if not adm and not step:
        print("\nNo records found. Check that diagnostics were enabled:")
        print('  extra_config {"lmcache.mp.admit_diagnostics": true}')
        print("  or env LMCACHE_MP_ADM_DIAG=1")
        print("and that the log level shows INFO.")
        return 1

    q1_hit_vs_tail(adm)
    q2_serialization(adm)
    q3_rank_spread(adm)
    q4_reasons(adm)
    q5_steps(step, it)

    if args.csv and adm:
        # Standard
        import csv

        keys = sorted({k for r in adm for k in r if not k.startswith("_")})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in adm:
                w.writerow(r)
        print(f"\nwrote {len(adm)} rows to {args.csv}")

    section("Summary")
    print("  Compare every VERDICT above against the PR's value assumption")
    print("  ('each prefill step carries ~600 ms of fixed overhead'). If E1")
    print("  shows d tracks hit, E2 shows serialization, and E3 shows the AND")
    print("  penalty, then the gate is estimating on the wrong variable, on")
    print("  queued durations, and paying max-over-ranks. No K/alpha/T setting")
    print("  fixes those; they are structural.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

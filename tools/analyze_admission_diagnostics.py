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
    adm, step = [], []
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
    return adm, step


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
    return s[idx]


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
    if total and no_sib / total > 0.4:
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


def q5_steps(step):
    """E4/E5: price c0 and check saturation."""
    section("E4/E5  Step pricing (c0) and saturation")
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
    adm, step = load(files)
    print(f"parsed {len(adm)} ADM_DIAG records, {len(step)} STEP_DIAG records")
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
    q5_steps(step)

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

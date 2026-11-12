#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster for make.fun bounty
roblox-up-and-coming-forecast-01.

Reads hourly-ish snapshots of Roblox's Up-and-Coming chart from /input
(history.ndjson + task.json) and writes /output/forecast.json:
  probabilities: [{universe_id, place (1-10), p}]
  top10: [10 universe ids in order]

Model: Monte Carlo over a latent score. Each game has a latent score
S = -log(rank). Rank is the order of S, so simulating S forward and
re-ranking each hour gives a proper chart. Per-game drift (momentum,
recent-weighted, decaying with horizon) + volatility (shrunk to the
global value) drive S. The forecast is the average over the 72 hourly
slots of the window, so probabilities are the mean over hours and
paths of "game X is at place k". A share of each place is left
unnamed for games that climb in from below the named set and for
games born after the cut (the chart turns over fast: only 1 of the
first snapshot's top 10 is still top 10 eight days later).

Python 3.13, standard library only. Deterministic (fixed seed).
"""
import json
import math
import os
import random
from collections import defaultdict

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunables (chosen from the rehearsal data; see README) ---
N_PATHS = 4000
SEED = 20261112
HOURS = 72                 # scored window length
DRIFT_TAU_H = 36.0         # momentum decay time constant (hours)
DRIFT_HALF_LIFE_SNAP = 3.0  # how fast old rank moves stop mattering (snapshots)
VOL_SHRINK = 6.0           # pseudo-count shrinking per-game vol to global
VOL_FLOOR = 0.35           # min per-hour score sd
VOL_CAP = 1.20             # max per-hour score sd
TEMP = 1.10                # temperature softening of place probabilities
UNNAMED_BASE = 0.05
UNNAMED_PER_PLACE = 0.005
MAX_ROWS_PER_PLACE = 80    # cap on named games per place
MIN_P = 0.001              # drop rows below this (keeps file small, mass -> unnamed)


def load_history(input_dir):
    task = json.load(open(os.path.join(input_dir, "task.json")))
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("ok", True):
                continue
            rows = d.get("up_and_coming")
            if not rows:
                continue
            snaps.append(d)
    return task, snaps


def parse_time(s):
    # ISO 8601 UTC -> epoch seconds
    from datetime import datetime, timezone
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def score_of(rank):
    return -math.log(max(rank, 1))


def main():
    task, snaps = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    window = task.get("window", {})
    hours = int(window.get("slots", HOURS))
    if hours <= 0:
        hours = HOURS

    # --- build per-game trajectories: (time_hours, score) ---
    t0 = parse_time(snaps[0]["at"])
    traj = defaultdict(list)  # uid -> [(t_h, score, rank)]
    for s in snaps:
        t_h = (parse_time(s["at"]) - t0) / 3600.0
        for r in s["up_and_coming"]:
            uid = r["universe_id"]
            traj[uid].append((t_h, score_of(r["rank"]), r["rank"]))
    for uid in traj:
        traj[uid].sort()

    last = snaps[-1]
    last_rank = {}
    last_score = {}
    for r in last["up_and_coming"]:
        last_rank[r["universe_id"]] = r["rank"]
        last_score[r["universe_id"]] = score_of(r["rank"])
    last_time_h = (parse_time(last["at"]) - t0) / 3600.0

    # --- per-game drift (score/hour) and volatility, from trajectory ---
    # drift: recency-weighted mean of per-hour score changes
    # vol: sd of per-hour score changes
    drift = {}
    vol = {}
    all_changes = []
    for uid, tr in traj.items():
        if len(tr) < 2:
            continue
        # per-step changes with time gaps
        ch = []  # (weight, dscore_per_hour)
        for j in range(1, len(tr)):
            dt = tr[j][0] - tr[j - 1][0]
            if dt <= 0:
                continue
            d = (tr[j][1] - tr[j - 1][1]) / dt
            # weight: recent steps matter more; age measured in snapshots back from end
            age_snaps = len(tr) - 1 - j
            w = 0.5 ** (age_snaps / DRIFT_HALF_LIFE_SNAP)
            ch.append((w, d))
            all_changes.append(d)
        if not ch:
            continue
        wsum = sum(w for w, _ in ch)
        mu = sum(w * d for w, d in ch) / wsum if wsum > 0 else 0.0
        # volatility: weighted sd around mu
        var = sum(w * (d - mu) ** 2 for w, d in ch) / wsum if wsum > 0 else 0.0
        drift[uid] = mu
        vol[uid] = math.sqrt(var) if var > 0 else 0.0

    # global volatility (per-hour score sd)
    if all_changes:
        m = sum(all_changes) / len(all_changes)
        gvar = sum((c - m) ** 2 for c in all_changes) / len(all_changes)
        gvol = math.sqrt(gvar)
    else:
        gvol = 0.6
    gvol = min(max(gvol, VOL_FLOOR), VOL_CAP)

    # shrink per-game vol toward global; games with little data get ~global
    for uid in list(drift.keys()):
        n = len(traj[uid]) - 1
        v = vol.get(uid, gvol)
        vol[uid] = (n * v + VOL_SHRINK * gvol) / (n + VOL_SHRINK)
        vol[uid] = min(max(vol[uid], VOL_FLOOR), VOL_CAP)

    # --- candidate games: everyone in the last snapshot, plus recent
    # returners (in history but not last snapshot) with a penalty ---
    # last snapshot games are the core; recent returners can re-enter.
    cand = {}
    for uid in last_score:
        cand[uid] = last_score[uid]
    # returners: last seen within the last 25% of the history
    hist_span = last_time_h
    for uid, tr in traj.items():
        if uid in cand:
            continue
        last_seen = tr[-1][0]
        if last_time_h - last_seen <= 0.25 * hist_span + 24:
            # re-enter near the bottom of the chart with high uncertainty
            cand[uid] = score_of(95)
            # give them a neutral-to-slightly-positive drift (they may climb back)
            drift.setdefault(uid, 0.0)
            vol.setdefault(uid, gvol)
    for uid in cand:
        drift.setdefault(uid, 0.0)
        vol.setdefault(uid, gvol)

    uids = sorted(cand.keys())
    s0 = [cand[u] for u in uids]
    mu = [drift[u] for u in uids]
    sg = [vol[u] for u in uids]

    rng = random.Random(SEED)
    n = len(uids)

    # --- Monte Carlo: simulate latent scores over the window ---
    # counts[place][uid] = number of (path, hour) where uid is at that place
    counts = [[0] * n for _ in range(places)]
    total_slots = 0
    for _ in range(N_PATHS):
        s = list(s0)
        for h in range(1, hours + 1):
            decay = math.exp(-h / DRIFT_TAU_H)
            for i in range(n):
                s[i] += mu[i] * decay + sg[i] * rng.gauss(0.0, 1.0)
            # rank by score desc; top `places` are the chart's top 10
            order = sorted(range(n), key=lambda i: -s[i])
            for k in range(places):
                counts[k][order[k]] += 1
            total_slots += 1

    # --- probabilities: mean over hours and paths ---
    probs = [[0.0] * n for _ in range(places)]
    for k in range(places):
        for i in range(n):
            probs[k][i] = counts[k][i] / total_slots if total_slots else 0.0

    # temperature softening (calibration): p^(1/T) renormalised per place
    for k in range(places):
        row = probs[k]
        tot = sum(row)
        if tot <= 0:
            continue
        soft = [p ** (1.0 / TEMP) if p > 0 else 0.0 for p in row]
        stot = sum(soft)
        if stot > 0:
            probs[k] = [v / stot * tot for v in soft]

    # --- emit: top MAX_ROWS_PER_PLACE games per place, leave unnamed mass ---
    probabilities = []
    top10 = []
    # top10: by highest P(place 1) then expected rank; use sum of P(top3) as strength
    strength = []
    for i in range(n):
        strength.append(sum(probs[k][i] for k in range(min(3, places))))
    top10_idx = sorted(range(n), key=lambda i: -strength[i])[:places]
    top10 = [uids[i] for i in top10_idx]

    for k in range(places):
        unnamed = min(UNNAMED_BASE + UNNAMED_PER_PLACE * k, 0.55)
        # named mass available
        named_mass = 1.0 - unnamed
        idx = sorted(range(n), key=lambda i: -probs[k][i])
        # take rows until we have MAX_ROWS_PER_PLACE or p < MIN_P
        chosen = [i for i in idx if probs[k][i] >= MIN_P][:MAX_ROWS_PER_PLACE]
        raw = sum(probs[k][i] for i in chosen)
        if raw <= 0:
            continue
        # rescale the CHOSEN rows so they sum to exactly named_mass
        for i in chosen:
            p = probs[k][i] / raw * named_mass
            if p < MIN_P:
                continue
            probabilities.append({"universe_id": uids[i], "place": k + 1, "p": round(p, 6)})

    out = {"probabilities": probabilities, "top10": top10}
    # --- output validation (fail loudly rather than emit a bad forecast) ---
    assert len(top10) == places and len(set(top10)) == places, \
        "top10 must hold %d distinct ids, got %d" % (places, len(top10))
    for row in probabilities:
        assert 0.0 < row["p"] <= 1.0, "bad probability %r" % row
    for k in range(places):
        s = sum(r["p"] for r in probabilities if r["place"] == k + 1)
        assert 0.0 < s <= 1.0 + 1e-6, "place %d sums to %f" % (k + 1, s)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out) + "\n")
    print("wrote %s: %d prob rows, top10=%s" % (
        os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")),
        len(probabilities), top10))


if __name__ == "__main__":
    main()

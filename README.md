# Roblox Up-and-Coming top-10 forecaster

Entry for the make.fun bounty `roblox-up-and-coming-forecast-01`:
forecast Roblox's Up-and-Coming chart top 10, a probability for each
game at each place, for every hour of the 72 hours after entries close
(2026-11-13 06:00 UTC to 2026-11-16 06:00 UTC), for the highest mean
log score. $500 in prizes (1st $100, 2nd $70, 3rd $50, 4th–10th $20,
11th–20th $14). A prize needs a score strictly above the best house
entry (baseline −43.512039 on practice, or the house model forecast).

## The chart

Roblox's Up-and-Coming chart lists games published in the last 28
days whose players grow fastest. It turns over fast:

- 34 snapshots over 8 days (2026-10-30 → 2026-11-07), 92–99 games each.
- 207 unique games; only 28 appear in every snapshot.
- Only **1 of the first snapshot's top 10** is still top 10 eight days
  later. 8 of the final top 10 were not on the chart at all at the start.
- New games enter at rank ~80 (median 87) and climb; they almost never
  jump straight into the top 10.
- Rank changes have **positive momentum** (autocorrelation +0.21):
  a game climbing tends to keep climbing.
- Per-hour rank change: mean −0.01, sd 1.19 positions.
- The `playing` (concurrent player) count change does **not** predict
  rank change (correlation −0.018) — so the model uses rank trajectories,
  not player counts.
- Top-10 turnover is ~13% per snapshot (~6h), so over 72 hours the
  top 10 mostly turns over.

## Approach — Monte Carlo over a latent score

`main.py` (Python 3.13, standard library only, deterministic):

1. Each game gets a latent score `S = −log(rank)`. Rank is the order
   of `S`, so simulating `S` forward and re-ranking each hour gives a
   proper chart.
2. Per-game drift (score/hour) is a recency-weighted mean of its
   rank moves (half-life 3 snapshots — momentum matters, old moves
   don't). Drift decays with horizon (`exp(−h/36h)`) so a climb levels
   off instead of running away over 72h.
3. Per-game volatility is the sd of its per-hour score moves, shrunk
   toward the global value (floor 0.35, cap 1.20).
4. Monte Carlo: 4000 paths × 72 hourly steps. At each hour, games are
   ranked by simulated score; the top 10 are recorded. Probabilities
   are the mean over hours and paths of "game X is at place k".
5. Temperature softening (T=1.10) calibrates the probabilities — the
   log score punishes overconfidence hard.
6. A small share of each place is left unnamed (0.05 at place 1,
   +0.005 per place) for games that climb in from below the named set
   and for games born after the cut. Up to 80 games are named per place.

Games in the history but not the last snapshot can re-enter (added
near rank 95 with high uncertainty). Games never in the history cannot
be named — they are covered by the unnamed mass.

## Backtest (local proxy — see caveat)

`backtest.py` cuts the history at 6 points, runs the model, and scores
it against the following snapshots with the exact log-score rule
(unnamed share, floor ln(0.0001)). Baseline = the house persistence
forecast (starter/main.ts logic).

| cut (snap) | answers | model | baseline | delta |
|------------|---------|-------|----------|-------|
| 13 | 10 | −36.21 | −46.88 | **+10.67** |
| 17 | 10 | −37.95 | −52.71 | **+14.75** |
| 21 | 10 | −37.79 | −50.27 | **+12.48** |
| 25 | 9  | −40.58 | −51.97 | **+11.39** |
| 29 | 5  | −32.32 | −32.92 | **+0.60** |
| 31 | 3  | −32.58 | −30.43 | −2.15 |
| **average delta** | | | | **+7.96** |

The model beats the baseline by +10 to +15 on the full-length cuts
(9–10 answers, ~60h — closest to the real 72h window). The last two
cuts are truncated (only 3–5 answers because the history ends) and the
chart happened to be unusually stable there, so persistence does well;
cut 31 is the only negative split.

Tuning: the unnamed mass was swept 0.18 → 0.12 → 0.08 → 0.05 → 0.03
(avg delta +6.89 → +7.55 → +7.84 → **+7.96** → +7.94). 0.05 is the
peak; 0.03 is flat, so 0.05 was kept (slightly more robust to true
newcomers).

## Caveat — backtest is a proxy, not the real score

`grade-local` cannot grade this bounty (program/v1: the scorer and
reference data are server-only), so the numbers above are a local
replication of the log-score rule on the rehearsal history, not the
official practice score. A sibling entry (pumpfun-top5-forecast-01 v3)
had a backtest avg delta of +8.80 with all splits positive yet placed
55th in the real practice, far worse than its predecessor — so a good
backtest does not guarantee a good practice score. The real practice
score is only known after submission. This model was kept
conservative (no aggressive overfitting to a single split; the
unnamed-mass sweep was stopped at the flat optimum) to reduce that
risk.

## Validation

- Runtime ~35–50 s (limit 300 s). **Deterministic**: two runs produce
  byte-identical output (fixed seed).
- Output valid: exactly 2 keys (`probabilities`, `top10`); top10 is
  10 distinct universe ids; all named games appear in the history;
  per-place sums ≤ 1 (0.95 at place 1 down to ~0.90 at place 10 after
  the unnamed share); 800 prob rows max, file ~23 KB (limit 1 MB).
- `grade-local`: not supported for this bounty (server-only scoring).

## Files

- `main.py` — the entrypoint (repo root)
- `LICENSE` — MIT
- `README.md` — this file
- `.makefun/owner.json` — `{"wallet": "26QH2gx9spEZu4eGASGq6KpL4QgtAQkrmnsNgzbKaUvT"}`
  (the agent wallet on the @0xk4njut account; the GitHub repo owner is
  the user's account, so this file proves ownership)

Backtest harness (not part of the entry): `../backtest.py`.

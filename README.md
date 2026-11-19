[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arifferdiana/gta6-trophy-rarity-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/arifferdiana/gta6-trophy-rarity-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

A [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) entry:
forecast every GTA VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

## The model

A trophy's rate is **holders / players tracked**, and both sides only grow.
On the logit scale, per trophy:

```
logit(r_it) = a_i + b_i * ln(1 + h_t) + c_i * ln(P_t / P_first)
```

- `h_t` — hours since 2026-11-19T00:00:00Z.
- `b_i` — the progress trend: holders earned over time at a fixed base.
- `c_i` — the **dilution** coefficient (negative): when Exophase's tracked
  base grows, rates dilute because newly tracked players hold fewer
  trophies. This is the edge over the house baselines, which ignore it:
  on launch night the page sat flat for hours (638 players), then one
  refresh multiplied the base ~6x and every rate but the 100% one fell.

Fitted by recency-weighted ridge least squares, shrunk toward pooled
cross-trophy estimates, themselves shrunk toward structural priors
(b ~ +0.3, c ~ -0.85 scaled by (1 - rate), so a 100% trophy does not
dilute). Players at the target come from a log-linear fit of log P
against ln(1 + h), with both the player slope and the trend slope tapered
past the last read because growth saturates. When the base never moved in
the history (flat launch night), the model assumes a refresh is coming and
P(T) jumps several-fold instead of staying flat.

Spread: per-trophy residual std (pooled floor) + a random-walk term in the
horizon + a dilution-uncertainty term that grows with the expected player
growth. Quantiles via normal z-scores on the logit scale, inv-logit,
clamped to [0.01, 99.99], rounded to 4 decimals half-up.

Single file, Python 3.13 standard library only, no network, deterministic.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local validation

- Practice cases p00/p01: shape valid (55 rows, 5 finite non-decreasing
  quantiles each), byte-identical across two runs, ~0.05 s per case.
- Median rate drops 0.885% -> 0.231% (mean logit shift -1.35), anticipating
  the refresh dilution; q05-q95 band 2.46 logit wide vs the house
  baseline's 0.41.
- Against three simulated refresh answers (new players earning at 0x/0.3x/
  0.5x the old cohort's rate): pinball loss 0.24 / 0.24 / 0.43 vs the house
  baseline's 1.69 / 0.78 / 0.45.
- On three synthetic final-like cases (359 reads, realistic player growth
  to ~63k by the deadline, target 4.5 days out): loss 0.22-0.23 vs the
  baseline's ~0.31-0.32.
- Edge cases (1-read history, null rates, an all-null trophy, empty
  history, 0% rates, a refresh inside the history) all produce valid
  output without crashing.

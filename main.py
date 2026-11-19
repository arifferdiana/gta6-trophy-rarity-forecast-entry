"""gta6-trophy-rarity-forecast-01 entry: holders / players dilution forecast.

A trophy's earn rate is holders / players_tracked, and both sides only grow.
On the logit scale, per trophy:

    logit(r_it) = a_i + b_i * ln(1 + h_t) + c_i * ln(P_t / P_first)

where h_t is hours since 2026-11-19T00:00:00Z and P_t the players tracked at
read t.  b is the progress trend (holders earned over time at a fixed base);
c is the dilution coefficient: when the tracked base grows, rates dilute
because newly tracked players hold fewer trophies, so c is negative.

Fitted by recency-weighted ridge least squares, shrunk toward pooled
cross-trophy estimates, themselves shrunk toward structural priors
(b ~ +0.3, c ~ -0.85 scaled by (1 - rate), so a 100% trophy does not
dilute).  Players at the target come from a log-linear fit of log P
against ln(1 + h), capped; when the base never moved in the history
(launch night: the page sat flat for hours) we assume a refresh is coming
and P(T) jumps several-fold instead of staying flat.

Spread: per-trophy residual std (pooled floor) plus a random-walk term in
the horizon plus a dilution-uncertainty term that grows with the expected
player growth.  Quantiles via normal z-scores on the logit scale, then
inv-logit, clamped to [0.01, 99.99], rounded to 4 decimals half-up.

Deterministic, standard library only, no network.  Reads every *.in in
$MAKEFUN_INPUT_DIR (else /input/), writes one /output/<id>.out per case
to $MAKEFUN_OUTPUT_DIR (else /output/).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Structural priors (used only where the history cannot identify them).
B_PRIOR = 0.3        # progress trend on the logit scale per ln(1+h)
C_PRIOR = -0.85      # dilution: new players hold far fewer trophies
RIDGE = 2.0          # ridge strength toward the pooled prior (weight units)
SIGMA_FLOOR = 0.30   # logit floor on a trophy's residual spread
SIGMA_RW = 0.35      # random-walk spread per sqrt(day) of horizon
DILUTION_UNC = 0.5   # spread per unit of expected log player growth
MAX_P_GROWTH = 20.0  # cap on P(T) / P_last from the players fit
P_TAPER = 0.5        # taper on the players slope past the last read
TREND_TAPER = 0.75   # taper on the trend slope past the last read


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0:
        p = 1.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        p = e / (1.0 + e)
    return clamp_pct(100.0 * p)


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript Math.round does it).

    Whole numbers come back as ints, so json writes 100, not 100.0.
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def inv_normal(tau: float) -> float:
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if tau < p_low:
        q = math.sqrt(-2.0 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if tau > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(text: str) -> float:
    dt = parse_time(text)
    return max(-0.99, (dt - LAUNCH).total_seconds() / 3600.0)


def solve3(mat, vec):
    """Solve a 3x3 linear system by Gaussian elimination with pivoting."""
    n = 3
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            aug[piv][col] = 1e-12
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            f = aug[r][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[r][k] -= f * aug[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))
        out[i] = s / aug[i][i]
    return out


def ridge3(points, prior, lam):
    """Weighted ridge least squares for y ~ 1 + x1 + x2.

    points: list of (x1, x2, y, w).  prior: (a0, b0, c0).  lam: ridge
    strength on the slopes (and a small one on the intercept).
    """
    lam_a, lam_b, lam_c = 0.05, lam, lam
    ata = [[0.0] * 3 for _ in range(3)]
    aty = [0.0] * 3
    for x1, x2, y, w in points:
        row = (1.0, x1, x2)
        for i in range(3):
            aty[i] += w * row[i] * y
            for j in range(3):
                ata[i][j] += w * row[i] * row[j]
    for i, l in enumerate((lam_a, lam_b, lam_c)):
        ata[i][i] += l
        aty[i] += l * prior[i]
    return solve3(ata, aty)


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zscores = [inv_normal(t) for t in levels]
    horizon_days = max(0.0, case.get("horizon_minutes", 0) / 1440.0)
    horizon_hours = max(0.0, case.get("horizon_minutes", 0) / 60.0)

    if not history:
        # No reads at all: a wide, low prior rather than a crash.
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit_pct(inv_normal(lv) * 2.0 + logit_pct(1.0)))
                                  for lv in levels]}
            for t in trophies]}

    # ---- reads: time, players, log players ratio -------------------------
    times = [hours_since_launch(r["at"]) for r in history]
    last_at = times[-1]
    players = []
    for r in history:
        p = r.get("players_tracked") or 0
        players.append(float(p) if p and p > 0 else None)
    first_p = next((p for p in players if p is not None), None)
    last_p = next((p for p in reversed(players) if p is not None), None)
    if first_p is None:
        first_p = last_p = 1.0
    if last_p is None:
        last_p = first_p
    x1 = [math.log1p(max(-0.99, h)) for h in times]
    x2 = [math.log(p / first_p) if p is not None else 0.0 for p in players]
    # Recency weights: 12-hour half-life from the last read.
    weights = [0.5 ** max(0.0, (last_at - t) / 12.0) for t in times]

    # ---- per-trophy data --------------------------------------------------
    n_tr = len(trophies)
    per_tr = []
    for i in range(n_tr):
        pts = []
        for j, r in enumerate(history):
            rate = r["rates"][i] if i < len(r.get("rates") or []) else None
            if rate is None:
                continue
            pts.append((x1[j], x2[j], logit_pct(rate), weights[j]))
        per_tr.append(pts)

    # ---- pooled (within-trophy demeaned) fit for the priors --------------
    pooled_pts = []
    for pts in per_tr:
        if len(pts) < 2:
            continue
        sw = sum(p[3] for p in pts)
        if sw <= 0:
            continue
        my = sum(p[3] * p[2] for p in pts) / sw
        mx1 = sum(p[3] * p[0] for p in pts) / sw
        mx2 = sum(p[3] * p[1] for p in pts) / sw
        for x1v, x2v, yv, wv in pts:
            pooled_pts.append((x1v - mx1, x2v - mx2, yv - my, wv))
    if pooled_pts:
        # Ridge on the demeaned design; intercept is 0 by construction.
        ata = [[0.0] * 2 for _ in range(2)]
        aty = [0.0] * 2
        for x1v, x2v, yv, wv in pooled_pts:
            for a, xv in enumerate((x1v, x2v)):
                aty[a] += wv * xv * yv
                for b, y2 in enumerate((x1v, x2v)):
                    ata[a][b] += wv * xv * y2
        ata[0][0] += RIDGE
        ata[1][1] += RIDGE
        aty[0] += RIDGE * B_PRIOR
        aty[1] += RIDGE * C_PRIOR
        det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
        if abs(det) > 1e-12:
            b_pool = (aty[0] * ata[1][1] - aty[1] * ata[0][1]) / det
            c_pool = (ata[0][0] * aty[1] - ata[1][0] * aty[0]) / det
        else:
            b_pool, c_pool = B_PRIOR, C_PRIOR
    else:
        b_pool, c_pool = B_PRIOR, C_PRIOR
    b_pool = max(-3.0, min(3.0, b_pool))
    c_pool = max(-1.5, min(0.5, c_pool))

    # ---- players at the target --------------------------------------------
    valid_p = [p for p in players if p is not None]
    if not valid_p:
        # No usable player counts at all: assume no growth.
        p_target = 1.0
        p_flat = True
    elif (max(valid_p) / min(valid_p)) < 1.05:
        # Launch night: the page sat unchanged for hours, then one refresh
        # multiplied the base ~6x. Expect a refresh inside the horizon.
        factor = 1.0 + 4.0 * min(1.0, horizon_hours / 1.5)
        p_target = last_p * factor
        p_flat = True
    else:
        # Log-linear fit of log P against ln(1 + h), recency weighted.
        # Growth saturates as the tracked base fills up, so the fitted
        # slope is tapered by half over the segment past the last read:
        # a straight line through the steep early phase would otherwise
        # overstate how much the base still grows by the target.
        p_flat = False
        pts = [(x1[j], math.log(players[j]), weights[j])
               for j in range(len(history)) if players[j] is not None]
        sw = sum(p[2] for p in pts)
        mx = sum(p[2] * p[0] for p in pts) / sw
        my = sum(p[2] * p[1] for p in pts) / sw
        sxx = sum(p[2] * (p[0] - mx) ** 2 for p in pts)
        sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts)
        slope = max(0.0, min(2.5, sxy / sxx)) if sxx > 1e-12 else 0.0
        inter = my - slope * mx
        h_target = hours_since_launch(case["target"])
        x1_target = math.log1p(max(-0.99, h_target))
        log_p_last = inter + slope * x1[-1]
        log_p_target = log_p_last + P_TAPER * slope * (x1_target - x1[-1])
        p_target = math.exp(log_p_target)
        p_target = max(last_p, min(last_p * MAX_P_GROWTH, p_target))
    log_p_growth = math.log(max(1.0, p_target / last_p)) if last_p > 0 else 0.0
    h_target = hours_since_launch(case["target"])
    x1_target = math.log1p(max(-0.99, h_target))
    x2_target = math.log(p_target / first_p)

    # ---- per-trophy fits, medians, spreads --------------------------------
    # Pooled median last rate, for trophies with no usable history.
    last_rates = []
    for i in range(n_tr):
        for r in reversed(history):
            rate = r["rates"][i] if i < len(r.get("rates") or []) else None
            if rate is not None:
                last_rates.append(rate)
                break
    pooled_last_logit = logit_pct(
        sorted(last_rates)[len(last_rates) // 2]) if last_rates else logit_pct(1.0)

    residuals = []
    fits = []
    for i, pts in enumerate(per_tr):
        if pts:
            # The trophy's own observed level anchors the intercept, even
            # from a single read; the slopes come from the pooled priors.
            sw = sum(p[3] for p in pts)
            a_prior = sum(p[3] * p[2] for p in pts) / sw
        else:
            a_prior = pooled_last_logit
        med_rate = None
        if pts:
            med_rate = sorted(p[2] for p in pts)[len(pts) // 2]
        rate_pct = 100.0 / (1.0 + math.exp(-med_rate)) if med_rate is not None else 1.0
        c_prior = c_pool * (1.0 - rate_pct / 100.0)
        if len(pts) >= 2:
            a, b, c = ridge3(pts, (a_prior, b_pool, c_prior), RIDGE)
        elif len(pts) == 1:
            a, b, c = pts[0][2], b_pool, c_prior
        else:
            a, b, c = a_prior, b_pool, c_prior
        b = max(-3.0, min(3.0, b))
        c = max(-1.5, min(0.5, c))
        fits.append((a, b, c))
        if len(pts) >= 3:
            sse = sum(p[3] * (p[2] - (a + b * p[0] + c * p[1])) ** 2 for p in pts)
            sw = sum(p[3] for p in pts)
            residuals.append(math.sqrt(max(0.0, sse / sw)))
    pooled_resid = sorted(residuals)[len(residuals) // 2] if residuals else 0.35
    pooled_resid = max(0.30, pooled_resid)

    sigma_rw = SIGMA_RW * math.sqrt(horizon_days)
    # The trend is extrapolated in ln(1 + h); growth saturates, so the
    # slope is tapered past the last read (TREND_TAPER = 1 keeps it whole).
    x1_trend_target = x1[-1] + TREND_TAPER * (x1_target - x1[-1])
    rows = []
    for i, trophy in enumerate(trophies):
        a, b, c = fits[i]
        pts = per_tr[i]
        if len(pts) >= 3:
            sse = sum(p[3] * (p[2] - (a + b * p[0] + c * p[1])) ** 2 for p in pts)
            sw = sum(p[3] for p in pts)
            sigma_fit = max(SIGMA_FLOOR, math.sqrt(max(0.0, sse / sw)))
        elif pts:
            sigma_fit = pooled_resid
        else:
            sigma_fit = pooled_resid * 1.5
        sigma_dil = DILUTION_UNC * abs(c) * log_p_growth
        sigma = math.sqrt(sigma_fit ** 2 + sigma_rw ** 2 + sigma_dil ** 2)
        median_logit = a + b * x1_trend_target + c * x2_target
        qs = []
        prev = None
        for z in zscores:
            q = round4(inv_logit_pct(median_logit + z * sigma))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

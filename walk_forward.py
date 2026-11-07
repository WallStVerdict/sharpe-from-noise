"""noise-sharpe, part 2: does the winner keep winning? A hold-out test and a walk-forward.

noise_sharpe.py showed that the best of 1,000 random strategies on pure noise scores a Sharpe near 3.
Those were coin flips, which you cannot trade forward. Here every strategy is a rule you can:

  A rule: "be in the market when the price is above (or below) its L-day average, by a band, acting d days later".
  1,000 rules = 25 lookbacks x trend or reverse x 4 bands x 5 delays.

A. Pure noise. Each rule trades its own random-walk price for 21 years (1 warm-up year + 20).
   1. Hold-out: pick the best rule on year 1, then run it unchanged on years 2-6 (data it never saw).
   2. Walk-forward: every year, re-pick the best rule on the year just ended and trade it the next year. 19 times.
real_market.py runs the same 1,000 rules on the real US stock market (not in the video).

Seeded. No data needed: every price here is simulated.
"""
import json
import sys
from pathlib import Path

import numpy as np

SEED = 2026
WARM = 250                 # warm-up days, so the 250-day average exists from day 1 of year 1
YEARS = 20
D = 252
LOOKBACKS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 40, 50, 60, 75, 90, 100, 120, 150, 180, 200, 225, 250]
MODES = ["trend", "reverse"]
BANDS = [0.0, 0.005, 0.01, 0.02]
DELAYS = [1, 2, 3, 4, 5]
HOLDOUT_YEARS = 5
OUT = Path("results")
BG, FG, MUTED, YELLOW, TEAL, CORAL, GRID = "#080d18", "#f4f6fa", "#a0acc0", "#ffcc33", "#4fd2c2", "#ff8a65", "#2a3a55"


def grid():
    return [(L, m, b, d) for L in LOOKBACKS for m in MODES for b in BANDS for d in DELAYS]


def describe(rule):
    L, m, b, d = rule
    side = "above" if m == "trend" else "below"
    band = f" by {b:.1%}" if b else ""
    return f"in when price is {side} its {L}-day average{band}, act {d} day{'s' if d > 1 else ''} later"


def positions(price, rule):
    """0/1 position for each day of a price path (1-D) from one rule. Uses closes up to day t - delay only."""
    L, m, b, d = rule
    c = np.concatenate([[0.0], np.cumsum(price)])
    ma = np.full(len(price), np.nan)
    ma[L - 1:] = (c[L:] - c[:-L]) / L
    sig = (price > ma * (1 + b)) if m == "trend" else (price < ma * (1 - b))
    sig = np.where(np.isnan(ma), False, sig).astype(float)
    pos = np.zeros_like(sig)
    pos[d:] = sig[:-d]
    return pos


def sharpe(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(D)) if sd > 0 else 0.0


# ---------------------------------------------------------------- A. pure noise
def part_a(rng, rules):
    n = len(rules)
    T = WARM + YEARS * D
    r = rng.normal(0.0, 0.01, size=(T, n))                 # each rule gets its own random walk: zero drift
    price = np.cumprod(1 + r, axis=0)
    strat = np.empty((T, n))
    for i, rule in enumerate(rules):
        strat[:, i] = positions(price[:, i], rule) * r[:, i]
    strat = strat[WARM:]
    yearly = np.array([[sharpe(strat[y * D:(y + 1) * D, i]) for i in range(n)] for y in range(YEARS)])

    # 1. hold-out: chosen on year 1, run unchanged on years 2..6
    w = int(yearly[0].argmax())
    hold = strat[D:(1 + HOLDOUT_YEARS) * D]
    oos_all = np.array([sharpe(hold[:, i]) for i in range(n)])
    top10 = np.argsort(yearly[0])[-10:]
    holdout = {"winner": w, "winner_rule": describe(rules[w]), "winner_is": float(yearly[0, w]),
               "winner_oos": float(oos_all[w]), "winner_oos_by_year": [float(v) for v in yearly[1:1 + HOLDOUT_YEARS, w]],
               "top10_is_mean": float(yearly[0, top10].mean()), "top10_oos_mean": float(oos_all[top10].mean()),
               "all_oos_mean": float(oos_all.mean()), "corr_is_oos": float(np.corrcoef(yearly[0], oos_all)[0, 1]),
               "oos_sharpe_sd": float(oos_all.std(ddof=1)),
               "winner_curve": np.cumprod(1 + strat[:(1 + HOLDOUT_YEARS) * D, w]).tolist(),
               "top10_curve": np.cumprod(1 + strat[:(1 + HOLDOUT_YEARS) * D][:, top10].mean(axis=1)).tolist()}

    # 2. walk-forward: re-pick the best on the year just ended, trade it the next year
    steps, chain = [], []
    for y in range(1, YEARS):
        k = int(yearly[y - 1].argmax())
        steps.append({"year": y + 1, "pick": k, "is": float(yearly[y - 1, k]), "oos": float(yearly[y, k])})
        chain.append(strat[y * D:(y + 1) * D, k])
    chain = np.concatenate(chain)
    wf = {"steps": steps, "is_mean": float(np.mean([s["is"] for s in steps])),
          "oos_mean": float(np.mean([s["oos"] for s in steps])), "chain_sharpe": sharpe(chain),
          "chain_t": sharpe(chain) * np.sqrt(len(chain) / D), "years": len(steps),
          "oos_positive_years": sum(s["oos"] > 0 for s in steps)}
    return holdout, wf


# ---------------------------------------------------------------- B. a real price
# ---------------------------------------------------------------- charts
def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120, facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=18)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.spines[["top", "right"]].set_visible(False)
    return plt, fig, ax


def _save(plt, fig, ax, path, pts):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    an = {"size": [w, h]}
    for k, (x, y) in pts.items():
        X, Y = ax.transData.transform((x, y))
        an[k] = [round(float(X)), round(float(h - Y))]
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    (path.parent / (path.stem + "_anchors.json")).write_text(json.dumps(an))


def chart_holdout(ho, path):
    """The 10 best rules of year 1, held as one equal-weight basket: the year they were picked, then 5 more."""
    plt, fig, ax = _fig()
    c = np.array(ho["top10_curve"]) * 100
    x = np.arange(1, len(c) + 1) / D
    ax.axvspan(0, 1, color="#122036", lw=0)
    ax.plot(x[:D], c[:D], color=YELLOW, lw=3.4)
    ax.plot(x[D - 1:], c[D - 1:], color=CORAL, lw=3.4)
    ax.axvline(1, color=FG, ls="--", lw=1.8)
    lo, hi = min(c) * 0.97, max(c) * 1.12
    ax.set_ylim(lo, hi)
    ax.text(0.5, hi * 0.985, f"PICKED\nSharpe {ho['top10_is_mean']:.2f}", color=YELLOW, fontsize=24,
            ha="center", va="top", weight="bold")
    ax.text(1 + HOLDOUT_YEARS / 2, hi * 0.985, f"THE NEXT {HOLDOUT_YEARS} YEARS, SAME RULES\nSharpe {ho['top10_oos_mean']:.2f}",
            color=CORAL, fontsize=24, ha="center", va="top", weight="bold")
    ax.set_xlim(0, 1 + HOLDOUT_YEARS)
    ax.set_xticks(range(0, HOLDOUT_YEARS + 2), ["start", "year 1"] + [f"year {k}" for k in range(2, HOLDOUT_YEARS + 2)])
    ax.set_ylabel("value of $100", color=MUTED, fontsize=20)
    fig.suptitle("The 10 best of 1,000 rules on noise: picked on year 1, then left alone", color=FG, fontsize=28,
                 x=0.03, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(plt, fig, ax, path, {"split": (1.0, c[D - 1]), "end": (x[-1], c[-1]), "start": (0.02, c[0]),
                               "is_label": (0.5, hi * 0.95), "oos_label": (1 + HOLDOUT_YEARS / 2, hi * 0.95)})


def chart_wf_noise(wf, path):
    plt, fig, ax = _fig()
    yrs = np.array([s["year"] for s in wf["steps"]])
    a = [s["is"] for s in wf["steps"]]
    b = [s["oos"] for s in wf["steps"]]
    ax.bar(yrs - 0.2, a, 0.4, color=YELLOW, label="Sharpe the year it was picked (best of 1,000)")
    ax.bar(yrs + 0.2, b, 0.4, color=[TEAL if v > 0 else CORAL for v in b], label="Sharpe the next year, same rule")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhline(wf["is_mean"], color=YELLOW, ls="--", lw=1.6)
    ax.axhline(wf["oos_mean"], color=TEAL, ls="--", lw=1.6)
    ax.text(yrs[-1] + 0.7, wf["is_mean"], f"avg {wf['is_mean']:.2f}", color=YELLOW, fontsize=20, va="center")
    ax.text(yrs[-1] + 0.7, wf["oos_mean"], f"avg {wf['oos_mean']:.2f}", color=TEAL, fontsize=20, va="center")
    ax.set_xlim(yrs[0] - 0.8, yrs[-1] + 2.2)
    ax.set_xticks(yrs)
    ax.set_xlabel("year traded", color=MUTED, fontsize=20)
    ax.set_ylabel("annualised Sharpe", color=MUTED, fontsize=20)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=FG, fontsize=18, ncol=2)
    ax.set_ylim(min(min(b), -2.5) - 0.3, max(a) + 1.4)
    fig.suptitle(f"Walk-forward on noise: re-pick the best of 1,000 every year, {len(yrs)} times", color=FG, fontsize=28,
                 x=0.03, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(plt, fig, ax, path, {"is_avg": (yrs[-1] + 1.2, wf["is_mean"]), "oos_avg": (yrs[-1] + 1.2, wf["oos_mean"]),
                               "first_is": (yrs[0] - 0.2, a[0] / 2), "first_oos": (yrs[0] + 0.2, b[0] / 2)})


def merge_summary(flat):
    """Add the headline numbers to results/summary.json (written by noise_sharpe.py), so the video reads one file."""
    p = OUT / "summary.json"
    base = json.loads(p.read_text()) if p.exists() else {}
    base.update({k: round(v, 4) if isinstance(v, float) else v for k, v in flat.items()})
    p.write_text(json.dumps(base, indent=2) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    rules = grid()
    rng = np.random.default_rng([SEED, 2])
    print(f"walk-forward | seed {SEED} | {len(rules):,} rules: {len(LOOKBACKS)} lookbacks x 2 directions x "
          f"{len(BANDS)} bands x {len(DELAYS)} delays")
    print(f"[1/2] pure noise: each rule on its own random price, {YEARS} years ...", flush=True)
    ho, wf = part_a(rng, rules)
    print(f"      hold-out: best on year 1 Sharpe {ho['winner_is']:.2f} -> next {HOLDOUT_YEARS} years {ho['winner_oos']:.2f}")
    print(f"      top 10 on year 1: Sharpe {ho['top10_is_mean']:.2f} -> next {HOLDOUT_YEARS} years {ho['top10_oos_mean']:.2f}")
    print(f"      walk-forward: picked at {wf['is_mean']:.2f} on average -> next year {wf['oos_mean']:.2f}")
    print("[2/2] charts", flush=True)
    chart_holdout(ho, OUT / "holdout_noise.png")
    chart_wf_noise(wf, OUT / "wf_noise.png")

    L = [f"Rules (N)                         {len(rules):,}",
         "PURE NOISE (each rule on its own random price)",
         f"Hold-out: best on year 1             Sharpe {ho['winner_is']:.2f}   ({ho['winner_rule']})",
         f"          same rule, next {HOLDOUT_YEARS} years     Sharpe {ho['winner_oos']:.2f}   by year: "
         + ", ".join(f"{v:.2f}" for v in ho["winner_oos_by_year"]),
         f"          top 10 on year 1           Sharpe {ho['top10_is_mean']:.2f} -> {ho['top10_oos_mean']:.2f} next {HOLDOUT_YEARS} years",
         f"          link year 1 -> next {HOLDOUT_YEARS}     correlation {ho['corr_is_oos']:.2f} (0 = none)",
         f"Walk-forward, {wf['years']} yearly re-picks   picked at {wf['is_mean']:.2f} on average -> {wf['oos_mean']:.2f} the next year",
         f"          all {wf['years']} traded years joined   Sharpe {wf['chain_sharpe']:.2f} (t {wf['chain_t']:.1f}); "
         f"{wf['oos_positive_years']} of {wf['years']} years above zero",
         "Education, not advice."]
    (OUT / "walk_forward.txt").write_text("\n".join(L) + "\n")
    ho.pop("winner_curve")
    ho.pop("top10_curve")
    flat = {"wf_rules": len(rules), "ho_is": ho["winner_is"], "ho_oos": ho["winner_oos"], "ho_corr": ho["corr_is_oos"],
            "ho_top10_is": ho["top10_is_mean"], "ho_top10_oos": ho["top10_oos_mean"],
            "wf_is_mean": wf["is_mean"], "wf_oos_mean": wf["oos_mean"], "wf_chain_sharpe": wf["chain_sharpe"],
            "wf_years": wf["years"], "wf_pos_years": wf["oos_positive_years"]}
    (OUT / "walk_forward.json").write_text(json.dumps({**flat, "noise_holdout": ho, "noise_walk_forward": wf}, indent=1) + "\n")
    merge_summary(flat)
    print("results/  walk_forward.txt  walk_forward.json  holdout_noise.png  wf_noise.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

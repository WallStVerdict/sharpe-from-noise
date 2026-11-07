"""noise-sharpe: the best of 1,000 strategies on pure noise looks brilliant. Is it?

1. Simulate 1,000 random long/flat strategies, each on its own random-walk price (no edge anywhere).
2. Compute every annualised Sharpe and pick the best one.
3. Compare it with the expected maximum Sharpe of N trials (Bailey & Lopez de Prado, 2014)
   and compute its Deflated Sharpe Ratio. Check the formula on 2,000 fresh noise worlds.

Stdlib + numpy + matplotlib only. Seeded: same numbers on every machine.
"""
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

SEED = 2026
N = 1_000          # strategies tried
T = 252            # trading days per strategy (1 year)
HOLD = 5           # mean holding period of a position, days
WORLDS = 2_000     # fresh noise worlds for the formula check
EULER = 0.5772156649015329
OUT = Path("results")
Z = NormalDist()


def simulate(rng, n=N, t=T):
    """Daily returns of n random long/flat strategies, each on its own random-walk price."""
    price_rets = rng.normal(0.0, 0.01, size=(t, n))           # pure random walk: zero drift
    flips = rng.random((t, n)) < 1.0 / HOLD                  # flip long <-> flat at random
    pos = (np.cumsum(flips, axis=0) + rng.integers(0, 2, n)) % 2
    return pos[:-1] * price_rets[1:]                         # trade on yesterday's position


def sharpe_ratios(rets):
    sharpe = rets.mean(axis=0) / rets.std(axis=0, ddof=1) * np.sqrt(252)
    return sharpe


def expected_max_sharpe(sr_std, n):
    """E[max Sharpe] of n trials with zero true skill (Bailey & Lopez de Prado 2014, eq. 4)."""
    return sr_std * ((1 - EULER) * Z.inv_cdf(1 - 1 / n) + EULER * Z.inv_cdf(1 - 1 / (n * np.e)))


def prob_sharpe(sr, sr0, t, skew, kurt):
    """Probabilistic Sharpe Ratio of a per-period Sharpe sr against a benchmark sr0."""
    z = (sr - sr0) * np.sqrt(t - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return Z.cdf(float(z))


def chart(sr, best, emax, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, fg, muted, yellow, teal = "#080d18", "#f4f6fa", "#a0acc0", "#ffcc33", "#4fd2c2"
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120, facecolor=bg)
    ax.set_facecolor(bg)
    ax.hist(sr, bins=60, color=teal, alpha=0.85, edgecolor=bg)
    ax.axvline(emax, color=muted, ls="--", lw=2)
    ax.text(emax, ax.get_ylim()[1] * 0.92, f"expected best of {N:,}: {emax:.2f}  ",
            color=muted, fontsize=20, va="top", ha="right")
    ax.plot([best], [1.5], "o", ms=20, color=yellow, markeredgecolor=bg, markeredgewidth=2)
    ax.set_title(f"Sharpe ratios of {N:,} strategies on pure noise", color=fg, fontsize=28, pad=18, loc="left")
    ax.set_xlabel("annualised Sharpe ratio", color=muted, fontsize=20)
    ax.set_ylabel("strategies", color=muted, fontsize=20)
    ax.tick_params(colors=muted, labelsize=18)
    for s in ax.spines.values():
        s.set_color("#2a3a55")
    fig.tight_layout()
    fig.canvas.draw()
    h = fig.canvas.get_width_height()[1]

    def px(x, y):                                            # data -> PNG pixel (origin top-left)
        X, Y = ax.transData.transform((x, y))
        return [round(float(X)), round(float(h - Y))]

    anchors = {"best": px(best, 1.5), "emax": px(emax, ax.get_ylim()[1] * 0.5), "size": list(fig.canvas.get_width_height())}
    fig.savefig(path, facecolor=bg)
    (path.parent / (path.stem + "_anchors.json")).write_text(json.dumps(anchors))


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    print(f"noise-sharpe | seed {SEED} | {N:,} strategies x {T} days of pure noise")

    print(f"[1/3] simulate {N:,} random long/flat strategies ...", end=" ", flush=True)
    rets = simulate(rng)
    sr = sharpe_ratios(rets)
    print("done")

    i = int(sr.argmax())
    best = float(sr[i])
    r = rets[:, i]
    skew = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
    kurt = float(((r - r.mean()) ** 4).mean() / r.std() ** 4)
    d = np.sqrt(252)
    psr = prob_sharpe(best / d, 0.0, len(r), skew, kurt)
    print(f"[2/3] best strategy: #{i + 1}  Sharpe {best:.2f}  (annualised)")
    print(f"      p-value if it were your only try: {1 - psr:.4f}")

    emax = expected_max_sharpe(float(sr.std(ddof=1)), N)
    dsr = prob_sharpe(best / d, emax / d, len(r), skew, kurt)
    sr_std = float(sr.std(ddof=1))                          # one year of noise: about +/- 1 Sharpe
    emax_10y = expected_max_sharpe(sr_std / np.sqrt(10), N)  # same N tries on 10 years: the noise shrinks by sqrt(10)

    print(f"[3/3] check the max-Sharpe formula on {WORLDS:,} fresh noise worlds")
    maxes = []
    for w in range(1, WORLDS + 1):
        maxes.append(float(sharpe_ratios(simulate(rng)).max()))
        if w % (WORLDS // 10) == 0:
            print(f"      world {w:>5,}/{WORLDS:,}   mean best Sharpe {np.mean(maxes):.2f}", flush=True)
    sim_max = float(np.mean(maxes))

    chart(sr, best, emax, OUT / "sharpe_hist.png")
    verdict = "SIGNIFICANT" if dsr >= 0.95 else "NOT SIGNIFICANT"
    lines = [
        f"Trials (N)                    {N:,}",
        f"Best Sharpe (annualised)      {best:.2f}",
        f"Expected max Sharpe, N={N:,}  {emax:.2f}   formula",
        f"Mean max Sharpe, simulated    {sim_max:.2f}   {WORLDS:,} noise worlds",
        f"Deflated Sharpe ratio         {dsr:.2f}   needs 0.95  (chance it is real: {dsr:.0%}, needs 95%)",
        f"VERDICT: {verdict}. Best of {N:,} tries on pure noise.",
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps({
        "seed": SEED, "n": N, "t": T, "best_index": i + 1, "best_sharpe": round(best, 4),
        "p_value_single_try": round(1 - psr, 6), "expected_max_sharpe": round(emax, 4),
        "simulated_mean_max_sharpe": round(sim_max, 4), "deflated_sharpe": round(dsr, 4),
        "deflated_pct": int(round(dsr * 100)), "sharpe_std_1y": round(sr_std, 4),
        "expected_max_sharpe_10y": round(emax_10y, 4),
        "expected_max_sharpe_unit": round(expected_max_sharpe(1.0, N), 4),  # Bailey et al. (2014), Proposition 1: 1 year, unit spread
        "verdict": verdict}, indent=2) + "\n")
    print("results/")
    print("  sharpe_hist.png   all 1,000 Sharpes")
    print("  summary.txt       expected max, deflated Sharpe, verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())

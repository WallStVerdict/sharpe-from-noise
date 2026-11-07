"""noise-sharpe, extra: the same 1,000 rules on a real price. Not in the video; the walk-forward beat on it was cut for time.

The US stock market (Ken French daily, 1926-2026), 5 bps a switch, out of the market = in T-bills.
Pick on 1990-2015, test 2016-2026; and a yearly walk-forward.
Also prints the market's own long-run Sharpe, the 0.45 level quoted in the video.

The Ken French file is not ours to redistribute: the first run downloads it into data/ and checks it against data.sha256.
The library rebuilds it monthly, so a newer vintage prints a WARNING and the numbers may move a little.
"""
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from walk_forward import OUT, BG, FG, MUTED, YELLOW, TEAL, grid, describe, positions, sharpe, _fig, _save, merge_summary

COST_BPS = 5.0             # each switch in or out of the market
DATA = Path(__file__).resolve().parent / "data" / "F-F_Research_Data_Factors_daily.csv"
SUMS = Path(__file__).resolve().parent / "data.sha256"
ZIP_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def fetch():
    if DATA.exists():
        return
    print(f"      download {ZIP_URL.rsplit('/', 1)[1]}", flush=True)
    DATA.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(ZIP_URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    member = next(m for m in z.namelist() if m.lower() == DATA.name.lower())
    DATA.write_bytes(z.read(member))


def market_sharpe_monthly(dates, ex, rf):
    """The market's own Sharpe, from monthly excess returns compounded from the daily file."""
    months = dates.astype("datetime64[M]")
    x = []
    for m in np.unique(months):
        k = months == m
        x.append(np.prod(1 + ex[k] + rf[k]) - np.prod(1 + rf[k]))
    x = np.array(x)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def load_market():
    h = hashlib.sha256(DATA.read_bytes()).hexdigest()
    ok = SUMS.exists() and h in SUMS.read_text()
    rows = []
    for line in DATA.read_text(errors="replace").splitlines():
        if re.match(r"^\s*\d{8}\s*,", line):
            p = [c.strip() for c in line.split(",")]
            rows.append((p[0], float(p[1]) / 100, float(p[4]) / 100))
    dates = np.array([np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}") for d, _, _ in rows])
    ex = np.array([x for _, x, _ in rows])       # market minus T-bill
    rf = np.array([x for _, _, x in rows])
    head = DATA.read_text(errors="replace")[:200]
    m = re.search(r"(\d{6}) CRSP", head)
    return dates, ex, rf, ok, f"Ken French data library, CRSP vintage {m.group(1) if m else '?'}"


def rule_excess(price, ex, rule):
    """Daily return above T-bills of one rule on the market, after COST_BPS on every switch."""
    pos = positions(price, rule)
    cost = np.abs(np.diff(pos, prepend=0.0)) * COST_BPS / 1e4
    return pos * ex - cost


def part_b(rules):
    dates, ex, rf, ok, vintage = load_market()
    price = np.cumprod(1 + ex + rf)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    ys = list(range(int(years[0]) + 1, int(years[-1]) + 1))          # 1927.. (1926 is warm-up)
    idx = {y: np.where(years == y)[0] for y in ys}
    R = np.empty((len(ex), len(rules)), dtype=np.float32)
    for i, rule in enumerate(rules):
        R[:, i] = rule_excess(price, ex, rule)
    yearly = {y: np.array([sharpe(R[idx[y], i].astype(float)) for i in range(len(rules))]) for y in ys}

    def span(a, b):
        return np.where((years >= a) & (years <= b))[0]

    # hold-out: pick on 1990-2015, test 2016-2026
    des, tst = span(1990, 2015), span(2016, int(years[-1]))
    s_des = np.array([sharpe(R[des, i].astype(float)) for i in range(len(rules))])
    s_tst = np.array([sharpe(R[tst, i].astype(float)) for i in range(len(rules))])
    w = int(s_des.argmax())
    holdout = {"rule": describe(rules[w]), "design_sharpe": float(s_des[w]), "test_sharpe": float(s_tst[w]),
               "bh_design_sharpe": sharpe(ex[des]), "bh_test_sharpe": sharpe(ex[tst]),
               "test_rank_of_winner": int((s_tst > s_tst[w]).sum()) + 1, "rules_beating_bh_test": int((s_tst > sharpe(ex[tst])).sum())}

    # walk-forward: train on the past `tw` years, trade the next one
    wfs = {}
    for tw in (1, 5):
        chain_i, steps = [], []
        for y in ys[tw:]:
            past = np.concatenate([idx[z] for z in range(y - tw, y)])
            sc = yearly[y - 1] if tw == 1 else np.array([sharpe(R[past, i].astype(float)) for i in range(len(rules))])
            k = int(sc.argmax())
            steps.append({"year": y, "rule": describe(rules[k]), "is": float(sc[k]), "oos": float(yearly[y][k])})
            chain_i.append((idx[y], k))
        chain = np.concatenate([R[ii, k].astype(float) for ii, k in chain_i])
        days = np.concatenate([ii for ii, _ in chain_i])
        res = {"start": ys[tw], "is_mean": float(np.mean([s["is"] for s in steps])),
               "oos_mean_yearly": float(np.mean([s["oos"] for s in steps])), "sharpe": sharpe(chain),
               "bh_sharpe": sharpe(ex[days]), "steps": steps}
        for lab, a, b in (("pre1990", ys[tw], 1989), ("post1990", 1990, int(years[-1])), ("last10", 2016, int(years[-1]))):
            m = (years[days] >= a) & (years[days] <= b)
            res[lab] = {"sharpe": sharpe(chain[m]), "bh_sharpe": sharpe(ex[days][m])}
        if tw == 1:
            res["curve"] = (np.cumprod(1 + chain + rf[days]), np.cumprod(1 + ex[days] + rf[days]), dates[days])
        wfs[tw] = res
    return holdout, wfs, ok, vintage, (dates[0], dates[-1])


def chart_wf_market(wf, path):
    plt, fig, ax = _fig()
    strat, bh, dates = wf["curve"]
    t = dates.astype("datetime64[D]").astype(float) / 365.25 + 1970
    ax.plot(t, bh, color=TEAL, lw=2.4, label="US stock market, buy and hold")
    ax.plot(t, strat, color=YELLOW, lw=3.0, label="last year's best rule of 1,000, after costs")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=18)
    for s, col, key in ((bh, TEAL, "end_bh"), (strat, YELLOW, "end_wf")):
        ax.annotate(f"${s[-1]:,.0f}", (t[-1], s[-1]), xytext=(8, 0), textcoords="offset points", color=col,
                    fontsize=19, va="center", weight="bold")
    ax.set_xlim(t[0], t[-1] + 6)
    ax.set_ylabel("value of $1 (log scale)", color=MUTED, fontsize=20)
    ax.axvspan(1990, t[-1], color="#122036", lw=0, zorder=0)
    ax.text(1991, ax.get_ylim()[0] * 1.6, "since 1990", color=FG, fontsize=20)
    fig.suptitle(f"US stock market: re-pick the best of 1,000 rules every year, {int(t[0])} to {int(t[-1])}", color=FG,
                 fontsize=28, x=0.03, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    i90 = int(np.searchsorted(t, 1990))
    _save(plt, fig, ax, path, {"end_bh": (t[-1], bh[-1]), "end_wf": (t[-1], strat[-1]), "y1990": (1990, strat[i90])})


def main():
    OUT.mkdir(exist_ok=True)
    fetch()
    rules = grid()
    print(f"real market | {len(rules):,} rules | US stock market 1926-2026, {COST_BPS:g} bps a switch ...", flush=True)
    dates, ex, rf, ok, _ = load_market()
    full = market_sharpe_monthly(dates, ex, rf)
    hb, wb, ok, vintage, (d0, d1) = part_b(rules)
    if not ok:
        print("      WARNING data file does not match data.sha256: these are not the published numbers")
    chart_wf_market(wb[1], OUT / "wf_market.png")
    L = [f"REAL PRICE: US stock market, {str(d0)[:4]}-{str(d1)[:4]}, {COST_BPS:g} bps a switch, out = T-bills ({vintage})",
         f"Market buy and hold, monthly        Sharpe {full:.2f}",
         f"Picked on 1990-2015                  Sharpe {hb['design_sharpe']:.2f}   ({hb['rule']})",
         f"          test 2016-2026             Sharpe {hb['test_sharpe']:.2f}   buy and hold {hb['bh_test_sharpe']:.2f};"
         f" {hb['rules_beating_bh_test']} of 1,000 rules beat buy and hold"]
    for tw, w in wb.items():
        L.append(f"Walk-forward, train {tw}y, {w['start']}-2026   picked at {w['is_mean']:.2f} -> Sharpe {w['sharpe']:.2f}"
                 f" vs buy and hold {w['bh_sharpe']:.2f}")
        L.append(f"          before 1990 {w['pre1990']['sharpe']:.2f} vs {w['pre1990']['bh_sharpe']:.2f} | "
                 f"1990-2026 {w['post1990']['sharpe']:.2f} vs {w['post1990']['bh_sharpe']:.2f} | "
                 f"2016-2026 {w['last10']['sharpe']:.2f} vs {w['last10']['bh_sharpe']:.2f}")
    L.append("Education, not advice.")
    (OUT / "real_market.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    for w in wb.values():
        w.pop("curve", None)
    flat = {"mkt_sharpe_full": full, "mkt_design": hb["design_sharpe"], "mkt_test": hb["test_sharpe"], "mkt_bh_test": hb["bh_test_sharpe"],
            "mkt_wf_sharpe": wb[1]["sharpe"], "mkt_wf_bh": wb[1]["bh_sharpe"],
            "mkt_wf_post1990": wb[1]["post1990"]["sharpe"], "mkt_wf_bh_post1990": wb[1]["post1990"]["bh_sharpe"]}
    (OUT / "real_market.json").write_text(json.dumps({**flat, "market_holdout": hb, "market_walk_forward": wb}, indent=1) + "\n")
    merge_summary(flat)
    print("results/  real_market.txt  real_market.json  wf_market.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

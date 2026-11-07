# Sharpe from noise: is a Sharpe of 3 real?

<!-- YOUTUBE THUMBNAIL GOES HERE. After upload replace the next line with:
     [![Watch the episode](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID) -->
**Episode video: link coming after upload.**

**The claim we test.** In 2014, Bailey, Borwein, López de Prado & Zhu showed that a backtest can look great just because enough strategies were tried.
Their formula (Proposition 1) gives the expected best Sharpe ratio when every strategy has no skill.
With 1,000 tries on one year of data, it comes to about 3 (3.25 on our arithmetic; the paper's own examples use other numbers of tries).

We test that with code you can run. Then we ask whether the winners keep winning.

## Reproduce

```
make reproduce        # builds .venv on first use, then runs everything (about two minutes)
make docker           # or in Docker (python:3.12-slim), nothing else needed. DOCKER=podman works too
```

Seeded (`SEED = 2026`), with pinned package versions, so you get the same numbers.
Parts 1 and 2 use simulated prices only, so they need no data.

The extra (`real_market.py`) uses the Ken French daily file.
That file is not ours to redistribute, so the first run downloads it into `data/` from the library.
`data.sha256` is the fingerprint of the vintage behind the numbers below (CRSP 202609).
The library is rebuilt every month. If a newer vintage is live you get a WARNING, and the extra's numbers may move a little.
Delete `data/` to download it again.

## How it works

1. `noise_sharpe.py`: the claim. 1,000 random long/flat strategies, each on its own random-walk price, so there is no edge anywhere.
   Pick the best Sharpe. Compare it with the expected best of 1,000 tries, compute its Deflated Sharpe Ratio, and check the formula on 2,000 fresh noise worlds.
2. `walk_forward.py`: does the winner keep winning? Coin flips can't be traded forward, so here we use 1,000 rules:
   "in when the price is above / below its L-day average, by a band, d days later" (25 lookbacks x 2 directions x 4 bands x 5 delays).
   Each rule trades its own random-walk price.
   - Hold-out: pick on year 1, run the same rules on years 2-6.
   - Walk-forward: every year, re-pick the best on the year just ended and trade it the next year, 19 times.
3. `real_market.py`, an extra that is **not in the video**: the same 1,000 rules on the US stock market, 1926-2026, at 5 bps a switch, in T-bills when out.
   It also computes the market's own long-run Sharpe (0.45), the level quoted in the video.

## Results

| On pure noise | |
|---|---|
| Best Sharpe of 1,000 tries | **3.19** |
| Expected best of 1,000 (formula, with our run's measured spread) | 3.32 |
| Mean best of 1,000, 2,000 simulated worlds | 3.27 |
| Deflated Sharpe ratio | 0.45 (the usual bar is 0.95) |
| 10 best rules on year 1, Sharpe then | 2.51 |
| Same 10 rules, next 5 years | **0.07** |
| Walk-forward: picked at / next year (19 re-picks, average) | 3.15 / **0.15** |
| Verdict on the claim | **Holds up**, with one year of data |

### Extra: real prices (not in the video)

| Last year's best of 1,000 rules, re-picked yearly | Rule | Buy and hold |
|---|---|---|
| 1928-1989 | 0.90 | 0.40 |
| 1990-2026 | 0.22 | 0.52 |
| 2016-2026 | 0.06 | 0.73 |

Read the early years with care.
Before 1990 most picks were very short trend rules, and old index data trends a little from day to day, partly because many stocks traded rarely (Lo & MacKinlay, 1988).
Trading the whole market at each day's close wasn't practical then, so that edge is mostly not tradeable.

Levels quoted in the video: US market 0.45 (`real_market.py`); Berkshire Hathaway 0.79 (Frazzini, Kabiller & Pedersen, 2018); "2 very good, 3 excellent" is a rule of thumb, as quoted in a footnote of Xiao, Sun, Luo & Wang, arXiv 2412.20138.

## Limits

- The claim is about one year of data and no skill. It doesn't say every high Sharpe is luck.
- The strategies and rules are independent by construction, on simulated prices. Real strategy variations are correlated, so their effective number of tries is lower.
- No costs on the noise tests. Costs would only lower the numbers.

## Prior work

This is not our discovery.

- Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance", Notices of the AMS 61(5), 458-471.
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management 40(5), 94-107.
- Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns", Review of Financial Studies 29(1), 5-68.
- Harvey & Liu (2015), "Backtesting", Journal of Portfolio Management, Fall 2015, 13-28.
- Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal 58(4), 36-52.

Data: [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
Education, not advice. Our code is MIT; see `LICENSE`. Third-party notes are in `NOTICE`.

# Project brief for coding agents

NCI BSHDS4 Neural Networks & Prescriptive Analytics individual project. **Submission: 4 December 2026.**
Read this file before every task. The full plan lives in the "UB Air Quality Project: Revised Plan" doc;
this file is the working summary. If a request conflicts with a rule below, stop and ask.

## Research question

How can neural-network forecasting and optimisation support action against Ulaanbaatar's winter PM2.5:
by warning of unhealthy days a day ahead, by estimating what the 2019 raw-coal ban achieved once weather is
removed, and by deciding how few stations of the new monitoring network are needed?

| Hypothesis | Statement | Where |
|---|---|---|
| H1 | The selected neural network has positive MAE skill over persistence on test smog-season days | notebooks 05, 06 |
| H2 | Weather features improve the neural network's validation MAE (ablation on identical rows) | 05, 06 |
| H3 | GA station subsets beat random, highest-mean and spatial-spread subsets on the test months and match or beat greedy | 03 |
| H4 | The GA finds the exhaustive optimum (or within 2%) wherever exhaustive search is feasible | 03 |
| H5 | A fatigue-constrained warning policy catches more unhealthy days per alert than a single-threshold rule on the test winter | 08 |
| H6 | Weather-adjusted winter PM2.5 is lower after the 15 May 2019 raw-coal ban than before | 07 |

## Data and splits (all dates in configs/study_config.yaml; the config is the single source of truth)

| Part | Data | Train | Validation | Test (sealed) |
|---|---|---|---|---|
| Forecasting (H1, H2, H5) | US Embassy monitor, OpenAQ 8160, hourly → daily | 2019-07-01 to 2023-06-30 | 2023/24 winter | 2024-07-01 to 2025-03-31 |
| Coal-ban estimate (H6) | Same station, Nov 2016 to Mar 2025, + ERA5 | pre-ban winters used here only | — | — |
| Network design (H3, H4) | agaar.gov.mn via data-mn, 13 eligible UB stations, daily | Oct–Dec 2025 | Jan 2026 | Feb–Mar 2026 |

Weather: hourly ERA5 from Open-Meteo, `models=era5`, requested in UTC (grid point 48.0N, 107.0E).

## Hard rules

1. **Test data is sealed until 23 November.** No code scores, plots or prints test-split PM2.5 before then.
   Test-period *weather* may be used only for the mildness check. Functions that predict the test split may
   be written and unit-tested, but not called.
2. **Forecasting fits start at the core date (2019-07-01).** Every fit set uses `target_date >= core.start`.
   Pre-ban data is used only by the coal-ban notebook (07).
3. **Fit on all training months, score on smog-season days (Nov–Feb).** Select models on the mean MAE of the
   two rolling folds (fold 1 validates 2021/22, fold 2 validates 2023/24; 2022/23 is an AirNow outage).
4. **Never impute PM2.5.** Missing days stay missing; rows without the target or required features are dropped.
5. **No leakage.** Scalers, imputers, thresholds and climatology are fitted on fit rows only. Every feature uses
   day D or earlier; `*_tplus1` weather columns exist only for the labelled upper-bound experiment.
6. **Same rows for comparisons.** With/without-weather ablations and skill vs persistence use identical rows.
7. **Rules are fixed before modelling and based on availability, never on scores.** Record each one in the
   config with a comment.
8. **Logic lives in `src/uaq/*.py`; notebooks only call it.** Every new module gets offline tests in `tests/`.
   Do not modify existing modules unless the task says so. No new dependencies without asking.
9. **Raw data is never committed** (`data/raw/**` is git-ignored). NAMEM data has no published terms: academic
   use with attribution, no redistribution.
10. Run `pytest -q` before every commit; all tests must pass.

## Decisions already made (do not reopen without asking)

- Station 8160 eligibility: ≥ 240 valid training smog days (it has 283) and ≥ 75% coverage in val and test.
- Boundary-layer-height features dropped: ERA5 lacks BLH for Jan–Jun 2024 (`features.drop_columns`).
- Feature missingness rule: among training rows (core period, all months) where `y` and `pm_lag0` are both
  present, drop features missing on > 10%. `pm_lag0` is never dropped.
- Event threshold: lowest official line exceeded on ≤ 60% of training days, else the training 75th percentile.
- Network: stations 12601 and 12602 stay as GA candidates (flagged in the write-up); 12602's above-PM10
  cleaning rule is kept.

## Notebooks

| # | Purpose | Status |
|---|---|---|
| 01 | Forecasting data audit (8160) | done |
| 02 | Network data audit (agaar) | done |
| 03 | GA vs alternatives on validation; sealed test cell | done (test cell off) |
| 04 | ERA5 weather and feature table | done |
| 05 | Baselines, ridge, boosting, MLP; worst-10% days; weather ablation; test-winter mildness check | next |
| 06 | LSTM or GRU on 14-day sequences, same folds and scoring rows as the MLP | 10–16 Nov |
| 07 | Weather-normalised coal-ban estimate (random forest + linear cross-check, bootstrap range, no date-trend variable) | 10–16 Nov |
| 08 | Fatigue-aware warning policy: two alert levels chosen on validation, max warnings per winter and max consecutive warning days | 17–22 Nov |

Model choice is frozen by 18 November and the commit tagged `pre-test`. The single test run is on 23 November.

## Out of scope (do not add)

Kraków or Beijing comparisons, a dashboard, quantile forecasts, an inversion-strength proxy, transfer learning
from other cities, hyperparameter searches beyond the grids in the config.

## Commits

One logical change per commit; a short imperative subject and a body listing what changed and why.

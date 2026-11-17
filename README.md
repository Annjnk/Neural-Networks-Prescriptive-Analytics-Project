# Ulaanbaatar winter PM2.5: day-ahead warnings, the coal-ban effect and a leaner monitoring network

BSHDS4 Neural Networks & Prescriptive Analytics, National College of Ireland.
How can neural-network forecasting and optimisation support action against Ulaanbaatar's winter PM2.5?
The project forecasts next-day PM2.5 at the US Embassy monitor and turns it into a fatigue-aware warning policy,
estimates what the 15 May 2019 raw-coal ban achieved once weather is removed, and uses a genetic algorithm to decide
how few stations of the new official monitoring network are needed.

## Results (scored once on the sealed test data)

| | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | The selected neural network (MLP) beats persistence | **Supported** | skill +0.189; MAE −6.9 µg/m³ (95% CI −10.7 to −3.3) |
| H2 | Weather features improve the neural network | **Supported** (small) | MLP MAE −0.52 µg/m³ with weather (ridge +0.88, boosting −0.90, LSTM +8.54) |
| H3 | GA station subsets beat random, highest-mean, spatial spread and match greedy | Not supported | GA wins at 5 of 10 network sizes |
| H4 | The GA finds the exhaustive optimum (within 2%) | **Supported** | 10 of 10 network sizes (validation) |
| H5 | The fatigue-aware policy catches more unhealthy days per alert | Not supported | 0.36 vs 0.36 at a matched alert budget |
| H6 | Weather-adjusted winter PM2.5 fell after the ban | Inconclusive | placebo +31.4% (rule ±15%); pooled effect −25.2% (−28.9 to −15.0) |

Sources: `outputs/results/test_summary.json` and `ban_effect.json`. Test winter 2024/25 (forecasting, warnings);
test months Feb–Mar 2026 (network).

## Repository map

```
configs/study_config.yaml       every date, threshold, grid and rule (frozen)
src/uaq/                        all logic; notebooks only call it
notebooks/                      six notebooks, run in order 1 → 6
tests/                          offline pytest suite
outputs/results/                validation and test results, test-run log
outputs/audit_forecast|network/ audit reports and figures
docs/data_source_register.csv   sources, licences, attribution
data/raw/                       downloads (git-ignored, never committed)
dev/                            synthetic-data generator for dry runs
```

| Notebook | Contents |
|---|---|
| `1_data_preparation` | Forecasting data audit (8160), network data audit (agaar), ERA5 weather and the feature table |
| `2_forecasting_models` | Baselines, ridge, boosting, MLP and LSTM selected on two rolling validation folds; weather ablation |
| `3_warning_policy` | Two-level warning policy with fatigue rule and alert budgets, chosen on validation |
| `4_coal_ban_effect` | Weather-normalised counterfactual (random forest, linear and dummy-regression cross-checks), placebo |
| `5_network_design_ga` | GA vs exhaustive, greedy, random, highest-mean and spatial-spread station subsets |
| `6_final_test_results` | The single test run: H1–H5 verdicts, test-winter figures, network test months |

| Old numbers | 01, 02, 04 | 05, 06 | 08 | 07 | 03 | 09 |
|---|---|---|---|---|---|---|
| Now | 1_data_preparation | 2_forecasting_models | 3 | 4 | 5 | 6 |

Comments in `src/` and `configs/` use the old numbers and were left unchanged so the hashes in
`outputs/results/test_run_log.json` still match.

## Reproduce (macOS)

```bash
conda create -n neural python=3.12 && conda activate neural
pip install -r requirements.txt && pip install -e . && pytest -q
brew install awscli    # OpenAQ station 8160
aws s3 cp --no-sign-request --region us-east-1 \
  s3://openaq-data-archive/records/csv.gz/locationid=8160/ data/raw/pm25/openaq/8160/ --recursive
git clone --depth 1 https://github.com/American-University-of-Mongolia/data-mn ../data-mn   # agaar archive
mkdir -p data/raw/agaar && cp -R ../data-mn/tools/sources/agaar/raw/ data/raw/agaar/
python -m uaq.weather  # hourly ERA5 from Open-Meteo (notebook 1 also downloads it if missing)
```

Run the notebooks in order 1 → 5. Notebook 6 refuses to run again: the test run is final and logged.

## Integrity

- Chronological splits: train winters 2019/20–2022/23, validation 2023/24 (two rolling folds), test 2024/25;
  every rule was fixed in the config before modelling, on availability, never on scores.
- The test data were sealed and scored once, at tag `test-run`, from code frozen at tag `pre-test-v2`; the run checks
  that no frozen file changed and logs SHA-256 hashes (`outputs/results/test_run_log.json`).
- Disclosed process notes: (1) an early data-audit figure showed the 2024/25 smog-season mean; it was corrected and
  no modelling choice used it; (2) the warning policy was revised on validation before the test (advisories had no
  budget; tags `pre-test` → `pre-test-v2`).

## Data sources and licences

| Source | Used for | Licence / terms |
|---|---|---|
| OpenAQ location 8160 (AirNow, US Diplomatic Post) | forecasting, warnings, coal ban | US Public Domain |
| agaar.gov.mn (NAMEM) via the data-mn archive | network design | no published data terms: academic use with attribution, not redistributed (repo code MIT) |
| ERA5 via Open-Meteo | weather features | attribution: Open-Meteo (Zippenfenig, 2023); ERA5, Copernicus Climate Change Service |
| NSO Mongolia monthly tables | optional cross-check | CC BY 4.0 |

Details and attribution text: `docs/data_source_register.csv`.

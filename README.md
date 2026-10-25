# Ulaanbaatar winter PM2.5: forecasting and monitoring-network design

BSHDS4 Neural Networks & Prescriptive Analytics project, National College of Ireland.

Two parts, each on the data that suits it:

| Part | Question | Data | Train | Validation | Test (sealed) |
|---|---|---|---|---|---|
| **Predictive** | How well can a neural network forecast tomorrow's mean PM2.5? | US Embassy monitor, OpenAQ location 8160, hourly | winters 2019/20–2022/23 | 2023/24 | 2024/25 |
| **Prescriptive** | Which k of the new official stations should a reduced network keep? | agaar.gov.mn (NAMEM) via the data-mn archive, 22 UB stations, daily | Oct–Dec 2025 | Jan 2026 | Feb–Mar 2026 |

The forecasting core starts after the raw-coal ban of 15 May 2019; station 8160's pre-ban
winters (Nov 2016–Jun 2019) are used only for the optional sensitivity experiment (H5).
All dates and rules are in `configs/study_config.yaml`.

## Setup (macOS)

Uses the existing conda environment `neural` (Python 3.12, Apple silicon; PyTorch uses `mps`).

```bash
conda activate neural
pip install -r requirements.txt
pip install -e .                   # makes `uaq` importable from notebooks and scripts
pytest -q                          # 19 tests
```

In VS Code the folder is preset to the `neural` interpreter (`.vscode/settings.json`); pick
`neural` as the kernel when you open a notebook. Or run `jupyter lab` from the activated terminal.

## Data (not committed; see docs/data_source_register.csv)

```
data/raw/
├── pm25/openaq/8160/      OpenAQ archive for the US Embassy station (forecasting)
├── agaar/                 copy of data-mn tools/sources/agaar/raw/ (network design)
│   ├── agaar-stations.csv
│   └── daily/agaar-daily-YYYY-MM.csv
└── nso/                   optional NSO monthly tables (cross-check)
```

```bash
# OpenAQ, station 8160 (brew install awscli first)
aws s3 cp --no-sign-request --region us-east-1 \
  s3://openaq-data-archive/records/csv.gz/locationid=8160/ data/raw/pm25/openaq/8160/ --recursive

# agaar.gov.mn archive (clone OUTSIDE this repo, then copy)
git clone --depth 1 https://github.com/American-University-of-Mongolia/data-mn ../data-mn
mkdir -p data/raw/agaar && cp -R ../data-mn/tools/sources/agaar/raw/ data/raw/agaar/
```

Dry run without OpenAQ data: `python scripts/make_synthetic_data.py` writes a FAKE station-8160
file; delete `data/raw/pm25/_synthetic.csv` before the real run.

**Known data gap:** AirNow filled station 8160's winter 2022/23 with `-999` (missing), and winter
2020/21 is only 54% complete, so only two training winters pass the 75% rule. See the forecasting audit.

## Notebooks (run in order)

| Notebook | What it does |
|---|---|
| `01_forecast_data_audit.ipynb` | Loads 8160, hourly → daily, coverage per winter, coal-ban shift, daily cycle, event threshold, baseline scores |
| `02_network_data_audit.ipynb` | Loads and cleans agaar data, coverage per split, eligible stations, map, correlation, odd-station check, threshold |
| `03_network_selection.ipynb` | GA vs exhaustive, greedy, highest-mean, spatial spread and random at every k; convergence; λ sensitivity; sealed test cell |

Notebooks only call code in `src/uaq/`, so the same steps also run from the command line:
`python -m uaq.audit_forecast` and `python -m uaq.audit_network`.

## Code

```
configs/study_config.yaml   every date, threshold and rule
src/uaq/io.py               OpenAQ loader (time zones, duplicates, flags)
src/uaq/daily.py            hourly → daily with the 18-of-24-hours rule
src/uaq/periods.py          season years, splits, rolling folds
src/uaq/thresholds.py       event-threshold rule (training data only)
src/uaq/baselines.py        persistence, 3-day mean, climatology
src/uaq/metrics.py          MAE, RMSE, R², event precision/recall
src/uaq/audit_forecast.py   forecasting audit
src/uaq/agaar.py            agaar loader + cleaning rules
src/uaq/audit_network.py    network audit
src/uaq/network.py          fitness function, GA, greedy, exhaustive, comparators
tests/                      pytest suite
```

## Data sources and licences

See `docs/data_source_register.csv`. OpenAQ 8160 is US public domain (AirNow). The agaar data
come from NAMEM's portal via the data-mn repository (MIT); record the data-mn commit id you copied.

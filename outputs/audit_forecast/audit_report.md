# Forecasting data audit

- Raw files read: 3181
- Units seen: ['µg/m³'] (expected µg/m³)
- Hourly rows: 61,607 across 1 station(s)
- Date range: 2016-11-09 17:00:00 → 2025-03-04 12:00:00
- Duplicate station-hours merged: 0
- Unparseable timestamps dropped: 0
- Hourly values flagged implausible (>1500): 0

## Decisions

- **Usable station(s):** 8160
- **Event threshold:** train_p75 = 170.8 µg/m³. Every official threshold is exceeded too often in training (mns_4585_2025 89%, mns_4585_2016 81%, who_it1_2021 66%), so a study-defined training percentile is used. Report the official rates as context.
- **Splits:** train 2019-07-01→2023-06-30, val 2023-07-01→2024-06-30, test 2024-07-01→2025-03-31 (sealed).

## Station eligibility

|   station_id |   good_train_winters |   train_valid_smog_days |   val_cov |   test_cov_min | eligible   |
|-------------:|---------------------:|------------------------:|----------:|---------------:|:-----------|
|         8160 |                    2 |                     283 |     0.893 |          0.925 | True       |

## Threshold base rates (training smog-season days)

| threshold     |   ug_m3 |   base_rate |
|:--------------|--------:|------------:|
| mns_4585_2025 |  37.500 |       0.894 |
| mns_4585_2016 |  50.000 |       0.813 |
| who_it1_2021  |  75.000 |       0.661 |
| train_p75     | 170.800 |       0.251 |

## Baselines, smog season (validation is the bar the neural nets must clear)

| group   | model       |   n |    mae |   rmse |     r2 |   events |   base_rate |   tp |   false_alarms |   missed |   precision |   recall |    f1 |
|:--------|:------------|----:|-------:|-------:|-------:|---------:|------------:|-----:|---------------:|---------:|------------:|---------:|------:|
| train   | persistence | 263 | 48.460 | 66.033 |  0.170 |       68 |       0.259 |   39 |             29 |       29 |       0.574 |    0.574 | 0.574 |
| val     | persistence | 105 | 49.240 | 62.469 | -0.018 |       22 |       0.210 |    8 |             14 |       14 |       0.364 |    0.364 | 0.364 |
| train   | mean3d      | 274 | 50.943 | 67.009 |  0.140 |       70 |       0.255 |   33 |             19 |       37 |       0.635 |    0.471 | 0.541 |
| val     | mean3d      | 106 | 51.166 | 61.576 |  0.004 |       22 |       0.208 |    8 |              7 |       14 |       0.533 |    0.364 | 0.432 |
| train   | climatology | 283 | 47.013 | 58.492 |  0.343 |       71 |       0.251 |   18 |             16 |       53 |       0.529 |    0.254 | 0.343 |
| val     | climatology | 108 | 50.876 | 61.762 |  0.004 |       22 |       0.204 |    4 |             11 |       18 |       0.267 |    0.182 | 0.216 |
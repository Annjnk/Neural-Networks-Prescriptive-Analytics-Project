Put the original downloads here, unchanged.

- `pm25/openaq/8160/` — OpenAQ archive for the US Embassy station (read by notebooks 1 and 4).
- `agaar/` — copy of data-mn `tools/sources/agaar/raw/` (read by notebooks 1, 5 and 6).
- `weather/` — hourly ERA5 from Open-Meteo, written by `python -m uaq.weather` (read by notebooks 1, 2 and 4).
- `nso/` — optional NSO monthly tables (cross-check only; not read by the notebooks).

Never edit these files; all cleaning happens in code. Log every source in
`docs/data_source_register.csv`. Raw files are git-ignored by default; the register plus
the download commands in the README are what the lecturer gets.

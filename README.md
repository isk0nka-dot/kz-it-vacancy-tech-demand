@"
# KZ IT vacancy technology demand (hh.kz)

Study of technology demand in software-related vacancies on hh.kz (Kazakhstan), Research Methods and Tools, AITU.

## Run
python collect.py --days 30
python analyze.py
python analyze.py --validate labeled/validation_sample.csv
python analyze.py --spotcheck labeled/title_spotcheck.csv
python extra_analysis.py

The raw data (data/raw/vacancies.jsonl) is not in the repository. Collected on 2026-10-27, SHA-256 bc7fb23b6261057cc5ccf66e3206d60742c19487892d9555d5b6f9272e9f10bd.
The hh API token is read from the environment and is never stored in the code.
"@ | Set-Content -Encoding utf8 README.md
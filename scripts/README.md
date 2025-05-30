# scripts/

This directory is for any helper or deployment scripts.

- `run_all.sh` — e.g., a bash script to sequentially execute all modules:
  ```bash
  #!/usr/bin/env bash
  python src/eda.py
  python src/sentiment.py
  python src/indicators.py

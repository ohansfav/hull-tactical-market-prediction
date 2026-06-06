# Hull Tactical - Market Prediction

<div align="center">

![Kaggle](https://img.shields.io/badge/Kaggle-Hull%20Tactical-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Project-Starter%20Repo-0F766E?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Time%20Series%20Alpha-1D4ED8?style=for-the-badge)

</div>

Starter repository for the Kaggle competition [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## What We Know So Far

- Competition type: Featured Code Competition
- Prompt: predict market returns / predictability under volatility constraints
- Time remaining at setup: 10 days
- Kaggle currently shows an account gate: identity verification is required before submission

## Current Objective

Build a repo that is ready for fast iteration once the competition data is available locally.

## Repo Structure

```text
hull-tactical-market-prediction/
├── pipeline.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
└── src/
    └── hull_tactical_market_prediction/
        └── __init__.py
```

## Baseline Capabilities

- Dataset inspection for CSV or Parquet train/test files
- Time-aware regression baseline with `TimeSeriesSplit`
- Mixed-type preprocessing for numeric and categorical features
- Multi-model benchmark to pick a first strong baseline
- Output artifacts for metrics and test predictions

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Inspect the competition dataset after download:

```bash
python pipeline.py inspect --data-dir data/raw
```

Run a first baseline after you identify the target column:

```bash
python pipeline.py train --data-dir data/raw --target-column target --time-column date
```

## Immediate Next Steps

1. Join the competition and complete Kaggle identity verification.
2. Download the official data into `data/raw/`.
3. Inspect schema, target definition, and any sample submission format.
4. Replace the generic baseline with competition-specific objective engineering.
5. Add stronger models and feature generation aimed at leaderboard performance.

## Winning Direction

For this type of problem, the likely edge will come from:

- leak-free temporal validation
- volatility-aware objective design
- lagged and rolling market features
- regime detection features
- strong boosting models and ensembles
- disciplined offline-to-leaderboard correlation tracking
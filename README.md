# Hull Tactical - Market Prediction

<div align="center">

![Kaggle](https://img.shields.io/badge/Kaggle-Hull%20Tactical-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Project-Starter%20Repo-0F766E?style=for-the-badge)
![Focus](https://img.shields.io/badge/Focus-Time%20Series%20Alpha-1D4ED8?style=for-the-badge)

</div>

Starter repository for the Kaggle competition [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction).

## What We Know So Far

- Competition type: Featured Code Competition
- Prompt: predict excess S&P 500 returns while respecting volatility constraints
- Metric: a Sharpe-ratio variant with penalties for poor return and excessive volatility
- Submission mode: official submissions must use the provided Kaggle evaluation API, not a normal CSV upload
- Prediction target at submission time: daily allocation to the S&P 500 in the valid range `[0, 2]`
- Time remaining at setup: 10 days
- Current gate on the account shown in browser: identity verification is incomplete

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
- Output artifacts for metrics and offline test predictions

## Competition-Specific Constraint

This competition is closer to a market-timing system than a plain regression leaderboard.

- Offline research can use local train/test files and benchmark predictive models.
- Official competition submission must be wrapped in the Kaggle evaluation API.
- The final model must emit a position size or allocation, not just a raw return forecast.
- Any serious attempt to win will need leak-free temporal validation that tracks the competition metric, not just RMSE.

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
4. Replace the generic baseline with competition-specific objective engineering and allocation logic.
5. Add stronger models and feature generation aimed at leaderboard performance.

## Winning Direction

For this type of problem, the likely edge will come from:

- leak-free temporal validation
- volatility-aware objective design
- lagged and rolling market features
- regime detection features
- strong boosting models and ensembles
- disciplined offline-to-leaderboard correlation tracking
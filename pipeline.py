from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOGGER = logging.getLogger("hull_tactical")


def build_logger() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def resolve_table_file(data_dir: Path, stem: str) -> Path:
    candidates = [
        data_dir / f"{stem}.parquet",
        data_dir / f"{stem}.csv",
        data_dir / f"{stem}.feather",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.csv / {stem}.parquet / {stem}.feather in {data_dir}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def inspect_dataset(data_dir: Path) -> None:
    train_path = resolve_table_file(data_dir, "train")
    train_df = read_table(train_path)

    summary = {
        "train_path": str(train_path),
        "shape": list(train_df.shape),
        "columns": train_df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in train_df.dtypes.items()},
        "missing_fraction": train_df.isna().mean().sort_values(ascending=False).head(20).to_dict(),
    }
    print(json.dumps(summary, indent=2))


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [column for column in frame.columns if column not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=8.0),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "hist_gbm": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=350,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, splitter) -> float:
    fold_scores: list[float] = []
    for train_idx, valid_idx in splitter.split(X):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_valid)
        score = root_mean_squared_error(y_valid, preds)
        fold_scores.append(score)
    return float(np.mean(fold_scores))


def train_baseline(data_dir: Path, output_dir: Path, target_column: str, time_column: str | None) -> None:
    train_path = resolve_table_file(data_dir, "train")
    train_df = read_table(train_path)

    test_df = None
    try:
        test_path = resolve_table_file(data_dir, "test")
        test_df = read_table(test_path)
    except FileNotFoundError:
        LOGGER.info("No test file found. Training only.")

    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in train data")

    if time_column and time_column in train_df.columns:
        train_df = train_df.sort_values(time_column).reset_index(drop=True)
        if test_df is not None and time_column in test_df.columns:
            test_df = test_df.sort_values(time_column).reset_index(drop=True)
        splitter = TimeSeriesSplit(n_splits=5)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)

    y = train_df[target_column]
    X = train_df.drop(columns=[target_column])

    preprocessor = build_preprocessor(X)
    metrics: list[dict[str, float | str]] = []
    best_name = ""
    best_score = float("inf")
    best_pipe: Pipeline | None = None

    for model_name, model in candidate_models().items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        score = evaluate_model(pipe, X, y, splitter)
        metrics.append({"model": model_name, "rmse": score})
        LOGGER.info("%s RMSE: %.6f", model_name, score)

        if score < best_score:
            best_score = score
            best_name = model_name
            best_pipe = pipe

    assert best_pipe is not None
    LOGGER.info("Selected model: %s (RMSE %.6f)", best_name, best_score)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics).sort_values("rmse")
    metrics_df.to_csv(output_dir / "cv_metrics.csv", index=False)

    best_pipe.fit(X, y)

    if test_df is not None:
        preds = best_pipe.predict(test_df)
        prediction_column = target_column if target_column not in test_df.columns else f"pred_{target_column}"
        submission = pd.DataFrame({prediction_column: preds})
        submission.to_csv(output_dir / "test_predictions.csv", index=False)
        LOGGER.info("Saved test predictions to %s", output_dir / "test_predictions.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hull Tactical starter pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect available training data")
    inspect_parser.add_argument("--data-dir", default="data/raw")

    train_parser = subparsers.add_parser("train", help="Run a baseline benchmark")
    train_parser.add_argument("--data-dir", default="data/raw")
    train_parser.add_argument("--output-dir", default="outputs")
    train_parser.add_argument("--target-column", required=True)
    train_parser.add_argument("--time-column")

    return parser.parse_args()


def main() -> None:
    build_logger()
    args = parse_args()

    if args.command == "inspect":
        inspect_dataset(Path(args.data_dir))
        return

    if args.command == "train":
        train_baseline(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            target_column=args.target_column,
            time_column=args.time_column,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
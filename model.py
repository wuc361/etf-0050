from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
import joblib

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def _make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

def walk_forward_evaluate(data, test_window=40, min_train=250):
    return evaluate_horizon(data, "target", horizon=1, test_window=test_window, min_train=min_train)

def fit_latest_model(data, target_col="target"):
    df = data.dropna(subset=[target_col]).copy()
    features = [c for c in df.columns if c.startswith("f_")]
    model = _make_model().fit(df[features], df[target_col].astype(int))
    joblib.dump({"model": model, "features": features, "target": target_col}, OUT / f"latest_model_{target_col}.joblib")
    return model, features

def save_report(report, predictions):
    OUT.mkdir(exist_ok=True)
    (OUT / "backtest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(OUT / "predictions.csv", index=False)

def evaluate_horizon(data, target_col, horizon=1, test_window=40, min_train=250):
    """Purged expanding-window evaluation for one cumulative forecast horizon."""
    df = data.dropna(subset=[target_col]).copy()
    features = [c for c in df.columns if c.startswith("f_")]
    df = df.dropna(subset=features, how="all")
    if len(df) < min_train + 20:
        raise ValueError(f"{target_col} 有效資料不足：{len(df)} 筆")
    predictions = []
    start = min_train + (horizon - 1)
    while start < len(df):
        end = min(start + test_window, len(df))
        train_end = max(0, start - (horizon - 1))
        train, test = df.iloc[:train_end], df.iloc[start:end]
        if len(train) < min_train:
            break
        estimator = _make_model()
        estimator.fit(train[features], train[target_col].astype(int))
        probabilities = estimator.predict_proba(test[features])[:, 1]
        for dt, probability, actual in zip(test.index, probabilities, test[target_col].astype(int)):
            predictions.append({
                "date": str(dt.date()),
                "up_probability": float(probability),
                "prediction": "UP" if probability >= .5 else "DOWN",
                "actual": "UP" if actual == 1 else "DOWN",
                "correct": bool((probability >= .5) == (actual == 1))
            })
        start = end
    if not predictions:
        raise ValueError(f"{target_col} 沒有可用的樣本外預測")
    pred = pd.DataFrame(predictions)
    actual = (pred["actual"] == "UP").astype(int)
    predicted = (pred["prediction"] == "UP").astype(int)
    report = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "up_recall": float(recall_score(actual, predicted, zero_division=0)),
        "base_rate": float(actual.mean()),
        "n_test": int(len(pred))
    }
    return report, pred

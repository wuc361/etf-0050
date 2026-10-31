from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, brier_score_loss
import joblib

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def _make_model(model_name="Logistic Regression", n_splits=3):
    """Chronologically calibrated candidates for comparable probability forecasts."""
    if model_name == "Logistic Regression":
        base = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight=None))
        ])
    elif model_name == "Random Forest":
        base = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=80, max_depth=5, min_samples_leaf=8,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ))
        ])
    elif model_name == "Extra Trees":
        base = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesClassifier(
                n_estimators=80, max_depth=5, min_samples_leaf=8,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return CalibratedClassifierCV(
        estimator=base, method="sigmoid",
        cv=TimeSeriesSplit(n_splits=n_splits), ensemble=True
    )

def _bounded_probabilities(values):
    # A noisy market-direction model should not communicate mathematical certainty.
    return np.clip(np.asarray(values, dtype=float), 0.01, 0.99)

def walk_forward_evaluate(data, test_window=40, min_train=250):
    return evaluate_horizon(data, "target", horizon=1, test_window=test_window, min_train=min_train)

def fit_latest_model(data, target_col="target", model_name="Logistic Regression"):
    df = data.dropna(subset=[target_col]).copy()
    features = [c for c in df.columns if c.startswith("f_")]
    model = _make_model(model_name, n_splits=5).fit(df[features], df[target_col].astype(int))
    joblib.dump({"model": model, "features": features, "target": target_col}, OUT / f"latest_model_{target_col}.joblib")
    return model, features

def save_report(report, predictions):
    OUT.mkdir(exist_ok=True)
    (OUT / "backtest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(OUT / "predictions.csv", index=False)

def evaluate_horizon(data, target_col, horizon=1, test_window=40, min_train=250, model_name="Logistic Regression"):
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
        estimator = _make_model(model_name)
        estimator.fit(train[features], train[target_col].astype(int))
        probabilities = _bounded_probabilities(estimator.predict_proba(test[features])[:, 1])
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
    accuracy = float(accuracy_score(actual, predicted))
    base_rate = float(actual.mean())
    majority_baseline = max(base_rate, 1.0 - base_rate)
    report = {
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "up_recall": float(recall_score(actual, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(actual, pred["up_probability"])),
        "majority_baseline_accuracy": majority_baseline,
        "accuracy_lift_vs_baseline": accuracy - majority_baseline,
        "base_rate": base_rate,
        "n_test": int(len(pred))
    }
    return report, pred


def compare_models_walk_forward(data, target_col="target_1d", test_window=40, min_train=250):
    """Compare calibrated candidates on identical expanding-window out-of-sample folds."""
    df = data.dropna(subset=[target_col]).copy()
    features = [col for col in df.columns if col.startswith("f_")]
    df = df.dropna(subset=features, how="all")
    if len(df) < min_train + 20:
        return {"status": "insufficient_data", "models": [], "selected_model": "Logistic Regression"}
    names = ["Logistic Regression", "Random Forest", "Extra Trees"]
    predictions = {name: [] for name in names}
    start = min_train
    while start < len(df):
        end = min(start + test_window, len(df))
        train, test = df.iloc[:start], df.iloc[start:end]
        for name in names:
            estimator = _make_model(name)
            estimator.fit(train[features], train[target_col].astype(int))
            probability = _bounded_probabilities(estimator.predict_proba(test[features])[:, 1])
            for dt, p, actual in zip(test.index, probability, test[target_col].astype(int)):
                predictions[name].append((str(dt.date()), float(p), int(actual)))
        start = end
    reports = []
    for name, rows in predictions.items():
        if not rows:
            continue
        pred = pd.DataFrame(rows, columns=["date", "probability", "actual"])
        direction = (pred["probability"] >= .5).astype(int)
        actual = pred["actual"]
        reports.append({
            "name": name,
            "accuracy": float(accuracy_score(actual, direction)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, direction)),
            "brier_score": float(brier_score_loss(actual, pred["probability"])),
            "n_test": int(len(pred)),
            "majority_baseline_accuracy": float(max(actual.mean(), 1 - actual.mean()))
        })
    # Prefer balanced accuracy, but reject a candidate that fails to beat the
    # majority baseline or has poor probability score. Fall back to Logistic.
    eligible = [r for r in reports if r["balanced_accuracy"] > 0.52 and r["brier_score"] < 0.25
                and r["balanced_accuracy"] > (2 * r["majority_baseline_accuracy"] - 1) * 0.5 + 0.5]
    selected = max(eligible, key=lambda r: (r["balanced_accuracy"], -r["brier_score"]))["name"] if eligible else "Logistic Regression"
    return {"status": "ok" if reports else "no_predictions", "target": target_col,
            "method": "identical expanding-window out-of-sample folds; sigmoid calibration",
            "selection_rule": "highest balanced accuracy among candidates passing baseline and Brier guardrails",
            "selected_model": selected, "production_model": selected, "models": reports}

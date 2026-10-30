from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
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
    """Fit final classifier and chronological out-of-fold Platt calibration."""
    df = data.dropna(subset=[target_col]).copy()
    features = [c for c in df.columns if c.startswith("f_")]
    df = df.dropna(subset=features, how="all")
    X, y = df[features], df[target_col].astype(int)
    oof_p, oof_y = [], []
    if len(df) >= 350 and y.nunique() == 2:
        horizon = int(target_col.removeprefix("target_").removesuffix("d")) if target_col.startswith("target_") else 1
        for train_idx, valid_idx in TimeSeriesSplit(n_splits=5, gap=max(0, horizon - 1)).split(X):
            if y.iloc[train_idx].nunique() < 2:
                continue
            fold = _make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_p.extend(fold.predict_proba(X.iloc[valid_idx])[:, 1].tolist())
            oof_y.extend(y.iloc[valid_idx].tolist())
    model = _make_model().fit(X, y)
    calibrator = None
    if len(oof_p) >= 100 and len(set(oof_y)) == 2:
        p = np.clip(np.asarray(oof_p), 1e-5, 1 - 1e-5)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(logits, np.asarray(oof_y, dtype=int))
    model.probability_calibrator_ = calibrator
    model.calibration_status_ = "chronological_oof_platt" if calibrator is not None else "uncalibrated_fallback"
    model.calibration_samples_ = len(oof_p)
    joblib.dump({"model": model, "features": features, "target": target_col}, OUT / f"latest_model_{target_col}.joblib")
    return model, features


def predict_calibrated_probability(model, row, features):
    raw = float(model.predict_proba(row[features])[:, 1][0])
    calibrator = getattr(model, "probability_calibrator_", None)
    if calibrator is not None:
        p = float(np.clip(raw, 1e-5, 1 - 1e-5))
        logit = np.log(p / (1 - p))
        probability = float(calibrator.predict_proba(np.asarray([[logit]]))[0, 1])
    else:
        probability = raw
    # Do not display numerical extremes as certainty.
    return float(np.clip(probability, 0.05, 0.95))

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
    """Compare models on identical purged expanding-window folds; keep Logistic as production baseline."""
    df = data.dropna(subset=[target_col]).copy()
    features = [col for col in df.columns if col.startswith("f_")]
    df = df.dropna(subset=features, how="all")
    if len(df) < min_train + 20:
        return {"status": "insufficient_data", "models": []}
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    factories = {
        "Logistic Regression": lambda: _make_model(),
        "Random Forest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=120, max_depth=5, min_samples_leaf=8,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ))
        ]),
        "Extra Trees": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesClassifier(
                n_estimators=120, max_depth=5, min_samples_leaf=8,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ])
    }
    predictions = {name: [] for name in factories}
    start = min_train
    while start < len(df):
        end = min(start + test_window, len(df))
        train, test = df.iloc[:start], df.iloc[start:end]
        if len(train) < min_train:
            break
        for name, factory in factories.items():
            estimator = factory()
            estimator.fit(train[features], train[target_col].astype(int))
            probability = estimator.predict_proba(test[features])[:, 1]
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
        accuracy = float(accuracy_score(actual, direction))
        reports.append({
            "name": name,
            "accuracy": accuracy,
            "balanced_accuracy": float(balanced_accuracy_score(actual, direction)),
            "brier_score": float(brier_score_loss(actual, pred["probability"])),
            "n_test": int(len(pred)),
            "majority_baseline_accuracy": float(max(actual.mean(), 1 - actual.mean()))
        })
    return {"status": "ok" if reports else "no_predictions", "target": target_col,
            "method": "identical purged expanding-window folds",
            "production_model": "Logistic Regression", "models": reports}

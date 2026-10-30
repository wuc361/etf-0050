from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

def _make_model():
    # Robust baseline; optional XGBoost/LightGBM can be added after benchmark.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

def walk_forward_evaluate(data, test_window=40, min_train=250):
    df = data.dropna(subset=["target"]).copy()
    features = [c for c in df.columns if c.startswith("f_")]
    df = df.dropna(subset=features, how="all")
    if len(df) < min_train + test_window:
        raise ValueError(f"有效資料不足：至少需要 {min_train + test_window} 筆，目前 {len(df)} 筆。")
    preds = []
    start = min_train
    while start < len(df):
        end = min(start + test_window, len(df))
        train = df.iloc[:start]
        test = df.iloc[start:end]
        model = _make_model()
        model.fit(train[features], train["target"].astype(int))
        prob = model.predict_proba(test[features])[:,1]
        # Probability threshold 0.5 for unfiltered direction metric.
        for dt, p, actual in zip(test.index, prob, test["target"].astype(int)):
            preds.append({"date":str(dt.date()),"up_probability":float(p),
                          "prediction":"UP" if p >= .5 else "DOWN",
                          "actual":"UP" if actual == 1 else "DOWN",
                          "correct":bool((p >= .5) == (actual == 1))})
        start = end
    p = pd.DataFrame(preds)
    y = (p["actual"] == "UP").astype(int)
    yp = (p["prediction"] == "UP").astype(int)
    report = {
        "accuracy":float(accuracy_score(y,yp)),
        "up_recall":float(recall_score(y,yp,zero_division=0)),
        "coverage":1.0,
        "base_rate":float(y.mean()),
        "n_test":int(len(p))
    }
    return report, p

def fit_latest_model(data, target_col="target"):
    df = data.dropna(subset=[target_col]).copy()
    features = [c for c in df.columns if c.startswith("f_")]
    model = _make_model().fit(df[features], df["target"].astype(int))
    joblib.dump({"model":model,"features":features}, OUT/"latest_model.joblib")
    return model, features

def save_report(report, predictions):
    OUT.mkdir(exist_ok=True)
    (OUT/"backtest_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    predictions.to_csv(OUT/"predictions.csv",index=False)


def evaluate_horizon(data, target_col, test_window=40, min_train=250):
    """Expanding-window evaluation for one independently trained forecast horizon."""
    df = data.dropna(subset=[target_col]).copy()
    features = [c for c in df.columns if c.startswith("f_")]
    df = df.dropna(subset=features, how="all")
    if len(df) < min_train + 20:
        raise ValueError(f"{target_col} 有效資料不足：{len(df)} 筆")
    predictions = []
    start = min_train
    while start < len(df):
        end = min(start + test_window, len(df))
        train, test = df.iloc[:start], df.iloc[start:end]
        estimator = _make_model()
        estimator.fit(train[features], train[target_col].astype(int))
        probabilities = estimator.predict_proba(test[features])[:, 1]
        for dt, probability, actual in zip(test.index, probabilities, test[target_col].astype(int)):
            predictions.append({"date": str(dt.date()), "up_probability": float(probability),
                                "prediction": "UP" if probability >= .5 else "DOWN",
                                "actual": "UP" if actual == 1 else "DOWN",
                                "correct": bool((probability >= .5) == (actual == 1))})
        start = end
    pred = pd.DataFrame(predictions)
    actual = (pred["actual"] == "UP").astype(int)
    predicted = (pred["prediction"] == "UP").astype(int)
    report = {"accuracy": float(accuracy_score(actual, predicted)),
              "up_recall": float(recall_score(actual, predicted, zero_division=0)),
              "base_rate": float(actual.mean()), "n_test": int(len(pred))}
    return report, pred

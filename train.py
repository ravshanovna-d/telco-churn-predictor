"""Train the customer churn model, evaluate it, and export it for the web demo.

Usage:  python train.py
Outputs: web/model.onnx, web/meta.json, web/assets/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from export_utils import check_onnx, examples, export_onnx, feature_schema, write_meta

ROOT = Path(__file__).parent
DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
SEED = 42


def load_data() -> pd.DataFrame:
    cache = ROOT / "data" / "telco_churn.csv"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(cache, index=False)
    df = pd.read_csv(cache)
    # TotalCharges is blank for brand-new customers (tenure 0): nothing billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.drop(columns="customerID")


def build_models(num_cols, cat_cols):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return {
        "Logistic Regression": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, C=0.5))]),
        "Random Forest": Pipeline([("prep", prep), ("model", RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", prep), ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED))]),
    }


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = df.drop(columns="Churn"), (df["Churn"] == "Yes").astype(int)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 1) Compare models with 5-fold cross-validation on the training set
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = build_models(num_cols, cat_cols)
    rows = []
    for name, model in models.items():
        r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                           scoring={"ROC-AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy"})
        rows.append({"model": name, **{k: float(np.mean(r[f"test_{k}"])) for k in ("ROC-AUC", "F1", "Accuracy")}})
        print(f"{name:22s} ROC-AUC={rows[-1]['ROC-AUC']:.3f}  F1={rows[-1]['F1']:.3f}")
    comparison = sorted(rows, key=lambda r: -r["ROC-AUC"])
    # Prefer the simplest model (dict order = simplest first) within 0.005 ROC-AUC of the top score.
    top = comparison[0]["ROC-AUC"]
    best_name = next(r["model"] for r in rows if r["ROC-AUC"] >= top - 0.005)
    best = models[best_name].fit(X_train, y_train)

    # 2) Final evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }
    print("Test:", {k: round(v, 3) for k, v in test_metrics.items()})

    # 3) Figures
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stays", "Churns"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "roc_curve.png", dpi=130); plt.close(fig)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Most important features")
    fig.tight_layout(); fig.savefig(ASSETS / "feature_importance.png", dpi=130); plt.close(fig)

    rate = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot.bar(ax=ax, color="#4f46e5", rot=0); ax.set_ylabel("Churn rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Churn rate by contract type")
    fig.tight_layout(); fig.savefig(ASSETS / "churn_by_contract.png", dpi=130); plt.close(fig)

    # 4) Export for the browser and verify it matches scikit-learn
    export_onnx(best, X_train, WEB / "model.onnx")
    err = check_onnx(WEB / "model.onnx", best, X_test)
    print(f"ONNX check OK (max diff {err:.2e})")

    labels = {
        "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has partner",
        "Dependents": "Has dependents", "tenure": "Tenure", "PhoneService": "Phone service",
        "MultipleLines": "Multiple lines", "InternetService": "Internet service",
        "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
        "DeviceProtection": "Device protection", "TechSupport": "Tech support",
        "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
        "Contract": "Contract", "PaperlessBilling": "Paperless billing",
        "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
    }
    helps = {"tenure": "months", "MonthlyCharges": "USD", "TotalCharges": "USD"}
    write_meta(WEB / "meta.json", {
        "title": "Customer Churn Predictor",
        "subtitle": ("Predicts how likely a telecom customer is to cancel their subscription, "
                     "so a retention team can contact the right people first."),
        "tags": ["Classification", "scikit-learn", best_name, "Cross-validation", "ONNX"],
        "task": "classification",
        "classes": [0, 1], "positive_class": 1,
        "output_label": "Probability this customer leaves",
        "verdicts": ["Low risk — likely to stay", "Medium risk — worth a check-in", "High risk — contact soon"],
        "thresholds": [0.3, 0.6],
        "result_note": "Try switching Contract to “Two year” or adding Tech support and watch the risk drop.",
        "features": feature_schema(X_train, labels, helps),
        "examples": examples(X_test),
        "model_name": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metric_formats": {k: "percent" for k in test_metrics if k != "ROC-AUC"},
        "figures": [
            {"src": "assets/churn_by_contract.png", "caption": "Month-to-month customers churn far more often."},
            {"src": "assets/feature_importance.png", "caption": "What drives the prediction the most."},
            {"src": "assets/roc_curve.png", "caption": "ROC curve on the test set."},
            {"src": "assets/confusion_matrix.png", "caption": "Correct and wrong predictions on the test set."},
        ],
        "steps": [
            f"Cleaned {len(df):,} customer records (IBM Telco dataset): fixed blank charges for new customers.",
            "Built one scikit-learn Pipeline: scaling for numbers, one-hot encoding for categories.",
            "Compared Logistic Regression, Random Forest and Gradient Boosting with 5-fold stratified cross-validation.",
            f"Picked {best_name} by ROC-AUC and evaluated it once on a held-out 20% test set.",
            "Explained it with permutation feature importance.",
            "Exported the whole pipeline to ONNX so it runs in the browser — no server needed.",
        ],
        "dataset": {"name": "IBM Telco Customer Churn", "url": "https://github.com/IBM/telco-customer-churn-on-icp4d", "rows": len(df)},
        "repo_url": "https://github.com/ravshanovna-d/telco-churn-predictor",
        "notebook_url": "https://github.com/ravshanovna-d/telco-churn-predictor/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    })
    print("Wrote web/model.onnx and web/meta.json")


if __name__ == "__main__":
    main()

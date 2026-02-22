from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

IN_PATH = Path("data/processed/features.parquet")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

TARGET_COL = "ChurnFlag"
ID_COL = "customerID"


def main():
    print("Running training pipeline...")

    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run: python src/features.py")

    df = pd.read_parquet(IN_PATH)

    # Drop ID + raw target label string (keep numeric target)
    drop_cols = [ID_COL, "Churn"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Preprocess
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # Model (baseline, interpretable)
    model = LogisticRegression(max_iter=200, class_weight="balanced")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    # Evaluate
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "roc_auc": float(auc),
        "classification_report": report,
        "confusion_matrix": cm,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_used": X.columns.tolist(),
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "threshold": 0.5
    }

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODEL_DIR / "model.pkl")

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model ->", (MODEL_DIR / "model.pkl").resolve())
    print("Saved metrics ->", (REPORTS_DIR / "metrics.json").resolve())
    print("ROC AUC:", round(auc, 4))
    print("Done.")


if __name__ == "__main__":
    main()
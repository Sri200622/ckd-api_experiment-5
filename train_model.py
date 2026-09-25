import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

FEATURES = ["age", "bp", "sg", "al", "bgr", "bu", "sc", "sod", "hemo", "wbcc"]

X, y = make_classification(
    n_samples=800, n_features=len(FEATURES), n_informative=6,
    n_redundant=2, n_classes=2, weights=[0.6, 0.4], random_state=42
)
X = pd.DataFrame(X, columns=FEATURES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
}
print("Evaluation metrics:", metrics)

joblib.dump({"pipeline": pipeline, "features": FEATURES, "metrics": metrics,
             "model_name": "CKD-RandomForest", "version": "1.0"},
            "model/ckd_pipeline.joblib")
print("Model saved to model/ckd_pipeline.joblib")
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import pandas as pd

from data_preprocessing import load_data, preprocess_data
from feature_engineering import add_features


DATA_PATH = "data/processed/telco_churn_cleaned.csv"


def train():
    df = load_data(DATA_PATH)
    df = add_features(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    mlflow.set_experiment("Customer_Churn_Prediction")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, preds)

            mlflow.log_param("model", model_name)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"{model_name} ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    train()

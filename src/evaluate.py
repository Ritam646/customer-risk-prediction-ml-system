import mlflow
import mlflow.sklearn
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_preprocessing import load_data, preprocess_data
from feature_engineering import add_features


DATA_PATH = "data/processed/telco_churn_cleaned.csv"
EXPERIMENT_NAME = "Customer_Churn_Prediction"


def load_best_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    return model


def evaluate():
    df = load_data(DATA_PATH)
    df = add_features(df)

    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    model = load_best_model()

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # SHAP Explainability
    print("Generating SHAP explanations...")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    X_test_transformed = preprocessor.transform(X_test)

    explainer = shap.Explainer(classifier, X_test_transformed)
    shap_values = explainer(X_test_transformed)

    shap.summary_plot(shap_values, X_test_transformed, show=True)


if __name__ == "__main__":
    evaluate()

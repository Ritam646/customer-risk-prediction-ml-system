import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

EXPERIMENT_NAME = "Customer_Churn_Prediction"

app = FastAPI(
    title="Customer Risk Prediction API",
    description="Predict customer churn risk using ML",
    version="1.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


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


model = load_best_model()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict_risk(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])

    # Feature engineering (same as training)
    input_df["AvgMonthlySpend"] = input_df["TotalCharges"] / (
        input_df["tenure"] + 1
    )
    input_df["IsLongTermCustomer"] = (input_df["tenure"] > 24).astype(int)

    churn_prob = model.predict_proba(input_df)[0][1]
    risk_score = round(churn_prob * 100, 2)

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_score": risk_score,
        "risk_label": "High Risk" if risk_score >= 60 else "Low Risk",
    }

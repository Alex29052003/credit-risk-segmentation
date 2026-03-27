import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit Risk Prediction Service")

# загружаем модель
with open("src/model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
columns = artifacts["columns"]


class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict(data: ClientData):
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])

    # добавим отсутствующие колонки, если надо
    for col in columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[columns]

    probability = model.predict_proba(df_input)[0][1]
    prediction = int(probability > 0.3)

    return {
        "default_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk_label": "high_risk" if prediction == 1 else "low_risk"
    }
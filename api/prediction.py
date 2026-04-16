from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .model_loader import get_models

router = APIRouter(prefix="/api", tags=["Prediction"])


class ChurnPredictionInput(BaseModel):
    """
    Pydantic model for validating incoming prediction request data.
    Matches the fields sent from the prediction form in prediction.html
    """

    gender: str
    seniorCitizen: int
    partner: str
    dependents: str
    phoneService: str
    multipleLines: str
    contract: str
    internetService: str
    onlineSecurity: str
    onlineBackup: str
    deviceProtection: str
    techSupport: str
    streamingTV: str
    streamingMovies: str
    paperlessBilling: str
    paymentMethod: str
    tenure: float
    monthlyCharges: float
    totalCharges: float


@router.post("/predict")
async def predict_churn(data: ChurnPredictionInput):
    """
    Predict customer churn probability based on input features.

    This endpoint:
    1. Loads the trained model, scaler, and column names.
    2. Converts the input JSON to a pandas DataFrame.
    3. Aligns columns to match the training data.
    4. Scales the features using the saved scaler.
    5. Returns the churn probability and risk classification.
    """
    try:
        # 1. Load model artifacts
        model, scaler, training_columns = get_models()

        # 2. Convert input to DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # 3. Map form inputs to exact training column names
        # The model was trained with specific one-hot encoded columns.
        # We must manually create these columns to match the training data exactly.
        input_data = data.model_dump()

        # Initialize all training columns to 0
        encoded = {col: 0 for col in training_columns}

        # Numerical columns (direct mapping)
        encoded["tenure"] = input_data["tenure"]
        encoded["MonthlyCharges"] = input_data["monthlyCharges"]
        encoded["TotalCharges"] = input_data["totalCharges"]

        # Binary columns (Yes/No or Male/Female -> 1/0)
        encoded["SeniorCitizen"] = input_data["seniorCitizen"]
        encoded["gender"] = 1 if input_data["gender"] == "Male" else 0
        encoded["Partner"] = 1 if input_data["partner"] == "Yes" else 0
        encoded["Dependents"] = 1 if input_data["dependents"] == "Yes" else 0
        encoded["PhoneService"] = 1 if input_data["phoneService"] == "Yes" else 0
        encoded["MultipleLines"] = 1 if input_data["multipleLines"] == "Yes" else 0
        encoded["PaperlessBilling"] = (
            1 if input_data["paperlessBilling"] == "Yes" else 0
        )

        # One-hot encoded columns (reference categories are implicitly 0)
        # Internet Service: DSL is reference
        if input_data["internetService"] == "Fiber optic":
            encoded["InternetService_Fiber optic"] = 1
        elif input_data["internetService"] == "No":
            encoded["InternetService_No"] = 1

        # Contract: Month-to-month is reference
        if input_data["contract"] == "One year":
            encoded["Contract_One year"] = 1
        elif input_data["contract"] == "Two year":
            encoded["Contract_Two year"] = 1

        # Payment Method: Bank transfer (automatic) is reference
        if input_data["paymentMethod"] == "Credit card (automatic)":
            encoded["PaymentMethod_Credit card (automatic)"] = 1
        elif input_data["paymentMethod"] == "Electronic check":
            encoded["PaymentMethod_Electronic check"] = 1
        elif input_data["paymentMethod"] == "Mailed check":
            encoded["PaymentMethod_Mailed check"] = 1

        # Services with "No internet service" option (No is reference)
        for col_prefix, form_key in [
            ("OnlineSecurity", "onlineSecurity"),
            ("OnlineBackup", "onlineBackup"),
            ("DeviceProtection", "deviceProtection"),
            ("TechSupport", "techSupport"),
            ("StreamingTV", "streamingTV"),
            ("StreamingMovies", "streamingMovies"),
        ]:
            val = input_data[form_key]
            if val == "Yes":
                encoded[f"{col_prefix}_Yes"] = 1
            elif val == "No internet service":
                encoded[f"{col_prefix}_No internet service"] = 1

        # MultipleLines not in form, defaults to 0

        # 4. Create DataFrame with exact column order
        input_df = pd.DataFrame([encoded])[training_columns]

        # 4. Scale features
        input_scaled = scaler.transform(input_df)
        # Convert back to DataFrame to preserve feature names for the model
        input_scaled_df = pd.DataFrame(input_scaled, columns=training_columns)

        # 5. Predict probability
        # predict_proba returns [[prob_class_0, prob_class_1]]
        probabilities = model.predict_proba(input_scaled_df)[0]
        churn_prob = float(probabilities[1])  # Probability of class 1 (Churn)

        # 6. Determine prediction and risk level
        prediction = "Yes" if churn_prob >= 0.5 else "No"

        if churn_prob >= 0.7:
            risk_level = "High"
            risk_class = "high"
            message = "High risk of churn. Immediate retention actions recommended."
        elif churn_prob >= 0.4:
            risk_level = "Medium"
            risk_class = "medium"
            message = "Moderate risk. Monitor customer engagement closely."
        else:
            risk_level = "Low"
            risk_class = "low"
            message = "Low risk. Customer appears satisfied and engaged."

        return {
            "probability": round(churn_prob, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "message": message,
        }

    except Exception as e:
        # Log error and return 500
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

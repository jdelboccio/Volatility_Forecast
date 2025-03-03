import shap
import numpy as np
from models.lstm_model import lstm_model

def compute_shap_values(X_test):
    """
    Compute SHAP values for explainability.
    """
    try:
        background_data = X_test[:50] if len(X_test) >= 50 else X_test
        explainer = shap.Explainer(lstm_model, background_data)
        shap_values = explainer(X_test)
        return shap_values
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return None

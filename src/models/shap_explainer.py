import numpy as np
import joblib
try:
    import shap
except ImportError:
    shap = None
    import warnings
    warnings.warn("SHAP library not installed. SHAP explanations will be skipped.")

from pathlib import Path

def get_shap_explainer(model, X_background=None):
    """Load or create a SHAP explainer for the given model."""
    # Determine path for shap_model.pkl relative to this file
    shap_file = Path(__file__).resolve().parent / "shap_model.pkl"
    explainer = None
    if shap is None:
        return None
    if shap_file.is_file():
        # Load existing SHAP explainer
        try:
            explainer = joblib.load(shap_file)
        except Exception as e:
            explainer = None
    if explainer is None and model is not None:
        # Create a new explainer (use TreeExplainer for tree-based models like Random Forest)
        try:
            explainer = shap.TreeExplainer(model, data=X_background)
        except Exception:
            explainer = shap.Explainer(model, X_background)  # fallback to generic Explainer
        # Save the explainer for future use
        try:
            joblib.dump(explainer, shap_file)
        except Exception as e:
            print(f"Could not save SHAP explainer: {e}")
    return explainer

def explain_model(model, data):
    """
    Compute SHAP values for the given model and dataset.
    Returns the SHAP values for the model predictions on the data.
    """
    if shap is None:
        return None
    if model is None or data is None:
        return None
    # Prepare background data (if too large, sample for efficiency)
    X = data.copy()
    # Only use feature columns (exclude 'ticker' or target if present)
    cols = [c for c in X.columns if c not in ['ticker', 'volatility']]
    X = X[cols]
    explainer = get_shap_explainer(model, X_background=X.head(100))  # use first 100 as background to speed up
    if explainer is None:
        return None
    # Compute SHAP values for the provided data
    try:
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        shap_values = None
    return shap_values

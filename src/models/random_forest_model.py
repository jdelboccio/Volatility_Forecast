import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_random_forest(data: pd.DataFrame, target_column: str, model_path: str = "random_forest_model.pkl"):
    """
    Train a Random Forest model for short-term volatility forecasting.

    :param data: DataFrame containing features and target volatility.
    :param target_column: The column to predict (volatility).
    :param model_path: Path to save the trained model.
    """
    # Drop rows with missing values
    data = data.dropna()

    # Define Features (X) and Target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and Train Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    predictions = model.predict(X_test)
    error = mean_squared_error(y_test, predictions)

    print(f"Random Forest Model Error (MSE): {error}")

    # Save Model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model

if __name__ == "__main__":
    # Example usage (replace with actual dataset)
    df = pd.read_csv("../data/volatility_data.csv")  # Ensure you have this dataset
    model = train_random_forest(df, target_column="volatility_10d")

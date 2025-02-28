from sklearn.metrics import mean_absolute_error
from src.models.lstm_model import lstm_model, X_test, y_test
from src.models.random_forest_model import rf_model
from src.models.garch_model import fit_garch_model

# Predict using trained models
y_pred_lstm = lstm_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Get GARCH benchmark
garch_vol = fit_garch_model(df)

# Compare MAE
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Display results
print(f"Model Performance:")
print(f"- LSTM MAE: {mae_lstm:.4f}")
print(f"- Random Forest MAE: {mae_rf:.4f}")
print(f"- GARCH(1,1) Forecast: {garch_vol:.4f}")

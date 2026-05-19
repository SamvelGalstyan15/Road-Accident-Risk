# Road Accident Risk Prediction

An end-to-end Machine Learning pipeline built to predict road accident risk based on environmental, temporal, and structural road conditions.

## 📊 Performance
* **Model:** `StandardScaler` + `SGDRegressor` (via `scikit-learn` Pipeline)
* **Cross-Validation Score (\(R^2\)):** `0.8051`
* **Mean Absolute Error (MAE):** `0.0583`

## 📈 Actual vs Predicted

![Actual vs Predicted](actual_vs_predicted.png)

## 🏗️ Pipeline Overview
1. **Preprocessing:** Vectorized conversion of booleans to `0/1` via `.astype(int)` and One-Hot Encoding for categorical features (`drop_first=True` to prevent multicollinearity).
2. **Optimization:** 3-fold `GridSearchCV` sweeping 27 configurations across feature scalers, regularization strengths (`alpha`), and learning rates (`eta0`).
3. **Deployment:** Full pipeline serialized into `road_accident_risk_model.pkl` via `joblib` for production readiness.

## 🚀 Quick Inference

```python
import joblib
import pandas as pd

model = joblib.load('road_accident_risk_model.pkl')
# Input data must match the encoded feature structure
predicted_risk = model.predict(pd.DataFrame([new_data_row]))
```

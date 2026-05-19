# Road Accident Risk Prediction

An end-to-end Machine Learning pipeline built to predict road accident risk based on environmental, temporal, and structural road conditions.

## 📊 Performance
* **Model:** `StandardScaler` + `SGDRegressor` (via `scikit-learn` Pipeline)
* **Cross-Validation Score (\(R^2\)):** `0.8051`
* **Mean Absolute Error (MAE):** `0.0583`

## 📈 Actual vs Predicted

<img width="599" height="471" alt="image" src="https://github.com/user-attachments/assets/58b8cd4f-252b-423d-8aad-04cdfd6f6631" />


## 🛠️ Installation & Requirements

The project dependencies are managed via `requirements.txt`. To set up the environment and run the notebook or inference script, execute:

```bash
pip install -r requirements.txt
```


### 1. Download Model Weights
Download the pre-trained model file and place it into the root directory of the project (next to `prediction.py`):
 
   **[Download road_accident_risk_model.pkl](https://google.com)** (Google Drive)


## 🐳 Running with Docker

### 1. Build the Docker Image
Run the following command in your terminal inside the project root folder:
```bash
docker build -t road-accident-risk .
```

### 2. Run Predictions
Execute the container and pass your input data (numbers separated by commas, no spaces) as an argument:
```bash
docker run --rm road-accident-risk 3,1,80,0.5
```


## 🏗️ Pipeline Overview
1. **Preprocessing:** Vectorized conversion of booleans to `0/1` via `.astype(int)` and One-Hot Encoding for categorical features (`drop_first=True` to prevent multicollinearity).
2. **Optimization:** 3-fold `GridSearchCV` sweeping 27 configurations across feature scalers, regularization strengths (`alpha`), and learning rates (`eta0`).
3. **Deployment:** Full pipeline serialized into `road_accident_risk_model.pkl` via `joblib` for production readiness.

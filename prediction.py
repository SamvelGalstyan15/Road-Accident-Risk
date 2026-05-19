import pandas as pd
import sys
import joblib

def predict(test_data):
  try:
    model = joblib.load('road_accident_risk_model.pkl')
    pred = model.predict(test_data)
    return pred[0]    
  except Exception as ex:
    print(type(ex).__name__)


if __name__ == '__main__':
  if len(sys.argv) > 1:
    predict(sys.argv[1])


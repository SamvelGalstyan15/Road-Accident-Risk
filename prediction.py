import pandas as pd
import sys
import joblib

def predict(test_data):
  try:
    model = joblib.load('road_accident_risk_model.pkl')

    if ',' in test_data:
      input = [[float(i) for i in test_data.split(',')]]
    else:
      input = [[float(test_data)]]
    pred = model.predict(input)
    return pred[0]    
  except Exception as ex:
    print(type(ex).__name__)


if __name__ == '__main__':
  if len(sys.argv) > 1:
    print(predict(sys.argv[1]))


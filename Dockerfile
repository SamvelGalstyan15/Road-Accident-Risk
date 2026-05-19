FROM python:3.10-slim

WORKDIR /app

# Кэшируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и веса модели
COPY . .

# При запуске контейнера проверяем работоспособность модели
CMD ["python", "-c", "import joblib; model=joblib.load('road_accident_risk_model.pkl')"]

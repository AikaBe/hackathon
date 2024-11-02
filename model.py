from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Загрузка данных о автобусных остановках
data = pd.read_csv('bus_stops.csv')  # Замените на путь к вашим данным

# Здесь вы должны подготовить и обучить вашу модель
# Пример простой модели
model = LinearRegression()
# Предполагая, что у вас есть необходимые данные для обучения
# X = features, y = target (время прибытия)
X = data[['stop_id']]  # замените на ваши реальные признаки
y = data['arrival_time']  # замените на ваши реальные метки
model.fit(X, y)

@app.route('/api/bus_stops', methods=['GET'])
def get_bus_stops():
    stops = data.to_dict(orient='records')
    return jsonify(stops)

@app.route('/api/predict_arrival', methods=['GET'])
def predict_arrival():
    stop_id = request.args.get('stop_id', type=int)
    prediction = model.predict(np.array([[stop_id]]))  # Пример предсказания
    return jsonify({'arrival_time': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

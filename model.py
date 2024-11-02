from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from geopy.distance import geodesic

app = Flask(__name__)

# Загрузка данных о автобусных остановках
data = pd.read_csv('bus_stops.csv', names=['stop_id', 'latitude', 'longitude'])

approx_time_between_stops = 5  # примерное фиксированное время между остановками

def find_nearest_stop(user_location):
    distances = data.apply(
        lambda row: geodesic(user_location, (row['latitude'], row['longitude'])).meters, axis=1
    )
    nearest_stop_index = distances.idxmin()
    return data.iloc[nearest_stop_index]

@app.route('/api/predict_arrival', methods=['GET'])
def predict_arrival():
    stop_id = request.args.get('stop_id', type=int)

    if stop_id is None:
        return jsonify({'error': 'Параметр stop_id обязателен'}), 400

    # Находим остановку по ID
    stop_data = data[data['stop_id'] == stop_id]

    if stop_data.empty:
        return jsonify({'error': 'Остановка не найдена'}), 404

    # Оценка времени прибытия
    # Здесь вы можете добавить свою логику для расчета времени прибытия
    estimated_arrival_time = approx_time_between_stops

    return jsonify({
        'stop_id': stop_id,
        'arrival_time_estimate': estimated_arrival_time
    })


if __name__ == '__main__':
    app.run(debug=True)

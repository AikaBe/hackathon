import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
# Загрузка данных
data = pd.read_csv('municipality_bus_utilization.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Проверка первых строк данных
print(data.head())

# Визуализация данных
plt.figure(figsize=(12, 6))
plt.plot(data['usage'], label='Фактическое использование', color='blue')
plt.title('Использование автобусных услуг')
plt.xlabel('Время')
plt.ylabel('Использование')
plt.legend()
plt.show()
def test_stationarity(timeseries):
    # Выполнение теста Дики-Фуллера
    result = adfuller(timeseries)
    print('Статистика теста:', result[0])
    print('p-значение:', result[1])
    
test_stationarity(data['usage'])
# Обучение модели SES
ses_model = ExponentialSmoothing(data['usage'], trend=None, seasonal=None, seasonal_periods=None).fit()
data['SES'] = ses_model.fittedvalues

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(data['usage'], label='Фактическое использование', color='blue')
plt.plot(data['SES'], label='SES', color='red')
plt.title('Простое экспоненциальное сглаживание')
plt.xlabel('Время')
plt.ylabel('Использование')
plt.legend()
plt.show()
# Определение параметров SARIMA
order = (1, 1, 1)  # (p, d, q)
seasonal_order = (1, 1, 1, 12)  # (P, D, Q, S)

# Обучение модели SARIMA
sarima_model = SARIMAX(data['usage'], order=order, seasonal_order=seasonal_order).fit()
data['SARIMA'] = sarima_model.fittedvalues

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(data['usage'], label='Фактическое использование', color='blue')
plt.plot(data['SARIMA'], label='SARIMA', color='green')
plt.title('Модель SARIMA')
plt.xlabel('Время')
plt.ylabel('Использование')
plt.legend()
plt.show()
# Прогноз на следующие 12 периодов
forecast = sarima_model.forecast(steps=12)
plt.figure(figsize=(12, 6))
plt.plot(data['usage'], label='Фактическое использование', color='blue')
plt.plot(forecast.index, forecast, label='Прогноз', color='orange')
plt.title('Прогноз использования автобусных услуг')
plt.xlabel('Время')
plt.ylabel('Использование')
plt.legend()
plt.show()
mae = mean_absolute_error(data['usage'][-12:], forecast)
print(f'Mean Absolute Error: {mae}')

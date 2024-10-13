import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error


warnings.filterwarnings("ignore")


# Функция для определения оптимального значения d с тестом Дики-Фуллера
def determine_d(data):
    d = 0
    while adfuller(data)[1] > 0.05:  # Пока p-value > 0.05, продолжаем дифференцирование
        data = data.diff().dropna()
        d += 1
    return d


# Оптимизация параметров ARIMA с подбором p, d, q
def optimize_arima(data, p_range, q_range):
    d = determine_d(data)  # Определяем оптимальное значение d
    best_aic, best_params, best_model = float("inf"), None, None

    # Перебор всех комбинаций p и q
    for p, q in itertools.product(p_range, q_range):
        try:
            model = ARIMA(data, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic, best_params, best_model = results.aic, (p, d, q), results
        except:
            continue
    print(f'Лучшие параметры (p,d,q): {best_params}, AIC: {best_aic}')
    return best_model, best_params


# Визуализация диагностики модели
def plot_diagnostics(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    results.plot_diagnostics(fig=fig)
    plt.tight_layout()
    plt.show()


# Рассчет метрик ошибок
def calculate_mape(y_true, y_pred):
    mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_rmse_mae(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae


# Оценка модели на исторических и независимых данных
def evaluate_model(df, model, train_end, forecast_start, forecast_end, real_data_path):
    # Оценка на исторических данных
    in_sample_forecast_mean, actual_in_sample = in_sample_forecast(model, df, train_end)
    in_sample_mape = calculate_mape(actual_in_sample, in_sample_forecast_mean)
    print(f"MAPE на исторических данных: {in_sample_mape:.2f}%\n\n")

    # Оценка на независимых данных
    real_data = preprocess_data(real_data_path)
    out_of_sample_forecast_mean = out_of_sample_forecast(model, df, forecast_start, forecast_end, real_data)
    actual_out_of_sample = real_data.loc[forecast_start:forecast_end, 'Цена']
    out_of_sample_mape = calculate_mape(actual_out_of_sample, out_of_sample_forecast_mean)
    print(f"MAPE на независимых данных: {out_of_sample_mape:.2f}%")

    # Рассчет RMSE и MAE
    rmse, mae = calculate_rmse_mae(actual_out_of_sample, out_of_sample_forecast_mean)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Возвращаем прогноз и фактические данные
    return out_of_sample_forecast_mean, actual_out_of_sample


# Прогноз на исторических данных
def in_sample_forecast(model, data, train_end):
    forecast = model.get_prediction(end=train_end)
    forecast_mean = forecast.predicted_mean
    actual = data.loc[:train_end, 'Цена']

    # Визуализация прогноза
    plot_forecast(data, forecast_mean, 'Прогноз на исторических данных')
    return forecast_mean, actual


# Прогноз на будущие данные
def out_of_sample_forecast(model, data, forecast_start, forecast_end, real_data):
    steps = len(pd.date_range(start=forecast_start, end=forecast_end))

    # Получение прогноза и явное задание временных меток
    forecast_mean = model.get_forecast(steps=steps).predicted_mean
    forecast_dates = pd.date_range(start=forecast_start, periods=steps, freq='D')
    forecast_mean.index = forecast_dates

    # Визуализация прогноза и реальных данных
    plot_forecast(data, forecast_mean, 'Прогноз vs Реальные данные', real_data, forecast_start, forecast_end)
    return forecast_mean


# Общая функция для визуализации прогноза
def plot_forecast(data, forecast_mean, title, real_data=None, forecast_start=None, forecast_end=None):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Цена'], label='Исторические данные')
    plt.plot(forecast_mean.index, forecast_mean, label='Прогноз', color='red')

    # Визуализация реальных данных
    if real_data is not None and forecast_start is not None and forecast_end is not None:
        real_data_subset = real_data.loc[forecast_start:forecast_end]
        if not real_data_subset.empty:
            plt.plot(real_data_subset.index, real_data_subset['Цена'], label='Реальные данные', color='green')

    plt.title(title)
    plt.legend()
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def train_arima_model(data, order):
    model = ARIMA(data, order=order)
    results = model.fit()
    print(results.summary())
    return results


# Загрузка и предобработка данных
def preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Дата'], dayfirst=True)
    df = df.sort_values('Дата').set_index('Дата')
    df['Цена'] = pd.to_numeric(df['Цена'], errors='coerce')
    return df[['Цена']]


# Тест Дики-Фуллера на стационарность
def adf_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-value : ", dftest[1])
    print("3. Num of lags : ", dftest[2])
    print("4. Number of Observations Used : ", dftest[3])
    print("5. Critical values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)
    if dftest[1] <= 0.05:
        print("Ряд является стационарным")
    else:
        print("Ряд не является стационарным")


def plot_exchange_rate(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Цена'])
    plt.title('Колебания валютного курса CNY/RUB')
    plt.xlabel('Дата')
    plt.ylabel('Курс')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

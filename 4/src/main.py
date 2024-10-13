from func import *
import pandas as pd

def main():
    # 1. Загрузка данных
    print("Загрузка и предобработка данных...")
    df = preprocess_data('data/training_data_2.csv')
    print(df.head())
    print(df.info())

    # 2. Визуализация данных
    print("\n\nВизуализация данных...")
    plot_exchange_rate(df)

    # 3. Проверка на стационарность
    print("\n\nПроверка на стационарность...")
    adf_test(df['Цена'])

    # 4. Подбор оптимальных параметров ARIMA
    print("\n\nПодбор оптимальных параметров ARIMA...")
    p_range, q_range = range(1, 7), range(0, 4)
    best_model, best_params = optimize_arima(df['Цена'], p_range, q_range)

    # 5. Обучение модели
    print("\n\nОбучение модели...")
    final_model = train_arima_model(df['Цена'], best_params)

    # 6. Диагностика модели
    print("\n\nВизуализация диагностики модели...")
    plot_diagnostics(final_model)

    # 7-8. Оценка модели на исторических и независимых данных 
    train_end = '31-12-2023'
    forecast_start = pd.to_datetime('01-01-2024', dayfirst=True)
    forecast_end = pd.to_datetime('01-03-2024', dayfirst=True)
    real_data_path = 'data/data_for_comparison_with_forecast.csv'

    out_of_sample_forecast_mean, actual_out_of_sample = evaluate_model(df, final_model, train_end, forecast_start, forecast_end, real_data_path)

    # Сохранение результатов прогноза
    forecast_df = pd.DataFrame({
        'Дата': out_of_sample_forecast_mean.index,
        'Прогноз курса': out_of_sample_forecast_mean,
        'Реальный курс': actual_out_of_sample
    })
    forecast_df.to_csv('results/forecast_results.csv', index=False)
    print("\n\nРезультаты прогноза сохранены в файл 'results/forecast_results.csv'")


if __name__ == "__main__":
    main()

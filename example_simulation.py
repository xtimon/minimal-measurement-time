#!/usr/bin/env python3
"""
Пример полного скрипта для запуска симуляции времени измерения информации

Этот скрипт демонстрирует базовое использование пакета measurement-time-simulator
для расчета минимального времени измерения информации в физической системе.

Запуск:
    python example_simulation.py
"""

import numpy as np
from measurement_time_simulator import (
    GPUInformationMeasurementSimulator,
    ResultExporter
)


def main():
    """
    Основная функция для запуска симуляции
    """
    # Инициализация симулятора
    print("=" * 60)
    print("ИНИЦИАЛИЗАЦИЯ СИМУЛЯТОРА")
    print("=" * 60)
    sim = GPUInformationMeasurementSimulator(
        temperature=300.0,  # Температура системы: 300 K (комнатная температура)
        use_gpu=True        # Использовать GPU если доступен
    )
    print(f"Температура системы: {sim.temperature} K")
    print(f"Режим работы: {'GPU' if sim.use_gpu else 'CPU'}")
    
    # Параметры системы
    print("\n" + "=" * 60)
    print("ПАРАМЕТРЫ СИСТЕМЫ")
    print("=" * 60)
    print("Тип системы: Ферми-газ (электроны)")
    print("Количество информации: 8 бит")
    print("Критическая температура: 100 K")
    print("Число частиц: 10,000")
    print("Температура Ферми: 10,000 K")
    print("Взаимодействие (U/W): 0.5")
    
    # Запуск симуляции
    print("\n" + "=" * 60)
    print("ЗАПУСК СИМУЛЯЦИИ")
    print("=" * 60)
    print("Выполняется расчет...")
    
    measurement_time, results = sim.simulate_single_system(
        delta_I=8.0,           # 8 бит информации
        T_c=100.0,             # Критическая температура 100 K
        N=10000.0,             # 10000 частиц
        stats_type=0,          # Фермионы
        T_F=10000.0,           # Температура Ферми 10000 K
        U=0.5,                 # Энергия взаимодействия
        W=1.0,                 # Ширина зоны
        T1=1e-3,               # Время релаксации (1 мс)
        T2=1e-4,               # Время дефазировки (100 мкс)
        noise_temperature=1.0,  # Эффективная температура шума
        shot_noise_factor=0.1,  # Фактор дробового шума
        technical_noise=0.02,   # Технический шум (2%)
        include_decoherence=True,
        include_noise=True
    )
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
    print("=" * 60)
    print(f"Минимальное время измерения: {measurement_time:.3e} с")
    
    # Дополнительная информация из результатов
    if isinstance(results, dict):
        print(f"\nДетальная информация:")
        print(f"  Сложность системы (Γ_total): {results['gamma_factors']['total']:.3f}")
        print(f"  Фактор декогеренции: {results['decoherence']['factor']:.3f}")
        print(f"  Фактор шума: {results['noise']['factor']:.3f}")
        print(f"  Идеальное время (без декогеренции/шума): {results['min_measurement_time_ideal']:.3e} с")
        
        # Интерпретация времени
        if measurement_time < 1e-12:
            time_scale = "ПИКОСЕКУНДЫ"
        elif measurement_time < 1e-9:
            time_scale = "НАНОСЕКУНДЫ"
        elif measurement_time < 1e-6:
            time_scale = "МИКРОСЕКУНДЫ"
        elif measurement_time < 1e-3:
            time_scale = "МИЛЛИСЕКУНДЫ"
        else:
            time_scale = "СЕКУНДЫ"
        
        print(f"\n  Интерпретация: {time_scale}")
        print(f"  Отчет сохранен в: {results.get('report_file', 'N/A')}")
    else:
        print(f"Отчет сохранен в: {results}")
    
    print("=" * 60)
    print("\nСимуляция завершена успешно!")


def example_batch_simulation():
    """
    Пример векторизованной симуляции множества систем
    """
    print("\n" + "=" * 60)
    print("ПРИМЕР ВЕКТОРИЗОВАННОЙ СИМУЛЯЦИИ")
    print("=" * 60)
    
    sim = GPUInformationMeasurementSimulator(temperature=300.0, use_gpu=True)
    
    # Подготовка данных для 1000 систем
    n_points = 1000
    print(f"Подготовка данных для {n_points} систем...")
    
    np.random.seed(42)  # Для воспроизводимости
    delta_I_array = np.full(n_points, 1.0)
    T_c_array = np.random.uniform(10, 1000, n_points)
    N_array = np.random.lognormal(10, 2, n_points)
    stats_type_array = np.random.choice([0, 1, 2], n_points)
    T_F_array = np.random.uniform(100, 10000, n_points)
    
    # Запуск симуляции
    print("Выполняется расчет...")
    min_times = sim.main_equation_batch(
        delta_I_array=delta_I_array,
        T_c_array=T_c_array,
        N_array=N_array,
        stats_type_array=stats_type_array,
        T_F_array=T_F_array,
        include_decoherence=True,
        include_noise=True
    )
    
    # Анализ результатов
    print("\n" + "=" * 60)
    print("СТАТИСТИКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    print(f"Рассчитано систем: {len(min_times)}")
    print(f"Минимальное время: {np.min(min_times):.3e} с")
    print(f"Максимальное время: {np.max(min_times):.3e} с")
    print(f"Среднее время: {np.mean(min_times):.3e} с")
    print(f"Медианное время: {np.median(min_times):.3e} с")
    print(f"Стандартное отклонение: {np.std(min_times):.3e} с")
    print("=" * 60)


if __name__ == "__main__":
    # Запуск основной симуляции
    main()
    
    # Опционально: запуск примера векторизованной симуляции
    # Раскомментируйте следующую строку для запуска:
    # example_batch_simulation()


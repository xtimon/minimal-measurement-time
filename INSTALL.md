# Инструкция по установке

## Установка пакета

### 1. Установка в режиме разработки (рекомендуется)

```bash
cd /Users/timurisanov/Documents/measurement-time-simulator
pip install -e .
```

### 2. Установка из текущей директории

```bash
pip install .
```

### 3. Установка с поддержкой GPU (опционально)

```bash
pip install -e .[gpu]
```

или

```bash
pip install .[gpu]
```

## Проверка установки

```python
from minimal_measurement_time import GPUInformationMeasurementSimulator

# Создание симулятора
sim = GPUInformationMeasurementSimulator(temperature=300.0)

# Простой тест
measurement_time, results = sim.simulate_single_system(
    delta_I=1.0,
    T_c=100.0,
    N=1000.0,
    stats_type=0,
    T_F=5000.0
)

print(f"Время измерения: {measurement_time:.3e} с")
```

## Использование в коде

После установки пакет можно использовать в любом Python скрипте:

```python
from minimal_measurement_time import (
    GPUInformationMeasurementSimulator,
    ResultExporter,
    ParticleStatistics
)

# Ваш код здесь
```

## Структура пакета

```
minimal_measurement_time/
├── __init__.py          # Основные экспорты
├── constants.py         # Физические константы
├── exporter.py          # Экспорт результатов
├── particle_statistics.py  # Типы статистики частиц
└── simulator.py         # Основной класс симулятора
```


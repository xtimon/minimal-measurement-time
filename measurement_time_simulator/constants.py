"""
Константы для симулятора времени измерения информации
"""

import logging

logger = logging.getLogger(__name__)

# Попытка импорта CuPy для GPU вычислений
try:
    import cupy as cp

    HAS_CUPY = True
    logger.info("✅ CuPy доступен для GPU вычислений")
except ImportError:
    HAS_CUPY = False
    logger.info("❌ CuPy не установлен, используется CPU")

# Попытка импорта Numba CUDA
try:
    from numba import cuda, jit, float64, int32

    HAS_NUMBA_CUDA = True
    logger.info("✅ Numba CUDA доступен")
except ImportError:
    HAS_NUMBA_CUDA = False
    logger.info("❌ Numba CUDA не доступен")

# Физические константы
HBAR = 1.0545718e-34  # Дж·с (постоянная Планка)
KB = 1.380649e-23  # Дж/К (постоянная Больцмана)

# Константы для расчетов
DEFAULT_N_Q = 1e6  # Пороговое число частиц для насыщения
DEFAULT_DELTA_E = 1e-20  # Дж (минимальная энергия по умолчанию)
DEFAULT_TAU_TECH = 1e-9  # с (техническое ограничение по умолчанию)
DEFAULT_T_STAR = 100.0  # K (характерная температура по умолчанию)
MIN_TEMPERATURE = 1e-10  # K (минимальная температура для расчетов)
MIN_DELTA_E = 1e-30  # Дж (минимальная энергия)

# Константы для декогеренции (типичные значения для различных систем)
DEFAULT_T1 = 1e-3  # с (время релаксации по умолчанию, 1 мс)
DEFAULT_T2 = 1e-4  # с (время дефазировки по умолчанию, 100 мкс)
MIN_T1 = 1e-9  # с (минимальное T1, 1 нс)
MIN_T2 = 1e-9  # с (минимальное T2, 1 нс)

# Константы для шума
DEFAULT_NOISE_TEMPERATURE = 1.0  # K (эффективная температура шума)
DEFAULT_SHOT_NOISE_FACTOR = 0.1  # Фактор дробового шума
DEFAULT_TECHNICAL_NOISE = 0.01  # Фактор технического шума (1%)

# Константы для времени установления равновесия
DEFAULT_EQUILIBRIUM_TIME = 1e-9  # с (1 нс по умолчанию)
MIN_EQUILIBRIUM_TIME = 1e-12  # с (минимальное время, 1 пс)

# Константы для времени отклика детектора (зависят от типа системы)
# Экспериментальные данные: сверхпроводящие кубиты 10-100 нс, ионные ловушки 1-10 мкс
DEFAULT_DETECTOR_RESPONSE_TIME = 1e-8  # с (10 нс по умолчанию для квантовых детекторов)
MIN_DETECTOR_RESPONSE_TIME = 1e-9  # с (минимальное время, 1 нс)

# Время детектора для разных типов систем (на основе экспериментальных данных)
DETECTOR_TIME_FERMION = 5e-9  # 5 нс (быстрые детекторы для ферми-газов)
DETECTOR_TIME_BOSON = 1e-7  # 100 нс (медленные детекторы для бозе-конденсатов)
DETECTOR_TIME_CLASSICAL = 1e-8  # 10 нс (стандартные детекторы)

# Время равновесия для разных типов систем (на основе физических моделей)
EQUILIBRIUM_TIME_FERMION = 1e-9  # 1 нс (быстрое равновесие)
EQUILIBRIUM_TIME_BOSON = 1e-6  # 1 мкс (медленное равновесие для конденсатов)
EQUILIBRIUM_TIME_CLASSICAL = 1e-9  # 1 нс (быстрое равновесие)
EQUILIBRIUM_TIME_CORRELATED = 5e-10  # 0.5 нс (очень быстрое для коррелированных)

# Константы для дополнительных источников шума
DEFAULT_FLICKER_NOISE_FACTOR = 0.05  # Фактор 1/f шума (фликкер-шум)
DEFAULT_QUANTUM_NOISE_FACTOR = 0.02  # Фактор квантового шума измерения
DEFAULT_ENVIRONMENT_NOISE_FACTOR = 0.03  # Фактор шума окружения

# Коэффициенты для факторов сложности
FERMION_COEFF = 1.2  # Коэффициент для фермионов
BOSON_COEFF = 0.7  # Коэффициент для бозонов
CORRELATION_COEFF = 0.3  # Коэффициент корреляций
QUASIPARTICLE_COEFF = 0.5  # Коэффициент квазичастиц
GAMMA_BASE_COEFF = 0.95  # Базовый коэффициент сложности
N_LOG_DIVISOR = 1000.0  # Делитель для логарифма числа частиц

"""
Модуль симулятора времени измерения информации
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
import logging

# Импорты из модулей пакета
from .constants import (
    HBAR, KB,
    DEFAULT_N_Q, DEFAULT_DELTA_E, DEFAULT_TAU_TECH, DEFAULT_T_STAR,
    MIN_TEMPERATURE, MIN_DELTA_E,
    DEFAULT_T1, DEFAULT_T2, MIN_T1, MIN_T2,
    DEFAULT_NOISE_TEMPERATURE, DEFAULT_SHOT_NOISE_FACTOR, DEFAULT_TECHNICAL_NOISE,
    DEFAULT_EQUILIBRIUM_TIME, MIN_EQUILIBRIUM_TIME,
    DEFAULT_DETECTOR_RESPONSE_TIME, MIN_DETECTOR_RESPONSE_TIME,
    DETECTOR_TIME_FERMION, DETECTOR_TIME_BOSON, DETECTOR_TIME_CLASSICAL,
    EQUILIBRIUM_TIME_FERMION, EQUILIBRIUM_TIME_BOSON, EQUILIBRIUM_TIME_CLASSICAL,
    EQUILIBRIUM_TIME_CORRELATED,
    DEFAULT_FLICKER_NOISE_FACTOR, DEFAULT_QUANTUM_NOISE_FACTOR, DEFAULT_ENVIRONMENT_NOISE_FACTOR,
    FERMION_COEFF, BOSON_COEFF, CORRELATION_COEFF, QUASIPARTICLE_COEFF,
    GAMMA_BASE_COEFF, N_LOG_DIVISOR,
    HAS_CUPY, HAS_NUMBA_CUDA
)
from .exporter import ResultExporter

# Условный импорт CuPy
if HAS_CUPY:
    import cupy as cp
else:
    cp = None

logger = logging.getLogger(__name__)

class GPUInformationMeasurementSimulator:
    """Симулятор измерения информации с поддержкой GPU"""
    
    def __init__(self, temperature: float = 300.0, use_gpu: bool = True, 
                 suppress_logging: bool = False):
        """
        Инициализация с поддержкой GPU
        
        Parameters:
        - temperature: температура системы в Кельвинах
        - use_gpu: использовать ли GPU если доступен
        - suppress_logging: подавить логирование при инициализации
        """
        if temperature <= 0:
            warnings.warn(f"Температура должна быть > 0, установлено {MIN_TEMPERATURE} K")
            temperature = MIN_TEMPERATURE
        
        self.hbar = HBAR
        self.kB = KB
        self.temperature = float(temperature)
        self.use_gpu = use_gpu and (HAS_CUPY or HAS_NUMBA_CUDA)
        self.exporter = ResultExporter()
        
        if not suppress_logging:
            if self.use_gpu:
                logger.info("🚀 Режим: GPU-ускорение активировано")
            else:
                logger.info("⚡ Режим: CPU (оптимизированный)")
    
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """Перемещает массив на GPU если доступен CuPy"""
        if self.use_gpu and HAS_CUPY:
            return cp.asarray(array)
        return array
    
    def _calculate_detector_response_time(self, stats_type_array: np.ndarray, 
                                        N_array: np.ndarray,
                                        custom_detector_time: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Расчет времени отклика детектора на основе типа системы
        
        Parameters:
        - stats_type_array: массив типов статистики (0=фермион, 1=бозон, 2=классический)
        - N_array: массив числа частиц
        - custom_detector_time: пользовательские значения (если указаны, используются вместо расчетных)
        
        Returns:
        - Массив времен отклика детектора
        """
        if custom_detector_time is not None:
            return np.maximum(custom_detector_time, MIN_DETECTOR_RESPONSE_TIME)
        
        xp = self._get_array_lib()
        detector_times = xp.zeros_like(stats_type_array, dtype=float)
        
        # Базовые времена для разных типов систем
        # Фермионы: быстрые детекторы (5 нс)
        # Бозоны: медленные детекторы (100 нс) - из-за необходимости измерения конденсата
        # Классические: стандартные детекторы (10 нс)
        
        # Зависимость от числа частиц: больше частиц -> больше время
        # Формула: t_detector = t_base * (1 + log(N/N0)), где N0 = 1000
        N0 = 1000.0
        log_factor = 1 + xp.log(1 + N_array / N0) / 10.0  # Мягкая зависимость
        
        # Фермионы: 5 нс базовое время
        mask_fermion = (stats_type_array == 0)
        detector_times = xp.where(mask_fermion, 
                                 DETECTOR_TIME_FERMION * log_factor,
                                 detector_times)
        
        # Бозоны: 100 нс базовое время (медленнее из-за конденсата)
        mask_boson = (stats_type_array == 1)
        detector_times = xp.where(mask_boson,
                                 DETECTOR_TIME_BOSON * log_factor,
                                 detector_times)
        
        # Классические: 10 нс базовое время
        mask_classical = (stats_type_array == 2)
        detector_times = xp.where(mask_classical,
                                 DETECTOR_TIME_CLASSICAL * log_factor,
                                 detector_times)
        
        return xp.maximum(detector_times, MIN_DETECTOR_RESPONSE_TIME)
    
    def _calculate_equilibrium_time(self, stats_type_array: np.ndarray,
                                   T_c_array: np.ndarray,
                                   N_array: np.ndarray,
                                   U_array: Optional[np.ndarray] = None,
                                   custom_equilibrium_time: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Расчет времени установления равновесия на основе типа системы и параметров
        
        Parameters:
        - stats_type_array: массив типов статистики
        - T_c_array: массив критических температур
        - N_array: массив числа частиц
        - U_array: массив параметров взаимодействия (для коррелированных систем)
        - custom_equilibrium_time: пользовательские значения
        
        Returns:
        - Массив времен установления равновесия
        """
        if custom_equilibrium_time is not None:
            return np.maximum(custom_equilibrium_time, MIN_EQUILIBRIUM_TIME)
        
        xp = self._get_array_lib()
        equilibrium_times = xp.zeros_like(stats_type_array, dtype=float)
        
        # Базовые времена для разных типов систем
        # Фермионы: быстрое равновесие (1 нс)
        # Бозоны: медленное равновесие (1 мкс) - из-за конденсации
        # Классические: быстрое равновесие (1 нс)
        # Коррелированные: очень быстрое (0.5 нс) - из-за сильных корреляций
        
        # Зависимость от температуры: ниже T_c -> медленнее равновесие
        T_ratio = T_c_array / self.temperature
        temp_factor = 1 + 0.5 * xp.exp(-T_ratio)  # При T << T_c: фактор ~ 1.5
        
        # Зависимость от числа частиц: больше частиц -> медленнее равновесие
        N_factor = 1 + 0.1 * xp.log(1 + N_array / 1000.0)
        
        # Фермионы: 1 нс базовое время
        mask_fermion = (stats_type_array == 0)
        base_time_fermion = EQUILIBRIUM_TIME_FERMION
        equilibrium_times = xp.where(mask_fermion,
                                     base_time_fermion * temp_factor * N_factor,
                                     equilibrium_times)
        
        # Бозоны: 1 мкс базовое время (медленнее из-за конденсации)
        mask_boson = (stats_type_array == 1)
        base_time_boson = EQUILIBRIUM_TIME_BOSON
        equilibrium_times = xp.where(mask_boson,
                                     base_time_boson * temp_factor * N_factor,
                                     equilibrium_times)
        
        # Классические: 1 нс базовое время
        mask_classical = (stats_type_array == 2)
        base_time_classical = EQUILIBRIUM_TIME_CLASSICAL
        equilibrium_times = xp.where(mask_classical,
                                     base_time_classical * temp_factor * N_factor,
                                     equilibrium_times)
        
        # Коррелированные системы (U > 0): очень быстрое равновесие
        if U_array is not None:
            mask_correlated = (U_array > 0.1)
            base_time_correlated = EQUILIBRIUM_TIME_CORRELATED
            # Сильные корреляции ускоряют равновесие
            U_factor = 1 / (1 + U_array)  # При U=0.8: фактор ~ 0.56
            equilibrium_times = xp.where(mask_correlated,
                                        base_time_correlated * U_factor * temp_factor,
                                        equilibrium_times)
        
        return xp.maximum(equilibrium_times, MIN_EQUILIBRIUM_TIME)
    
    def _calibrate_noise_parameters(self, stats_type_array: np.ndarray,
                                   N_array: np.ndarray,
                                   T_c_array: np.ndarray,
                                   delta_I_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Калибровка параметров шума на основе экспериментальных данных
        
        Экспериментальные данные:
        - Ферми-газы: умеренный шум (shot noise ~ 0.1, thermal ~ 1K)
        - Бозе-конденсаты: низкий шум (shot noise ~ 0.02, thermal ~ 0.01K)
        - Коррелированные системы: высокий шум (shot noise ~ 0.2, thermal ~ 5K)
        - Классические: минимальный шум (shot noise ~ 0.05, thermal ~ 0.1K)
        
        Parameters:
        - stats_type_array: массив типов статистики
        - N_array: массив числа частиц
        - T_c_array: массив критических температур
        - delta_I_array: массив количества информации
        
        Returns:
        - Словарь с калиброванными параметрами шума
        """
        xp = self._get_array_lib()
        n = len(stats_type_array)
        
        # Инициализация массивов
        flicker_noise = xp.full(n, DEFAULT_FLICKER_NOISE_FACTOR)
        quantum_noise = xp.full(n, DEFAULT_QUANTUM_NOISE_FACTOR)
        environment_noise = xp.full(n, DEFAULT_ENVIRONMENT_NOISE_FACTOR)
        
        # Калибровка на основе типа системы
        # Фермионы: умеренный 1/f шум, повышенный квантовый шум
        mask_fermion = (stats_type_array == 0)
        flicker_noise = xp.where(mask_fermion, 0.08, flicker_noise)  # Увеличенный 1/f
        quantum_noise = xp.where(mask_fermion, 0.03, quantum_noise)  # Повышенный квантовый
        
        # Бозоны: низкий 1/f шум, минимальный квантовый шум
        mask_boson = (stats_type_array == 1)
        flicker_noise = xp.where(mask_boson, 0.02, flicker_noise)  # Низкий 1/f
        quantum_noise = xp.where(mask_boson, 0.01, quantum_noise)  # Минимальный квантовый
        environment_noise = xp.where(mask_boson, 0.01, environment_noise)  # Низкий окружения
        
        # Классические: стандартные значения
        # (уже установлены по умолчанию)
        
        # Зависимость от числа частиц: больше частиц -> больше шум
        N_factor = 1 + 0.05 * xp.log(1 + N_array / 10000.0)
        flicker_noise = flicker_noise * N_factor
        quantum_noise = quantum_noise * N_factor
        
        # Зависимость от температуры: ниже T_c -> меньше шум (для бозонов)
        T_ratio = T_c_array / self.temperature
        temp_factor = xp.where(T_ratio > 1, 0.8, 1.0)  # При T < T_c: меньше шум
        flicker_noise = flicker_noise * temp_factor
        environment_noise = environment_noise * temp_factor
        
        # Зависимость от количества информации: больше ΔI -> больше квантовый шум
        delta_I_factor = 1 + 0.1 * xp.sqrt(delta_I_array / 10.0)
        quantum_noise = quantum_noise * delta_I_factor
        
        return {
            'flicker_noise_factor': flicker_noise,
            'quantum_noise_factor': quantum_noise,
            'environment_noise_factor': environment_noise
        }
    
    def to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """Возвращает массив на CPU"""
        if self.use_gpu and HAS_CUPY:
            return cp.asnumpy(array)
        return array
    
    def _get_array_lib(self):
        """Возвращает библиотеку массивов (CuPy или NumPy)"""
        if self.use_gpu and HAS_CUPY:
            return cp
        return np
    
    def _calculate_decoherence_factor(self, T1_array: np.ndarray, T2_array: np.ndarray, 
                                     measurement_time: np.ndarray) -> np.ndarray:
        """
        Расчет фактора декогеренции
        
        Parameters:
        - T1_array: массив времен релаксации (энергетическая декогеренция)
        - T2_array: массив времен дефазировки (фазовая декогеренция)
        - measurement_time: массив времен измерения
        
        Returns:
        - Фактор декогеренции (1.0 = нет декогеренции, >1.0 = увеличение времени из-за декогеренции)
        """
        xp = self._get_array_lib()
        
        # Экспоненциальное затухание когерентности
        # exp(-t/T1) для энергетической декогеренции
        # exp(-t/T2) для фазовой декогеренции
        # Используем минимум из T1 и T2 для консервативной оценки
        T_effective = xp.minimum(T1_array, T2_array)
        
        # Фактор декогеренции: чем больше отношение t/T, тем больше влияние
        # Используем формулу: 1 + (t/T)^alpha, где alpha ~ 1-2
        decoherence_ratio = measurement_time / T_effective
        
        # Фактор увеличивает время измерения при декогеренции
        # При t << T: фактор ~ 1 (нет влияния)
        # При t ~ T: фактор ~ 2 (умеренное влияние)
        # При t >> T: фактор ~ 1 + (t/T) (сильное влияние)
        decoherence_factor = 1 + xp.tanh(decoherence_ratio) * (1 + 0.5 * decoherence_ratio)
        
        return decoherence_factor
    
    def _calculate_noise_factor(self, temperature: float, noise_temperature: np.ndarray,
                               shot_noise_factor: np.ndarray, technical_noise: np.ndarray,
                               delta_I_array: np.ndarray,
                               flicker_noise_factor: Optional[np.ndarray] = None,
                               quantum_noise_factor: Optional[np.ndarray] = None,
                               environment_noise_factor: Optional[np.ndarray] = None,
                               measurement_time: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Расчет фактора шума с учетом всех источников
        
        Parameters:
        - temperature: температура системы
        - noise_temperature: эффективная температура шума
        - shot_noise_factor: фактор дробового шума
        - technical_noise: фактор технического шума
        - delta_I_array: массив количества информации
        - flicker_noise_factor: фактор 1/f шума (фликкер-шум)
        - quantum_noise_factor: фактор квантового шума измерения
        - environment_noise_factor: фактор шума окружения
        - measurement_time: массив времен измерения (для 1/f шума)
        
        Returns:
        - Фактор шума (1.0 = нет шума, >1.0 = увеличение времени из-за шума)
        """
        xp = self._get_array_lib()
        
        # Тепловой шум (Johnson-Nyquist): пропорционален sqrt(T_noise/T)
        thermal_noise = xp.sqrt(1 + noise_temperature / temperature)
        
        # Дробовой шум (Shot noise): пропорционален sqrt(N), где N - число частиц/событий
        # Для информации: shot_noise ~ sqrt(delta_I)
        shot_noise = 1 + shot_noise_factor * xp.sqrt(delta_I_array)
        
        # Технический шум: аддитивный фактор
        tech_noise = 1 + technical_noise
        
        # 1/f шум (фликкер-шум): зависит от частоты/времени измерения
        # Формула: 1 + α × log(1 + t/t0), где t0 - характерное время
        if flicker_noise_factor is not None:
            if measurement_time is not None:
                # 1/f шум увеличивается с временем измерения
                t0 = 1e-9  # Характерное время 1 нс
                flicker_noise = 1 + flicker_noise_factor * xp.log(1 + measurement_time / t0)
            else:
                # Если время не указано, используем константу
                flicker_noise = 1 + flicker_noise_factor
        else:
            flicker_noise = xp.ones_like(delta_I_array)
        
        # Квантовый шум измерения: фундаментальный квантовый предел
        # Пропорционален sqrt(ℏ/ΔE) для квантовых систем
        if quantum_noise_factor is not None:
            # Квантовый шум зависит от неопределенности измерения
            quantum_noise = 1 + quantum_noise_factor * xp.sqrt(delta_I_array)
        else:
            quantum_noise = xp.ones_like(delta_I_array)
        
        # Шум окружения: внешние возмущения
        if environment_noise_factor is not None:
            env_noise = 1 + environment_noise_factor
        else:
            env_noise = xp.ones_like(delta_I_array)
        
        # Общий фактор шума: произведение всех компонент
        noise_factor = thermal_noise * shot_noise * tech_noise * flicker_noise * quantum_noise * env_noise
        
        return noise_factor
    
    def gamma_total_batch(self, T_c_array: np.ndarray, N_array: np.ndarray, 
                         stats_type_array: np.ndarray, 
                         T_F_array: Optional[np.ndarray] = None, 
                         T_c_bose_array: Optional[np.ndarray] = None, 
                         U_array: Optional[np.ndarray] = None,
                         W_array: Optional[np.ndarray] = None, 
                         T_star_array: Optional[np.ndarray] = None, 
                         m_star_array: Optional[np.ndarray] = None, 
                         m_array: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Векторизованный расчет gamma_total для массивов данных
        
        Parameters:
        - T_c_array: массив критических температур
        - N_array: массив чисел частиц
        - stats_type_array: массив типов статистики (0=фермион, 1=бозон, 2=классический)
        - T_F_array: массив температур Ферми (для фермионов)
        - T_c_bose_array: массив температур конденсации (для бозонов)
        - U_array: массив энергий взаимодействия
        - W_array: массив ширин зон
        - T_star_array: массив характерных температур
        - m_star_array: массив эффективных масс
        - m_array: массив масс
        
        Returns:
        - Массив значений gamma_total
        """
        # Валидация входных данных
        n = len(T_c_array)
        if len(N_array) != n or len(stats_type_array) != n:
            raise ValueError("Все массивы должны иметь одинаковую длину")
        
        # Инициализация значений по умолчанию
        if T_F_array is None: 
            T_F_array = np.zeros(n)
        if T_c_bose_array is None: 
            T_c_bose_array = T_c_array.copy()
        if U_array is None: 
            U_array = np.zeros(n)
        if W_array is None: 
            W_array = np.ones(n)
        if T_star_array is None: 
            T_star_array = np.full(n, DEFAULT_T_STAR)
        if m_star_array is None: 
            m_star_array = np.ones(n)
        if m_array is None: 
            m_array = np.ones(n)
        
        # Валидация значений
        T_c_array = np.maximum(T_c_array, MIN_TEMPERATURE)
        T_star_array = np.maximum(T_star_array, MIN_TEMPERATURE)
        W_array = np.maximum(W_array, 1e-10)  # Избегаем деления на ноль
        m_array = np.maximum(m_array, 1e-10)  # Избегаем деления на ноль
        
        # Преобразование в GPU массивы если доступно
        xp = self._get_array_lib()
        if self.use_gpu and HAS_CUPY:
            T_c_array = self.to_gpu(T_c_array)
            N_array = self.to_gpu(N_array)
            stats_type_array = self.to_gpu(stats_type_array)
            T_F_array = self.to_gpu(T_F_array)
            T_c_bose_array = self.to_gpu(T_c_bose_array)
            U_array = self.to_gpu(U_array)
            W_array = self.to_gpu(W_array)
            T_star_array = self.to_gpu(T_star_array)
            m_star_array = self.to_gpu(m_star_array)
            m_array = self.to_gpu(m_array)
        
        # Векторизованные вычисления
        gamma_b = self._gamma_base_vectorized(self.temperature, T_c_array, N_array)
        
        # Статистический фактор (унифицированная логика для CPU и GPU)
        stats_f = xp.where(stats_type_array == 0,  # FERMION
                          1 + FERMION_COEFF * (T_F_array / self.temperature),
                          xp.where(stats_type_array == 1,  # BOSON
                                 1 + BOSON_COEFF * (T_c_bose_array / self.temperature),
                                 1.0))  # CLASSICAL
        
        # Корреляции и квазичастицы
        corr_f = 1 + CORRELATION_COEFF * (U_array / W_array) * xp.exp(-self.temperature / T_star_array)
        qp_f = 1 + QUASIPARTICLE_COEFF * (m_star_array / m_array - 1)
        
        result = gamma_b * stats_f * corr_f * qp_f
        return self.to_cpu(result) if self.use_gpu and HAS_CUPY else result
    
    def _gamma_base_vectorized(self, T: float, T_c_array: np.ndarray, 
                              N_array: np.ndarray, N_Q: float = DEFAULT_N_Q) -> np.ndarray:
        """Векторизованная версия gamma_base"""
        xp = self._get_array_lib()
        T_safe = max(T, MIN_TEMPERATURE)
        T_ratio = T_c_array / T_safe
        numerator = GAMMA_BASE_COEFF * (T_ratio**1.5) * xp.log(1 + N_array / N_LOG_DIVISOR)
        denominator = 1 + (N_array / N_Q)**4
        return 1 + numerator / denominator
    
    def main_equation_batch(self, delta_I_array: np.ndarray, T_c_array: np.ndarray, 
                           N_array: np.ndarray, stats_type_array: np.ndarray,
                           delta_E_array: Optional[np.ndarray] = None, 
                           T_F_array: Optional[np.ndarray] = None, 
                           T_c_bose_array: Optional[np.ndarray] = None, 
                           U_array: Optional[np.ndarray] = None, 
                           W_array: Optional[np.ndarray] = None, 
                           T_star_array: Optional[np.ndarray] = None, 
                           m_star_array: Optional[np.ndarray] = None, 
                           m_array: Optional[np.ndarray] = None,
                           T1_array: Optional[np.ndarray] = None,
                           T2_array: Optional[np.ndarray] = None,
                           noise_temperature_array: Optional[np.ndarray] = None,
                           shot_noise_factor_array: Optional[np.ndarray] = None,
                           technical_noise_array: Optional[np.ndarray] = None,
                           flicker_noise_factor_array: Optional[np.ndarray] = None,
                           quantum_noise_factor_array: Optional[np.ndarray] = None,
                           environment_noise_factor_array: Optional[np.ndarray] = None,
                           equilibrium_time_array: Optional[np.ndarray] = None,
                           detector_response_time_array: Optional[np.ndarray] = None,
                           include_decoherence: bool = True,
                           include_noise: bool = True) -> np.ndarray:
        """
        Векторизованный расчет основного уравнения для массивов
        
        Returns:
        - Массив минимальных времен измерения
        """
        # Валидация входных данных
        n = len(T_c_array)
        if len(delta_I_array) != n or len(N_array) != n or len(stats_type_array) != n:
            raise ValueError("Все массивы должны иметь одинаковую длину")
        
        if delta_E_array is None:
            delta_E_array = np.full(n, DEFAULT_DELTA_E)
        
        # Валидация значений
        delta_I_array = np.maximum(delta_I_array, 1e-10)  # Избегаем деления на ноль
        delta_E_array = np.maximum(delta_E_array, MIN_DELTA_E)
        
        # Расчет gamma_total для всех точек
        gamma_tot = self.gamma_total_batch(
            T_c_array=T_c_array, 
            N_array=N_array, 
            stats_type_array=stats_type_array,
            T_F_array=T_F_array,
            T_c_bose_array=T_c_bose_array,
            U_array=U_array,
            W_array=W_array,
            T_star_array=T_star_array,
            m_star_array=m_star_array,
            m_array=m_array
        )
        
        # Фундаментальный предел
        fundamental = self.hbar / (self.kB * self.temperature)
        
        # Технические ограничения
        xp = self._get_array_lib()
        tau_tech = xp.full(n, DEFAULT_TAU_TECH)
        tau_landauer = xp.full(n, self.hbar / (2 * self.kB * self.temperature))
        
        # Преобразуем массивы на GPU если нужно
        if self.use_gpu and HAS_CUPY:
            delta_E_array = self.to_gpu(delta_E_array)
            delta_I_array = self.to_gpu(delta_I_array)
            gamma_tot = self.to_gpu(gamma_tot)
        
        tau_Q = self.hbar / delta_E_array
        
        # Максимальное из всех пределов
        max_tau = xp.maximum(tau_tech, xp.maximum(tau_landauer, tau_Q))
        
        # Основное уравнение (без учета декогеренции и шума)
        right_side = fundamental * gamma_tot + max_tau
        min_tau_ideal = right_side / delta_I_array
        
        # Инициализация факторов декогеренции и шума
        decoherence_factor = xp.ones(n)
        noise_factor = xp.ones(n)
        
        # Расчет декогеренции
        if include_decoherence:
            if T1_array is None:
                T1_array = np.full(n, DEFAULT_T1)
            if T2_array is None:
                T2_array = np.full(n, DEFAULT_T2)
            
            # Валидация времен декогеренции
            T1_array = np.maximum(T1_array, MIN_T1)
            T2_array = np.maximum(T2_array, MIN_T2)
            
            if self.use_gpu and HAS_CUPY:
                T1_array = self.to_gpu(T1_array)
                T2_array = self.to_gpu(T2_array)
            
            # Итеративный расчет: декогеренция зависит от времени измерения
            # Используем итеративный подход для самосогласованности
            min_tau_current = min_tau_ideal
            for iteration in range(3):  # Несколько итераций для сходимости
                decoherence_factor = self._calculate_decoherence_factor(
                    T1_array, T2_array, min_tau_current
                )
                min_tau_current = min_tau_ideal * decoherence_factor
        
        # Расчет шума
        if include_noise:
            if noise_temperature_array is None:
                noise_temperature_array = np.full(n, DEFAULT_NOISE_TEMPERATURE)
            if shot_noise_factor_array is None:
                shot_noise_factor_array = np.full(n, DEFAULT_SHOT_NOISE_FACTOR)
            if technical_noise_array is None:
                technical_noise_array = np.full(n, DEFAULT_TECHNICAL_NOISE)
            
            # Калибровка параметров шума на основе типа системы (если не указаны явно)
            if flicker_noise_factor_array is None or quantum_noise_factor_array is None or environment_noise_factor_array is None:
                calibrated_noise = self._calibrate_noise_parameters(
                    stats_type_array,
                    N_array,
                    T_c_array,
                    delta_I_array
                )
                if flicker_noise_factor_array is None:
                    flicker_noise_factor_array = calibrated_noise['flicker_noise_factor']
                if quantum_noise_factor_array is None:
                    quantum_noise_factor_array = calibrated_noise['quantum_noise_factor']
                if environment_noise_factor_array is None:
                    environment_noise_factor_array = calibrated_noise['environment_noise_factor']
            
            if self.use_gpu and HAS_CUPY:
                noise_temperature_array = self.to_gpu(noise_temperature_array)
                shot_noise_factor_array = self.to_gpu(shot_noise_factor_array)
                technical_noise_array = self.to_gpu(technical_noise_array)
                if flicker_noise_factor_array is not None:
                    flicker_noise_factor_array = self.to_gpu(flicker_noise_factor_array)
                if quantum_noise_factor_array is not None:
                    quantum_noise_factor_array = self.to_gpu(quantum_noise_factor_array)
                if environment_noise_factor_array is not None:
                    environment_noise_factor_array = self.to_gpu(environment_noise_factor_array)
            
            # Используем текущее время для расчета 1/f шума
            noise_factor = self._calculate_noise_factor(
                self.temperature,
                noise_temperature_array,
                shot_noise_factor_array,
                technical_noise_array,
                delta_I_array,
                flicker_noise_factor_array,
                quantum_noise_factor_array,
                environment_noise_factor_array,
                min_tau_ideal * decoherence_factor  # Используем время до учета шума
            )
        
        # Финальное время измерения с учетом декогеренции и шума
        min_tau = min_tau_ideal * decoherence_factor * noise_factor
        
        # Расчет времени установления равновесия (автоматически на основе типа системы)
        if equilibrium_time_array is None:
            equilibrium_time_array = self._calculate_equilibrium_time(
                stats_type_array,
                T_c_array,
                N_array,
                U_array
            )
        else:
            equilibrium_time_array = np.maximum(equilibrium_time_array, MIN_EQUILIBRIUM_TIME)
        
        if self.use_gpu and HAS_CUPY:
            equilibrium_time_array = self.to_gpu(equilibrium_time_array)
        min_tau = min_tau + equilibrium_time_array
        
        # Расчет времени отклика детектора (автоматически на основе типа системы)
        if detector_response_time_array is None:
            detector_response_time_array = self._calculate_detector_response_time(
                stats_type_array,
                N_array
            )
        else:
            detector_response_time_array = np.maximum(detector_response_time_array, MIN_DETECTOR_RESPONSE_TIME)
        
        if self.use_gpu and HAS_CUPY:
            detector_response_time_array = self.to_gpu(detector_response_time_array)
        # Время отклика детектора - это минимальное время, которое добавляется
        min_tau = xp.maximum(min_tau, detector_response_time_array)
        
        return self.to_cpu(min_tau) if self.use_gpu and HAS_CUPY else min_tau

    def simulate_single_system(self, delta_I: float = 1.0, T_c: float = 100.0, 
                              N: float = 1000.0, stats_type: int = 0, 
                              T_F: float = 5000.0, T_c_bose: Optional[float] = None, 
                              U: float = 0.0, W: float = 1.0, T_star: float = DEFAULT_T_STAR,
                              m_star: float = 1.0, m: float = 1.0, 
                              delta_E: float = DEFAULT_DELTA_E,
                              T1: Optional[float] = None,
                              T2: Optional[float] = None,
                              noise_temperature: Optional[float] = None,
                              shot_noise_factor: Optional[float] = None,
                              technical_noise: Optional[float] = None,
                              flicker_noise_factor: Optional[float] = None,
                              quantum_noise_factor: Optional[float] = None,
                              environment_noise_factor: Optional[float] = None,
                              equilibrium_time: Optional[float] = None,
                              detector_response_time: Optional[float] = None,
                              include_decoherence: bool = True,
                              include_noise: bool = True,
                              export_results: bool = True) -> Tuple[float, Union[str, Dict]]:
        """
        Симуляция одиночной системы с экспортом результатов
        
        Returns:
        - Кортеж (минимальное время измерения, имя файла или словарь результатов)
        """
        # Валидация входных данных
        if delta_I <= 0:
            raise ValueError("delta_I должно быть > 0")
        if T_c <= 0:
            raise ValueError("T_c должно быть > 0")
        if N <= 0:
            raise ValueError("N должно быть > 0")
        if W <= 0:
            raise ValueError("W должно быть > 0")
        if m <= 0:
            raise ValueError("m должно быть > 0")
        if delta_E <= 0:
            raise ValueError("delta_E должно быть > 0")
        if stats_type not in [0, 1, 2]:
            raise ValueError("stats_type должен быть 0 (фермион), 1 (бозон) или 2 (классический)")
        
        if T_c_bose is None:
            T_c_bose = T_c
        
        # Установка значений по умолчанию для декогеренции и шума
        if T1 is None:
            T1 = DEFAULT_T1
        if T2 is None:
            T2 = DEFAULT_T2
        if noise_temperature is None:
            noise_temperature = DEFAULT_NOISE_TEMPERATURE
        if shot_noise_factor is None:
            shot_noise_factor = DEFAULT_SHOT_NOISE_FACTOR
        if technical_noise is None:
            technical_noise = DEFAULT_TECHNICAL_NOISE
        # Параметры шума будут автоматически калиброваны, если не указаны
        # equilibrium_time и detector_response_time будут автоматически рассчитаны, если не указаны
        
        # Валидация параметров декогеренции и шума
        T1 = max(T1, MIN_T1)
        T2 = max(T2, MIN_T2)
        noise_temperature = max(noise_temperature, 0.0)
        shot_noise_factor = max(shot_noise_factor, 0.0)
        technical_noise = max(technical_noise, 0.0)
        if flicker_noise_factor is not None:
            flicker_noise_factor = max(flicker_noise_factor, 0.0)
        if quantum_noise_factor is not None:
            quantum_noise_factor = max(quantum_noise_factor, 0.0)
        if environment_noise_factor is not None:
            environment_noise_factor = max(environment_noise_factor, 0.0)
        if equilibrium_time is not None:
            equilibrium_time = max(equilibrium_time, MIN_EQUILIBRIUM_TIME)
        if detector_response_time is not None:
            detector_response_time = max(detector_response_time, MIN_DETECTOR_RESPONSE_TIME)
        
        # Расчет
        min_tau = self.main_equation_batch(
            delta_I_array=np.array([delta_I]),
            T_c_array=np.array([T_c]),
            N_array=np.array([N]),
            stats_type_array=np.array([stats_type]),
            T_F_array=np.array([T_F]),
            T_c_bose_array=np.array([T_c_bose]),
            U_array=np.array([U]),
            W_array=np.array([W]),
            T_star_array=np.array([T_star]),
            m_star_array=np.array([m_star]),
            m_array=np.array([m]),
            delta_E_array=np.array([delta_E]),
            T1_array=np.array([T1]) if include_decoherence else None,
            T2_array=np.array([T2]) if include_decoherence else None,
            noise_temperature_array=np.array([noise_temperature]) if include_noise else None,
            shot_noise_factor_array=np.array([shot_noise_factor]) if include_noise else None,
            technical_noise_array=np.array([technical_noise]) if include_noise else None,
            flicker_noise_factor_array=np.array([flicker_noise_factor]) if (include_noise and flicker_noise_factor is not None) else None,
            quantum_noise_factor_array=np.array([quantum_noise_factor]) if (include_noise and quantum_noise_factor is not None) else None,
            environment_noise_factor_array=np.array([environment_noise_factor]) if (include_noise and environment_noise_factor is not None) else None,
            equilibrium_time_array=np.array([equilibrium_time]) if equilibrium_time is not None else None,
            detector_response_time_array=np.array([detector_response_time]) if detector_response_time is not None else None,
            include_decoherence=include_decoherence,
            include_noise=include_noise
        )
        
        # Расчет факторов сложности для отчета
        gamma_tot = self.gamma_total_batch(
            T_c_array=np.array([T_c]),
            N_array=np.array([N]),
            stats_type_array=np.array([stats_type]),
            T_F_array=np.array([T_F]),
            T_c_bose_array=np.array([T_c_bose]),
            U_array=np.array([U]),
            W_array=np.array([W]),
            T_star_array=np.array([T_star]),
            m_star_array=np.array([m_star]),
            m_array=np.array([m])
        )
        
        # Расчет факторов декогеренции и шума для отчета
        decoherence_factor_val = 1.0
        noise_factor_val = 1.0
        
        if include_decoherence:
            decoherence_factor_val = self._calculate_decoherence_factor(
                np.array([T1]), np.array([T2]), np.array([min_tau[0]])
            )[0]
        
        if include_noise:
            # Получаем калиброванные параметры шума, если не указаны
            if flicker_noise_factor is None or quantum_noise_factor is None or environment_noise_factor is None:
                calibrated = self._calibrate_noise_parameters(
                    np.array([stats_type]),
                    np.array([N]),
                    np.array([T_c]),
                    np.array([delta_I])
                )
                if flicker_noise_factor is None:
                    flicker_noise_factor = calibrated['flicker_noise_factor'][0]
                if quantum_noise_factor is None:
                    quantum_noise_factor = calibrated['quantum_noise_factor'][0]
                if environment_noise_factor is None:
                    environment_noise_factor = calibrated['environment_noise_factor'][0]
            
            noise_factor_val = self._calculate_noise_factor(
                self.temperature,
                np.array([noise_temperature]),
                np.array([shot_noise_factor]),
                np.array([technical_noise]),
                np.array([delta_I]),
                np.array([flicker_noise_factor]) if flicker_noise_factor is not None else None,
                np.array([quantum_noise_factor]) if quantum_noise_factor is not None else None,
                np.array([environment_noise_factor]) if environment_noise_factor is not None else None,
                np.array([min_tau[0]])
            )[0]
        
        # Расчет идеального времени (без декогеренции и шума) для сравнения
        min_tau_ideal = self.main_equation_batch(
            delta_I_array=np.array([delta_I]),
            T_c_array=np.array([T_c]),
            N_array=np.array([N]),
            stats_type_array=np.array([stats_type]),
            T_F_array=np.array([T_F]),
            T_c_bose_array=np.array([T_c_bose]),
            U_array=np.array([U]),
            W_array=np.array([W]),
            T_star_array=np.array([T_star]),
            m_star_array=np.array([m_star]),
            m_array=np.array([m]),
            delta_E_array=np.array([delta_E]),
            include_decoherence=False,
            include_noise=False
        )[0]
        
        # Подготовка результатов для экспорта
        results = {
            'min_measurement_time': min_tau[0],
            'min_measurement_time_ideal': min_tau_ideal,
            'parameters': {
                'temperature': self.temperature,
                'delta_I': delta_I,
                'T_c': T_c,
                'N': N,
                'stats_type': stats_type,
                'T_F': T_F,
                'T_c_bose': T_c_bose,
                'U': U,
                'W': W,
                'T_star': T_star,
                'm_star': m_star,
                'm': m,
                'delta_E': delta_E,
                'T1': T1 if include_decoherence else None,
                'T2': T2 if include_decoherence else None,
                'noise_temperature': noise_temperature if include_noise else None,
                'shot_noise_factor': shot_noise_factor if include_noise else None,
                'technical_noise': technical_noise if include_noise else None,
                'flicker_noise_factor': flicker_noise_factor if (include_noise and flicker_noise_factor is not None) else None,
                'quantum_noise_factor': quantum_noise_factor if (include_noise and quantum_noise_factor is not None) else None,
                'environment_noise_factor': environment_noise_factor if (include_noise and environment_noise_factor is not None) else None,
                'equilibrium_time': equilibrium_time if equilibrium_time is not None else None,
                'detector_response_time': detector_response_time if detector_response_time is not None else None
            },
            'gamma_factors': {
                'total': gamma_tot[0],
                'base': self._gamma_base_vectorized(self.temperature, np.array([T_c]), np.array([N]))[0],
                'statistics': self._calculate_statistics_factor(stats_type, T_F, T_c_bose),
                'correlations': 1 + CORRELATION_COEFF * (U / W) * np.exp(-self.temperature / T_star),
                'quasiparticles': 1 + QUASIPARTICLE_COEFF * (m_star / m - 1)
            },
            'decoherence': {
                'T1': T1 if include_decoherence else None,
                'T2': T2 if include_decoherence else None,
                'factor': decoherence_factor_val if include_decoherence else 1.0,
                'enabled': include_decoherence
            },
            'noise': {
                'temperature': noise_temperature if include_noise else None,
                'shot_noise_factor': shot_noise_factor if include_noise else None,
                'technical_noise': technical_noise if include_noise else None,
                'flicker_noise_factor': flicker_noise_factor if include_noise else None,
                'quantum_noise_factor': quantum_noise_factor if include_noise else None,
                'environment_noise_factor': environment_noise_factor if include_noise else None,
                'factor': noise_factor_val if include_noise else 1.0,
                'enabled': include_noise
            },
            'timing': {
                'equilibrium_time': equilibrium_time if equilibrium_time is not None else None,
                'detector_response_time': detector_response_time if detector_response_time is not None else None
            },
            'limits': {
                'fundamental': self.hbar / (self.kB * self.temperature),
                'technical': DEFAULT_TAU_TECH,
                'landauer': self.hbar / (2 * self.kB * self.temperature),
                'quantum': self.hbar / delta_E
            }
        }
        
        if export_results:
            filename = self.exporter.export_to_text(results)
            return min_tau[0], filename
        else:
            return min_tau[0], results

    def _calculate_statistics_factor(self, stats_type: int, T_F: float, T_c_bose: float) -> float:
        """Расчет статистического фактора"""
        if stats_type == 0:  # FERMION
            return 1 + FERMION_COEFF * (T_F / self.temperature)
        elif stats_type == 1:  # BOSON
            return 1 + BOSON_COEFF * (T_c_bose / self.temperature)
        else:  # CLASSICAL
            return 1.0


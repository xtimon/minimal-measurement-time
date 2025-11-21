#!/usr/bin/env python3
"""
План симуляций для всех сред с различными параметрами

Этот скрипт генерирует и выполняет комплексный план симуляций для анализа
влияния различных параметров на время измерения информации в различных физических системах.
"""

import numpy as np
import json
import os
import gzip
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import partial
from minimal_measurement_time import GPUInformationMeasurementSimulator


def _run_single_simulation(config: Dict[str, Any], use_gpu: bool) -> Optional[Dict[str, Any]]:
    """
    Выполнение одной симуляции (используется для параллельной обработки)

    Parameters:
    - config: Словарь с параметрами конфигурации
    - use_gpu: Использовать GPU

    Returns:
    - Словарь с результатами или None в случае ошибки
    """
    try:
        # Создание симулятора с нужной температурой
        sim = GPUInformationMeasurementSimulator(
            temperature=config["temperature"], use_gpu=use_gpu, suppress_logging=True
        )

        # Подготовка параметров
        sim_params = {
            "delta_I": config["delta_I"],
            "T_c": config["T_c"],
            "N": config["N"],
            "stats_type": config["stats_type"],
            "U": config["U"],
            "W": config["W"],
            "include_decoherence": config["include_decoherence"],
            "include_noise": config["include_noise"],
            "export_results": False,  # Не сохранять отдельные файлы
        }

        # Добавление специфичных параметров
        if config["stats_type"] == 0:  # Фермион
            sim_params["T_F"] = config["T_F"]
        elif config["stats_type"] == 1:  # Бозон
            sim_params["T_c_bose"] = config.get("T_c_bose", config["T_c"])

        # Параметры декогеренции
        if config.get("T1") is not None:
            sim_params["T1"] = config["T1"]
        if config.get("T2") is not None:
            sim_params["T2"] = config["T2"]

        # Выполнение симуляции
        measurement_time, result_dict = sim.simulate_single_system(**sim_params)

        # Формирование результата
        result = {
            "config": config,
            "measurement_time": float(measurement_time),
            "gamma_total": float(result_dict["gamma_factors"]["total"]),
            "decoherence_factor": float(result_dict["decoherence"]["factor"]),
            "noise_factor": float(result_dict["noise"]["factor"]),
            "min_measurement_time_ideal": float(result_dict["min_measurement_time_ideal"]),
        }

        return result

    except Exception as e:
        # Возвращаем None в случае ошибки (ошибка будет обработана в вызывающем коде)
        return None


class SimulationPlan:
    """Класс для создания и выполнения плана симуляций"""

    def __init__(self, output_dir: str = "results/simulation_plan"):
        """
        Инициализация плана симуляций

        Parameters:
        - output_dir: Директория для сохранения результатов
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        self.plan = []

    def generate_plan(self) -> List[Dict[str, Any]]:
        """
        Генерация плана симуляций для всех сред

        Returns:
        - Список словарей с параметрами симуляций
        """
        plan = []

        # Определение параметров для варьирования
        # 1. Количество бит информации (логарифмическая шкала)
        delta_I_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904, 9223372036854775808, 18446744073709551616]

        # 2. Количество частиц (логарифмическая шкала)
        N_values = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30]

        # 3. Температуры системы (различные диапазоны)
        temperature_values = [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]

        # 4. Критические температуры (зависят от типа системы)
        T_c_fermion = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]
        T_c_boson = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        T_c_classical = [10.0, 50.0, 100.0, 300.0, 500.0, 1000.0]

        # 5. Температуры Ферми (для фермионов)
        T_F_values = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]

        # 6. Параметры взаимодействия
        U_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        W_values = [0.5, 1.0, 2.0, 5.0]

        # 7. Параметры декогеренции
        T1_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # от 1 мкс до 10 мс
        T2_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]  # от 100 нс до 1 мс

        # Генерация плана для каждой среды

        # ========== ФЕРМИОНЫ (stats_type=0) ==========
        print("Генерация плана для ФЕРМИОНОВ...")
        fermion_configs = self._generate_fermion_configs(
            delta_I_values, N_values, temperature_values, T_c_fermion, T_F_values, U_values, W_values, T1_values, T2_values
        )
        plan.extend(fermion_configs)
        print(f"  Создано {len(fermion_configs)} конфигураций для фермионов")

        # ========== БОЗОНЫ (stats_type=1) ==========
        print("Генерация плана для БОЗОНОВ...")
        boson_configs = self._generate_boson_configs(
            delta_I_values, N_values, temperature_values, T_c_boson, U_values, W_values, T1_values, T2_values
        )
        plan.extend(boson_configs)
        print(f"  Создано {len(boson_configs)} конфигураций для бозонов")

        # ========== КЛАССИЧЕСКИЕ СИСТЕМЫ (stats_type=2) ==========
        print("Генерация плана для КЛАССИЧЕСКИХ СИСТЕМ...")
        classical_configs = self._generate_classical_configs(
            delta_I_values, N_values, temperature_values, T_c_classical, U_values, W_values, T1_values, T2_values
        )
        plan.extend(classical_configs)
        print(f"  Создано {len(classical_configs)} конфигураций для классических систем")

        self.plan = plan
        print(f"\nВсего создано {len(plan)} конфигураций для симуляций")
        return plan

    def _generate_fermion_configs(
        self, delta_I_values, N_values, temperature_values, T_c_values, T_F_values, U_values, W_values, T1_values, T2_values
    ) -> List[Dict[str, Any]]:
        """Генерация конфигураций для фермионов"""
        configs = []

        # Базовые конфигурации (без взаимодействия)
        for delta_I in delta_I_values:
            for N in N_values:
                for T in temperature_values:
                    for T_c in T_c_values:
                        for T_F in T_F_values:
                            # Пропускаем нефизические комбинации
                            if T_F < T_c:
                                continue

                            configs.append({
                                "stats_type": 0,  # Фермион
                                "delta_I": delta_I,
                                "N": N,
                                "temperature": T,
                                "T_c": T_c,
                                "T_F": T_F,
                                "U": 0.0,
                                "W": 1.0,
                                "T1": None,  # Автоматический расчет
                                "T2": None,
                                "include_decoherence": True,
                                "include_noise": True,
                                "description": f"Фермион: ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K, T_F={T_F}K"
                            })

        # Конфигурации с взаимодействием (выборочно, чтобы не перегружать)
        # Используем подмножество параметров для взаимодействующих систем
        delta_I_interaction = [1, 8, 64, 512]
        N_interaction = [1e3, 1e4, 1e5]
        T_interaction = [1.0, 10.0, 100.0, 300.0]
        T_c_interaction = [50.0, 100.0, 500.0, 1000.0]
        T_F_interaction = [1000.0, 5000.0, 10000.0]

        for delta_I in delta_I_interaction:
            for N in N_interaction:
                for T in T_interaction:
                    for T_c in T_c_interaction:
                        for T_F in T_F_interaction:
                            if T_F < T_c:
                                continue
                            for U in U_values[1:]:  # Пропускаем U=0 (уже есть выше)
                                for W in W_values:
                                    configs.append({
                                        "stats_type": 0,
                                        "delta_I": delta_I,
                                        "N": N,
                                        "temperature": T,
                                        "T_c": T_c,
                                        "T_F": T_F,
                                        "U": U,
                                        "W": W,
                                        "T1": None,
                                        "T2": None,
                                        "include_decoherence": True,
                                        "include_noise": True,
                                        "description": f"Фермион (взаимодействие): ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K, T_F={T_F}K, U={U}, W={W}"
                                    })

        # Конфигурации с явными параметрами декогеренции (выборочно)
        delta_I_decoherence = [1, 8, 64]
        N_decoherence = [1e3, 1e4, 1e5]
        T_decoherence = [0.01, 1.0, 100.0, 300.0]
        T_c_decoherence = [10.0, 100.0, 1000.0]
        T_F_decoherence = [1000.0, 10000.0]

        for delta_I in delta_I_decoherence:
            for N in N_decoherence:
                for T in T_decoherence:
                    for T_c in T_c_decoherence:
                        for T_F in T_F_decoherence:
                            if T_F < T_c:
                                continue
                            for T1 in T1_values:
                                for T2 in T2_values:
                                    if T2 >= T1:  # T2 обычно меньше T1
                                        continue
                                    configs.append({
                                        "stats_type": 0,
                                        "delta_I": delta_I,
                                        "N": N,
                                        "temperature": T,
                                        "T_c": T_c,
                                        "T_F": T_F,
                                        "U": 0.0,
                                        "W": 1.0,
                                        "T1": T1,
                                        "T2": T2,
                                        "include_decoherence": True,
                                        "include_noise": True,
                                        "description": f"Фермион (декогеренция): ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K, T_F={T_F}K, T1={T1:.2e}s, T2={T2:.2e}s"
                                    })

        return configs

    def _generate_boson_configs(
        self, delta_I_values, N_values, temperature_values, T_c_values, U_values, W_values, T1_values, T2_values
    ) -> List[Dict[str, Any]]:
        """Генерация конфигураций для бозонов"""
        configs = []

        # Базовые конфигурации
        for delta_I in delta_I_values:
            for N in N_values:
                for T in temperature_values:
                    for T_c in T_c_values:
                        # Для бозонов T_c_bose обычно близко к T_c
                        T_c_bose = T_c

                        configs.append({
                            "stats_type": 1,  # Бозон
                            "delta_I": delta_I,
                            "N": N,
                            "temperature": T,
                            "T_c": T_c,
                            "T_c_bose": T_c_bose,
                            "U": 0.0,
                            "W": 1.0,
                            "T1": None,
                            "T2": None,
                            "include_decoherence": True,
                            "include_noise": True,
                            "description": f"Бозон: ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K"
                        })

        # Конфигурации с взаимодействием (выборочно)
        delta_I_interaction = [1, 8, 64, 512]
        N_interaction = [1e3, 1e4, 1e5, 1e6]
        T_interaction = [0.001, 0.01, 0.1, 1.0, 10.0]
        T_c_interaction = [0.001, 0.01, 0.1, 1.0, 10.0]

        for delta_I in delta_I_interaction:
            for N in N_interaction:
                for T in T_interaction:
                    for T_c in T_c_interaction:
                        T_c_bose = T_c
                        for U in U_values[1:]:
                            for W in W_values:
                                configs.append({
                                    "stats_type": 1,
                                    "delta_I": delta_I,
                                    "N": N,
                                    "temperature": T,
                                    "T_c": T_c,
                                    "T_c_bose": T_c_bose,
                                    "U": U,
                                    "W": W,
                                    "T1": None,
                                    "T2": None,
                                    "include_decoherence": True,
                                    "include_noise": True,
                                    "description": f"Бозон (взаимодействие): ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K, U={U}, W={W}"
                                })

        # Конфигурации с декогеренцией (выборочно)
        delta_I_decoherence = [1, 8, 64]
        N_decoherence = [1e3, 1e4, 1e5, 1e6]
        T_decoherence = [0.001, 0.01, 1.0, 10.0]
        T_c_decoherence = [0.001, 0.01, 1.0, 10.0]

        for delta_I in delta_I_decoherence:
            for N in N_decoherence:
                for T in T_decoherence:
                    for T_c in T_c_decoherence:
                        T_c_bose = T_c
                        for T1 in T1_values:
                            for T2 in T2_values:
                                if T2 >= T1:
                                    continue
                                configs.append({
                                    "stats_type": 1,
                                    "delta_I": delta_I,
                                    "N": N,
                                    "temperature": T,
                                    "T_c": T_c,
                                    "T_c_bose": T_c_bose,
                                    "U": 0.0,
                                    "W": 1.0,
                                    "T1": T1,
                                    "T2": T2,
                                    "include_decoherence": True,
                                    "include_noise": True,
                                    "description": f"Бозон (декогеренция): ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K, T1={T1:.2e}s, T2={T2:.2e}s"
                                })

        return configs

    def _generate_classical_configs(
        self, delta_I_values, N_values, temperature_values, T_c_values, U_values, W_values, T1_values, T2_values
    ) -> List[Dict[str, Any]]:
        """Генерация конфигураций для классических систем"""
        configs = []

        # Базовые конфигурации
        for delta_I in delta_I_values:
            for N in N_values:
                for T in temperature_values:
                    for T_c in T_c_values:
                        configs.append({
                            "stats_type": 2,  # Классический
                            "delta_I": delta_I,
                            "N": N,
                            "temperature": T,
                            "T_c": T_c,
                            "U": 0.0,
                            "W": 1.0,
                            "T1": None,
                            "T2": None,
                            "include_decoherence": True,
                            "include_noise": True,
                            "description": f"Классический: ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K"
                        })

        # Конфигурации с взаимодействием (выборочно)
        delta_I_interaction = [1, 8, 64, 512]
        N_interaction = [1e2, 1e3, 1e4, 1e5]
        T_interaction = [10.0, 100.0, 300.0, 1000.0]
        T_c_interaction = [50.0, 100.0, 300.0, 500.0, 1000.0]

        for delta_I in delta_I_interaction:
            for N in N_interaction:
                for T in T_interaction:
                    for T_c in T_c_interaction:
                        for U in U_values[1:]:
                            for W in W_values:
                                configs.append({
                                    "stats_type": 2,
                                    "delta_I": delta_I,
                                    "N": N,
                                    "temperature": T,
                                    "T_c": T_c,
                                    "U": U,
                                    "W": W,
                                    "T1": None,
                                    "T2": None,
                                    "include_decoherence": True,
                                    "include_noise": True,
                                    "description": f"Классический (взаимодействие): ΔI={delta_I}, N={N:.0e}, T={T}K, T_c={T_c}K, U={U}, W={W}"
                                })

        return configs

    def save_plan(self, filename: str = None):
        """Сохранение плана симуляций в файл"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"simulation_plan_{timestamp}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.plan, f, indent=2, ensure_ascii=False)

        print(f"\nПлан сохранен в: {filename}")
        print(f"Всего конфигураций: {len(self.plan)}")
        return filename

    def execute_plan(
        self,
        max_simulations: int = None,
        use_gpu: bool = False,
        save_every: int = 100,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ):
        """
        Выполнение плана симуляций с поддержкой параллельной обработки

        Parameters:
        - max_simulations: Максимальное количество симуляций (None = все)
        - use_gpu: Использовать GPU (при использовании GPU параллелизм ограничен)
        - save_every: Сохранять промежуточные результаты каждые N симуляций
        - verbose: Выводить прогресс
        - n_jobs: Количество параллельных процессов (None = автоматически, все ядра)
        """
        if not self.plan:
            print("План не создан. Вызовите generate_plan() сначала.")
            return

        total = len(self.plan) if max_simulations is None else min(max_simulations, len(self.plan))
        
        # Определение количества процессов
        if n_jobs is None:
            n_jobs = cpu_count()
        
        # Для GPU используем меньше процессов или последовательную обработку
        if use_gpu:
            # GPU не может использоваться в нескольких процессах одновременно
            # Используем batch processing или ограничиваем параллелизм
            if n_jobs > 1:
                print("⚠️  Предупреждение: GPU не поддерживает параллельную обработку в нескольких процессах.")
                print("   Используется последовательная обработка с batch mode для GPU.")
                n_jobs = 1
                use_parallel = False
            else:
                use_parallel = False
        else:
            # Для CPU используем параллельную обработку
            use_parallel = n_jobs > 1

        print(f"\n{'='*70}")
        print(f"ВЫПОЛНЕНИЕ ПЛАНА СИМУЛЯЦИЙ")
        print(f"{'='*70}")
        print(f"Всего симуляций: {total}")
        print(f"GPU: {'Да' if use_gpu else 'Нет'}")
        print(f"Параллельная обработка: {'Да' if use_parallel else 'Нет'}")
        if use_parallel:
            print(f"Количество процессов: {n_jobs} (используются все доступные ядра)")
        elif not use_gpu:
            print(f"Количество процессов: {n_jobs} (последовательная обработка)")
        print(f"{'='*70}\n")

        if use_parallel:
            results = self._execute_plan_parallel(total, n_jobs, save_every, verbose)
        else:
            results = self._execute_plan_sequential(total, use_gpu, save_every, verbose)

        self.results = results

        print(f"\n{'='*70}")
        print(f"ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*70}")
        print(f"Успешно выполнено: {len(results)}/{total}")
        print(f"{'='*70}\n")

        return results

    def _execute_plan_sequential(
        self, total: int, use_gpu: bool, save_every: int, verbose: bool
    ) -> List[Dict[str, Any]]:
        """Последовательное выполнение плана (для GPU или однопоточного режима)"""
        results = []
        current_sim = 0

        for idx, config in enumerate(self.plan[:total]):
            try:
                if verbose and (idx + 1) % 10 == 0:
                    print(f"Прогресс: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)")

                # Выполнение одной симуляции
                result = _run_single_simulation(config, use_gpu)
                if result is not None:
                    results.append(result)
                    current_sim = idx + 1

                    # Промежуточное сохранение
                    if (idx + 1) % save_every == 0:
                        self._save_results(results, current_sim, total)

            except Exception as e:
                if verbose:
                    print(f"\nОшибка при симуляции {idx + 1}: {config.get('description', 'N/A')}")
                    print(f"  Ошибка: {str(e)}")
                continue

        # Финальное сохранение
        if results:
            self._save_results(results, current_sim, total)

        return results

    def _execute_plan_parallel(
        self, total: int, n_jobs: int, save_every: int, verbose: bool
    ) -> List[Dict[str, Any]]:
        """Параллельное выполнение плана (для CPU)"""
        results = []
        completed = 0
        configs_to_process = self.plan[:total]

        # Используем ProcessPoolExecutor для параллельной обработки
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Запускаем все задачи
            future_to_config = {
                executor.submit(_run_single_simulation, config, False): (idx, config)
                for idx, config in enumerate(configs_to_process)
            }

            # Обрабатываем результаты по мере их готовности
            for future in as_completed(future_to_config):
                idx, config = future_to_config[future]
                completed += 1

                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)

                    # Промежуточное сохранение
                    if completed % save_every == 0:
                        self._save_results(results, completed, total)

                    if verbose and completed % 10 == 0:
                        print(f"Прогресс: {completed}/{total} ({100*completed/total:.1f}%)")

                except Exception as e:
                    if verbose:
                        print(f"\nОшибка при симуляции {idx + 1}: {config.get('description', 'N/A')}")
                        print(f"  Ошибка: {str(e)}")

        # Финальное сохранение
        if results:
            self._save_results(results, completed, total)

        return results

    def _save_results(self, results: List[Dict], current: int, total: int):
        """Сохранение промежуточных результатов (с сжатием)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"results_{timestamp}_partial_{current}of{total}.json.gz")

        # Сохранение с gzip сжатием
        with gzip.open(filename, "wt", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if current % 100 == 0:
            # Получаем размер файла для информации
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"  Промежуточное сохранение: {filename} ({len(results)} результатов, {file_size:.2f} MB)")

    def save_final_results(self, filename: str = None):
        """Сохранение финальных результатов"""
        if not self.results:
            print("Нет результатов для сохранения.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"final_results_{timestamp}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\nФинальные результаты сохранены в: {filename}")
        print(f"Всего результатов: {len(self.results)}")

        # Создание сводной статистики
        self._create_summary_statistics(filename.replace(".json", "_summary.txt"))

        return filename

    def _create_summary_statistics(self, filename: str):
        """Создание сводной статистики по результатам"""
        if not self.results:
            return

        measurement_times = [r["measurement_time"] for r in self.results]
        gamma_totals = [r["gamma_total"] for r in self.results]

        # Группировка по типам систем
        by_stats_type = {0: [], 1: [], 2: []}
        for r in self.results:
            stats_type = r["config"]["stats_type"]
            by_stats_type[stats_type].append(r["measurement_time"])

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("СВОДНАЯ СТАТИСТИКА РЕЗУЛЬТАТОВ СИМУЛЯЦИЙ\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Всего симуляций: {len(self.results)}\n\n")

            f.write("ОБЩАЯ СТАТИСТИКА:\n")
            f.write(f"  Минимальное время: {np.min(measurement_times):.3e} с\n")
            f.write(f"  Максимальное время: {np.max(measurement_times):.3e} с\n")
            f.write(f"  Среднее время: {np.mean(measurement_times):.3e} с\n")
            f.write(f"  Медианное время: {np.median(measurement_times):.3e} с\n")
            f.write(f"  Стандартное отклонение: {np.std(measurement_times):.3e} с\n\n")

            f.write("СТАТИСТИКА ПО ТИПАМ СИСТЕМ:\n")
            stats_names = {0: "Фермионы", 1: "Бозоны", 2: "Классические"}
            for stats_type, times in by_stats_type.items():
                if times:
                    f.write(f"\n  {stats_names[stats_type]}:\n")
                    f.write(f"    Количество: {len(times)}\n")
                    f.write(f"    Минимум: {np.min(times):.3e} с\n")
                    f.write(f"    Максимум: {np.max(times):.3e} с\n")
                    f.write(f"    Среднее: {np.mean(times):.3e} с\n")
                    f.write(f"    Медиана: {np.median(times):.3e} с\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"Сводная статистика сохранена в: {filename}")


def main():
    """Основная функция для создания и выполнения плана симуляций"""
    print("=" * 70)
    print("ГЕНЕРАЦИЯ И ВЫПОЛНЕНИЕ ПЛАНА СИМУЛЯЦИЙ")
    print("=" * 70)

    # Создание плана
    plan = SimulationPlan()

    # Генерация плана
    print("\n1. ГЕНЕРАЦИЯ ПЛАНА СИМУЛЯЦИЙ")
    print("-" * 70)
    plan.generate_plan()

    # Сохранение плана
    print("\n2. СОХРАНЕНИЕ ПЛАНА")
    print("-" * 70)
    plan.save_plan()

    # Выполнение плана (можно ограничить для тестирования)
    print("\n3. ВЫПОЛНЕНИЕ ПЛАНА")
    print("-" * 70)
    print("ВНИМАНИЕ: Полный план может содержать тысячи симуляций.")
    print("Для тестирования рекомендуется ограничить количество.")
    print("\nДля выполнения всех симуляций используйте:")
    print("  plan.execute_plan(max_simulations=None)")
    print("\nДля тестирования используйте:")
    print("  plan.execute_plan(max_simulations=100)")

    # Запрашиваем у пользователя
    response = input("\nВыполнить все симуляции? (y/n, по умолчанию n): ").strip().lower()
    if response == "y":
        max_sims = None
    else:
        try:
            max_sims = int(input("Введите количество симуляций для выполнения (по умолчанию 100): ") or "100")
        except ValueError:
            max_sims = 100

    # Определение режима работы
    use_gpu_response = input("Использовать GPU? (y/n, по умолчанию n для параллельной обработки): ").strip().lower()
    use_gpu = use_gpu_response == "y"
    
    # Для CPU автоматически используем все ядра
    n_jobs = None if not use_gpu else 1  # Для GPU используем последовательную обработку
    
    plan.execute_plan(max_simulations=max_sims, use_gpu=use_gpu, save_every=100, n_jobs=n_jobs)

    # Сохранение финальных результатов
    print("\n4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 70)
    plan.save_final_results()

    print("\n" + "=" * 70)
    print("ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"\nРезультаты сохранены в: {plan.output_dir}")


if __name__ == "__main__":
    main()


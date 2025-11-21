#!/usr/bin/env python3
"""
Скрипт для анализа результатов симуляций

Этот скрипт анализирует результаты выполнения плана симуляций и создает
детальные отчеты для дальнейшего исследования.
"""

import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any

# Попытка импорта matplotlib (опционально)
try:
    import matplotlib
    matplotlib.use('Agg')  # Использовать backend без GUI
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Предупреждение: matplotlib не установлен. Графики не будут созданы.")


class SimulationAnalyzer:
    """Класс для анализа результатов симуляций"""

    def __init__(self, results_file: str, output_dir: str = "results/analysis"):
        """
        Инициализация анализатора

        Parameters:
        - results_file: Путь к JSON файлу с результатами
        - output_dir: Директория для сохранения результатов анализа
        """
        self.results_file = results_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        self.load_results()

    def load_results(self):
        """Загрузка результатов из файла (поддерживает сжатые .gz файлы)"""
        print(f"Загрузка результатов из: {self.results_file}")
        import gzip
        
        # Определяем, является ли файл сжатым
        if self.results_file.endswith('.gz'):
            with gzip.open(self.results_file, "rt", encoding="utf-8") as f:
                self.results = json.load(f)
        else:
            with open(self.results_file, "r", encoding="utf-8") as f:
                self.results = json.load(f)
        print(f"Загружено {len(self.results)} результатов")

    def analyze_by_parameter(self, parameter: str) -> Dict[str, Any]:
        """
        Анализ зависимости результатов от параметра

        Parameters:
        - parameter: Имя параметра для анализа (например, 'delta_I', 'N', 'temperature')

        Returns:
        - Словарь с результатами анализа
        """
        analysis = {
            "parameter": parameter,
            "values": [],
            "measurement_times": [],
            "gamma_totals": [],
            "stats_by_value": {}
        }

        # Группировка по значениям параметра
        grouped = {}
        for result in self.results:
            config = result["config"]
            if parameter in config:
                value = config[parameter]
                if value not in grouped:
                    grouped[value] = []
                grouped[value].append(result)

        # Анализ для каждого значения
        for value in sorted(grouped.keys()):
            times = [r["measurement_time"] for r in grouped[value]]
            gammas = [r["gamma_total"] for r in grouped[value]]

            analysis["values"].append(value)
            analysis["measurement_times"].append(times)
            analysis["gamma_totals"].append(gammas)

            analysis["stats_by_value"][value] = {
                "count": len(times),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "std": float(np.std(times)),
                "gamma_mean": float(np.mean(gammas)),
            }

        return analysis

    def analyze_by_stats_type(self) -> Dict[str, Any]:
        """Анализ результатов по типам систем"""
        stats_names = {0: "Фермионы", 1: "Бозоны", 2: "Классические"}
        analysis = {}

        for stats_type in [0, 1, 2]:
            type_results = [r for r in self.results if r["config"]["stats_type"] == stats_type]
            if not type_results:
                continue

            times = [r["measurement_time"] for r in type_results]
            gammas = [r["gamma_total"] for r in type_results]

            analysis[stats_names[stats_type]] = {
                "count": len(type_results),
                "measurement_time": {
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                    "mean": float(np.mean(times)),
                    "median": float(np.median(times)),
                    "std": float(np.std(times)),
                },
                "gamma_total": {
                    "min": float(np.min(gammas)),
                    "max": float(np.max(gammas)),
                    "mean": float(np.mean(gammas)),
                    "median": float(np.median(gammas)),
                },
            }

        return analysis

    def create_plots(self):
        """Создание графиков для визуализации результатов"""
        if not HAS_MATPLOTLIB:
            print("\nПропуск создания графиков (matplotlib не установлен)")
            print("Установите matplotlib для создания графиков: pip install matplotlib")
            return

        print("\nСоздание графиков...")

        # 1. Распределение времен измерения по типам систем
        self._plot_by_stats_type()

        # 2. Зависимость от количества бит
        self._plot_vs_delta_I()

        # 3. Зависимость от количества частиц
        self._plot_vs_N()

        # 4. Зависимость от температуры
        self._plot_vs_temperature()

        # 5. Корреляция между параметрами
        self._plot_correlations()

        print(f"Графики сохранены в: {self.output_dir}")

    def _plot_by_stats_type(self):
        """График распределения времен по типам систем"""
        stats_names = {0: "Фермионы", 1: "Бозоны", 2: "Классические"}
        data = {name: [] for name in stats_names.values()}

        for result in self.results:
            stats_type = result["config"]["stats_type"]
            name = stats_names[stats_type]
            data[name].append(result["measurement_time"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Гистограмма
        for name, times in data.items():
            if times:
                ax1.hist(np.log10(times), bins=30, alpha=0.6, label=name, edgecolor='black')
        ax1.set_xlabel("log₁₀(Время измерения, с)")
        ax1.set_ylabel("Частота")
        ax1.set_title("Распределение времен измерения по типам систем")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        box_data = [np.log10(times) for times in data.values() if times]
        box_labels = [name for name, times in data.items() if times]
        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_ylabel("log₁₀(Время измерения, с)")
        ax2.set_title("Сравнение времен измерения")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "by_stats_type.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_vs_delta_I(self):
        """График зависимости от количества бит"""
        stats_names = {0: "Фермионы", 1: "Бозоны", 2: "Классические"}
        
        fig, ax = plt.subplots(figsize=(12, 8))

        for stats_type in [0, 1, 2]:
            type_results = [r for r in self.results if r["config"]["stats_type"] == stats_type]
            if not type_results:
                continue

            delta_I_values = []
            mean_times = []
            std_times = []

            grouped = {}
            for result in type_results:
                delta_I = result["config"]["delta_I"]
                if delta_I not in grouped:
                    grouped[delta_I] = []
                grouped[delta_I].append(result["measurement_time"])

            for delta_I in sorted(grouped.keys()):
                times = grouped[delta_I]
                delta_I_values.append(delta_I)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))

            ax.errorbar(
                delta_I_values, mean_times, yerr=std_times,
                label=stats_names[stats_type], marker='o', capsize=5, capthick=2
            )

        ax.set_xlabel("Количество бит (ΔI)", fontsize=12)
        ax.set_ylabel("Время измерения (с)", fontsize=12)
        ax.set_title("Зависимость времени измерения от количества бит", fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "vs_delta_I.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_vs_N(self):
        """График зависимости от количества частиц"""
        stats_names = {0: "Фермионы", 1: "Бозоны", 2: "Классические"}
        
        fig, ax = plt.subplots(figsize=(12, 8))

        for stats_type in [0, 1, 2]:
            type_results = [r for r in self.results if r["config"]["stats_type"] == stats_type]
            if not type_results:
                continue

            N_values = []
            mean_times = []
            std_times = []

            grouped = {}
            for result in type_results:
                N = result["config"]["N"]
                if N not in grouped:
                    grouped[N] = []
                grouped[N].append(result["measurement_time"])

            for N in sorted(grouped.keys()):
                times = grouped[N]
                N_values.append(N)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))

            ax.errorbar(
                N_values, mean_times, yerr=std_times,
                label=stats_names[stats_type], marker='s', capsize=5, capthick=2
            )

        ax.set_xlabel("Количество частиц (N)", fontsize=12)
        ax.set_ylabel("Время измерения (с)", fontsize=12)
        ax.set_title("Зависимость времени измерения от количества частиц", fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "vs_N.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_vs_temperature(self):
        """График зависимости от температуры"""
        stats_names = {0: "Фермионы", 1: "Бозоны", 2: "Классические"}
        
        fig, ax = plt.subplots(figsize=(12, 8))

        for stats_type in [0, 1, 2]:
            type_results = [r for r in self.results if r["config"]["stats_type"] == stats_type]
            if not type_results:
                continue

            T_values = []
            mean_times = []
            std_times = []

            grouped = {}
            for result in type_results:
                T = result["config"]["temperature"]
                if T not in grouped:
                    grouped[T] = []
                grouped[T].append(result["measurement_time"])

            for T in sorted(grouped.keys()):
                times = grouped[T]
                T_values.append(T)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))

            ax.errorbar(
                T_values, mean_times, yerr=std_times,
                label=stats_names[stats_type], marker='^', capsize=5, capthick=2
            )

        ax.set_xlabel("Температура (K)", fontsize=12)
        ax.set_ylabel("Время измерения (с)", fontsize=12)
        ax.set_title("Зависимость времени измерения от температуры", fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "vs_temperature.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlations(self):
        """График корреляций между параметрами"""
        # Подготовка данных
        delta_I = [r["config"]["delta_I"] for r in self.results]
        N = [r["config"]["N"] for r in self.results]
        T = [r["config"]["temperature"] for r in self.results]
        times = [r["measurement_time"] for r in self.results]
        gammas = [r["gamma_total"] for r in self.results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Корреляция: время vs delta_I
        axes[0, 0].scatter(delta_I, times, alpha=0.3, s=10)
        axes[0, 0].set_xlabel("ΔI (бит)")
        axes[0, 0].set_ylabel("Время измерения (с)")
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title("Время vs Количество бит")
        axes[0, 0].grid(True, alpha=0.3)

        # Корреляция: время vs N
        axes[0, 1].scatter(N, times, alpha=0.3, s=10)
        axes[0, 1].set_xlabel("N (частиц)")
        axes[0, 1].set_ylabel("Время измерения (с)")
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title("Время vs Количество частиц")
        axes[0, 1].grid(True, alpha=0.3)

        # Корреляция: время vs температура
        axes[1, 0].scatter(T, times, alpha=0.3, s=10)
        axes[1, 0].set_xlabel("Температура (K)")
        axes[1, 0].set_ylabel("Время измерения (с)")
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title("Время vs Температура")
        axes[1, 0].grid(True, alpha=0.3)

        # Корреляция: время vs gamma_total
        axes[1, 1].scatter(gammas, times, alpha=0.3, s=10)
        axes[1, 1].set_xlabel("Γ_total")
        axes[1, 1].set_ylabel("Время измерения (с)")
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title("Время vs Сложность системы")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlations.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self):
        """Генерация полного отчета"""
        print("\nГенерация отчета...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"analysis_report_{timestamp}.md")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# ОТЧЕТ ПО АНАЛИЗУ РЕЗУЛЬТАТОВ СИМУЛЯЦИЙ\n\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Источник данных: {self.results_file}\n")
            f.write(f"Всего результатов: {len(self.results)}\n\n")
            f.write("=" * 70 + "\n\n")

            # Анализ по типам систем
            f.write("## АНАЛИЗ ПО ТИПАМ СИСТЕМ\n\n")
            stats_analysis = self.analyze_by_stats_type()
            for name, data in stats_analysis.items():
                f.write(f"### {name}\n\n")
                f.write(f"- Количество симуляций: {data['count']}\n")
                f.write(f"- Время измерения:\n")
                f.write(f"  - Минимум: {data['measurement_time']['min']:.3e} с\n")
                f.write(f"  - Максимум: {data['measurement_time']['max']:.3e} с\n")
                f.write(f"  - Среднее: {data['measurement_time']['mean']:.3e} с\n")
                f.write(f"  - Медиана: {data['measurement_time']['median']:.3e} с\n")
                f.write(f"  - Стандартное отклонение: {data['measurement_time']['std']:.3e} с\n")
                f.write(f"- Сложность системы (Γ_total):\n")
                f.write(f"  - Среднее: {data['gamma_total']['mean']:.3f}\n")
                f.write(f"  - Медиана: {data['gamma_total']['median']:.3f}\n\n")

            # Анализ по параметрам
            f.write("## АНАЛИЗ ПО ПАРАМЕТРАМ\n\n")
            for param in ["delta_I", "N", "temperature"]:
                f.write(f"### Зависимость от {param}\n\n")
                analysis = self.analyze_by_parameter(param)
                f.write(f"Количество уникальных значений: {len(analysis['values'])}\n\n")
                f.write("| Значение | Количество | Мин. время | Макс. время | Среднее время |\n")
                f.write("|----------|-------------|------------|-------------|---------------|\n")
                for value, stats in analysis["stats_by_value"].items():
                    f.write(f"| {value} | {stats['count']} | {stats['min']:.3e} | {stats['max']:.3e} | {stats['mean']:.3e} |\n")
                f.write("\n")

            f.write("## ГРАФИКИ\n\n")
            f.write("Графики сохранены в директории анализа:\n")
            f.write("- `by_stats_type.png` - Распределение по типам систем\n")
            f.write("- `vs_delta_I.png` - Зависимость от количества бит\n")
            f.write("- `vs_N.png` - Зависимость от количества частиц\n")
            f.write("- `vs_temperature.png` - Зависимость от температуры\n")
            f.write("- `correlations.png` - Корреляции между параметрами\n")

        print(f"Отчет сохранен в: {report_file}")
        return report_file


def main():
    """Основная функция для анализа результатов"""
    import sys

    if len(sys.argv) < 2:
        print("Использование: python analyze_simulation_results.py <results_file.json>")
        print("\nПример:")
        print("  python analyze_simulation_results.py results/simulation_plan/final_results_20241120_120000.json")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Ошибка: Файл {results_file} не найден")
        sys.exit(1)

    print("=" * 70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ СИМУЛЯЦИЙ")
    print("=" * 70)

    analyzer = SimulationAnalyzer(results_file)

    # Создание графиков
    analyzer.create_plots()

    # Генерация отчета
    analyzer.generate_report()

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 70)
    print(f"\nРезультаты сохранены в: {analyzer.output_dir}")


if __name__ == "__main__":
    main()


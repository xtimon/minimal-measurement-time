"""
Модуль для экспорта результатов симуляции
"""

import numpy as np
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ResultExporter:
    """Класс для экспорта результатов симуляции в текстовом формате"""
    
    RESULTS_DIR = "results"
    
    @staticmethod
    def _ensure_results_dir() -> str:
        """
        Создает директорию results если её нет и возвращает путь к ней
        
        Returns:
        - Путь к директории results
        """
        results_dir = ResultExporter.RESULTS_DIR
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            logger.info(f"📁 Создана директория для результатов: {results_dir}")
        return results_dir
    
    @staticmethod
    def export_to_text(results: Dict[str, Any], filename: Optional[str] = None, 
                      format_type: str = "detailed") -> str:
        """
        Экспорт результатов в текстовый файл
        
        Parameters:
        - results: словарь с результатами симуляции
        - filename: имя файла (если None, генерируется автоматически)
        - format_type: "detailed" (подробный) или "summary" (краткий)
        
        Returns:
        - Полный путь к сохраненному файлу
        """
        # Создаем директорию results если её нет
        results_dir = ResultExporter._ensure_results_dir()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.txt"
        
        # Если filename не содержит путь, добавляем results/
        if os.path.dirname(filename) == "":
            filepath = os.path.join(results_dir, filename)
        else:
            filepath = filename
        
        try:
            report_content = ResultExporter._generate_report(results, format_type)
            if not report_content or len(report_content.strip()) == 0:
                logger.warning(f"⚠️ Сгенерированный отчет пуст для файла {filepath}")
                report_content = "⚠️ ОТЧЕТ ПУСТ - НЕТ ДАННЫХ ДЛЯ ОТОБРАЖЕНИЯ\n"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"📄 Результаты сохранены в файл: {filepath} ({len(report_content)} символов)")
            return filepath
        except IOError as e:
            logger.error(f"Ошибка при сохранении файла {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при генерации отчета для {filepath}: {e}")
            raise
    
    @staticmethod
    def _generate_report(results: Dict[str, Any], format_type: str) -> str:
        """Генерация текстового отчета"""
        report = []
        
        # Заголовок
        report.append("=" * 70)
        report.append("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ОСНОВНОГО УРАВНЕНИЯ")
        report.append("Информация × Время ≥ Квантовый предел × Сложность системы + Технические ограничения")
        report.append(f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")
        
        if format_type == "detailed":
            report.extend(ResultExporter._generate_detailed_report(results))
        else:
            report.extend(ResultExporter._generate_summary_report(results))
        
        return '\n'.join(report)
    
    @staticmethod
    def _generate_detailed_report(results: Dict[str, Any]) -> List[str]:
        """Генерация подробного отчета"""
        report = []
        
        # Основные результаты
        if 'min_measurement_time' in results:
            report.append("📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
            report.append(f"   Минимальное время измерения: {results['min_measurement_time']:.3e} с")
            
            # Сравнение с идеальным временем (без декогеренции и шума)
            if 'min_measurement_time_ideal' in results:
                ideal_time = results['min_measurement_time_ideal']
                actual_time = results['min_measurement_time']
                if ideal_time > 0:
                    ratio = actual_time / ideal_time
                    report.append(f"   Идеальное время (без декогеренции/шума): {ideal_time:.3e} с")
                    report.append(f"   Влияние декогеренции/шума: {ratio:.2f}x")
            report.append("")
        
        # Статистика для батч-симуляций
        if 'statistics' in results:
            stats = results['statistics']
            report.append("📈 СТАТИСТИКА СИМУЛЯЦИИ:")
            
            n_points = stats.get('n_points', 'N/A')
            if isinstance(n_points, (int, float)):
                report.append(f"   Количество точек: {n_points:,}")
            else:
                report.append(f"   Количество точек: {n_points}")
            
            min_time = stats.get('min_time', 'N/A')
            if isinstance(min_time, (int, float)):
                report.append(f"   Минимальное время: {min_time:.3e} с")
            else:
                report.append(f"   Минимальное время: {min_time} с")
            
            max_time = stats.get('max_time', 'N/A')
            if isinstance(max_time, (int, float)):
                report.append(f"   Максимальное время: {max_time:.3e} с")
            else:
                report.append(f"   Максимальное время: {max_time} с")
            
            median_time = stats.get('median_time', 'N/A')
            if isinstance(median_time, (int, float)):
                report.append(f"   Медианное время: {median_time:.3e} с")
            else:
                report.append(f"   Медианное время: {median_time} с")
            
            mean_time = stats.get('mean_time', 'N/A')
            if isinstance(mean_time, (int, float)):
                report.append(f"   Среднее время: {mean_time:.3e} с")
            else:
                report.append(f"   Среднее время: {mean_time} с")
            
            std_time = stats.get('std_time', None)
            if std_time is not None and isinstance(std_time, (int, float)):
                report.append(f"   Стандартное отклонение: {std_time:.3e} с")
            
            report.append("")
        
        # Параметры системы
        if 'parameters' in results:
            params = results['parameters']
            report.append("⚙️ ПАРАМЕТРЫ СИСТЕМЫ:")
            report.append(f"   Температура: {params.get('temperature', 'N/A')} K")
            report.append(f"   Количество информации: {params.get('delta_I', 'N/A')} бит")
            report.append(f"   Число частиц: {params.get('N', 'N/A'):,}")
            report.append(f"   Критическая температура: {params.get('T_c', 'N/A')} K")
            
            stats_type = params.get('stats_type', 'N/A')
            if stats_type == 0:
                report.append(f"   Тип статистики: Фермионы")
                report.append(f"   Температура Ферми: {params.get('T_F', 'N/A')} K")
            elif stats_type == 1:
                report.append(f"   Тип статистики: Бозоны")
                report.append(f"   Температура конденсации: {params.get('T_c_bose', 'N/A')} K")
            else:
                report.append(f"   Тип статистики: Классическая")
            
            report.append(f"   Сила взаимодействия (U/W): {params.get('U', 0)}")
            
            # Параметры декогеренции
            if params.get('T1') is not None or params.get('T2') is not None:
                report.append("")
                report.append("   🔬 ПАРАМЕТРЫ ДЕКОГЕРЕНЦИИ:")
                if params.get('T1') is not None:
                    report.append(f"   Время релаксации T1: {params.get('T1'):.3e} с")
                if params.get('T2') is not None:
                    report.append(f"   Время дефазировки T2: {params.get('T2'):.3e} с")
            
            # Параметры шума
            if (params.get('noise_temperature') is not None or 
                params.get('shot_noise_factor') is not None or 
                params.get('technical_noise') is not None):
                report.append("")
                report.append("   🔊 ПАРАМЕТРЫ ШУМА:")
                if params.get('noise_temperature') is not None:
                    report.append(f"   Эффективная температура шума: {params.get('noise_temperature'):.2f} K")
                if params.get('shot_noise_factor') is not None:
                    report.append(f"   Фактор дробового шума: {params.get('shot_noise_factor'):.3f}")
                if params.get('technical_noise') is not None:
                    report.append(f"   Технический шум: {params.get('technical_noise'):.3f}")
                if params.get('flicker_noise_factor') is not None:
                    report.append(f"   Фактор 1/f шума (фликкер): {params.get('flicker_noise_factor'):.3f}")
                if params.get('quantum_noise_factor') is not None:
                    report.append(f"   Фактор квантового шума: {params.get('quantum_noise_factor'):.3f}")
                if params.get('environment_noise_factor') is not None:
                    report.append(f"   Фактор шума окружения: {params.get('environment_noise_factor'):.3f}")
            
            # Временные параметры
            if params.get('equilibrium_time') is not None or params.get('detector_response_time') is not None:
                report.append("")
                report.append("   ⏱️ ВРЕМЕННЫЕ ПАРАМЕТРЫ:")
                if params.get('equilibrium_time') is not None:
                    report.append(f"   Время установления равновесия: {params.get('equilibrium_time'):.3e} с")
                if params.get('detector_response_time') is not None:
                    report.append(f"   Время отклика детектора: {params.get('detector_response_time'):.3e} с")
            
            report.append("")
        
        # Факторы сложности
        if 'gamma_factors' in results:
            gamma = results['gamma_factors']
            report.append("🔬 ФАКТОРЫ СЛОЖНОСТИ СИСТЕМЫ:")
            report.append(f"   Общая сложность Γ_total: {gamma.get('total', 'N/A'):.3f}")
            report.append(f"   Базовая сложность: {gamma.get('base', 'N/A'):.3f}")
            report.append(f"   Статистический фактор: {gamma.get('statistics', 'N/A'):.3f}")
            report.append(f"   Фактор корреляций: {gamma.get('correlations', 'N/A'):.3f}")
            report.append(f"   Фактор квазичастиц: {gamma.get('quasiparticles', 'N/A'):.3f}")
            report.append("")
        
        # Декогеренция
        if 'decoherence' in results:
            decoherence = results['decoherence']
            if decoherence.get('enabled', False):
                report.append("🔬 ДЕКОГЕРЕНЦИЯ:")
                if decoherence.get('T1') is not None:
                    report.append(f"   Время релаксации T1: {decoherence.get('T1'):.3e} с")
                if decoherence.get('T2') is not None:
                    report.append(f"   Время дефазировки T2: {decoherence.get('T2'):.3e} с")
                factor = decoherence.get('factor', 1.0)
                report.append(f"   Фактор декогеренции: {factor:.3f}")
                if factor > 1.01:
                    report.append(f"   ⚠️ Декогеренция увеличивает время измерения в {factor:.2f} раз")
                report.append("")
        
        # Шум
        if 'noise' in results:
            noise = results['noise']
            if noise.get('enabled', False):
                report.append("🔊 ШУМ:")
                if noise.get('temperature') is not None:
                    report.append(f"   Эффективная температура шума: {noise.get('temperature'):.2f} K")
                if noise.get('shot_noise_factor') is not None:
                    report.append(f"   Фактор дробового шума: {noise.get('shot_noise_factor'):.3f}")
                if noise.get('technical_noise') is not None:
                    report.append(f"   Технический шум: {noise.get('technical_noise'):.3f}")
                if noise.get('flicker_noise_factor') is not None:
                    report.append(f"   Фактор 1/f шума (фликкер): {noise.get('flicker_noise_factor'):.3f}")
                if noise.get('quantum_noise_factor') is not None:
                    report.append(f"   Фактор квантового шума: {noise.get('quantum_noise_factor'):.3f}")
                if noise.get('environment_noise_factor') is not None:
                    report.append(f"   Фактор шума окружения: {noise.get('environment_noise_factor'):.3f}")
                factor = noise.get('factor', 1.0)
                report.append(f"   Фактор шума: {factor:.3f}")
                if factor > 1.01:
                    report.append(f"   ⚠️ Шум увеличивает время измерения в {factor:.2f} раз")
                report.append("")
        
        # Временные параметры
        if 'timing' in results:
            timing = results['timing']
            report.append("⏱️ ВРЕМЕННЫЕ ПАРАМЕТРЫ:")
            if timing.get('equilibrium_time') is not None:
                report.append(f"   Время установления равновесия: {timing.get('equilibrium_time'):.3e} с")
            if timing.get('detector_response_time') is not None:
                report.append(f"   Время отклика детектора: {timing.get('detector_response_time'):.3e} с")
            report.append("")
        
        # Физические пределы
        if 'limits' in results:
            limits = results['limits']
            report.append("🎯 ФИЗИЧЕСКИЕ ПРЕДЕЛЫ:")
            report.append(f"   Фундаментальный предел: {limits.get('fundamental', 'N/A'):.3e} с")
            report.append(f"   Техническое ограничение: {limits.get('technical', 'N/A'):.3e} с")
            report.append(f"   Предел Ландауэра: {limits.get('landauer', 'N/A'):.3e} с")
            report.append(f"   Квантовый предел: {limits.get('quantum', 'N/A'):.3e} с")
            report.append("")
        
        # Интерпретация результатов
        report.append("💡 ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
        time_val = None
        
        # Пытаемся получить время из разных источников
        if 'min_measurement_time' in results:
            time_val = results['min_measurement_time']
        elif 'statistics' in results:
            stats = results['statistics']
            # Используем медианное время для интерпретации
            if 'median_time' in stats and stats['median_time'] != 'N/A':
                time_val = stats['median_time']
            elif 'mean_time' in stats and stats['mean_time'] != 'N/A':
                time_val = stats['mean_time']
        
        if time_val is not None:
            if time_val < 1e-12:
                interpretation = "ПИКОСЕКУНДЫ: Сверхбыстрое квантовое измерение"
            elif time_val < 1e-9:
                interpretation = "НАНОСЕКУНДЫ: Быстрое измерение в квантовых системах"
            elif time_val < 1e-6:
                interpretation = "МИКРОСЕКУНДЫ: Типичное время для квантовых сенсоров"
            elif time_val < 1e-3:
                interpretation = "МИЛЛИСЕКУНДЫ: Измерение в макроскопических системах"
            elif time_val < 1:
                interpretation = "СЕКУНДЫ: Медленное измерение в сложных системах"
            else:
                interpretation = "ДЛИТЕЛЬНОЕ ВРЕМЯ: Системы с высокой сложностью"
            report.append(f"   {interpretation}")
        else:
            report.append("   Недостаточно данных для интерпретации")
        
        # Проверка, что отчет не пуст
        if len(report) == 0:
            report.append("⚠️ Нет данных для отображения")
        
        return report
    
    @staticmethod
    def _generate_summary_report(results: Dict[str, Any]) -> List[str]:
        """Генерация краткого отчета"""
        report = []
        
        report.append("📋 КРАТКИЙ ОТЧЕТ:")
        
        if 'min_measurement_time' in results:
            report.append(f"Время измерения: {results['min_measurement_time']:.3e} с")
        
        if 'parameters' in results:
            params = results['parameters']
            stats_type = {0: 'Фермионы', 1: 'Бозоны', 2: 'Классическая'}.get(params.get('stats_type'), 'N/A')
            report.append(f"Система: {stats_type}, T={params.get('temperature', 'N/A')}K, N={params.get('N', 'N/A'):,}")
        
        if 'gamma_factors' in results:
            report.append(f"Сложность системы: {results['gamma_factors'].get('total', 'N/A'):.2f}")
        
        return report
    
    @staticmethod
    def export_comparison(systems_data: List[Dict[str, Any]], 
                         filename: Optional[str] = None) -> str:
        """
        Экспорт сравнения нескольких систем
        
        Parameters:
        - systems_data: список словарей с данными систем
        - filename: имя файла (если None, генерируется автоматически)
        
        Returns:
        - Полный путь к сохраненному файлу
        """
        # Создаем директорию results если её нет
        results_dir = ResultExporter._ensure_results_dir()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_comparison_{timestamp}.txt"
        
        # Если filename не содержит путь, добавляем results/
        if os.path.dirname(filename) == "":
            filepath = os.path.join(results_dir, filename)
        else:
            filepath = filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(ResultExporter._generate_comparison_report(systems_data))
            logger.info(f"📊 Сравнение систем сохранено в: {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Ошибка при сохранении файла {filepath}: {e}")
            raise
    
    @staticmethod
    def _generate_comparison_report(systems_data: List[Dict[str, Any]]) -> str:
        """Генерация отчета сравнения систем"""
        report = []
        
        report.append("=" * 80)
        report.append("СРАВНЕНИЕ ФИЗИЧЕСКИХ СИСТЕМ")
        report.append("=" * 80)
        report.append("")
        
        # Таблица сравнения
        report.append(f"{'Система':<30} {'Время (с)':<15} {'Сложность':<12} {'Тип':<10}")
        report.append("-" * 80)
        
        for system in systems_data:
            name = system.get('name', 'N/A')
            time_val = system.get('measurement_time', 0)
            complexity = system.get('complexity', 0)
            system_type = system.get('type', 'N/A')
            
            report.append(f"{name:<30} {time_val:<15.2e} {complexity:<12.2f} {system_type:<10}")
        
        report.append("")
        report.append("💡 ВЫВОДЫ:")
        
        # Анализ результатов
        times = [s.get('measurement_time', 0) for s in systems_data]
        if times:
            fastest_idx = np.argmin(times)
            slowest_idx = np.argmax(times)
            report.append(f"Самая быстрая система: {systems_data[fastest_idx].get('name')}")
            report.append(f"Самая медленная система: {systems_data[slowest_idx].get('name')}")
            report.append(f"Разница во времени: {times[slowest_idx]/times[fastest_idx]:.1f}x")
        
        return '\n'.join(report)


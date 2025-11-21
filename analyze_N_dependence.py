#!/usr/bin/env python3
"""
Анализ зависимости времени измерения от количества частиц N
для понимания, почему время падает и стабилизируется после 10^5
"""

import numpy as np
import matplotlib.pyplot as plt
from minimal_measurement_time import (
    GPUInformationMeasurementSimulator,
    DEFAULT_N_Q,
    N_LOG_DIVISOR,
    GAMMA_BASE_COEFF,
)

# Параметры для анализа
temperature = 300.0
T_c = 100.0
delta_I = 1.0
stats_type = 0  # Фермион
T_F = 5000.0

# Диапазон N для анализа
N_values = np.logspace(2, 8, 100)  # от 10^2 до 10^8

# Создаем симулятор
sim = GPUInformationMeasurementSimulator(temperature=temperature, use_gpu=False, suppress_logging=True)

# Результаты
measurement_times = []
gamma_bases = []
gamma_totals = []
detector_times = []
equilibrium_times = []

print("Анализ зависимости от N...")
print(f"N_Q (порог насыщения) = {DEFAULT_N_Q:.0e}")
print(f"N_LOG_DIVISOR = {N_LOG_DIVISOR:.0f}")
print(f"GAMMA_BASE_COEFF = {GAMMA_BASE_COEFF}\n")

for N in N_values:
    result_time, result_dict = sim.simulate_single_system(
        delta_I=delta_I,
        T_c=T_c,
        N=N,
        stats_type=stats_type,
        T_F=T_F,
        include_decoherence=True,
        include_noise=True,
        export_results=False,
    )
    
    measurement_times.append(result_time)
    gamma_bases.append(result_dict["gamma_factors"]["base"])
    gamma_totals.append(result_dict["gamma_factors"]["total"])
    detector_times.append(result_dict.get("detector_response_time", 0))
    equilibrium_times.append(result_dict.get("equilibrium_time", 0))

measurement_times = np.array(measurement_times)
gamma_bases = np.array(gamma_bases)
gamma_totals = np.array(gamma_totals)
detector_times = np.array(detector_times)
equilibrium_times = np.array(equilibrium_times)

# Анализ формулы gamma_base
T_ratio = T_c / temperature
numerator_analytical = GAMMA_BASE_COEFF * (T_ratio**1.5) * np.log(1 + N_values / N_LOG_DIVISOR)
denominator_analytical = 1 + (N_values / DEFAULT_N_Q) ** 4
gamma_base_analytical = 1 + numerator_analytical / denominator_analytical

# Создание графиков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Время измерения vs N
ax1 = axes[0, 0]
ax1.loglog(N_values, measurement_times, 'b-', linewidth=2, label='Время измерения')
ax1.axvline(x=DEFAULT_N_Q, color='r', linestyle='--', alpha=0.5, label=f'N_Q = {DEFAULT_N_Q:.0e}')
ax1.axvline(x=1e5, color='g', linestyle='--', alpha=0.5, label='N = 10^5')
ax1.set_xlabel('Количество частиц N')
ax1.set_ylabel('Время измерения (с)')
ax1.set_title('Зависимость времени измерения от N')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. gamma_base vs N
ax2 = axes[0, 1]
ax2.semilogx(N_values, gamma_bases, 'b-', linewidth=2, label='gamma_base (вычисленный)')
ax2.semilogx(N_values, gamma_base_analytical, 'r--', linewidth=1, alpha=0.7, label='gamma_base (аналитический)')
ax2.axvline(x=DEFAULT_N_Q, color='r', linestyle='--', alpha=0.5, label=f'N_Q = {DEFAULT_N_Q:.0e}')
ax2.axvline(x=1e5, color='g', linestyle='--', alpha=0.5, label='N = 10^5')
ax2.set_xlabel('Количество частиц N')
ax2.set_ylabel('gamma_base')
ax2.set_title('Зависимость gamma_base от N')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Компоненты времени измерения
ax3 = axes[1, 0]
# Вычисляем идеальное время (без декогеренции и шума)
fundamental = sim.hbar / (sim.kB * temperature)
tau_landauer = sim.hbar / (2 * sim.kB * temperature)
tau_tech = 1e-9
tau_Q = sim.hbar / (1e-20)  # Примерное delta_E
max_tau = np.maximum(tau_tech, np.maximum(tau_landauer, tau_Q))
ideal_times = (fundamental * gamma_totals + max_tau) / delta_I

ax3.loglog(N_values, measurement_times, 'b-', linewidth=2, label='Финальное время')
ax3.loglog(N_values, ideal_times, 'g--', linewidth=1, alpha=0.7, label='Идеальное время (без декогеренции/шума)')
ax3.loglog(N_values, detector_times, 'r--', linewidth=1, alpha=0.7, label='Время детектора')
ax3.axvline(x=DEFAULT_N_Q, color='r', linestyle='--', alpha=0.5)
ax3.axvline(x=1e5, color='g', linestyle='--', alpha=0.5)
ax3.set_xlabel('Количество частиц N')
ax3.set_ylabel('Время (с)')
ax3.set_title('Компоненты времени измерения')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Анализ знаменателя и числителя gamma_base
ax4 = axes[1, 1]
ax4.loglog(N_values, numerator_analytical, 'b-', linewidth=2, label='Числитель (растет как log(N))')
ax4.loglog(N_values, denominator_analytical, 'r-', linewidth=2, label='Знаменатель (1 + (N/N_Q)^4)')
ax4.axvline(x=DEFAULT_N_Q, color='r', linestyle='--', alpha=0.5, label=f'N_Q = {DEFAULT_N_Q:.0e}')
ax4.axvline(x=1e5, color='g', linestyle='--', alpha=0.5, label='N = 10^5')
ax4.set_xlabel('Количество частиц N')
ax4.set_ylabel('Значение')
ax4.set_title('Числитель и знаменатель gamma_base')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('N_dependence_analysis.png', dpi=150, bbox_inches='tight')
print("График сохранен в N_dependence_analysis.png")

# Детальный анализ в точке N = 10^5
idx_1e5 = np.argmin(np.abs(N_values - 1e5))
N_1e5 = N_values[idx_1e5]

print("\n" + "="*70)
print("ДЕТАЛЬНЫЙ АНАЛИЗ В ТОЧКЕ N = 10^5")
print("="*70)
print(f"N = {N_1e5:.2e}")
print(f"Время измерения = {measurement_times[idx_1e5]:.3e} с")
print(f"gamma_base = {gamma_bases[idx_1e5]:.3f}")
print(f"gamma_total = {gamma_totals[idx_1e5]:.3f}")
print(f"Время детектора = {detector_times[idx_1e5]:.3e} с")
print(f"Время равновесия = {equilibrium_times[idx_1e5]:.3e} с")
print(f"\nАнализ gamma_base:")
print(f"  N / N_Q = {N_1e5 / DEFAULT_N_Q:.3f}")
print(f"  (N / N_Q)^4 = {(N_1e5 / DEFAULT_N_Q) ** 4:.3e}")
print(f"  Знаменатель = {denominator_analytical[idx_1e5]:.3e}")
print(f"  Числитель = {numerator_analytical[idx_1e5]:.3f}")
print(f"  gamma_base = {gamma_base_analytical[idx_1e5]:.3f}")

# Анализ в точке N = 10^6
idx_1e6 = np.argmin(np.abs(N_values - DEFAULT_N_Q))
N_1e6 = N_values[idx_1e6]

print("\n" + "="*70)
print("ДЕТАЛЬНЫЙ АНАЛИЗ В ТОЧКЕ N = N_Q = 10^6")
print("="*70)
print(f"N = {N_1e6:.2e}")
print(f"Время измерения = {measurement_times[idx_1e6]:.3e} с")
print(f"gamma_base = {gamma_bases[idx_1e6]:.3f}")
print(f"gamma_total = {gamma_totals[idx_1e6]:.3f}")
print(f"\nАнализ gamma_base:")
print(f"  N / N_Q = {N_1e6 / DEFAULT_N_Q:.3f}")
print(f"  (N / N_Q)^4 = {(N_1e6 / DEFAULT_N_Q) ** 4:.3e}")
print(f"  Знаменатель = {denominator_analytical[idx_1e6]:.3e}")
print(f"  Числитель = {numerator_analytical[idx_1e6]:.3f}")
print(f"  gamma_base = {gamma_base_analytical[idx_1e6]:.3f}")

# Поиск точки, где время начинает стабилизироваться
# Находим производную (логарифмическую)
dlog_tau = np.diff(np.log(measurement_times)) / np.diff(np.log(N_values))
# Находим, где производная становится близкой к нулю (стабилизация)
threshold = 0.1  # Порог для "стабилизации"
stabilization_idx = np.where(np.abs(dlog_tau) < threshold)[0]
if len(stabilization_idx) > 0:
    stabilization_N = N_values[stabilization_idx[0]]
    print("\n" + "="*70)
    print("АНАЛИЗ СТАБИЛИЗАЦИИ")
    print("="*70)
    print(f"Время начинает стабилизироваться при N ≈ {stabilization_N:.2e}")
    print(f"Это происходит, когда (N/N_Q)^4 начинает доминировать в знаменателе")
    print(f"При N = {stabilization_N:.2e}, (N/N_Q)^4 = {(stabilization_N / DEFAULT_N_Q) ** 4:.2e}")

print("\n" + "="*70)
print("ВЫВОДЫ")
print("="*70)
print("1. Формула gamma_base содержит знаменатель: 1 + (N/N_Q)^4")
print(f"2. При N > N_Q = {DEFAULT_N_Q:.0e}, знаменатель быстро растет")
print("3. Это приводит к насыщению gamma_base → 1")
print("4. Когда gamma_base насыщается, вклад в общее время измерения уменьшается")
print("5. При больших N начинает доминировать время детектора (растет как log(N))")
print("6. Поэтому общее время измерения стабилизируется на уровне времени детектора")


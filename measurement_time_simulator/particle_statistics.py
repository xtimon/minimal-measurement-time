"""
Типы статистики частиц
"""

from enum import Enum

class ParticleStatistics(Enum):
    """Типы статистики частиц в физических системах"""
    FERMION = "fermion"
    BOSON = "boson" 
    CLASSICAL = "classical"


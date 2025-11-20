"""
Cosmic Sim - Библиотека для симуляции космических явлений и орбитальной механики
"""

from .core import CosmicSim
from .orbital import OrbitalSimulator
from .visualization import ParallaxVisualizer

__version__ = "0.1.0"
__all__ = ["CosmicSim", "OrbitalSimulator", "ParallaxVisualizer"]


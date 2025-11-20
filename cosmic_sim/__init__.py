"""
Cosmic Sim - Библиотека для симуляции космических явлений и орбитальной механики
"""

from .core import CosmicSim
from .orbital import OrbitalSimulator
from .visualization import ParallaxVisualizer
from .body import Body
from .nbody import NBodySimulator
from .visualization_advanced import AdvancedVisualizer
from .presets import SystemPresets
from .save_load import SimulationIO

__version__ = "0.2.0"
__all__ = [
    "CosmicSim",
    "OrbitalSimulator", 
    "ParallaxVisualizer",
    "Body",
    "NBodySimulator",
    "AdvancedVisualizer",
    "SystemPresets",
    "SimulationIO"
]


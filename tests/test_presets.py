"""
Тесты для модуля presets
"""
import pytest
import numpy as np
from cosmic_sim import SystemPresets, Body


class TestSystemPresets:
    """Тесты для класса SystemPresets"""
    
    def test_create_solar_system(self):
        """Тест создания Солнечной системы"""
        presets = SystemPresets()
        
        # Внутренние планеты
        bodies = presets.create_solar_system(include_outer_planets=False)
        assert len(bodies) == 5  # Солнце + 4 планеты
        
        # С внешними планетами
        bodies_full = presets.create_solar_system(include_outer_planets=True)
        assert len(bodies_full) == 9  # Солнце + 8 планет
        
        # Проверить, что есть Солнце
        sun = [b for b in bodies if b.name == "Солнце"]
        assert len(sun) == 1
        assert sun[0].mass > 0
    
    def test_create_binary_star_system(self):
        """Тест создания двойной звездной системы"""
        presets = SystemPresets()
        bodies = presets.create_binary_star_system(separation_au=10.0)
        
        assert len(bodies) == 2
        assert all(b.name.startswith("Звезда") for b in bodies)
        assert all(b.mass > 0 for b in bodies)
    
    def test_create_earth_moon_system(self):
        """Тест создания системы Земля-Луна"""
        presets = SystemPresets()
        bodies = presets.create_earth_moon_system()
        
        assert len(bodies) == 2
        names = [b.name for b in bodies]
        assert "Земля" in names
        assert "Луна" in names


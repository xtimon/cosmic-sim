"""
Тесты для модуля nbody
"""
import pytest
import numpy as np
from cosmic_sim import NBodySimulator, Body, CosmicSim


class TestNBodySimulator:
    """Тесты для класса NBodySimulator"""
    
    def test_initialization(self):
        """Тест инициализации симулятора"""
        bodies = [
            Body("Body1", 1e24, np.array([0, 0, 0]), np.array([0, 0, 0])),
            Body("Body2", 1e24, np.array([1e11, 0, 0]), np.array([0, 0, 0]))
        ]
        simulator = NBodySimulator(bodies)
        
        assert len(simulator.bodies) == 2
        assert simulator.time == 0.0
    
    def test_add_remove_body(self):
        """Тест добавления и удаления тел"""
        simulator = NBodySimulator()
        
        body = Body("Test", 1e24, np.array([0, 0, 0]), np.array([0, 0, 0]))
        simulator.add_body(body)
        assert len(simulator.bodies) == 1
        
        simulator.remove_body("Test")
        assert len(simulator.bodies) == 0
    
    def test_get_body(self):
        """Тест получения тела по имени"""
        body = Body("Test", 1e24, np.array([0, 0, 0]), np.array([0, 0, 0]))
        simulator = NBodySimulator([body])
        
        found = simulator.get_body("Test")
        assert found is not None
        assert found.name == "Test"
        
        not_found = simulator.get_body("NonExistent")
        assert not_found is None
    
    def test_center_of_mass(self):
        """Тест вычисления центра масс"""
        cosmic = CosmicSim()
        bodies = [
            Body("Sun", cosmic.mass_sun, np.array([0, 0, 0]), np.array([0, 0, 0])),
            Body("Earth", 5.972e24, np.array([cosmic.AU, 0, 0]), np.array([0, 0, 0]))
        ]
        simulator = NBodySimulator(bodies)
        
        com = simulator.get_center_of_mass()
        assert len(com) == 3
        # Центр масс должен быть ближе к Солнцу
        assert abs(com[0]) < cosmic.AU / 2
    
    def test_total_energy(self):
        """Тест вычисления полной энергии"""
        cosmic = CosmicSim()
        bodies = [
            Body("Sun", cosmic.mass_sun, np.array([0, 0, 0]), np.array([0, 0, 0])),
            Body("Earth", 5.972e24, np.array([cosmic.AU, 0, 0]), 
                 np.array([0, cosmic.orbital_velocity(cosmic.mass_sun, cosmic.AU), 0]))
        ]
        simulator = NBodySimulator(bodies)
        
        energy = simulator.get_total_energy()
        assert np.isfinite(energy)
        # Энергия должна быть отрицательной для связанной системы
        assert energy < 0
    
    def test_simple_simulation(self):
        """Тест простой симуляции"""
        cosmic = CosmicSim()
        mass_sun = cosmic.mass_sun
        mass_earth = 5.972e24
        
        # Создать простую систему Солнце-Земля
        sun = Body("Sun", mass_sun, np.array([0, 0, 0]), np.array([0, 0, 0]))
        earth = Body(
            "Earth",
            mass_earth,
            np.array([cosmic.AU, 0, 0]),
            np.array([0, cosmic.orbital_velocity(mass_sun, cosmic.AU), 0])
        )
        
        simulator = NBodySimulator([sun, earth])
        
        # Короткая симуляция (1 день)
        one_day = 24 * 3600
        times, states = simulator.simulate(
            t_span=(0, one_day),
            n_points=100,
            save_trajectory=True
        )
        
        assert len(times) > 0
        assert states.shape[0] == 12  # 6 координат для каждого тела
        assert len(sun.trajectory) > 0
        assert len(earth.trajectory) > 0


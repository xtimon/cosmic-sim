"""
Тесты для класса Body
"""
import pytest
import numpy as np
from cosmic_sim import Body


class TestBody:
    """Тесты для класса Body"""
    
    def test_initialization(self):
        """Тест инициализации тела"""
        body = Body(
            name="Test",
            mass=1e24,
            position=np.array([1e11, 0, 0]),
            velocity=np.array([0, 30000, 0]),
            radius=1e6,
            color='blue'
        )
        
        assert body.name == "Test"
        assert body.mass == 1e24
        assert np.allclose(body.position, [1e11, 0, 0])
        assert np.allclose(body.velocity, [0, 30000, 0])
        assert body.radius == 1e6
        assert body.color == 'blue'
    
    def test_get_state(self):
        """Тест получения состояния"""
        body = Body("Test", 1e24, np.array([1, 2, 3]), np.array([4, 5, 6]))
        state = body.get_state()
        
        assert len(state) == 6
        assert np.allclose(state[:3], [1, 2, 3])
        assert np.allclose(state[3:], [4, 5, 6])
    
    def test_set_state(self):
        """Тест установки состояния"""
        body = Body("Test", 1e24, np.array([0, 0, 0]), np.array([0, 0, 0]))
        new_state = np.array([1, 2, 3, 4, 5, 6])
        body.set_state(new_state)
        
        assert np.allclose(body.position, [1, 2, 3])
        assert np.allclose(body.velocity, [4, 5, 6])
    
    def test_kinetic_energy(self):
        """Тест вычисления кинетической энергии"""
        body = Body("Test", 1e24, np.array([0, 0, 0]), np.array([1000, 0, 0]))
        ke = body.get_kinetic_energy()
        
        expected = 0.5 * 1e24 * 1000**2
        assert abs(ke - expected) < 1e10
    
    def test_distance_to(self):
        """Тест вычисления расстояния до другого тела"""
        body1 = Body("Test1", 1e24, np.array([0, 0, 0]), np.array([0, 0, 0]))
        body2 = Body("Test2", 1e24, np.array([3, 4, 0]), np.array([0, 0, 0]))
        
        distance = body1.get_distance_to(body2)
        assert abs(distance - 5.0) < 0.1
    
    def test_trajectory(self):
        """Тест сохранения траектории"""
        body = Body("Test", 1e24, np.array([0, 0, 0]), np.array([0, 0, 0]))
        
        body.add_to_trajectory()
        assert len(body.trajectory) == 1
        
        body.position = np.array([1, 0, 0])
        body.add_to_trajectory()
        assert len(body.trajectory) == 2
        
        body.clear_trajectory()
        assert len(body.trajectory) == 0


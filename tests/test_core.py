"""
Базовые тесты для модуля core
"""
import numpy as np
from cosmic_sim import CosmicSim


class TestCosmicSim:
    """Тесты для класса CosmicSim"""
    
    def test_initialization(self):
        """Тест инициализации"""
        cosmic = CosmicSim()
        assert cosmic.AU > 0
        assert cosmic.G > 0
        assert cosmic.c > 0
        assert cosmic.mass_sun > 0
    
    def test_parallax_distance(self):
        """Тест вычисления расстояния через параллакс"""
        cosmic = CosmicSim()
        baseline = 2 * cosmic.AU
        parallax_angle = np.radians(1.0 / 3600)  # 1 угловая секунда
        
        distance = cosmic.parallax_distance(baseline, parallax_angle)
        assert distance > 0
        assert np.isfinite(distance)
    
    def test_orbital_velocity(self):
        """Тест вычисления орбитальной скорости"""
        cosmic = CosmicSim()
        mass_sun = cosmic.mass_sun
        distance = cosmic.AU
        
        velocity = cosmic.orbital_velocity(mass_sun, distance)
        assert velocity > 0
        assert np.isfinite(velocity)
        # Земля движется примерно 30 км/с
        assert 20000 < velocity < 40000
    
    def test_kepler_third_law(self):
        """Тест третьего закона Кеплера"""
        cosmic = CosmicSim()
        mass_sun = cosmic.mass_sun
        mass_earth = 5.972e24
        semi_major_axis = cosmic.AU
        
        period = cosmic.kepler_third_law(semi_major_axis, mass_sun, mass_earth)
        assert period > 0
        # Период должен быть около 1 года
        days = period / (24 * 3600)
        assert 300 < days < 400
    
    def test_coordinate_transforms(self):
        """Тест преобразований координат"""
        cosmic = CosmicSim()
        
        # Тест сферические -> декартовы
        distance = 1.5e11  # 1 а.е.
        ra = np.radians(45)
        dec = np.radians(30)
        
        x, y, z = cosmic.spherical_to_cartesian(distance, ra, dec)
        assert len([x, y, z]) == 3
        assert all(np.isfinite([x, y, z]))
        
        # Тест обратного преобразования
        dist_back, ra_back, dec_back = cosmic.cartesian_to_spherical(x, y, z)
        assert abs(dist_back - distance) < 1e6  # Допустимая погрешность
        assert abs(ra_back - ra) < 0.01
        assert abs(dec_back - dec) < 0.01
    
    def test_gravitational_force(self):
        """Тест вычисления силы гравитации"""
        cosmic = CosmicSim()
        m1 = cosmic.mass_sun
        m2 = 5.972e24  # Земля
        r = cosmic.AU
        
        force = cosmic.gravitational_force(m1, m2, r)
        assert force > 0
        assert np.isfinite(force)


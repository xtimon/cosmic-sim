"""
Основной модуль с физическими константами и базовыми вычислениями
"""
import numpy as np


class CosmicSim:
    """Класс для работы с физическими константами и базовыми космическими вычислениями"""
    
    def __init__(self):
        # Физические константы
        self.AU = 1.495978707e11  # астрономическая единица (м)
        self.G = 6.67430e-11      # гравитационная постоянная
        self.c = 299792458        # скорость света
        self.R_earth = 6371000    # радиус Земли (м)
        self.mass_sun = 1.989e30  # масса Солнца (кг)
        
    def parallax_distance(self, baseline, parallax_angle_rad):
        """
        Вычисление расстояния через параллакс
        D = B / θ
        
        Parameters:
        -----------
        baseline : float
            База измерения (м)
        parallax_angle_rad : float
            Параллактический угол в радианах
            
        Returns:
        --------
        float
            Расстояние до объекта (м)
        """
        return baseline / parallax_angle_rad
    
    def angular_size(self, physical_size, distance):
        """
        Вычисление углового размера объекта
        θ = R / D
        
        Parameters:
        -----------
        physical_size : float
            Физический размер объекта (м)
        distance : float
            Расстояние до объекта (м)
            
        Returns:
        --------
        float
            Угловой размер в радианах
        """
        return physical_size / distance
    
    def spherical_to_cartesian(self, distance, ra_rad, dec_rad):
        """
        Преобразование сферических координат в декартовы
        
        Parameters:
        -----------
        distance : float
            Расстояние (м)
        ra_rad : float
            Прямое восхождение в радианах
        dec_rad : float
            Склонение в радианах
            
        Returns:
        --------
        numpy.ndarray
            Декартовы координаты [x, y, z]
        """
        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad) 
        z = distance * np.sin(dec_rad)
        return np.array([x, y, z])
    
    def cartesian_to_spherical(self, x, y, z):
        """
        Преобразование декартовых координат в сферические
        
        Parameters:
        -----------
        x, y, z : float
            Декартовы координаты (м)
            
        Returns:
        --------
        tuple
            (distance, ra, dec) - расстояние, прямое восхождение, склонение
        """
        distance = np.sqrt(x**2 + y**2 + z**2)
        ra = np.arctan2(y, x)
        dec = np.arcsin(z / distance)
        return distance, ra, dec
    
    def gravitational_force(self, m1, m2, r):
        """
        Сила гравитации между двумя массами
        F = G * m1 * m2 / r^2
        
        Parameters:
        -----------
        m1, m2 : float
            Массы объектов (кг)
        r : float
            Расстояние между объектами (м)
            
        Returns:
        --------
        float
            Сила гравитации (Н)
        """
        return self.G * m1 * m2 / r**2
    
    def orbital_velocity(self, central_mass, distance):
        """
        Орбитальная скорость для круговой орбиты
        v = sqrt(G * M / r)
        
        Parameters:
        -----------
        central_mass : float
            Масса центрального тела (кг)
        distance : float
            Расстояние от центрального тела (м)
            
        Returns:
        --------
        float
            Орбитальная скорость (м/с)
        """
        return np.sqrt(self.G * central_mass / distance)
    
    def kepler_third_law(self, semi_major_axis, mass1, mass2):
        """
        Третий закон Кеплера
        T^2 = (4π^2 / G(M1 + M2)) * a^3
        
        Parameters:
        -----------
        semi_major_axis : float
            Большая полуось орбиты (м)
        mass1, mass2 : float
            Массы двух тел (кг)
            
        Returns:
        --------
        float
            Орбитальный период (с)
        """
        total_mass = mass1 + mass2
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (self.G * total_mass))
        return period


"""
Предустановленные системы для симуляции
"""
import numpy as np
from .body import Body
from .core import CosmicSim


class SystemPresets:
    """Класс с предустановленными системами"""
    
    def __init__(self):
        self.cosmic = CosmicSim()
    
    def create_solar_system(self, include_outer_planets: bool = True) -> list:
        """
        Создать Солнечную систему
        
        Parameters:
        -----------
        include_outer_planets : bool
            Включать ли внешние планеты (Юпитер, Сатурн, Уран, Нептун)
            
        Returns:
        --------
        List[Body]
            Список тел Солнечной системы
        """
        bodies = []
        
        # Массы (кг)
        mass_sun = 1.989e30
        mass_mercury = 3.301e23
        mass_venus = 4.867e24
        mass_earth = 5.972e24
        mass_mars = 6.417e23
        mass_jupiter = 1.898e27
        mass_saturn = 5.683e26
        mass_uranus = 8.681e25
        mass_neptune = 1.024e26
        
        # Средние расстояния от Солнца (м)
        r_mercury = 0.387 * self.cosmic.AU
        r_venus = 0.723 * self.cosmic.AU
        r_earth = 1.0 * self.cosmic.AU
        r_mars = 1.524 * self.cosmic.AU
        r_jupiter = 5.203 * self.cosmic.AU
        r_saturn = 9.537 * self.cosmic.AU
        r_uranus = 19.191 * self.cosmic.AU
        r_neptune = 30.069 * self.cosmic.AU
        
        # Солнце в центре
        bodies.append(Body(
            name="Солнце",
            mass=mass_sun,
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            radius=6.96e8,  # Радиус Солнца
            color='yellow'
        ))
        
        # Внутренние планеты
        planets_data = [
            ("Меркурий", mass_mercury, r_mercury, 'gray'),
            ("Венера", mass_venus, r_venus, 'orange'),
            ("Земля", mass_earth, r_earth, 'blue'),
            ("Марс", mass_mars, r_mars, 'red'),
        ]
        
        if include_outer_planets:
            planets_data.extend([
                ("Юпитер", mass_jupiter, r_jupiter, 'brown'),
                ("Сатурн", mass_saturn, r_saturn, 'gold'),
                ("Уран", mass_uranus, r_uranus, 'cyan'),
                ("Нептун", mass_neptune, r_neptune, 'navy'),
            ])
        
        for name, mass, distance, color in planets_data:
            # Орбитальная скорость для круговой орбиты
            v_orbital = self.cosmic.orbital_velocity(mass_sun, distance)
            
            bodies.append(Body(
                name=name,
                mass=mass,
                position=np.array([distance, 0, 0]),
                velocity=np.array([0, v_orbital, 0]),
                radius=distance * 0.01,  # Для визуализации
                color=color
            ))
        
        return bodies
    
    def create_binary_star_system(self, separation_au: float = 10.0) -> list:
        """
        Создать двойную звездную систему
        
        Parameters:
        -----------
        separation_au : float
            Расстояние между звездами в астрономических единицах
            
        Returns:
        --------
        List[Body]
            Список тел двойной системы
        """
        bodies = []
        
        # Массы звезд (солнечные массы)
        mass1 = 1.0 * self.cosmic.mass_sun
        mass2 = 0.8 * self.cosmic.mass_sun
        
        separation = separation_au * self.cosmic.AU
        
        # Центр масс
        total_mass = mass1 + mass2
        r1 = separation * mass2 / total_mass
        r2 = separation * mass1 / total_mass
        
        # Орбитальная скорость
        v1 = np.sqrt(self.cosmic.G * mass2 * separation / (total_mass * r1))
        v2 = np.sqrt(self.cosmic.G * mass1 * separation / (total_mass * r2))
        
        bodies.append(Body(
            name="Звезда 1",
            mass=mass1,
            position=np.array([-r1, 0, 0]),
            velocity=np.array([0, -v1, 0]),
            radius=6.96e8,
            color='yellow'
        ))
        
        bodies.append(Body(
            name="Звезда 2",
            mass=mass2,
            position=np.array([r2, 0, 0]),
            velocity=np.array([0, v2, 0]),
            radius=5.5e8,
            color='orange'
        ))
        
        return bodies
    
    def create_earth_moon_system(self) -> list:
        """
        Создать систему Земля-Луна
        
        Returns:
        --------
        List[Body]
            Список тел системы Земля-Луна
        """
        bodies = []
        
        mass_earth = 5.972e24
        mass_moon = 7.342e22
        r_moon = 3.844e8  # Среднее расстояние до Луны (м)
        
        # Используем центр масс системы для более реалистичной симуляции
        total_mass = mass_earth + mass_moon
        r_earth_cm = r_moon * mass_moon / total_mass  # Расстояние Земли от центра масс
        r_moon_cm = r_moon * mass_earth / total_mass  # Расстояние Луны от центра масс
        
        # Орбитальная скорость для круговой орбиты вокруг центра масс
        # Для круговой орбиты: v = sqrt(G * M_other / R_total)
        # где M_other - масса другого тела, R_total - расстояние между телами
        # Это упрощенная формула, которая работает для круговых орбит
        v_earth = np.sqrt(self.cosmic.G * mass_moon / r_moon)
        v_moon = np.sqrt(self.cosmic.G * mass_earth / r_moon)
        
        # Земля
        bodies.append(Body(
            name="Земля",
            mass=mass_earth,
            position=np.array([-r_earth_cm, 0, 0]),
            velocity=np.array([0, -v_earth, 0]),
            radius=self.cosmic.R_earth,
            color='blue'
        ))
        
        # Луна
        bodies.append(Body(
            name="Луна",
            mass=mass_moon,
            position=np.array([r_moon_cm, 0, 0]),
            velocity=np.array([0, v_moon, 0]),
            radius=1.737e6,  # Радиус Луны
            color='gray'
        ))
        
        return bodies
    
    def create_custom_system(self, bodies_data: list) -> list:
        """
        Создать пользовательскую систему
        
        Parameters:
        -----------
        bodies_data : List[dict]
            Список словарей с данными тел:
            {'name': str, 'mass': float, 'position': array, 
             'velocity': array, 'radius': float, 'color': str}
            
        Returns:
        --------
        List[Body]
            Список тел
        """
        bodies = []
        for data in bodies_data:
            bodies.append(Body(**data))
        return bodies


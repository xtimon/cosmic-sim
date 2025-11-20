"""
Модуль для представления небесных тел
"""
import numpy as np
from typing import Optional, Tuple


class Body:
    """Класс для представления небесного тела"""
    
    def __init__(self, name: str, mass: float, position: np.ndarray, 
                 velocity: np.ndarray, radius: float = 0.0, color: str = 'blue'):
        """
        Инициализация небесного тела
        
        Parameters:
        -----------
        name : str
            Название тела
        mass : float
            Масса тела (кг)
        position : np.ndarray
            Начальная позиция [x, y, z] (м)
        velocity : np.ndarray
            Начальная скорость [vx, vy, vz] (м/с)
        radius : float
            Радиус тела для визуализации (м)
        color : str
            Цвет для визуализации
        """
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = radius
        self.color = color
        
        # История траектории
        self.trajectory = []
        self.energy_history = []
        
    def get_state(self) -> np.ndarray:
        """
        Получить вектор состояния [x, y, z, vx, vy, vz]
        
        Returns:
        --------
        np.ndarray
            Вектор состояния
        """
        return np.concatenate([self.position, self.velocity])
    
    def set_state(self, state: np.ndarray):
        """
        Установить состояние тела
        
        Parameters:
        -----------
        state : np.ndarray
            Вектор состояния [x, y, z, vx, vy, vz]
        """
        self.position = state[:3]
        self.velocity = state[3:6]
    
    def get_kinetic_energy(self) -> float:
        """
        Вычислить кинетическую энергию
        
        Returns:
        --------
        float
            Кинетическая энергия (Дж)
        """
        v_mag = np.linalg.norm(self.velocity)
        return 0.5 * self.mass * v_mag**2
    
    def get_distance_to(self, other: 'Body') -> float:
        """
        Вычислить расстояние до другого тела
        
        Parameters:
        -----------
        other : Body
            Другое тело
            
        Returns:
        --------
        float
            Расстояние (м)
        """
        return np.linalg.norm(self.position - other.position)
    
    def add_to_trajectory(self):
        """Добавить текущую позицию в историю траектории"""
        self.trajectory.append(self.position.copy())
    
    def clear_trajectory(self):
        """Очистить историю траектории"""
        self.trajectory = []
    
    def get_trajectory_array(self) -> np.ndarray:
        """
        Получить массив траектории
        
        Returns:
        --------
        np.ndarray
            Массив позиций формы (n_points, 3)
        """
        if len(self.trajectory) == 0:
            return np.array([]).reshape(0, 3)
        return np.array(self.trajectory)


"""
Модуль для симуляции орбитальной механики
"""
import numpy as np
from scipy.integrate import solve_ivp
from .core import CosmicSim


class OrbitalSimulator:
    """Класс для симуляции орбитальной динамики"""
    
    def __init__(self):
        self.cosmic = CosmicSim()
    
    def two_body_derivatives(self, t, y, mass1, mass2):
        """
        Производные для системы двух тел
        
        Parameters:
        -----------
        t : float
            Время
        y : array_like
            Вектор состояния [r1, r2, v1, v2]
        mass1, mass2 : float
            Массы двух тел (кг)
            
        Returns:
        --------
        numpy.ndarray
            Производные вектора состояния
        """
        r1 = y[:3]
        r2 = y[3:6]
        v1 = y[6:9]
        v2 = y[9:12]
        
        r_vec = r2 - r1
        r_mag = np.linalg.norm(r_vec)
        
        # Ускорения от гравитации
        a1 = self.cosmic.G * mass2 * r_vec / r_mag**3
        a2 = -self.cosmic.G * mass1 * r_vec / r_mag**3
        
        return np.concatenate([v1, v2, a1, a2])
    
    def simulate_two_body(self, mass1, mass2, initial_pos1, initial_pos2, 
                         initial_vel1, initial_vel2, t_span, n_points=1000):
        """
        Симуляция системы двух тел
        
        Parameters:
        -----------
        mass1, mass2 : float
            Массы двух тел (кг)
        initial_pos1, initial_pos2 : array_like
            Начальные позиции тел (м)
        initial_vel1, initial_vel2 : array_like
            Начальные скорости тел (м/с)
        t_span : tuple
            Интервал времени (t_start, t_end) в секундах
        n_points : int
            Количество точек для вычисления
            
        Returns:
        --------
        tuple
            (t, y) - массив времени и массив состояний
        """
        y0 = np.concatenate([initial_pos1, initial_pos2, initial_vel1, initial_vel2])
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        solution = solve_ivp(self.two_body_derivatives, t_span, y0, 
                           args=(mass1, mass2), t_eval=t_eval, rtol=1e-8)
        
        return solution.t, solution.y


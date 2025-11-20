"""
Модуль для полномасштабной симуляции N тел
"""
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional, Callable
from .core import CosmicSim
from .body import Body


class NBodySimulator:
    """Класс для симуляции системы N тел"""
    
    def __init__(self, bodies: Optional[List[Body]] = None):
        """
        Инициализация симулятора
        
        Parameters:
        -----------
        bodies : List[Body], optional
            Список небесных тел
        """
        self.cosmic = CosmicSim()
        self.bodies = bodies if bodies is not None else []
        self.time = 0.0
        self.history = []
        
    def add_body(self, body: Body):
        """
        Добавить тело в систему
        
        Parameters:
        -----------
        body : Body
            Небесное тело
        """
        self.bodies.append(body)
    
    def remove_body(self, name: str):
        """
        Удалить тело из системы
        
        Parameters:
        -----------
        name : str
            Название тела
        """
        self.bodies = [b for b in self.bodies if b.name != name]
    
    def get_body(self, name: str) -> Optional[Body]:
        """
        Получить тело по имени
        
        Parameters:
        -----------
        name : str
            Название тела
            
        Returns:
        --------
        Body or None
            Тело или None если не найдено
        """
        for body in self.bodies:
            if body.name == name:
                return body
        return None
    
    def _n_body_derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Производные для системы N тел
        
        Parameters:
        -----------
        t : float
            Время
        y : np.ndarray
            Вектор состояния всех тел [r1, r2, ..., rN, v1, v2, ..., vN]
            
        Returns:
        --------
        np.ndarray
            Производные вектора состояния
        """
        n_bodies = len(self.bodies)
        n_dof = 3  # 3D пространство
        
        # Извлечь позиции и скорости
        positions = y[:n_bodies * n_dof].reshape(n_bodies, n_dof)
        velocities = y[n_bodies * n_dof:].reshape(n_bodies, n_dof)
        
        # Вычислить ускорения для каждого тела
        accelerations = np.zeros_like(positions)
        
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    # Избежать деления на ноль
                    if r_mag > 1e-10:
                        accelerations[i] += (self.cosmic.G * self.bodies[j].mass * 
                                           r_vec / r_mag**3)
        
        # Вернуть производные [v1, v2, ..., vN, a1, a2, ..., aN]
        return np.concatenate([velocities.flatten(), accelerations.flatten()])
    
    def _update_bodies_from_state(self, y: np.ndarray):
        """
        Обновить состояния тел из вектора состояния
        
        Parameters:
        -----------
        y : np.ndarray
            Вектор состояния
        """
        n_bodies = len(self.bodies)
        n_dof = 3
        
        positions = y[:n_bodies * n_dof].reshape(n_bodies, n_dof)
        velocities = y[n_bodies * n_dof:].reshape(n_bodies, n_dof)
        
        for i, body in enumerate(self.bodies):
            body.position = positions[i]
            body.velocity = velocities[i]
    
    def simulate(self, t_span: Tuple[float, float], n_points: int = 1000, 
                 rtol: float = 1e-8, save_trajectory: bool = True,
                 callback: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запустить симуляцию системы N тел
        
        Parameters:
        -----------
        t_span : Tuple[float, float]
            Интервал времени (t_start, t_end) в секундах
        n_points : int
            Количество точек для вычисления
        rtol : float
            Относительная точность для решателя
        save_trajectory : bool
            Сохранять ли траектории тел
        callback : Callable, optional
            Функция обратного вызова на каждом шаге
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (t, states) - массив времени и массив состояний всех тел
        """
        if len(self.bodies) == 0:
            raise ValueError("Нет тел в системе для симуляции")
        
        # Собрать начальное состояние
        initial_state = np.concatenate([b.get_state() for b in self.bodies])
        
        # Очистить траектории
        if save_trajectory:
            for body in self.bodies:
                body.clear_trajectory()
        
        # Временные точки для вычисления
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Решить систему дифференциальных уравнений
        solution = solve_ivp(
            self._n_body_derivatives,
            t_span,
            initial_state,
            t_eval=t_eval,
            rtol=rtol,
            method='RK45'
        )
        
        # Обновить состояния тел и сохранить траектории
        states = []
        for i, t in enumerate(solution.t):
            state_vector = solution.y[:, i]
            self._update_bodies_from_state(state_vector)
            self.time = t
            
            if save_trajectory:
                for body in self.bodies:
                    body.add_to_trajectory()
            
            # Сохранить состояние
            states.append({
                'time': t,
                'bodies': {b.name: b.get_state().copy() for b in self.bodies}
            })
            
            # Вызвать callback если есть
            if callback:
                callback(t, self.bodies)
        
        self.history = states
        return solution.t, solution.y
    
    def get_total_energy(self) -> float:
        """
        Вычислить полную энергию системы
        
        Returns:
        --------
        float
            Полная энергия (Дж)
        """
        # Кинетическая энергия
        kinetic = sum(body.get_kinetic_energy() for body in self.bodies)
        
        # Потенциальная энергия
        potential = 0.0
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i+1:]:
                r = body1.get_distance_to(body2)
                if r > 1e-10:
                    potential -= self.cosmic.G * body1.mass * body2.mass / r
        
        return kinetic + potential
    
    def get_center_of_mass(self) -> np.ndarray:
        """
        Вычислить центр масс системы
        
        Returns:
        --------
        np.ndarray
            Позиция центра масс [x, y, z]
        """
        total_mass = sum(body.mass for body in self.bodies)
        if total_mass == 0:
            return np.zeros(3)
        
        com = np.zeros(3)
        for body in self.bodies:
            com += body.mass * body.position
        return com / total_mass
    
    def get_total_momentum(self) -> np.ndarray:
        """
        Вычислить полный импульс системы
        
        Returns:
        --------
        np.ndarray
            Полный импульс [px, py, pz]
        """
        momentum = np.zeros(3)
        for body in self.bodies:
            momentum += body.mass * body.velocity
        return momentum
    
    def reset(self):
        """Сбросить симуляцию"""
        self.time = 0.0
        self.history = []
        for body in self.bodies:
            body.clear_trajectory()


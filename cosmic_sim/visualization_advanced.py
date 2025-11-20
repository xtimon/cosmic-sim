"""
Продвинутая визуализация для полномасштабных симуляций
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
from .body import Body
from .core import CosmicSim


class AdvancedVisualizer:
    """Класс для продвинутой визуализации симуляций"""
    
    def __init__(self):
        self.cosmic = CosmicSim()
    
    def plot_3d_trajectories(self, bodies: List[Body], 
                            figsize: Tuple[int, int] = (12, 10),
                            show_orbits: bool = True,
                            show_bodies: bool = True,
                            title: str = "3D Траектории тел") -> plt.Figure:
        """
        Визуализация 3D траекторий тел
        
        Parameters:
        -----------
        bodies : List[Body]
            Список тел с траекториями
        figsize : Tuple[int, int]
            Размер фигуры
        show_orbits : bool
            Показывать ли орбиты
        show_bodies : bool
            Показывать ли текущие позиции тел
        title : str
            Заголовок графика
            
        Returns:
        --------
        matplotlib.figure.Figure
            Объект фигуры
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Найти границы для масштабирования
        all_positions = []
        for body in bodies:
            if len(body.trajectory) > 0:
                traj = body.get_trajectory_array()
                all_positions.append(traj)
        
        if len(all_positions) > 0:
            all_positions = np.vstack(all_positions)
            max_range = np.max(np.abs(all_positions))
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
        
        # Построить траектории
        if show_orbits:
            for body in bodies:
                if len(body.trajectory) > 0:
                    traj = body.get_trajectory_array()
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                           color=body.color, alpha=0.6, linewidth=1.5,
                           label=f'{body.name} (орбита)')
        
        # Показать текущие позиции
        if show_bodies:
            for body in bodies:
                pos = body.position
                ax.scatter(pos[0], pos[1], pos[2], 
                          c=body.color, s=100, marker='o',
                          label=body.name, edgecolors='black', linewidths=1)
        
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_2d_projection(self, bodies: List[Body], 
                         plane: str = 'xy',
                         figsize: Tuple[int, int] = (10, 10),
                         title: str = "Проекция орбит") -> plt.Figure:
        """
        Визуализация 2D проекции орбит
        
        Parameters:
        -----------
        bodies : List[Body]
            Список тел
        plane : str
            Плоскость проекции ('xy', 'xz', 'yz')
        figsize : Tuple[int, int]
            Размер фигуры
        title : str
            Заголовок графика
            
        Returns:
        --------
        matplotlib.figure.Figure
            Объект фигуры
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Индексы для плоскостей
        plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        if plane not in plane_map:
            plane = 'xy'
        idx1, idx2 = plane_map[plane]
        
        # Построить траектории
        for body in bodies:
            if len(body.trajectory) > 0:
                traj = body.get_trajectory_array()
                ax.plot(traj[:, idx1], traj[:, idx2], 
                       color=body.color, alpha=0.7, linewidth=2,
                       label=f'{body.name}')
                # Текущая позиция
                ax.scatter(body.position[idx1], body.position[idx2],
                          c=body.color, s=150, marker='o',
                          edgecolors='black', linewidths=2, zorder=5)
        
        labels = ['X', 'Y', 'Z']
        ax.set_xlabel(f'{labels[idx1]} (м)')
        ax.set_ylabel(f'{labels[idx2]} (м)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        return fig
    
    def animate_simulation(self, bodies: List[Body], 
                          interval: int = 50,
                          figsize: Tuple[int, int] = (12, 10),
                          save_path: Optional[str] = None) -> FuncAnimation:
        """
        Создать анимацию симуляции
        
        Parameters:
        -----------
        bodies : List[Body]
            Список тел с траекториями
        interval : int
            Интервал между кадрами (мс)
        figsize : Tuple[int, int]
            Размер фигуры
        save_path : str, optional
            Путь для сохранения анимации
            
        Returns:
        --------
        matplotlib.animation.FuncAnimation
            Анимация
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Найти максимальную длину траектории
        max_traj_len = max(len(b.trajectory) for b in bodies) if bodies else 0
        
        if max_traj_len == 0:
            raise ValueError("Нет траекторий для анимации")
        
        # Найти границы
        all_positions = []
        for body in bodies:
            if len(body.trajectory) > 0:
                traj = body.get_trajectory_array()
                all_positions.append(traj)
        
        if len(all_positions) > 0:
            all_positions = np.vstack(all_positions)
            max_range = np.max(np.abs(all_positions))
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
        
        # Инициализировать линии и точки
        lines = []
        points = []
        for body in bodies:
            if len(body.trajectory) > 0:
                line, = ax.plot([], [], [], color=body.color, alpha=0.6, 
                               linewidth=1.5, label=f'{body.name}')
                point, = ax.plot([], [], [], 'o', color=body.color, 
                                markersize=10, markeredgecolor='black')
                lines.append((line, body))
                points.append((point, body))
        
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_title('Анимация симуляции')
        ax.legend()
        ax.grid(True)
        
        def animate(frame):
            for line, body in lines:
                if frame < len(body.trajectory):
                    traj = np.array(body.trajectory[:frame+1])
                    line.set_data(traj[:, 0], traj[:, 1])
                    line.set_3d_properties(traj[:, 2])
            
            for point, body in points:
                if frame < len(body.trajectory):
                    pos = body.trajectory[frame]
                    point.set_data([pos[0]], [pos[1]])
                    point.set_3d_properties([pos[2]])
            
            return [l[0] for l in lines] + [p[0] for p in points]
        
        anim = FuncAnimation(fig, animate, frames=max_traj_len, 
                           interval=interval, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=20)
        
        return anim
    
    def plot_energy_conservation(self, simulator, times: np.ndarray) -> plt.Figure:
        """
        Построить график сохранения энергии
        
        Parameters:
        -----------
        simulator : NBodySimulator
            Симулятор с историей
        times : np.ndarray
            Массив времени
            
        Returns:
        --------
        matplotlib.figure.Figure
            Объект фигуры
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        energies = []
        for state in simulator.history:
            # Временно обновить состояния тел
            for body_name, body_state in state['bodies'].items():
                body = simulator.get_body(body_name)
                if body:
                    body.set_state(body_state)
            energies.append(simulator.get_total_energy())
        
        ax.plot(times, energies, 'b-', linewidth=2)
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Полная энергия (Дж)')
        ax.set_title('Сохранение энергии системы')
        ax.grid(True)
        
        # Показать относительное изменение
        if len(energies) > 0:
            initial_energy = energies[0]
            relative_change = (energies[-1] - initial_energy) / abs(initial_energy) * 100
            ax.text(0.02, 0.98, f'Изменение энергии: {relative_change:.2e}%',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig


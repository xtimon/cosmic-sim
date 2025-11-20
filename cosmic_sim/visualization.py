"""
Модуль для визуализации космических явлений
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .core import CosmicSim


class ParallaxVisualizer:
    """Класс для визуализации параллакса"""
    
    def __init__(self):
        self.cosmic = CosmicSim()
    
    def create_parallax_scene(self, star_distance_ly, baseline_au=2):
        """
        Создание сцены для демонстрации параллакса
        
        Parameters:
        -----------
        star_distance_ly : float
            Расстояние до звезды в световых годах
        baseline_au : float
            База измерения в астрономических единицах
            
        Returns:
        --------
        tuple
            (earth_pos1, earth_pos2, star_pos, parallax_angle)
        """
        # Преобразование в метры
        star_distance = star_distance_ly * 9.461e15
        baseline = baseline_au * self.cosmic.AU
        
        # Параллаксы с двух точек наблюдения
        parallax_angle = baseline / star_distance
        
        # Координаты
        earth_pos1 = np.array([-baseline/2, 0, 0])
        earth_pos2 = np.array([baseline/2, 0, 0])
        star_pos = np.array([0, star_distance, 0])
        
        return earth_pos1, earth_pos2, star_pos, parallax_angle
    
    def plot_parallax(self, star_distance_ly, baseline_au=2):
        """
        Визуализация параллакса
        
        Parameters:
        -----------
        star_distance_ly : float
            Расстояние до звезды в световых годах
        baseline_au : float
            База измерения в астрономических единицах
            
        Returns:
        --------
        matplotlib.figure.Figure
            Объект фигуры matplotlib
        """
        earth1, earth2, star, parallax_angle = self.create_parallax_scene(
            star_distance_ly, baseline_au)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D вид
        ax1.plot([earth1[0], star[0]], [earth1[1], star[1]], 'b--', alpha=0.7, label='Наблюдение 1')
        ax1.plot([earth2[0], star[0]], [earth2[1], star[1]], 'r--', alpha=0.7, label='Наблюдение 2')
        ax1.plot(earth1[0], earth1[1], 'bo', markersize=10, label='Земля (позиция 1)')
        ax1.plot(earth2[0], earth2[1], 'ro', markersize=10, label='Земля (позиция 2)')
        ax1.plot(star[0], star[1], 'y*', markersize=15, label='Звезда')
        ax1.set_xlabel('X (м)')
        ax1.set_ylabel('Y (м)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title(f'Параллакс: угол = {parallax_angle*206265:.2f} угловых секунд')
        
        # 3D вид
        ax3 = fig.add_subplot(122, projection='3d')
        ax3.plot([earth1[0], star[0]], [earth1[1], star[1]], [earth1[2], star[2]], 
                'b--', alpha=0.7)
        ax3.plot([earth2[0], star[0]], [earth2[1], star[1]], [earth2[2], star[2]], 
                'r--', alpha=0.7)
        ax3.scatter(*earth1, c='b', s=100, label='Земля 1')
        ax3.scatter(*earth2, c='r', s=100, label='Земля 2')
        ax3.scatter(*star, c='y', s=200, marker='*', label='Звезда')
        ax3.set_xlabel('X (м)')
        ax3.set_ylabel('Y (м)')
        ax3.set_zlabel('Z (м)')
        ax3.legend()
        ax3.set_title('3D вид параллакса')
        
        plt.tight_layout()
        return fig


"""
Продвинутая визуализация для полномасштабных симуляций
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
from .body import Body
from .core import CosmicSim


class AdvancedVisualizer:
    """Класс для продвинутой визуализации симуляций"""
    
    def __init__(self):
        self.cosmic = CosmicSim()
    
    def _create_sphere(self, x: float, y: float, z: float, 
                      size: float, color: str, resolution: int = 15,
                      moon_distance: float = 3.844e8) -> Poly3DCollection:
        """
        Создать 3D сферу
        
        Parameters:
        -----------
        x, y, z : float
            Центр сферы
        size : float
            Размер сферы (пропорциональный размеру точки)
        color : str
            Цвет сферы
        resolution : int
            Разрешение сферы (количество сегментов)
        moon_distance : float
            Расстояние до Луны в метрах (для масштабирования)
            
        Returns:
        --------
        Poly3DCollection
            Объект сферы для отображения
        """
        # Размер сферы масштабируется относительно расстояния между телами
        # Для больших систем (солнечная система) используем меньший множитель
        # Для малых систем (земля-луна) используем больший множитель
        max_size = 1000  # Максимальный размер точки
        # Нормализуем размер точки
        normalized_size = min(size / max_size, 1.0)  # Ограничиваем до 1.0
        
        # Определить масштаб системы с тремя категориями для оптимальной видимости
        # Для Солнца используем больший множитель для лучшей видимости
        is_sun = 'yellow' in color.lower() or size > 200  # Приблизительное определение Солнца
        
        if moon_distance > 5e11:  # Больше 5 а.е. - это солнечная система
            # Для солнечной системы: размер сферы = очень малая доля от расстояния
            max_sphere_radius = moon_distance * 0.001  # 0.1% от расстояния между телами
            sphere_radius = normalized_size * max_sphere_radius
            # Увеличить для Солнца
            if is_sun:
                sphere_radius = normalized_size * moon_distance * 0.005  # 0.5% для Солнца
        elif moon_distance > 1e10:  # От 0.1 до 5 а.е. - двойные звезды и средние системы
            # Для средних систем (двойные звезды): используем средний множитель
            max_sphere_radius = moon_distance * 0.01  # 1% от расстояния между телами
            sphere_radius = normalized_size * max_sphere_radius
            # Увеличить для звезд
            if is_sun:
                sphere_radius = normalized_size * moon_distance * 0.02  # 2% для звезд
        else:
            # Для малых систем (земля-луна): используем больший множитель
            sphere_radius = normalized_size * 0.1 * moon_distance
        
        # Параметрические уравнения сферы
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v)) + x
        y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v)) + y
        z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        
        # Создать поверхность сферы
        # Преобразуем в формат для Poly3DCollection
        verts = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                verts.append([
                    [x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]],
                    [x_sphere[i+1, j], y_sphere[i+1, j], z_sphere[i+1, j]],
                    [x_sphere[i+1, j+1], y_sphere[i+1, j+1], z_sphere[i+1, j+1]],
                    [x_sphere[i, j+1], y_sphere[i, j+1], z_sphere[i, j+1]]
                ])
        
        sphere = Poly3DCollection(verts, facecolor=color, edgecolor='black', 
                                 linewidths=0.3, alpha=0.8)
        return sphere
    
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
        
        # Найти границы для масштабирования (включая текущие позиции)
        all_positions = []
        for body in bodies:
            if len(body.trajectory) > 0:
                traj = body.get_trajectory_array()
                # Фильтровать невалидные значения (NaN, Inf, слишком большие)
                traj = traj[np.isfinite(traj).all(axis=1)]
                if len(traj) > 0:
                    all_positions.append(traj)
            # Добавить текущую позицию (если валидна)
            if np.all(np.isfinite(body.position)) and np.all(np.abs(body.position) < 1e15):
                all_positions.append(body.position.reshape(1, -1))
        
        # Вычислить центр масс системы для правильного центрирования
        total_mass = sum(body.mass for body in bodies)
        center = np.zeros(3)
        if total_mass > 0:
            for body in bodies:
                if np.all(np.isfinite(body.position)) and np.all(np.abs(body.position) < 1e12):
                    center += body.mass * body.position
            center /= total_mass
        
        # Вычислить максимальное расстояние между начальными позициями тел
        max_dist = 0
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i+1:]:
                if (np.all(np.isfinite(b1.position)) and np.all(np.isfinite(b2.position)) and
                    np.all(np.abs(b1.position) < 1e12) and np.all(np.abs(b2.position) < 1e12)):
                    dist = np.linalg.norm(b1.position - b2.position)
                    if dist < 1e12:  # Разумный предел
                        max_dist = max(max_dist, dist)
        
        # Использовать максимальное расстояние с оптимальным множителем
        # Используем те же параметры, что и в анимации для консистентности
        scale_factor = 1.2  # Тот же масштаб, что в анимации
        padding_factor = 0.3  # Тот же отступ, что в анимации
        
        if max_dist > 0:
            max_range = max_dist * scale_factor
        else:
            max_range = 5e8  # Fallback
        
        # Отступ для видимости
        padding = max_range * padding_factor
        
        # Установить границы относительно центра масс
        ax.set_xlim([center[0] - max_range - padding, center[0] + max_range + padding])
        ax.set_ylim([center[1] - max_range - padding, center[1] + max_range + padding])
        ax.set_zlim([center[2] - max_range - padding, center[2] + max_range + padding])
        
        # Построить траектории
        if show_orbits:
            for body in bodies:
                if len(body.trajectory) > 0:
                    traj = body.get_trajectory_array()
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                           color=body.color, alpha=0.6, linewidth=1.5,
                           label=f'{body.name}')
        
        # Показать текущие позиции в виде сфер (как в анимации)
        if show_bodies:
            for body in bodies:
                pos = body.position
                if np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 1e12):
                    # Вычислить размер сферы пропорционально радиусу тела
                    if body.radius > 0:
                        if max_dist > 0:
                            size_factor = (body.radius / max_dist) * scale_factor * 1000
                            size = max(50, min(int(size_factor), 1000))
                        else:
                            size = 200 if body.mass > 1e29 else 100
                    else:
                        size = 200 if body.mass > 1e29 else 100
                    
                    # Использовать расстояние между телами для масштабирования сфер
                    moon_distance = max_dist if max_dist > 0 else 3.844e8
                    sphere = self._create_sphere(pos[0], pos[1], pos[2], size, body.color, 
                                                moon_distance=moon_distance)
                    ax.add_collection3d(sphere)
        
        ax.set_xlabel('X (м)', fontsize=10)
        ax.set_ylabel('Y (м)', fontsize=10)
        ax.set_zlabel('Z (м)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Улучшенная легенда
        if show_orbits:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        
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
        
        # Вычислить центр для центрирования
        all_pos = []
        for body in bodies:
            if len(body.trajectory) > 0:
                all_pos.append(body.get_trajectory_array())
            all_pos.append(body.position.reshape(1, -1))
        
        if len(all_pos) > 0:
            all_pos = np.vstack(all_pos)
            center = np.mean(all_pos, axis=0)
        else:
            center = np.zeros(3)
        
        # Вычислить масштаб на основе начальных позиций (до построения графиков)
        max_dist = 0
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i+1:]:
                if (np.all(np.isfinite(b1.position)) and np.all(np.isfinite(b2.position))):
                    dist = np.linalg.norm(b1.position - b2.position)
                    if dist < 1e12:
                        max_dist = max(max_dist, dist)
        
        # Используем те же параметры, что и в анимации для консистентности
        scale_factor = 1.2  # Тот же масштаб, что в анимации
        padding_factor = 0.3  # Тот же отступ, что в анимации
        
        # Сначала построить траектории (относительно центра)
        for body in bodies:
            if len(body.trajectory) > 0:
                traj = body.get_trajectory_array()
                # Можно оставить абсолютные координаты, масштаб уже настроен
                ax.plot(traj[:, idx1], traj[:, idx2], 
                       color=body.color, alpha=0.7, linewidth=2,
                       label=f'{body.name}', zorder=1)
        
        # Затем показать текущие позиции (поверх траекторий)
        # Используем круги большего размера для лучшей видимости
        for body in bodies:
            # Вычислить размер пропорционально радиусу тела
            if body.radius > 0:
                if max_dist > 0:
                    size_factor = (body.radius / max_dist) * scale_factor * 1000
                    size = max(100, min(int(size_factor), 2000))
                else:
                    size = 300 if body.mass > 1e29 else 150
            else:
                size = 300 if body.mass > 1e29 else 150
            
            ax.scatter(body.position[idx1], body.position[idx2],
                      c=body.color, s=size, marker='o',
                      edgecolors='black', linewidths=2, zorder=10,
                      label=None)  # Не добавлять в легенду, чтобы избежать дублирования
        
        # Установить масштаб с оптимальным отступом
        if max_dist > 0:
            max_range = max_dist * scale_factor
            padding = max_range * padding_factor
            # Получить текущие границы
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
            ax.set_xlim([center_x - max_range - padding, center_x + max_range + padding])
            ax.set_ylim([center_y - max_range - padding, center_y + max_range + padding])
        
        labels = ['X', 'Y', 'Z']
        ax.set_xlabel(f'{labels[idx1]} (м)', fontsize=11)
        ax.set_ylabel(f'{labels[idx2]} (м)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return fig
    
    def _find_slowest_orbital_period(self, bodies: List[Body]) -> Tuple[Optional[Body], float]:
        """
        Найти планету с самым большим орбитальным периодом (самую медленную)
        
        Parameters:
        -----------
        bodies : List[Body]
            Список тел в системе
            
        Returns:
        --------
        Tuple[Optional[Body], float]
            (самая медленная планета, её орбитальный период в секундах)
        """
        if len(bodies) < 2:
            return None, 0.0
        
        # Найти центральное тело (обычно самое массивное)
        central_body = max(bodies, key=lambda b: b.mass)
        central_mass = central_body.mass
        
        slowest_body = None
        max_period = 0.0
        
        for body in bodies:
            if body == central_body:
                continue
            
            # Вычислить расстояние от центрального тела
            distance = np.linalg.norm(body.position - central_body.position)
            
            if distance > 0:
                # Вычислить орбитальный период по третьему закону Кеплера
                period = self.cosmic.kepler_third_law(distance, central_mass, body.mass)
                
                if period > max_period:
                    max_period = period
                    slowest_body = body
        
        return slowest_body, max_period
    
    def animate_simulation(self, bodies: List[Body], 
                          interval: int = 50,
                          figsize: Tuple[int, int] = (12, 10),
                          save_path: Optional[str] = None,
                          use_initial_positions: bool = True,
                          scale_factor: float = 25.0,
                          padding_factor: float = 1.5,
                          rotate_camera: bool = True,
                          camera_rotation_speed: float = 0.5,
                          sync_with_slowest: bool = False,
                          simulation_time_span: Optional[float] = None,
                          follow_central_body: bool = True,
                          show_center_of_mass: bool = False) -> FuncAnimation:
        """
        Создать анимацию симуляции
        
        Parameters:
        -----------
        bodies : List[Body]
            Список тел с траекториями
        interval : int
            Интервал между кадрами в миллисекундах
        sync_with_slowest : bool
            Синхронизировать скорость анимации с планетой с самым большим орбитальным периодом
        simulation_time_span : Optional[float]
            Общая длительность симуляции в секундах (нужно для синхронизации)
        follow_central_body : bool
            Следовать ли камерой за центральным телом (Солнце/Земля). 
            Если False, камера остается неподвижной, показывая движение всей системы
        show_center_of_mass : bool
            Показывать ли центр масс системы и его траекторию.
            Полезно для понимания движения системы Земля-Луна
            Список тел с траекториями
        interval : int
            Интервал между кадрами (мс)
        figsize : Tuple[int, int]
            Размер фигуры
        save_path : str, optional
            Путь для сохранения анимации
        scale_factor : float
            Множитель масштаба для визуализации
        padding_factor : float
            Множитель отступа вокруг тел
        rotate_camera : bool
            Вращать ли камеру вокруг системы для лучшего обзора
        camera_rotation_speed : float
            Скорость вращения камеры (градусов на кадр)
            
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
        
        # Синхронизация скорости анимации с самой медленной планетой
        if sync_with_slowest and simulation_time_span is not None:
            slowest_body, slowest_period = self._find_slowest_orbital_period(bodies)
            if slowest_body and slowest_period > 0:
                # Вычислить время симуляции на один кадр
                time_per_frame = simulation_time_span / max_traj_len
                
                # Желаемое время анимации для одного оборота самой медленной планеты (в секундах)
                # Например, 20 секунд анимации для одного оборота
                desired_animation_time_per_orbit = 20.0  # секунд
                
                # Вычислить, сколько кадров нужно для одного оборота
                frames_per_orbit = slowest_period / time_per_frame
                
                # Вычислить интервал между кадрами для синхронизации
                # interval в миллисекундах
                interval = int((desired_animation_time_per_orbit * 1000) / frames_per_orbit)
                
                # Ограничить интервал разумными значениями (10-500 мс)
                interval = max(10, min(500, interval))
                
                print(f"\nСинхронизация анимации:")
                print(f"  Самая медленная планета: {slowest_body.name}")
                print(f"  Орбитальный период: {slowest_period/(365.25*24*3600):.2f} лет")
                print(f"  Время на кадр симуляции: {time_per_frame:.2e} с")
                print(f"  Кадров на один оборот: {frames_per_orbit:.1f}")
                print(f"  Интервал между кадрами: {interval} мс")
                print(f"  Время анимации для одного оборота: {desired_animation_time_per_orbit:.1f} с")
            else:
                print("Не удалось найти планету для синхронизации, используется стандартный interval")
        else:
            # Вычислить временные метки для каждого кадра (если есть информация о времени)
            # Предполагаем, что время равномерно распределено между кадрами
            time_per_frame = 1.0  # По умолчанию 1 секунда на кадр
            if hasattr(bodies[0], 'trajectory') and len(bodies[0].trajectory) > 1:
                # Если есть информация о времени симуляции, можно использовать её
                # Пока используем просто номер кадра
                pass
        
        # Сохранить начальные позиции из траекторий для масштабирования
        # Используем первую точку траектории, если она есть, иначе текущую позицию
        initial_positions = []
        for body in bodies:
            if len(body.trajectory) > 0:
                initial_positions.append(body.trajectory[0].copy())
            else:
                initial_positions.append(body.position.copy())
        
        # Найти границы на основе начальных позиций и валидных траекторий
        all_positions = []
        valid_trajectories = []
        
        # Сначала добавить начальные позиции
        for pos in initial_positions:
            if np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 1e12):
                all_positions.append(pos.reshape(1, -1))
        
        # Затем добавить валидные части траекторий
        for body in bodies:
            if len(body.trajectory) > 0:
                traj = body.get_trajectory_array()
                # Фильтровать невалидные значения (NaN, Inf, слишком большие)
                # Используем разумный предел - не больше 1 а.е. (1.5e11 м)
                valid_mask = (np.isfinite(traj).all(axis=1) & 
                             (np.abs(traj) < 1e12).all(axis=1))
                traj_valid = traj[valid_mask]
                
                if len(traj_valid) > 0:
                    all_positions.append(traj_valid)
                    valid_trajectories.append((body, traj_valid))
        
        # Вычислить центр масс системы на основе НАЧАЛЬНЫХ позиций
        total_mass = sum(body.mass for body in bodies)
        center = np.zeros(3)
        if total_mass > 0:
            for i, body in enumerate(bodies):
                pos = initial_positions[i]
                if np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 1e12):
                    center += body.mass * pos
            center /= total_mass
        
        # Вычислить максимальное расстояние между телами для лучшего масштабирования
        # Это гарантирует, что оба тела будут видны раздельно
        max_dist_between_bodies = 0
        for i, pos1 in enumerate(initial_positions):
            for pos2 in initial_positions[i+1:]:
                if (np.all(np.isfinite(pos1)) and np.all(np.isfinite(pos2)) and
                    np.all(np.abs(pos1) < 1e12) and np.all(np.abs(pos2) < 1e12)):
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < 1e12:
                        max_dist_between_bodies = max(max_dist_between_bodies, dist)
        
        # Если не нашли расстояние между телами, использовать расстояние от центра масс
        if max_dist_between_bodies == 0:
            for pos in initial_positions:
                if np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 1e12):
                    dist_from_center = np.linalg.norm(pos - center)
                    if dist_from_center < 1e12:
                        max_dist_between_bodies = max(max_dist_between_bodies, dist_from_center * 2)
        
        # Использовать расстояние между телами с множителем масштаба
        if max_dist_between_bodies > 0:
            # Использовать переданный множитель масштаба
            max_range = max_dist_between_bodies * scale_factor
        else:
            # Fallback - использовать фиксированный размер
            max_range = 5e8  # 500,000 км
        
        # Увеличить отступ для лучшей видимости (использовать переданный множитель)
        padding = max_range * padding_factor
        
        # Вычислить границы относительно центра масс
        x_min = center[0] - max_range - padding
        x_max = center[0] + max_range + padding
        y_min = center[1] - max_range - padding
        y_max = center[1] + max_range + padding
        z_min = center[2] - max_range - padding
        z_max = center[2] + max_range + padding
        
        # Установить границы
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        # Отладочная информация (можно убрать после проверки)
        print(f"Масштабирование: расстояние между телами={max_dist_between_bodies/1000:.2f} км, "
              f"max_range={max_range/1000:.2f} км, padding={padding/1000:.2f} км")
        print(f"Границы: X=[{x_min/1000:.2f}, {x_max/1000:.2f}] км, "
              f"Y=[{y_min/1000:.2f}, {y_max/1000:.2f}] км, "
              f"Z=[{z_min/1000:.2f}, {z_max/1000:.2f}] км")
        
        # Сохранить центр для использования в анимации
        animation_center = center
        
        # Найти центральное тело для центрирования камеры (Солнце для солнечной системы, Земля для земля-луна)
        central_body = None
        # Сначала ищем Солнце
        for body in bodies:
            if 'солнце' in body.name.lower() or 'sun' in body.name.lower():
                central_body = body
                break
        # Если не нашли Солнце, ищем Землю
        if central_body is None:
            for body in bodies:
                if 'земля' in body.name.lower() or 'earth' in body.name.lower():
                    central_body = body
                    break
        # Если не нашли, используем самое массивное тело (обычно центральное)
        if central_body is None and len(bodies) > 0:
            central_body = max(bodies, key=lambda b: b.mass)
        
        # Инициализировать линии и точки
        lines = []
        points = []  # Используем постоянные точки вместо создания новых
        body_info = []
        
        # Для визуализации центра масс
        com_trajectory = []  # Траектория центра масс
        com_line = None
        com_point = None
        
        for body in bodies:
            if len(body.trajectory) > 0:
                # Линия для траектории
                line, = ax.plot([], [], [], color=body.color, alpha=0.6, 
                               linewidth=1.5, label=f'{body.name}')
                
                lines.append((line, body))
                
                # Создать постоянную точку для тела (будем обновлять её позицию)
                # Размер точек будет масштабироваться пропорционально расстояниям в системе
                # Для Солнца делаем сферу больше для лучшей видимости
                is_sun = 'солнце' in body.name.lower() or 'sun' in body.name.lower()
                
                if body.radius > 0:
                    if max_dist_between_bodies > 0:
                        size_factor = (body.radius / max_dist_between_bodies) * scale_factor * 1000
                        size = max(50, min(int(size_factor), 1000))
                        # Увеличить размер для Солнца
                        if is_sun:
                            size = max(200, int(size * 3))  # В 3 раза больше минимум
                    else:
                        min_radius = min(b.radius for b in bodies if b.radius > 0)
                        if min_radius > 0:
                            base_size = 150
                            size = int(base_size * (body.radius / min_radius) ** 0.6)
                            size = min(size, 500)
                            # Увеличить размер для Солнца
                            if is_sun:
                                size = max(300, int(size * 2))
                        else:
                            size = 150
                            if is_sun:
                                size = 300
                else:
                    size = 300 if body.mass > 1e29 else 150
                    # Увеличить размер для Солнца
                    if is_sun:
                        size = 500
                
                # Создать начальную сферу (будем обновлять её позицию)
                # Используем расстояние до Луны для масштабирования
                moon_distance = max_dist_between_bodies if max_dist_between_bodies > 0 else 3.844e8
                sphere = self._create_sphere(0, 0, 0, size, body.color, moon_distance=moon_distance)
                ax.add_collection3d(sphere)
                points.append((sphere, body))
                body_info.append((body, body.color, size, moon_distance))
        
        # Инициализировать визуализацию центра масс, если нужно
        if show_center_of_mass:
            com_line, = ax.plot([], [], [], color='red', alpha=0.4,
                               linewidth=2, linestyle='--', label='Центр масс')
            com_point, = ax.plot([], [], [], color='red', marker='x',
                                markersize=10, markeredgewidth=2, label='Центр масс (текущий)')
        
        ax.set_xlabel('X (м)', fontsize=10)
        ax.set_ylabel('Y (м)', fontsize=10)
        ax.set_zlabel('Z (м)', fontsize=10)
        ax.set_title('Анимация симуляции', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Начальный угол обзора камеры (фиксированный, очень близкий)
        initial_elev = 15  # Высота камеры (градусы) - еще уменьшено для максимально близкого обзора
        initial_azim = 45  # Азимут камеры (градусы)
        ax.view_init(elev=initial_elev, azim=initial_azim)
        
        def animate(frame):
            nonlocal com_trajectory  # Использовать внешнюю переменную
            frame_start_time = time.time()
            artists = []
            
            # Вычислить центр масс для текущего кадра
            if show_center_of_mass:
                total_mass = sum(body.mass for body in bodies)
                com = np.zeros(3)
                if total_mass > 0:
                    for body in bodies:
                        if frame < len(body.trajectory):
                            pos = body.trajectory[frame]
                            if np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 1e12):
                                com += body.mass * pos
                    com /= total_mass
                    com_trajectory.append(com.copy())
                    
                    # Обновить траекторию центра масс
                    if len(com_trajectory) > 1:
                        com_array = np.array(com_trajectory)
                        com_line.set_data(com_array[:, 0], com_array[:, 1])
                        com_line.set_3d_properties(com_array[:, 2])
                        artists.append(com_line)
                    
                    # Обновить текущую позицию центра масс
                    com_point.set_data([com[0]], [com[1]])
                    com_point.set_3d_properties([com[2]])
                    artists.append(com_point)
            
            # Обновить центр камеры на позицию центрального тела (Солнце или Земля)
            # Только если включено следование за центральным телом
            if follow_central_body and central_body and frame < len(central_body.trajectory):
                central_pos = central_body.trajectory[frame]
                if (np.all(np.isfinite(central_pos)) and np.all(np.abs(central_pos) < 1e12)):
                    # Обновить границы графика так, чтобы центральное тело было в центре
                    x_min = central_pos[0] - max_range - padding
                    x_max = central_pos[0] + max_range + padding
                    y_min = central_pos[1] - max_range - padding
                    y_max = central_pos[1] + max_range + padding
                    z_min = central_pos[2] - max_range - padding
                    z_max = central_pos[2] + max_range + padding
                    
                    ax.set_xlim([x_min, x_max])
                    ax.set_ylim([y_min, y_max])
                    ax.set_zlim([z_min, z_max])
            
            # Вращать камеру вокруг системы только если включено
            if rotate_camera:
                azim = initial_azim + frame * camera_rotation_speed
                azim = azim % 360
                ax.view_init(elev=initial_elev, azim=azim)
            
            # Обновить траектории (только валидные)
            # Оптимизация: ограничиваем количество точек для отрисовки
            for line, body in lines:
                if frame < len(body.trajectory):
                    # Ограничиваем количество точек для отрисовки (последние N точек)
                    # Это ускоряет рендеринг при большом количестве точек
                    # Увеличить количество точек траектории для лучшей видимости следа
                    # Для солнечной системы показываем больше точек
                    max_traj_points = min(frame + 1, 2000)  # Максимум 2000 точек для отрисовки (увеличено)
                    start_idx = max(0, frame + 1 - max_traj_points)
                    
                    traj = np.array(body.trajectory[start_idx:frame+1])
                    valid_mask = (np.isfinite(traj).all(axis=1) & 
                                 (np.abs(traj) < 1e12).all(axis=1))
                    traj_valid = traj[valid_mask]
                    
                    if len(traj_valid) > 0:
                        line.set_data(traj_valid[:, 0], traj_valid[:, 1])
                        line.set_3d_properties(traj_valid[:, 2])
                        artists.append(line)
            
            # Обновить позиции сфер
            for i, (sphere, body) in enumerate(points):
                if frame < len(body.trajectory):
                    pos = body.trajectory[frame]
                    if (np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 1e12)):
                        # Удалить старую сферу
                        try:
                            sphere.remove()
                        except:
                            pass
                        
                        # Найти размер сферы и расстояние до Луны из body_info
                        info = next(((s, md) for b, c, s, md in body_info if b == body), (100, 3.844e8))
                        size, moon_distance = info
                        # Создать новую сферу в новой позиции
                        new_sphere = self._create_sphere(pos[0], pos[1], pos[2], size, body.color, 
                                                        moon_distance=moon_distance)
                        ax.add_collection3d(new_sphere)
                        # Заменить в списке
                        points[i] = (new_sphere, body)
                        artists.append(new_sphere)
            
            # Вывести информацию о кадре с временной меткой
            frame_end_time = time.time()
            frame_time = frame_end_time - frame_start_time
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            
            if max_traj_len > 1:
                progress = (frame / max_traj_len) * 100
                print(f"[{timestamp}] Кадр {frame}/{max_traj_len-1} ({progress:.1f}%) | "
                      f"Время обработки: {frame_time*1000:.2f} мс", end='\r')
            
            # Для важных кадров выводим подробную информацию
            if frame == 0 or frame == max_traj_len - 1 or frame % 100 == 0:
                print()  # Новая строка
                for point, body in points:
                    if frame < len(body.trajectory):
                        pos = body.trajectory[frame]
                        if (np.all(np.isfinite(pos)) and np.all(np.abs(pos) < 1e12)):
                            dist_from_center = np.linalg.norm(pos - animation_center)
                            print(f"  {body.name}: позиция={pos/1000}, "
                                  f"расстояние от центра={dist_from_center/1000:.2f} км")
            
            return artists
        
        # Оптимизация: пропускаем кадры для ускорения анимации
        # Показываем каждый N-й кадр вместо всех кадров
        frame_skip = 1  # Показывать каждый кадр (можно увеличить до 2-5 для ускорения)
        frames_to_show = list(range(0, max_traj_len, frame_skip))
        
        # blit=True ускоряет рендеринг, обновляя только измененные части
        anim = FuncAnimation(fig, animate, frames=frames_to_show, 
                           interval=interval, blit=True, repeat=True, cache_frame_data=False)
        
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
        
        ax.plot(times, energies, 'b-', linewidth=2, label='Полная энергия')
        ax.set_xlabel('Время (с)', fontsize=11)
        ax.set_ylabel('Полная энергия (Дж)', fontsize=11)
        ax.set_title('Сохранение энергии системы', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Показать относительное изменение
        if len(energies) > 0:
            initial_energy = energies[0]
            relative_change = (energies[-1] - initial_energy) / abs(initial_energy) * 100
            # Форматировать время в более читаемый вид
            if times[-1] > 86400:  # Больше дня
                time_unit = "дней"
                time_value = times[-1] / 86400
            elif times[-1] > 3600:  # Больше часа
                time_unit = "часов"
                time_value = times[-1] / 3600
            else:
                time_unit = "секунд"
                time_value = times[-1]
            
            info_text = f'Время симуляции: {time_value:.2f} {time_unit}\n'
            info_text += f'Изменение энергии: {relative_change:.2e}%'
            
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   fontsize=9)
        
        ax.legend(loc='best', fontsize=9)
        
        return fig


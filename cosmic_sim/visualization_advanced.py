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
        # Размер сферы масштабируется относительно расстояния до Луны
        # Оптимальный размер для видимости без перекрытия
        # Размер пропорционален размеру точки
        max_size = 1000  # Максимальный размер точки
        # Нормализуем размер точки
        normalized_size = min(size / max_size, 1.0)  # Ограничиваем до 1.0
        # Используем 3 расстояния до Луны как оптимальный размер
        sphere_radius = normalized_size * 3.0 * moon_distance
        
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
        
        # Использовать максимальное расстояние с ОЧЕНЬ большим множителем
        if max_dist > 0:
            max_range = max_dist * 25.0
        else:
            max_range = 5e8  # Fallback
        
        # Увеличить отступ для лучшей видимости (теперь 1.5 = 150%)
        padding = max_range * 1.5
        
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
        
        # Показать текущие позиции (с большим размером для центральных тел)
        if show_bodies:
            for body in bodies:
                pos = body.position
                # Увеличить размер для тел с большой массой (например, Солнце)
                size = 200 if body.mass > 1e29 else 100
                ax.scatter(pos[0], pos[1], pos[2], 
                          c=body.color, s=size, marker='o',
                          edgecolors='black', linewidths=1.5, zorder=10)
        
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
        
        # Сначала построить траектории (относительно центра)
        for body in bodies:
            if len(body.trajectory) > 0:
                traj = body.get_trajectory_array()
                # Можно оставить абсолютные координаты, масштаб уже настроен
                ax.plot(traj[:, idx1], traj[:, idx2], 
                       color=body.color, alpha=0.7, linewidth=2,
                       label=f'{body.name}', zorder=1)
        
        # Затем показать текущие позиции (поверх траекторий)
        for body in bodies:
            # Увеличить размер для тел с большой массой
            size = 300 if body.mass > 1e29 else 150
            ax.scatter(body.position[idx1], body.position[idx2],
                      c=body.color, s=size, marker='o',
                      edgecolors='black', linewidths=2, zorder=10,
                      label=None)  # Не добавлять в легенду, чтобы избежать дублирования
        
        # Вычислить масштаб на основе начальных позиций
        max_dist = 0
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i+1:]:
                if (np.all(np.isfinite(b1.position)) and np.all(np.isfinite(b2.position))):
                    dist = np.linalg.norm(b1.position - b2.position)
                    if dist < 1e12:
                        max_dist = max(max_dist, dist)
        
        # Установить масштаб с большим отступом
        if max_dist > 0:
            max_range = max_dist * 25.0
            padding = max_range * 1.5
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
    
    def animate_simulation(self, bodies: List[Body], 
                          interval: int = 50,
                          figsize: Tuple[int, int] = (12, 10),
                          save_path: Optional[str] = None,
                          use_initial_positions: bool = True,
                          scale_factor: float = 25.0,
                          padding_factor: float = 1.5,
                          rotate_camera: bool = True,
                          camera_rotation_speed: float = 0.5) -> FuncAnimation:
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
        
        # Найти Землю для центрирования камеры на ней
        earth_body = None
        for body in bodies:
            if 'земля' in body.name.lower() or 'earth' in body.name.lower():
                earth_body = body
                break
        # Если не нашли по имени, используем первое тело (обычно это Земля)
        if earth_body is None and len(bodies) > 0:
            earth_body = bodies[0]
        
        # Инициализировать линии и точки
        lines = []
        points = []  # Используем постоянные точки вместо создания новых
        body_info = []
        
        for body in bodies:
            if len(body.trajectory) > 0:
                # Линия для траектории
                line, = ax.plot([], [], [], color=body.color, alpha=0.6, 
                               linewidth=1.5, label=f'{body.name}')
                
                lines.append((line, body))
                
                # Создать постоянную точку для тела (будем обновлять её позицию)
                # Размер точек будет масштабироваться пропорционально расстояниям в системе
                if body.radius > 0:
                    if max_dist_between_bodies > 0:
                        size_factor = (body.radius / max_dist_between_bodies) * scale_factor * 1000
                        size = max(50, min(int(size_factor), 1000))
                    else:
                        min_radius = min(b.radius for b in bodies if b.radius > 0)
                        if min_radius > 0:
                            base_size = 150
                            size = int(base_size * (body.radius / min_radius) ** 0.6)
                            size = min(size, 500)
                        else:
                            size = 150
                else:
                    size = 300 if body.mass > 1e29 else 150
                
                # Создать начальную сферу (будем обновлять её позицию)
                # Используем расстояние до Луны для масштабирования
                moon_distance = max_dist_between_bodies if max_dist_between_bodies > 0 else 3.844e8
                sphere = self._create_sphere(0, 0, 0, size, body.color, moon_distance=moon_distance)
                ax.add_collection3d(sphere)
                points.append((sphere, body))
                body_info.append((body, body.color, size, moon_distance))
        
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
            frame_start_time = time.time()
            artists = []
            
            # Обновить центр камеры на позицию Земли
            if earth_body and frame < len(earth_body.trajectory):
                earth_pos = earth_body.trajectory[frame]
                if (np.all(np.isfinite(earth_pos)) and np.all(np.abs(earth_pos) < 1e12)):
                    # Обновить границы графика так, чтобы Земля была в центре
                    x_min = earth_pos[0] - max_range - padding
                    x_max = earth_pos[0] + max_range + padding
                    y_min = earth_pos[1] - max_range - padding
                    y_max = earth_pos[1] + max_range + padding
                    z_min = earth_pos[2] - max_range - padding
                    z_max = earth_pos[2] + max_range + padding
                    
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
                    max_traj_points = min(frame + 1, 300)  # Максимум 300 точек для отрисовки (уменьшено)
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


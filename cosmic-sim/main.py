import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import math

class CosmicSim:
    def __init__(self):
        # Физические константы
        self.AU = 1.495978707e11  # астрономическая единица (м)
        self.G = 6.67430e-11      # гравитационная постоянная
        self.c = 299792458        # скорость света
        self.R_earth = 6371000    # радиус Земли (м)
        
    def parallax_distance(self, baseline, parallax_angle_rad):
        """
        Вычисление расстояния через параллакс
        D = B / θ
        """
        return baseline / parallax_angle_rad
    
    def angular_size(self, physical_size, distance):
        """
        Вычисление углового размера объекта
        θ = R / D
        """
        return physical_size / distance
    
    def spherical_to_cartesian(self, distance, ra_rad, dec_rad):
        """
        Преобразование сферических координат в декартовы
        """
        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad) 
        z = distance * np.sin(dec_rad)
        return np.array([x, y, z])
    
    def cartesian_to_spherical(self, x, y, z):
        """
        Преобразование декартовых координат в сферические
        """
        distance = np.sqrt(x**2 + y**2 + z**2)
        ra = np.arctan2(y, x)
        dec = np.arcsin(z / distance)
        return distance, ra, dec
    
    def gravitational_force(self, m1, m2, r):
        """
        Сила гравитации между двумя массами
        F = G * m1 * m2 / r^2
        """
        return self.G * m1 * m2 / r**2
    
    def orbital_velocity(self, central_mass, distance):
        """
        Орбитальная скорость для круговой орбиты
        v = sqrt(G * M / r)
        """
        return np.sqrt(self.G * central_mass / distance)
    
    def kepler_third_law(self, semi_major_axis, mass1, mass2):
        """
        Третий закон Кеплера
        T^2 = (4π^2 / G(M1 + M2)) * a^3
        """
        total_mass = mass1 + mass2
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (self.G * total_mass))
        return period

class OrbitalSimulator:
    def __init__(self):
        self.cosmic = CosmicSim()
    
    def two_body_derivatives(self, t, y, mass1, mass2):
        """
        Производные для системы двух тел
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
        """
        y0 = np.concatenate([initial_pos1, initial_pos2, initial_vel1, initial_vel2])
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        solution = solve_ivp(self.two_body_derivatives, t_span, y0, 
                           args=(mass1, mass2), t_eval=t_eval, rtol=1e-8)
        
        return solution.t, solution.y

class ParallaxVisualizer:
    def __init__(self):
        self.cosmic = CosmicSim()
    
    def create_parallax_scene(self, star_distance_ly, baseline_au=2):
        """
        Создание сцены для демонстрации параллакса
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

# Примеры использования библиотеки
def demo_parallax():
    """Демонстрация вычисления параллакса"""
    cosmic = CosmicSim()
    visualizer = ParallaxVisualizer()
    
    # Пример: параллакс ближайшей звезды
    star_distance = 4.37  # Расстояние до Проксимы Центавры в световых годах
    baseline = 2 * cosmic.AU  # Диаметр орбиты Земли
    
    parallax_distance = cosmic.parallax_distance(baseline, 1)  # Для угла в 1 радиан
    actual_parallax_angle = baseline / (star_distance * 9.461e15)
    
    print("=== ДЕМОНСТРАЦИЯ ПАРАЛЛАКСА ===")
    print(f"Расстояние до звезды: {star_distance} световых лет")
    print(f"База измерения: {baseline/cosmic.AU:.1f} а.е.")
    print(f"Параллактический угол: {actual_parallax_angle*206265:.2f} угловых секунд")
    print(f"Расстояние через параллакс: {cosmic.parallax_distance(baseline, actual_parallax_angle)/9.461e15:.2f} световых лет")
    
    # Визуализация
    visualizer.plot_parallax(star_distance)
    plt.show()

def demo_orbital_mechanics():
    """Демонстрация орбитальной механики"""
    cosmic = CosmicSim()
    simulator = OrbitalSimulator()
    
    # Параметры системы Земля-Солнце
    mass_sun = 1.989e30
    mass_earth = 5.972e24
    
    # Начальные условия (Земля на круговой орбите)
    earth_distance = cosmic.AU
    orbital_velocity = cosmic.orbital_velocity(mass_sun, earth_distance)
    
    initial_pos_earth = np.array([earth_distance, 0, 0])
    initial_vel_earth = np.array([0, orbital_velocity, 0])
    initial_pos_sun = np.array([0, 0, 0])
    initial_vel_sun = np.array([0, 0, 0])
    
    # Симуляция на 1 год
    t_span = [0, 365.25 * 24 * 3600]  # 1 год в секундах
    
    print("\n=== ДЕМОНСТРАЦИЯ ОРБИТАЛЬНОЙ МЕХАНИКИ ===")
    print(f"Орбитальная скорость Земли: {orbital_velocity/1000:.1f} км/с")
    print(f"Период орбиты (расчетный): {cosmic.kepler_third_law(earth_distance, mass_sun, mass_earth)/(24*3600):.1f} дней")
    
    # Запуск симуляции
    t, y = simulator.simulate_two_body(
        mass_sun, mass_earth,
        initial_pos_sun, initial_pos_earth,
        initial_vel_sun, initial_vel_earth,
        t_span, n_points=1000
    )
    
    # Визуализация орбиты
    earth_trajectory = y[3:6].T  # Позиция Земли
    
    plt.figure(figsize=(10, 8))
    plt.plot(earth_trajectory[:, 0], earth_trajectory[:, 1], 'b-', label='Орбита Земли')
    plt.plot(0, 0, 'yo', markersize=15, label='Солнце')
    plt.plot(earth_trajectory[0, 0], earth_trajectory[0, 1], 'go', label='Начальное положение')
    plt.xlabel('X (м)')
    plt.ylabel('Y (м)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Орбита Земли вокруг Солнца')
    plt.show()

def demo_combined_effects():
    """Демонстрация комбинированных эффектов"""
    cosmic = CosmicSim()
    
    # Пример: измерение массы через параллакс + орбитальную динамику
    print("\n=== КОМБИНИРОВАННЫЕ ЭФФЕКТЫ ===")
    
    # Для двойной звездной системы
    orbital_period = 10 * 365.25 * 24 * 3600  # 10 лет в секундах
    semi_major_axis = 10 * cosmic.AU  # 10 а.е.
    
    # Вычисление суммарной массы через закон Кеплера
    total_mass = (4 * np.pi**2 * semi_major_axis**3) / (cosmic.G * orbital_period**2)
    
    print(f"Период орбиты: 10 лет")
    print(f"Большая полуось: 10 а.е.")
    print(f"Суммарная масса системы: {total_mass/cosmic.mass_sun:.2f} масс Солнца")

if __name__ == "__main__":
    # Запуск демонстраций
    demo_parallax()
    demo_orbital_mechanics() 
    demo_combined_effects()

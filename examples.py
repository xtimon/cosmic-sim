"""
Примеры использования библиотеки cosmic-sim
"""
import numpy as np
import matplotlib.pyplot as plt
from cosmic_sim import CosmicSim, OrbitalSimulator, ParallaxVisualizer


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


"""
Полномасштабная симуляция космических систем
"""
import numpy as np
import matplotlib.pyplot as plt
from cosmic_sim import (
    NBodySimulator, 
    SystemPresets, 
    AdvancedVisualizer,
    Body,
    CosmicSim
)


def simulate_solar_system():
    """Полномасштабная симуляция Солнечной системы"""
    print("=" * 60)
    print("ПОЛНОМАСШТАБНАЯ СИМУЛЯЦИЯ СОЛНЕЧНОЙ СИСТЕМЫ")
    print("=" * 60)
    
    # Создать предустановленную систему
    presets = SystemPresets()
    bodies = presets.create_solar_system(include_outer_planets=False)  # Только внутренние планеты
    
    # Создать симулятор
    simulator = NBodySimulator(bodies)
    
    # Параметры симуляции
    # Симулируем 1 год (365.25 дней)
    one_year = 365.25 * 24 * 3600  # секунд
    t_span = (0, one_year)
    n_points = 2000  # Больше точек для плавности
    
    print(f"\nЗапуск симуляции на {one_year/(365.25*24*3600):.2f} лет...")
    print(f"Количество тел: {len(bodies)}")
    print(f"Точек вычисления: {n_points}")
    
    # Начальная энергия
    initial_energy = simulator.get_total_energy()
    print(f"\nНачальная полная энергия: {initial_energy:.2e} Дж")
    
    # Запустить симуляцию
    times, states = simulator.simulate(t_span, n_points=n_points, save_trajectory=True)
    
    # Финальная энергия
    final_energy = simulator.get_total_energy()
    energy_change = (final_energy - initial_energy) / abs(initial_energy) * 100
    print(f"Финальная полная энергия: {final_energy:.2e} Дж")
    print(f"Изменение энергии: {energy_change:.6f}%")
    
    # Центр масс
    com = simulator.get_center_of_mass()
    print(f"\nЦентр масс системы: [{com[0]/simulator.cosmic.AU:.6f}, "
          f"{com[1]/simulator.cosmic.AU:.6f}, {com[2]/simulator.cosmic.AU:.6f}] а.е.")
    
    # Визуализация
    visualizer = AdvancedVisualizer()
    
    # 3D траектории
    print("\nСоздание 3D визуализации...")
    fig1 = visualizer.plot_3d_trajectories(bodies, title="Солнечная система - 3D траектории")
    plt.savefig('solar_system_3d.png', dpi=150, bbox_inches='tight')
    print("✓ Сохранено: solar_system_3d.png")
    
    # 2D проекция XY
    print("Создание 2D проекции...")
    fig2 = visualizer.plot_2d_projection(bodies, plane='xy', 
                                        title="Солнечная система - проекция XY")
    plt.savefig('solar_system_2d.png', dpi=150, bbox_inches='tight')
    print("✓ Сохранено: solar_system_2d.png")
    
    # Сохранение энергии
    print("Создание графика сохранения энергии...")
    fig3 = visualizer.plot_energy_conservation(simulator, times)
    plt.savefig('energy_conservation.png', dpi=150, bbox_inches='tight')
    print("✓ Сохранено: energy_conservation.png")
    
    plt.show()
    
    return simulator, bodies


def simulate_binary_star_system():
    """Симуляция двойной звездной системы"""
    print("\n" + "=" * 60)
    print("СИМУЛЯЦИЯ ДВОЙНОЙ ЗВЕЗДНОЙ СИСТЕМЫ")
    print("=" * 60)
    
    presets = SystemPresets()
    bodies = presets.create_binary_star_system(separation_au=5.0)
    
    simulator = NBodySimulator(bodies)
    
    # Симулируем несколько орбитальных периодов
    # Приблизительный период для двойной системы
    total_mass = sum(b.mass for b in bodies)
    separation = np.linalg.norm(bodies[1].position - bodies[0].position)
    period = 2 * np.pi * np.sqrt(separation**3 / (simulator.cosmic.G * total_mass))
    
    t_span = (0, 3 * period)  # 3 периода
    n_points = 1500
    
    print(f"\nОрбитальный период: {period/(365.25*24*3600):.2f} лет")
    print(f"Симуляция на {3} периода...")
    
    times, states = simulator.simulate(t_span, n_points=n_points, save_trajectory=True)
    
    visualizer = AdvancedVisualizer()
    
    # 3D визуализация
    fig1 = visualizer.plot_3d_trajectories(bodies, title="Двойная звездная система")
    plt.savefig('binary_star_3d.png', dpi=150, bbox_inches='tight')
    
    # 2D проекция
    fig2 = visualizer.plot_2d_projection(bodies, plane='xy', 
                                        title="Двойная звездная система - проекция XY")
    plt.savefig('binary_star_2d.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return simulator, bodies


def simulate_custom_system():
    """Симуляция пользовательской системы"""
    print("\n" + "=" * 60)
    print("СИМУЛЯЦИЯ ПОЛЬЗОВАТЕЛЬСКОЙ СИСТЕМЫ")
    print("=" * 60)
    
    cosmic = CosmicSim()
    
    # Создать систему с центральным телом и несколькими спутниками
    bodies = []
    
    # Центральное тело (звезда)
    bodies.append(Body(
        name="Центральная звезда",
        mass=1.0 * cosmic.mass_sun,
        position=np.array([0, 0, 0]),
        velocity=np.array([0, 0, 0]),
        radius=6.96e8,
        color='yellow'
    ))
    
    # Несколько планет на разных орбитах
    planet_data = [
        (0.5, 'blue', "Планета 1"),
        (1.0, 'green', "Планета 2"),
        (1.5, 'red', "Планета 3"),
    ]
    
    for distance_au, color, name in planet_data:
        distance = distance_au * cosmic.AU
        v_orbital = cosmic.orbital_velocity(cosmic.mass_sun, distance)
        
        # Добавить небольшой наклон орбиты
        angle = len(bodies) * np.pi / 6  # Разные углы
        
        bodies.append(Body(
            name=name,
            mass=1e24,  # Масса планеты
            position=np.array([distance * np.cos(angle), distance * np.sin(angle), 0]),
            velocity=np.array([-v_orbital * np.sin(angle), v_orbital * np.cos(angle), 0]),
            radius=distance * 0.01,
            color=color
        ))
    
    simulator = NBodySimulator(bodies)
    
    one_year = 365.25 * 24 * 3600
    t_span = (0, one_year)
    n_points = 2000
    
    print(f"\nСимуляция пользовательской системы...")
    print(f"Количество тел: {len(bodies)}")
    
    times, states = simulator.simulate(t_span, n_points=n_points, save_trajectory=True)
    
    visualizer = AdvancedVisualizer()
    
    fig1 = visualizer.plot_3d_trajectories(bodies, title="Пользовательская система")
    fig2 = visualizer.plot_2d_projection(bodies, plane='xy', title="Пользовательская система - XY")
    
    plt.show()
    
    return simulator, bodies


def create_animation_example():
    """Создать анимированную симуляцию"""
    print("\n" + "=" * 60)
    print("СОЗДАНИЕ АНИМАЦИИ")
    print("=" * 60)
    
    presets = SystemPresets()
    bodies = presets.create_earth_moon_system()
    
    simulator = NBodySimulator(bodies)
    
    # Симулируем 1 месяц (период обращения Луны ~27 дней)
    one_month = 27 * 24 * 3600
    t_span = (0, one_month)
    n_points = 500  # Меньше точек для анимации
    
    print("\nЗапуск симуляции системы Земля-Луна...")
    times, states = simulator.simulate(t_span, n_points=n_points, save_trajectory=True)
    
    visualizer = AdvancedVisualizer()
    
    print("Создание анимации...")
    anim = visualizer.animate_simulation(bodies, interval=50)
    
    print("✓ Анимация создана! Закройте окно для продолжения.")
    plt.show()
    
    return anim


def main():
    """Главная функция для запуска различных симуляций"""
    print("\n" + "=" * 60)
    print("ПОЛНОМАСШТАБНАЯ СИМУЛЯЦИЯ КОСМИЧЕСКИХ СИСТЕМ")
    print("=" * 60)
    print("\nДоступные симуляции:")
    print("1. Солнечная система (внутренние планеты)")
    print("2. Двойная звездная система")
    print("3. Пользовательская система")
    print("4. Анимация системы Земля-Луна")
    print("5. Все симуляции")
    
    choice = input("\nВыберите симуляцию (1-5): ").strip()
    
    if choice == "1":
        simulate_solar_system()
    elif choice == "2":
        simulate_binary_star_system()
    elif choice == "3":
        simulate_custom_system()
    elif choice == "4":
        create_animation_example()
    elif choice == "5":
        simulate_solar_system()
        simulate_binary_star_system()
        simulate_custom_system()
    else:
        print("Неверный выбор. Запуск симуляции Солнечной системы по умолчанию...")
        simulate_solar_system()


if __name__ == "__main__":
    # Можно запустить конкретную симуляцию напрямую
    # или использовать интерактивное меню
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "solar":
            simulate_solar_system()
        elif sys.argv[1] == "binary":
            simulate_binary_star_system()
        elif sys.argv[1] == "custom":
            simulate_custom_system()
        elif sys.argv[1] == "animation":
            create_animation_example()
        else:
            main()
    else:
        # Запуск по умолчанию - Солнечная система
        simulate_solar_system()


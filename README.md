# Cosmic Sim

Библиотека для симуляции космических явлений и орбитальной механики на Python.

## Описание

Cosmic Sim предоставляет инструменты для:
- Вычисления параллакса звезд
- Симуляции орбитальной механики систем двух тел
- Преобразования координат (сферические ↔ декартовы)
- Визуализации космических явлений
- Работы с физическими константами и законами небесной механики

## Установка

### Установка из исходников

```bash
# Клонируйте репозиторий
git clone https://github.com/yourusername/cosmic-sim.git
cd cosmic-sim

# Установите пакет
pip install .
```

### Установка в режиме разработки

```bash
pip install -e .
```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

## Быстрый старт

```python
from cosmic_sim import CosmicSim, OrbitalSimulator, ParallaxVisualizer
import numpy as np

# Создание экземпляра для работы с физическими константами
cosmic = CosmicSim()

# Вычисление орбитальной скорости
mass_sun = 1.989e30  # кг
distance = cosmic.AU  # 1 астрономическая единица
velocity = cosmic.orbital_velocity(mass_sun, distance)
print(f"Орбитальная скорость: {velocity/1000:.1f} км/с")

# Симуляция орбиты
simulator = OrbitalSimulator()
mass_earth = 5.972e24  # кг

# Начальные условия
initial_pos_earth = np.array([cosmic.AU, 0, 0])
initial_vel_earth = np.array([0, velocity, 0])
initial_pos_sun = np.array([0, 0, 0])
initial_vel_sun = np.array([0, 0, 0])

# Симуляция на 1 год
t_span = [0, 365.25 * 24 * 3600]
t, y = simulator.simulate_two_body(
    mass_sun, mass_earth,
    initial_pos_sun, initial_pos_earth,
    initial_vel_sun, initial_vel_earth,
    t_span
)

# Визуализация параллакса
visualizer = ParallaxVisualizer()
fig = visualizer.plot_parallax(star_distance_ly=4.37)  # Проксима Центавра
```

## Структура библиотеки

### `CosmicSim`

Основной класс с физическими константами и базовыми вычислениями:

- `parallax_distance(baseline, parallax_angle_rad)` - вычисление расстояния через параллакс
- `angular_size(physical_size, distance)` - вычисление углового размера
- `spherical_to_cartesian(distance, ra_rad, dec_rad)` - преобразование координат
- `cartesian_to_spherical(x, y, z)` - обратное преобразование
- `gravitational_force(m1, m2, r)` - сила гравитации
- `orbital_velocity(central_mass, distance)` - орбитальная скорость
- `kepler_third_law(semi_major_axis, mass1, mass2)` - третий закон Кеплера

### `OrbitalSimulator`

Класс для симуляции орбитальной динамики:

- `simulate_two_body(...)` - симуляция системы двух тел

### `ParallaxVisualizer`

Класс для визуализации параллакса:

- `plot_parallax(star_distance_ly, baseline_au)` - визуализация параллакса

## Примеры

Полные примеры использования находятся в файле `examples.py`:

```bash
python examples.py
```

## Требования

- Python >= 3.7
- numpy >= 1.19.0
- scipy >= 1.5.0
- matplotlib >= 3.3.0

## Лицензия

MIT License

## Автор

Your Name

## Вклад в проект

Приветствуются pull requests и issues!


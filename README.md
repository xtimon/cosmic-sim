# Cosmic Sim 🌌

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Библиотека для полномасштабной симуляции космических явлений и орбитальной механики на Python.

## ✨ Основные возможности

- **🚀 Полномасштабная симуляция N тел** - симуляция произвольного количества небесных тел с высокой точностью
- **🌍 Предустановленные системы** - готовые конфигурации (Солнечная система, двойные звезды, Земля-Луна)
- **📊 Продвинутая визуализация** - 3D траектории, 2D проекции, анимации, графики энергии
- **💾 Сохранение и загрузка** - экспорт состояний в JSON, траекторий в CSV
- **🔬 Физические вычисления** - параллакс, орбитальная механика, законы Кеплера
- **📐 Преобразования координат** - сферические ↔ декартовы
- **⚡ Оптимизированные алгоритмы** - численное интегрирование с настраиваемой точностью

## 📦 Установка

### Установка из исходников

```bash
# Клонируйте репозиторий
git clone https://github.com/xtimon/cosmic-sim.git
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

## 🚀 Быстрый старт

### Полномасштабная симуляция Солнечной системы

```python
from cosmic_sim import NBodySimulator, SystemPresets, AdvancedVisualizer
import matplotlib.pyplot as plt

# Создать Солнечную систему
presets = SystemPresets()
bodies = presets.create_solar_system(include_outer_planets=False)

# Запустить симуляцию на 1 год
simulator = NBodySimulator(bodies)
times, states = simulator.simulate(
    t_span=(0, 365.25*24*3600),  # 1 год в секундах
    n_points=2000,
    save_trajectory=True
)

# Визуализация 3D траекторий
visualizer = AdvancedVisualizer()
fig = visualizer.plot_3d_trajectories(bodies, title="Солнечная система")
plt.show()

# Анализ энергии
total_energy = simulator.get_total_energy()
print(f"Полная энергия системы: {total_energy:.2e} Дж")
```

### Создание пользовательской системы

```python
from cosmic_sim import Body, NBodySimulator, AdvancedVisualizer
import numpy as np
import matplotlib.pyplot as plt

# Создать центральное тело
star = Body(
    name="Звезда",
    mass=1.989e30,  # кг
    position=np.array([0, 0, 0]),
    velocity=np.array([0, 0, 0]),
    color='yellow'
)

# Создать планету
planet = Body(
    name="Планета",
    mass=5.972e24,  # кг
    position=np.array([1.5e11, 0, 0]),  # 1 а.е.
    velocity=np.array([0, 30000, 0]),  # м/с
    color='blue'
)

# Симуляция
simulator = NBodySimulator([star, planet])
times, states = simulator.simulate(
    t_span=(0, 365.25*24*3600),
    n_points=1000,
    save_trajectory=True
)

# Визуализация
visualizer = AdvancedVisualizer()
fig = visualizer.plot_2d_projection([star, planet], plane='xy')
plt.show()
```

### Базовые вычисления

```python
from cosmic_sim import CosmicSim
import numpy as np

cosmic = CosmicSim()

# Орбитальная скорость
mass_sun = 1.989e30  # кг
distance = cosmic.AU  # 1 астрономическая единица
velocity = cosmic.orbital_velocity(mass_sun, distance)
print(f"Орбитальная скорость на 1 а.е.: {velocity/1000:.1f} км/с")

# Период орбиты (закон Кеплера)
period = cosmic.kepler_third_law(distance, mass_sun, 5.972e24)
print(f"Период орбиты: {period/(365.25*24*3600):.2f} лет")

# Параллакс
baseline = 2 * cosmic.AU  # Диаметр орбиты Земли
parallax_angle = 0.772  # угловых секунд для Проксимы Центавры
distance = cosmic.parallax_distance(baseline, np.radians(parallax_angle/3600))
print(f"Расстояние: {distance/9.461e15:.2f} световых лет")
```

## 📚 Структура библиотеки

### Основные классы

#### `NBodySimulator` ⭐

Класс для полномасштабной симуляции произвольного количества тел:

```python
simulator = NBodySimulator(bodies)
times, states = simulator.simulate(
    t_span=(t_start, t_end),
    n_points=1000,
    rtol=1e-8,
    save_trajectory=True
)

# Анализ системы
energy = simulator.get_total_energy()
com = simulator.get_center_of_mass()
momentum = simulator.get_total_momentum()
```

**Методы:**
- `simulate()` - запуск симуляции
- `get_total_energy()` - полная энергия системы
- `get_center_of_mass()` - центр масс
- `get_total_momentum()` - полный импульс
- `add_body()` / `remove_body()` - управление телами

#### `Body`

Класс для представления небесного тела:

```python
body = Body(
    name="Земля",
    mass=5.972e24,
    position=np.array([x, y, z]),
    velocity=np.array([vx, vy, vz]),
    radius=6371000,
    color='blue'
)

# Методы
state = body.get_state()
kinetic_energy = body.get_kinetic_energy()
distance = body.get_distance_to(other_body)
```

#### `SystemPresets`

Предустановленные системы для быстрого старта:

```python
presets = SystemPresets()

# Солнечная система
bodies = presets.create_solar_system(include_outer_planets=True)

# Двойная звездная система
bodies = presets.create_binary_star_system(separation_au=10.0)

# Система Земля-Луна
bodies = presets.create_earth_moon_system()
```

#### `AdvancedVisualizer`

Продвинутая визуализация симуляций:

```python
visualizer = AdvancedVisualizer()

# 3D траектории
fig = visualizer.plot_3d_trajectories(bodies)

# 2D проекции
fig = visualizer.plot_2d_projection(bodies, plane='xy')

# Анимация
anim = visualizer.animate_simulation(bodies, interval=50)

# Сохранение энергии
fig = visualizer.plot_energy_conservation(simulator, times)
```

#### `CosmicSim`

Физические константы и базовые вычисления:

- `AU`, `G`, `c`, `R_earth`, `mass_sun` - физические константы
- `parallax_distance()` - вычисление расстояния через параллакс
- `orbital_velocity()` - орбитальная скорость
- `kepler_third_law()` - третий закон Кеплера
- `gravitational_force()` - сила гравитации
- `spherical_to_cartesian()` / `cartesian_to_spherical()` - преобразования координат

#### `SimulationIO`

Сохранение и загрузка состояний:

```python
# Сохранение
SimulationIO.save_bodies(bodies, 'system.json')
SimulationIO.save_simulation_state(simulator, 'state.json')
SimulationIO.export_trajectories_csv(bodies, 'trajectories.csv')

# Загрузка
bodies = SimulationIO.load_bodies('system.json')
simulator = SimulationIO.load_simulation_state('state.json')
```

## 📖 Примеры использования

### Запуск готовых симуляций

```bash
# Интерактивное меню
python full_scale_simulation.py

# Конкретные симуляции
python full_scale_simulation.py solar      # Солнечная система
python full_scale_simulation.py binary     # Двойная звезда
python full_scale_simulation.py custom     # Пользовательская система
python full_scale_simulation.py animation  # Анимация Земля-Луна
```

### Базовые примеры

```bash
python examples.py
```

### Дополнительная документация

- [Полномасштабная симуляция](README_FULL_SCALE.md) - подробное описание возможностей
- [Инструкция по установке](INSTALL.md) - детали установки и сборки

## 🔬 Физические возможности

### Анализ энергии

Библиотека автоматически отслеживает сохранение энергии:

```python
initial_energy = simulator.get_total_energy()
# ... симуляция ...
final_energy = simulator.get_total_energy()
energy_change = (final_energy - initial_energy) / abs(initial_energy) * 100
print(f"Изменение энергии: {energy_change:.6f}%")
```

### Центр масс и импульс

```python
# Центр масс системы
com = simulator.get_center_of_mass()
print(f"Центр масс: {com} м")

# Полный импульс
momentum = simulator.get_total_momentum()
print(f"Импульс: {momentum} кг·м/с")
```

## ⚙️ Требования

- **Python** >= 3.7
- **numpy** >= 1.19.0
- **scipy** >= 1.5.0
- **matplotlib** >= 3.3.0

## 🎯 Производительность

Для оптимальной производительности:

- **Малые системы** (< 5 тел): `n_points=2000-5000`, `rtol=1e-8`
- **Средние системы** (5-10 тел): `n_points=1000-2000`, `rtol=1e-6`
- **Большие системы** (> 10 тел): `n_points=500-1000`, `rtol=1e-6`

Для очень больших систем рекомендуется отключить сохранение траекторий на промежуточных шагах.

## 🛠️ Разработка

### Установка в режиме разработки

```bash
pip install -e ".[dev]"
```

### Запуск тестов

```bash
pytest
```

### Форматирование кода

```bash
black cosmic_sim/
```

## 📝 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

## 🤝 Вклад в проект

Приветствуются pull requests и issues! 

1. Fork проекта
2. Создайте ветку для новой функции (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📧 Контакты

- **Автор**: Timur Isanov
- **Email**: tisanov@yahoo.com
- **GitHub**: [@xtimon](https://github.com/xtimon)

## 🙏 Благодарности

- Использует `scipy.integrate.solve_ivp` для численного интегрирования
- Визуализация на основе `matplotlib`
- Физические константы из актуальных астрономических данных

---

⭐ Если проект полезен, поставьте звезду!

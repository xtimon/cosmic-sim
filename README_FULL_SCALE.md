# Полномасштабная симуляция

Этот документ описывает возможности полномасштабной симуляции космических систем.

## Новые возможности

### 1. Симуляция N тел

Библиотека теперь поддерживает симуляцию произвольного количества небесных тел:

```python
from cosmic_sim import NBodySimulator, Body, SystemPresets
import numpy as np

# Создать систему
presets = SystemPresets()
bodies = presets.create_solar_system()

# Запустить симуляцию
simulator = NBodySimulator(bodies)
times, states = simulator.simulate(
    t_span=(0, 365.25*24*3600),  # 1 год
    n_points=2000
)
```

### 2. Класс Body

Представляет небесное тело с полной информацией:

```python
from cosmic_sim import Body

body = Body(
    name="Земля",
    mass=5.972e24,  # кг
    position=np.array([1.5e11, 0, 0]),  # м
    velocity=np.array([0, 30000, 0]),  # м/с
    radius=6371000,  # м
    color='blue'
)
```

### 3. Предустановленные системы

Готовые конфигурации для быстрого старта:

- **Солнечная система** (`create_solar_system()`)
- **Двойная звездная система** (`create_binary_star_system()`)
- **Система Земля-Луна** (`create_earth_moon_system()`)

### 4. Продвинутая визуализация

```python
from cosmic_sim import AdvancedVisualizer

visualizer = AdvancedVisualizer()

# 3D траектории
fig = visualizer.plot_3d_trajectories(bodies)

# 2D проекция
fig = visualizer.plot_2d_projection(bodies, plane='xy')

# Анимация
anim = visualizer.animate_simulation(bodies)
```

### 5. Сохранение и загрузка

```python
from cosmic_sim import SimulationIO

# Сохранить систему
SimulationIO.save_bodies(bodies, 'solar_system.json')

# Загрузить систему
bodies = SimulationIO.load_bodies('solar_system.json')

# Экспорт траекторий в CSV
SimulationIO.export_trajectories_csv(bodies, 'trajectories.csv')
```

## Примеры использования

### Быстрый старт

```bash
# Запустить полномасштабную симуляцию
python full_scale_simulation.py

# Или конкретную симуляцию
python full_scale_simulation.py solar
python full_scale_simulation.py binary
python full_scale_simulation.py custom
python full_scale_simulation.py animation
```

### Программное использование

```python
from cosmic_sim import NBodySimulator, SystemPresets, AdvancedVisualizer
import matplotlib.pyplot as plt

# Создать Солнечную систему
presets = SystemPresets()
bodies = presets.create_solar_system(include_outer_planets=False)

# Симуляция
simulator = NBodySimulator(bodies)
times, states = simulator.simulate(
    t_span=(0, 365.25*24*3600),
    n_points=2000,
    save_trajectory=True
)

# Визуализация
visualizer = AdvancedVisualizer()
fig = visualizer.plot_3d_trajectories(bodies)
plt.show()
```

## Физические возможности

### Анализ энергии

```python
# Получить полную энергию системы
total_energy = simulator.get_total_energy()

# Кинетическая энергия каждого тела
for body in bodies:
    kinetic = body.get_kinetic_energy()
    print(f"{body.name}: {kinetic:.2e} Дж")
```

### Центр масс

```python
com = simulator.get_center_of_mass()
print(f"Центр масс: {com}")
```

### Импульс системы

```python
momentum = simulator.get_total_momentum()
print(f"Полный импульс: {momentum}")
```

## Производительность

Для больших систем рекомендуется:

1. Уменьшить количество точек (`n_points`)
2. Использовать более низкую точность (`rtol=1e-6`)
3. Симулировать меньшие временные интервалы
4. Отключить сохранение траекторий для промежуточных шагов

## Ограничения

- Симуляция использует численные методы, точность зависит от параметров
- Для очень больших систем (>10 тел) может потребоваться оптимизация
- Анимации могут быть медленными для длинных траекторий

## Дальнейшее развитие

Возможные улучшения:

- [ ] Поддержка релятивистских эффектов
- [ ] Столкновения тел
- [ ] Газовые облака и туманности
- [ ] Оптимизация с использованием GPU
- [ ] Интеграция с реальными астрономическими данными


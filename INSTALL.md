# Инструкция по установке

## Установка пакета

### 1. Установка из исходников (рекомендуется для разработки)

```bash
# Перейдите в директорию проекта
cd cosmic-sim

# Установите пакет
pip install .

# Или в режиме разработки (изменения применяются сразу)
pip install -e .
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Проверка установки

```python
from cosmic_sim import CosmicSim, OrbitalSimulator, ParallaxVisualizer
print("✓ Пакет установлен успешно!")
```

## Сборка дистрибутива

### Создание исходного дистрибутива (sdist)

```bash
python setup.py sdist
```

### Создание wheel-пакета

```bash
pip install wheel
python setup.py bdist_wheel
```

### Установка из собранного пакета

```bash
pip install dist/cosmic-sim-0.1.0.tar.gz
# или
pip install dist/cosmic_sim-0.1.0-py3-none-any.whl
```

## Публикация в PyPI (опционально)

```bash
# Установите twine
pip install twine

# Загрузите пакет в PyPI
twine upload dist/*
```

## Запуск примеров

```bash
python examples.py
```


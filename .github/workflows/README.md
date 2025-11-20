# GitHub Actions Workflows

Этот каталог содержит GitHub Actions workflows для автоматизации CI/CD процессов.

## Доступные workflows

### 1. `ci.yml` - Continuous Integration
Основной CI пайплайн, который запускается при каждом push и pull request:
- Тестирование на Python 3.9-3.11
- Тестирование на Ubuntu, macOS, Windows
- Проверка импортов
- Базовые тесты функциональности
- Линтинг кода (flake8, black)
- Сборка пакета

### 2. `publish.yml` - Публикация в PyPI
Автоматическая публикация пакета в PyPI при создании release:
- Сборка wheel и source distribution
- Проверка пакета (twine check)
- Публикация в PyPI (требует секрет `PYPI_API_TOKEN`)

**Настройка:**
1. Создайте API token на [pypi.org](https://pypi.org/manage/account/token/)
2. Добавьте секрет `PYPI_API_TOKEN` в настройках репозитория (Settings → Secrets)

### 3. `codeql.yml` - CodeQL Security Analysis
Анализ безопасности кода:
- Автоматический анализ уязвимостей
- Запускается при push и еженедельно

### 4. `docs.yml` - Проверка документации
Проверка качества документации:
- Проверка форматирования README
- Проверка работоспособности примеров кода

### 5. `dependabot.yml` - Автоматическое обновление зависимостей
Автоматическое обновление зависимостей через Dependabot:
- Автоматическое создание PR для обновлений
- Автомerge для patch-обновлений

## Статус workflows

Вы можете проверить статус всех workflows на вкладке [Actions](https://github.com/xtimon/cosmic-sim/actions) репозитория.

## Локальный запуск тестов

Для запуска тестов локально:

```bash
# Установить зависимости для разработки
pip install -e ".[dev]"

# Запустить тесты
pytest tests/ -v

# С покрытием кода
pytest tests/ -v --cov=cosmic_sim --cov-report=html
```

## Требования для публикации

Перед публикацией убедитесь, что:
1. Версия обновлена в `pyproject.toml` и `setup.py`
2. Все тесты проходят
3. Документация актуальна
4. Создан release на GitHub


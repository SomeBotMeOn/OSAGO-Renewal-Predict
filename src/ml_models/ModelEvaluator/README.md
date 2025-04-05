# Model Evaluator (0.0.2)

Гибкий инструмент для оценки моделей машинного обучения с поддержкой ансамблей, оптимизации гиперпараметров и валидации.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Optuna](https://img.shields.io/badge/Optuna-4.2-2C5D92)](https://optuna.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-FF6B4A)](https://catboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-FFD700)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-9B4F96)](https://xgboost.readthedocs.io)

#### Автор: [Ерофеев Олег](https://github.com/SomeBotMeOn)

---

## 🔥 Возможности

- **10+ моделей**: CatBoost, XGBoost, LightGBM, Логистическая регрессия, Случайный лес и другие
- **Ансамбли**: Блендинг, Бэггинг, Стекинг, Голосование (Hard/Soft)
- **Оптимизация**: Интеграция с Optuna для автоматического подбора параметров
- **Валидация**: K-Fold, Стратифицированный K-Fold, Leave-One-Out, Train-Test Split
- **Метрики кластеризации**: Расчет силуэта, Calinski-Harabasz, Davies-Bouldin
- **Контроль качества**:
  - Предварительная проверка конфигурации
  - Соответствие PEP-8
  - Логирование ошибок
  - Автосохранение моделей

---

## 🆕 Новости:
* **March 28, 2025**: Добавлена оценка кластеризации (ModelEvaluator v0.0.2).
* **February 13, 2025**: Запуск ModelEvaluator 0.0.1 для задач классификации с поддержкой ансамблей и Optuna.

---

## Установка

1. Создайте окружение Anaconda:
```bash
conda create -n model_eval python=3.10
conda activate model_eval
```

2. Скачайте файл `environment.yml`
3. Установите зависимости:
```bash
conda env update -f environment.yml
```

---

## Быстрый старт

```python
from model_evaluator import ModelEvaluator

# Инициализация
evaluator = ModelEvaluator(
    data=df, # Ваш датасет
    target_column='Cluster', # Целевая переменная
)

link = 'models/' # Папка для сохранения моделей

# Пример оценки моделей
results = evaluator.evaluate_models(
    save_models_to=link
)
  
results # Вывод результатов
```

---

## Особенности

1. Быстрая проверка введенных параметров и конфликтов в самом начале работы кода!
2. Поддержка мультикласса и бинарной классификации
3. Автоматическое определение типа классификации
4. Поддержка метрик для многоклассовых задач
5. Расчет метрик качества кластеризации
6. Возможность сохранения моделей в файл
7. Поддержка кастомных параметров для моделей
8. Возможность оценки ансамблей моделей
9. Оптимизация гиперпараметров с помощью Optuna
10. Возможность выбора метода валидации
11. Возможность использовать конкретные модели для оценки
12. Возможность исключить модели из оценки

---

## Подробнее о методам

### Метод `evaluate_models()`

| Параметр           | Тип   | Описание                                                                                     |
|--------------------|-------|----------------------------------------------------------------------------------------------|
| `selected_models`  | dict  | Словарь с моделями для оценки: `{'Модель': {параметры}}`                                     |
| `unselected_models`| dict  | Модели для исключения из оценки: `{'Модель': {}}`                                            |
| `custom_params`    | dict  | Пользовательские параметры для всех моделей: `{'Модель': {параметры}}`                       |
| `cv_method`        | str   | Метод валидации: `KFold`, `Stratified`, `LeaveOneOut`, `train_test_split` (default: `KFold`) |
| `cv_params`        | dict  | Параметры для выбранной валидации                                                            |
| `save_models_to`   | str   | Путь для сохранения обученных моделей                                                        |

#### Заметки: 
1. Передавать можно один из трех параметров: `selected_models`, `unselected_models`, `custom_params`.
2. Если `selected_models` или `custom_params` не переданы, то будут использованы все модели по умолчанию
(с использованием зафиксированного `seed`, где это возможно)

#### Возвращает:
- Датасет с результатами оценки моделей
- Во время работы выводятся оформленные промежуточные результаты, а также информация о процессе обучения

#### Пример:
```python
link = '../../models/'

custom_params = {
    'CatBoost': {'verbose': 0, 'random_state': 42},
    'XGBoost': {'verbosity': 0, 'random_state': 42},
    'LightGBM': {'verbosity': -1, 'random_state': 42},
}

results_df = evaluator.evaluate_models(
    custom_params=custom_params,
    cv_method='KFold',
    cv_params={'n_splits': 5},
    save_models_to=link
)

results_df
```

### Метод `evaluate_clustering()`

Вычисляет метрики качества кластеризации и возвращает результаты в виде таблицы.

| Параметр           | Тип            | Описание                                                                                                                      |
|--------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| `metrics`          | list, optional | Список метрик для расчета. Доступные: `['silhouette', 'calinski_harabasz', 'davies_bouldin']`. По умолчанию используются все. |

#### Возвращает:
- `pd.DataFrame`: DataFrame с результатами расчета метрик, округленными до трех знаков.

#### Пример:
```python
evaluator = ModelEvaluator(data=data, target_column='Cluster')

# Расчет метрик
metrics_df = evaluator.evaluate_clustering()
print(metrics_df.round(3))
```
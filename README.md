# Прогнозирование пролонгации страховых договоров

## Оглавление

- [Описание проекта](#описание-проекта)
- [Структура проекта](#структура-проекта)
- [Основные этапы](#основные-этапы)
- [Результаты](#результаты)

## Описание проекта

Цель — предсказать вероятность продления договора ОСАГО для новых клиентов. Проект включает анализ данных, генерацию признаков и построение ML-моделей.

## Структура проекта

```
├── data/
│   ├── raw_data/                       # Исходные данные
│   ├── data_after_EDA/                 # Данные после EDA
│   ├── data_after_Feature_Generation/  # Сгенерированные признаки
│   ├── data_after_Feature_Selection/   # Отобранные признаки
│   └── new_data_preds/                 # Прогнозы для новых данных
│
├── models/
│   ├── 01_classic_ml_models/      # Базовые модели (LogisticRegression, CatBoost и др.)
│   ├── 02_ensemble_models/        # Ансамбли (Stacking, Voting, Bagging, Blending)
│   └── 03_final_model/            # Финальная модель
│
├── notebooks/
│   ├── 01_EDA/
│       └── 01_data_analysis.ipynb       # Анализ данных
│   ├── 02_feature_engineering/
│       └── 01_feature_generation.ipynb  # Генерация признаков
│   ├── 03_feature_selection/
│       └── 01_feature_selection.ipynb   # Отбор признаков
│   ├── 04_ml_models/
│       ├── 01_classic_ml_models.ipynb               # Базовые модели
│       ├── 02_tuning_optuna_classic_ml_models.ipynb # Оптимизация гиперпараметров
│       ├── 03_ensemble_default_models.ipynb         # Ансамбли (Stacking, Voting, Bagging, Blending)
│       └── 04_final_model.ipynb                     # Финальная модель
│   └── 05_results/
│       └── 01_pipeline.ipynb                        # Pipeline прогнозирования + прогноз для новых данных
│
├── reports/                       # Графики
│   
├── src/
│   ├── data_utils/                # Утилиты для работы с данными
│   ├── visualizations_utils/      # Утилиты для работы c визуализацией
│   └── ml_models/                 # Утилиты для работы с моделями ML
│
├── README.md                      # Документация
```

## Основные этапы

Предобработка данных: устранение аномалий, заполнение пропусков, кодирование категорий через Mean Target Encoding со сдвигом на глобальное среднее.

Генерация признаков: создание новых признаков через арифметические операции.

Отбор признаков: CatBoost (RecursiveByLossFunctionChange) + анализ корреляций.

Обучение моделей: тестирование классических (LogisticRegression, RandomForest) и ансамблевых подходов (Voting, Stacking, Bagging, Blending).

## Результаты
Лучшая модель — **Logistic Regression**. Метрики на тестовом наборе данных:

Balanced Accuracy: 0.6794

Accuracy: 0.6821

F1 Score: 0.5651

Precision: 0.4872

Recall: 0.6726

ROC AUC: 0.7393
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from ...resources.constants.all_models import \
    ALL_MODELS
from ...resources.constants.models_without_random_state import \
    MODELS_WITHOUT_RANDOM_STATE


def blending_func(
        X: pd.DataFrame,
        y: pd.Series,
        ensemble_config: dict,
        random_state: int = 42
) -> tuple[str, float, str, str, str]:
    """
    Реализация алгоритма блендинга с поддержкой множественных случайных разбиений данных.

    Параметры
    ----------
    X : pd.DataFrame
        Матрица признаков для обучения.
    y : pd.Series
        Целевая переменная.
    ensemble_config : dict
        Конфигурация ансамбля с параметрами:
        - base_models: Словарь базовых моделей в формате {'название_модели': параметры}
        - meta_model: Мета-модель в формате {'название_модели': параметры}
        - test_size: Доля тестовой выборки (по умолчанию 0.2)
        - n_splits: Количество случайных разбиений (по умолчанию 1)
        - custom_name: Название метода (по умолчанию 'Blending')
    random_state : int, опционально
        Seed для воспроизводимости (по умолчанию 42)

    Возвращает
    -------
    tuple
        Кортеж с агрегированными результатами:
            - Название метода ('Blending')
            - Средняя метрика по разбиениям
            - N/A (заглушка для совместимости)
            - N/A (заглушка для совместимости)
            - N/A (заглушка для совместимости)

    Пример
    --------
    >>> config = {
    ...     'Blending': {
    ...         'base_models': {
    ...             'RandomForest': {'n_estimators': 100},
    ...             'LogisticRegression': {'C': 0.1}
    ...         },
    ...         'meta_model': {'DecisionTree': {'max_depth': 3}},
    ...         'test_size': 0.2,
    ...         'n_splits': 3
    ...     }
    ... }
    """
    # Извлекаем параметры блендинга
    params = ensemble_config.get('Blending')
    base_models = params.get('base_models')
    meta_model = params.get('meta_model')
    custom_name = params.get('custom_name', 'Blending')
    test_size = params.get('test_size', 0.2)
    n_splits = params.get('n_splits', 1)

    scores = []

    # Цикл по случайным разбиениям данных
    for split_idx in range(n_splits):
        # Разделение данных с уникальным seed для каждого разбиения
        current_seed = random_state + split_idx
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=test_size,
            random_state=current_seed
        )

        # Инициализация и обучение базовых моделей
        base_instances = {}
        for name, config in base_models.items():
            model_config = config.copy()
            # Добавляем random_state если не задан
            if ('random_state' not in model_config
                    and name not in MODELS_WITHOUT_RANDOM_STATE):
                model_config['random_state'] = current_seed
            base_instances[name] = ALL_MODELS[name](**model_config).fit(
                X_train, y_train)

        # Генерация мета-признаков
        meta_matrix = np.column_stack([
            model.predict(X_holdout)
            for model in base_instances.values()
        ])

        # Обучение и оценка мета-модели
        meta_name, meta_params = list(meta_model.items())[0]
        meta_params = meta_params.copy()
        # Добавляем random_state если не задан
        if 'random_state' not in meta_params:
            meta_params['random_state'] = current_seed
        meta_clf = ALL_MODELS[meta_name](**meta_params).fit(meta_matrix,
                                                            y_holdout)

        y_pred = meta_clf.predict(meta_matrix)
        scores.append(balanced_accuracy_score(y_holdout, y_pred))

    # Рассчет агрегированных статистик
    avg_score = round(np.mean(scores), 2)
    min_score = round(np.min(scores), 2) if len(scores) > 1 else 'N/A'
    max_score = round(np.max(scores), 2) if len(scores) > 1 else 'N/A'
    std_score = round(np.std(scores), 2) if len(scores) > 1 else 'N/A'

    return custom_name, avg_score, min_score, max_score, std_score

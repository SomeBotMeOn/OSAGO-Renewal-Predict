import pandas as pd
import numpy as np

from sklearn.metrics import balanced_accuracy_score

from ...resources.constants.all_models import \
    ALL_MODELS
from ...resources.constants.models_without_random_state import \
    MODELS_WITHOUT_RANDOM_STATE
from ...utils.general_utils.cross_validation import (
    get_cv_method
)


def stacking_func(
        X: pd.DataFrame,
        y: pd.Series,
        ensemble_config: dict,
        random_state: int = 42
) -> tuple[str, float, float, float, float]:
    """
    Реализация алгоритма стекинга с использованием кросс-валидации.

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
        - cv_method: Метод кросс-валидации (по умолчанию 'KFold')
        - cv_params: Параметры CV (по умолчанию {'n_splits': 5})
        - custom_name: Название метода (по умолчанию 'Stacking')
    random_state : int, опционально
        Seed для воспроизводимости (по умолчанию 42)

    Возвращает
    -------
    tuple
        Кортеж с результатами:
            - Название метода ('Stacking')
            - Средняя метрика по фолдам
            - Минимальная метрика
            - Максимальная метрика
            - Стандартное отклонение метрик

    Пример
    --------
    >>> config = {
    ...     'Stacking': {
    ...         'base_models': {
    ...             'RandomForest': {'n_estimators': 100},
    ...             'LogisticRegression': {'C': 0.1}
    ...         },
    ...         'meta_model': {'DecisionTree': {'max_depth': 3}},
    ...         'cv_method': 'KFold',
    ...         'cv_params': {'n_splits': 5}
    ...     }
    ... }
    """
    # Извлечение параметров из конфигурации
    params = ensemble_config.get('Stacking', {})
    base_models = params.get('base_models', {})
    meta_model = params.get('meta_model', {})
    custom_name = params.get('custom_name', 'Stacking')
    cv_method = params.get('cv_method', 'KFold')
    cv_params = params.get('cv_params', {'n_splits': 5})

    # Инициализация CV
    cv, _ = get_cv_method(
        cv_method=cv_method,
        cv_params=cv_params,
        X=X,
        y=y,
        random_state=random_state  # Передаем random_state в CV
    )

    # Создание экземпляров базовых моделей
    base_models_instances = {}
    for model_name, model_params in base_models.items():
        params_copy = model_params.copy()
        # Добавляем random_state если не задан
        if ('random_state' not in params_copy
                and model_name not in MODELS_WITHOUT_RANDOM_STATE):
            params_copy['random_state'] = random_state
        base_models_instances[model_name] = ALL_MODELS.get(model_name)(
            **params_copy)

    # Инициализация структур данных
    fold_scores = []
    meta_features_train = np.zeros((X.shape[0], len(base_models)))

    # Основной цикл кросс-валидации
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Матрица для мета-признаков текущего фолда
        fold_meta_features = np.zeros((X_val.shape[0], len(base_models)))

        # Обучение базовых моделей и генерация мета-признаков
        for model_idx, (model_name, model) in enumerate(
                base_models_instances.items()):
            model.fit(X_train, y_train)
            fold_meta_features[:, model_idx] = model.predict(X_val).ravel()

        # Сохранение мета-признаков
        meta_features_train[val_idx] = fold_meta_features

        # Обучение мета-модели
        meta_model_name, meta_model_params = list(meta_model.items())[0]
        meta_params = meta_model_params.copy()
        # Добавляем random_state если не задан
        if 'random_state' not in meta_params:
            meta_params['random_state'] = random_state
        meta_clf = ALL_MODELS.get(meta_model_name)(**meta_params)
        meta_clf.fit(fold_meta_features, y_val)

        # Расчет метрики
        y_meta_pred = meta_clf.predict(fold_meta_features)
        fold_score = balanced_accuracy_score(y_val, y_meta_pred)
        fold_scores.append(fold_score)

    # Расчет финальных метрик
    avg_score = round(np.mean(fold_scores), 2)
    min_score = round(np.min(fold_scores), 2)
    max_score = round(np.max(fold_scores), 2)
    std_score = round(np.std(fold_scores), 2)

    return custom_name, avg_score, min_score, max_score, std_score

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from collections import Counter

from ...resources.constants.all_models import \
    ALL_MODELS
from ...utils.general_utils.cross_validation import \
    get_cv_method


def voting_func(
        X: pd.DataFrame,
        y: pd.Series,
        ensemble_config: dict,
        random_state: int = 42
) -> tuple[str, float, float, float, float]:
    """
    Реализация алгоритма Voting с кросс-валидацией для базовых моделей.

    Параметры
    ----------
    X : pd.DataFrame
        Матрица признаков.
    y : pd.Series
        Целевая переменная.
    ensemble_config : dict
        Конфигурация:
        - base_models: Словарь моделей {'название': параметры}
        - voting_type: 'hard' (по умолчанию) или 'soft'
        - cv_method: Метод кросс-валидации (например, 'KFold')
        - cv_params: Параметры CV (например, {'n_splits': 5})
        - custom_name: Название метода (по умолчанию 'Voting').
    random_state : int
        Seed для воспроизводимости.

    Возвращает
    -------
    tuple
        Кортеж с агрегированными результатами:
            - Название метода ('Voting')
            - Значение метрики
            - N/A (заглушка для совместимости)
            - N/A (заглушка для совместимости)
            - N/A (заглушка для совместимости)

    Пример
    --------
    >>> config = {
    ...     'Voting': {
    ...         'base_models': {
    ...             'RandomForest': {'n_estimators': 100},
    ...             'LogisticRegression': {'C': 0.1}
    ...         },
    ...         'voting_type': 'soft',
    ...         'cv_method': 'KFold',
    ...         'cv_params': {'n_splits': 5}
    ...     }
    ... }
    """
    # Извлечение параметров
    params = ensemble_config.get('Voting', {})
    base_models = params.get('base_models', {})
    custom_name = params.get('custom_name', 'Voting')
    voting_type = params.get('voting_type', 'hard')
    cv_method = params.get('cv_method', 'KFold')
    cv_params = params.get('cv_params', {'n_splits': 5})

    # Инициализация кросс-валидации
    cv, _ = get_cv_method(
        cv_method=cv_method,
        cv_params=cv_params,
        X=X,
        y=y,
        random_state=random_state
    )

    fold_scores = []
    full_preds = np.zeros(len(y))

    # Основной цикл по фолдам
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Сбор out-of-fold предсказаний для каждой модели
        model_preds = []
        for model_name, model_params in base_models.items():
            # Внутренняя CV для базовой модели (например, GridSearchCV)
            model = ALL_MODELS[model_name](
                **model_params
            ).fit(X_train, y_train)

            # Предсказание на валидационной части
            if voting_type == 'hard':
                preds = model.predict(X_val)
            else:
                preds = model.predict_proba(X_val)
            model_preds.append(preds)

        # Агрегация предсказаний
        if voting_type == 'hard':
            final_pred = [
                Counter(preds).most_common(1)[0][0]
                for preds in zip(*model_preds)
            ]
        else:
            final_pred = np.argmax(np.mean(model_preds, axis=0), axis=1)

        # Расчет метрики для фолда
        fold_score = balanced_accuracy_score(y_val, final_pred)
        fold_scores.append(fold_score)
        full_preds[val_idx] = final_pred

    # Агрегация результатов
    avg_score = round(np.mean(fold_scores), 2)
    min_score = round(np.min(fold_scores), 2)
    max_score = round(np.max(fold_scores), 2)
    std_score = round(np.std(fold_scores), 2)

    return custom_name, avg_score, min_score, max_score, std_score

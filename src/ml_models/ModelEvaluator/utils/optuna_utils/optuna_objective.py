import pandas as pd
from typing import Any, Dict
import numpy as np

import optuna
from sklearn.metrics import get_scorer

from ...errors.optuna_errors.model_creation_error import  raise_model_creation_error
from ...errors.optuna_errors.parameter_suggestion_error import raise_parameter_suggestion_error
from ...resources.constants.all_models import ALL_MODELS


def objective_wrapper(
        trial: optuna.Trial,
        model_name: str,
        optuna_config: Dict[str, Any],
        scoring: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv_method: Any
) -> float:
    """
    Целевая функция для оптимизации гиперпараметров с использованием Optuna.

    Параметры:
        trial (optuna.Trial): Испытание Optuna.
        model_name (str): Название модели.
        optuna_config (Dict[str, Any]): Конфигурация гиперпараметров.
        scoring (str): Метрика для оптимизации.
        X (pd.DataFrame): Признаки.
        y (pd.Series): Целевая переменная.
        cv_method (Any): Метод кросс-валидации.

    Возвращает:
        float: Среднее значение метрики по кросс-валидации.
    """
    model = None
    params = {}

    for param_name, param_value in optuna_config[model_name].items():
        try:
            # Если параметр представлен строкой - исполняем код через eval
            if isinstance(param_value, str):
                params[param_name] = eval(
                    param_value,
                    {'trial': trial, 'optuna': optuna}
                )
            # Иначе используем значение напрямую (статический параметр)
            else:
                params[param_name] = param_value
        except Exception as e:
            raise_parameter_suggestion_error(model_name, param_name, e)

    # Создание модели с предложенными параметрами
    model_class = ALL_MODELS[model_name]
    try:
        model = model_class(**params)
    except Exception as e:
        raise_model_creation_error(model_name, params, e)

    # Вычисление метрики с использованием кросс-валидации
    scores = []
    for train_idx, val_idx in cv_method.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        scorer = get_scorer(scoring)
        score = scorer(model, X_val, y_val)
        scores.append(score)

    return np.mean(scores) if scores else float('-inf')
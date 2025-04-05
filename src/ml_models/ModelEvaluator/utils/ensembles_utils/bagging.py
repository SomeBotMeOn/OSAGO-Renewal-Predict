import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score
from collections import Counter

from ...resources.constants.all_models import ALL_MODELS


def bagging_func(
        X: pd.DataFrame,
        y: pd.Series,
        ensemble_config: dict,
        random_state: int = 42
) -> tuple[str, float, str, str, str]:
    """
    Реализация алгоритма Bagging с поддержкой бутстрэп выборок и агрегации предсказаний.

    Параметры
    ----------
    X : pd.DataFrame
        Матрица признаков
    y : pd.Series
        Целевая переменная
    ensemble_config : dict
        Конфигурация с параметрами:
        - base_models: Базовая модель в формате {'название_модели': параметры}
        - n_estimators: Количество моделей в ансамбле (по умолчанию 10)
        - max_samples: Размер бутстрэп выборки (по умолчанию 1.0)
        - max_features: Доля признаков для выборки (по умолчанию 1.0)
        - bootstrap: Флаг бутстрэпа (по умолчанию True)
        - bootstrap_features: Флаг бутстрэпа признаков (по умолчанию False)
        - custom_name: Название метода (по умолчанию 'Bagging')
    random_state : int, опционально
        Seed для воспроизводимости

    Возвращает
    -------
    tuple
        Результаты в формате:
        - Название метода ('Bagging')
        - Значение метрики
        - N/A (заглушка для совместимости)
        - N/A (заглушка для совместимости)
        - N/A (заглушка для совместимости)

    Пример
    --------
    >>> config = {
    ...     'Bagging': {
    ...         'base_models': {'DecisionTree': {'max_depth': 5}},
    ...         'n_estimators': 20,
    ...         'max_samples': 0.8
    ...     }
    ... }
    """
    # Извлечение параметров
    params = ensemble_config.get('Bagging')
    base_models = params.get('base_models')
    custom_name = params.get('custom_name', 'Bagging')
    n_estimators = params.get('n_estimators', 10)
    max_samples = params.get('max_samples', 1.0)
    max_features = params.get('max_features', 1.0)
    bootstrap = params.get('bootstrap', True)
    bootstrap_features = params.get('bootstrap_features', False)

    # Инициализация базовой модели
    model_name, model_params = list(base_models.items())[0]
    BaseModel = ALL_MODELS[model_name]

    estimators = []
    rng = np.random.RandomState(random_state)

    for i in range(n_estimators):
        # Уникальный seed для каждой итерации
        iter_seed = rng.randint(0, 10 ** 6)

        # Бутстрэп выборка
        X_sample, y_sample = resample(
            X, y,
            replace=bootstrap,
            n_samples=int(max_samples * len(X)),
            random_state=iter_seed  # Динамический seed
        )

        # Выбор признаков
        if max_features < 1.0:
            n_features = int(max_features * X.shape[1])
            features = rng.choice(
                X.columns,
                size=n_features,
                replace=bootstrap_features
            )
            X_sample = X_sample[features]

        # Динамический random_state для модели
        model_params_iter = model_params.copy()
        if 'random_state' in model_params_iter:
            model_params_iter['random_state'] = iter_seed

        model = BaseModel(**model_params_iter)
        model.fit(X_sample, y_sample)
        estimators.append((model, features if max_features < 1.0 else None))

    # Прогнозирование и агрегация
    all_preds = []
    for model, features in estimators:
        if features is not None:
            X_pred = X[features]
        else:
            X_pred = X
        all_preds.append(model.predict(X_pred))

    # Голосование большинства
    final_pred = []
    for preds in zip(*all_preds):
        counter = Counter(preds)
        final_pred.append(counter.most_common(1)[0][0])

    # Расчет метрики
    score = balanced_accuracy_score(y, final_pred)

    return custom_name, round(score, 2), 'N/A', 'N/A', 'N/A'

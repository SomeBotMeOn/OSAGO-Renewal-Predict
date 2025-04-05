from inspect import signature
from typing import Dict, List, Any

from ...resources.constants.all_models import ALL_MODELS


def raise_validate_models_parameters(
        model_params: Dict[str, Dict[str, Any]]) -> None:
    """Проверяет валидность параметров для указанных моделей машинного обучения.

    Параметры:
    ----------
    model_params : Dict[str, Dict[str, Any]]
        Словарь параметров моделей в формате:
        {
            'название_модели': {
                'параметр1': значение,
                'параметр2': значение,
                ...
            }
        }
        Название модели должно присутствовать в ALL_MODELS

    Возвращает:
    -----------
    None

    Выбрасывает:
    ------------
    ValueError
        Если найдены модели с невалидными параметрами

    Пример:
    --------
    >>> params = {'CatBoost': {'iterations': 100}}
    >>> raise_validate_models_parameters(params)

    Заметка:
    --------
    - Для проверки используются параметры конструктора класса модели
    - Учитываются параметры, полученные через get_params() экземпляра модели
    """
    errors = []

    for model_name, params in model_params.items():
        if model_name not in ALL_MODELS:
            continue

        model_class = ALL_MODELS[model_name]
        constructor_params = _get_model_params(model_class)

        # Специальная обработка для LightGBM
        if model_name == 'LightGBM':
            if 'verbosity' not in constructor_params:
                constructor_params.append('verbosity')

        invalid_params = [p for p in params if p not in constructor_params]
        if invalid_params:
            errors.append(
                f"{model_name}: недопустимые параметры - {', '.join(invalid_params)}"
            )

    if errors:
        raise ValueError("\n".join(errors))


def _get_model_params(model_class: type) -> List[str]:
    """Возвращает список допустимых параметров для класса модели.

    Параметры:
    ----------
    model_class : type
        Класс модели машинного обучения (например, CatBoostClassifier)

    Возвращает:
    -----------
    list[str]
        Список допустимых параметров конструктора (без служебных параметров)

    Пример:
    --------
    >>> _get_model_params(LogisticRegression)
    ['penalty', 'dual', 'tol', 'C', ...]

    Заметка:
    --------
    - Использует комбинацию методов get_params() и анализа сигнатуры конструктора
    - Игнорирует служебные параметры: self, args, kwargs
    """
    # Попытка получить параметры через экземпляр модели
    instance = model_class()
    if hasattr(instance, 'get_params'):
        params = instance.get_params()
        if params:
            return list(params.keys())

    # Получение параметров через сигнатуру конструктора
    sig = signature(model_class.__init__)
    return [
        param
        for param in sig.parameters
        if param not in ('self', 'args', 'kwargs')
    ]
from ...resources.constants.validation_strategies import (
    SUPPORTED_VALIDATION_STRATEGIES)


def raise_unsupported_validation_strategies(cv_method: str) -> None:
    """
    Вызывает исключение, если переданный метод кросс-валидации не поддерживается.

    Параметры:
    ----------
    cv_method : str
        Название метода кросс-валидации.

    Возвращает:
    ------
    ValueError
        Если переданный метод кросс-валидации отсутствует в AVAILABLE_CV_METHODS.
    """
    if cv_method not in SUPPORTED_VALIDATION_STRATEGIES:
        raise ValueError(
            f"Неподдерживаемый метод валидации! Доступные: "
            f"{SUPPORTED_VALIDATION_STRATEGIES}")

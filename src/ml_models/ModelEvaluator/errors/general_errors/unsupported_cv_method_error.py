from ...resources.constants.cross_validation_methods import (
    AVAIBLE_CROSS_VALIDATION_METHODS)


def raise_unsupported_cv_method_error(cv_method: str) -> None:
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
    if cv_method not in AVAIBLE_CROSS_VALIDATION_METHODS:
        raise ValueError(
            f"Неподдерживаемый метод кросс-валидации! Доступные: "
            f"{AVAIBLE_CROSS_VALIDATION_METHODS}")

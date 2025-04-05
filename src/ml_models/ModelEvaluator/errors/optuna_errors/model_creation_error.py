from sklearn.ensemble import RandomForestClassifier


def raise_model_creation_error(
        model_name: str,
        params: dict,
        original_error: Exception
) -> None:
    """
    Генерирует ошибку создания модели

    Параметры
    ----------
    model_name : str
        Название модели, которую не удалось создать
    params : dict
        Параметры, использованные для создания модели
    original_error : Exception
        Исходное исключение из конструктора модели

    Возвращает
    -------
    None
        Всегда вызывает исключение

    Пример
    -------
    >>> try:
    ...     RandomForestClassifier(max_depth=-1)
    ... except Exception as e:
    ...     raise_model_creation_error('RandomForest', {'max_depth': -1}, e)
    """
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    error_msg = (
        f"Ошибка подбора параметра для модели {model_name} "
        f"с параметрами [{params_str}]: {str(original_error)}"
    )
    raise RuntimeError(error_msg) from original_error

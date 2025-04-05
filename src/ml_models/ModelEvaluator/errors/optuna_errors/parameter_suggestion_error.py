def raise_parameter_suggestion_error(
        model_name: str,
        param_name: str,
        original_error: Exception
) -> None:
    """
    Генерирует ошибку предложения параметра для Optuna

    Параметры:
    ----------
    model_name : str
        Название модели, для которой настраиваются параметры
    param_name : str
        Имя параметра, вызвавшего ошибку
    original_error : Exception
        Исходное исключение, вызвавшее проблему

    Возвращает:
    -------
    None
        Функция всегда вызывает исключение

    Пример:
    -------
    >>> try:
    ...     eval("trial.suggest_int('n_estimators', 0, 10)")
    ... except Exception as e:
    ...     raise_parameter_suggestion_error('RandomForest', 'n_estimators', e)
    """
    error_msg = (
        f"Ошибка подбора параметра для {model_name} "
        f"в параметре '{param_name}': {str(original_error)}"
    )
    raise ValueError(error_msg) from original_error

def raise_model_training_failed(fold_scores: list) -> None:
    """
    Вызывает исключение, если обучение модели не удалось (список пуст).

    Параметры:
    ----------
    fold_scores : list
        Список оценок модели по фолдам.

    Исключения:
    ------
    ValueError
        Если обучение модели не удалось.
    """
    if not fold_scores:
        raise ValueError("Не удалось обучить модель!")

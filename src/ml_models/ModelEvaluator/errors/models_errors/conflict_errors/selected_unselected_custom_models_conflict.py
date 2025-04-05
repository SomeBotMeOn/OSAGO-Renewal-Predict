def raise_selected_unselected_custom_models_conflict(
        selected_models: dict | None = None,
        unselected_models: dict | None = None,
        custom_params: dict | None = None
) -> None:
    """
    Вызывает исключение, если передано два или более конфликтующих параметра.

    Параметры:
    ----------
    selected_models : dict
        Словарь с выбранными моделями.
    unselected_models : dict
        Словарь с исключенными моделями.
    custom_params : dict
        Словарь с кастомными параметрами моделей.

    Исключения:
    -----------
    ValueError
        Если передано два или более противоречащих друг другу параметра одновременно.
    """
    # Собираем список непустых параметров
    non_empty = []
    if selected_models:
        non_empty.append("'selected_models'")
    if unselected_models:
        non_empty.append("'unselected_models'")
    if custom_params:
        non_empty.append("'custom_params'")

    # Проверяем конфликт
    if len(non_empty) >= 2:
        error_message = (
            f"Были переданы {len(non_empty)} одновременно противоречащих "
            f"друг другу параметра. "
            f"Выберите только один из: [{', '.join(non_empty)}]"
        )
        raise ValueError(error_message)

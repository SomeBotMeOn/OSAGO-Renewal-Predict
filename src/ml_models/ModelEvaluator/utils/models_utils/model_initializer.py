def initialize_models(selected_models: dict, unselected_models: dict,
                      default_params: dict, all_models: dict,
                      custom_params: dict) -> dict:
    """
    Инициализация доступных моделей с заданными параметрами.

    Parameters
    ----------
    selected_models : dict
        Словарь с выбранными моделями и их параметрами.
    unselected_models : dict
        Список моделей, которые необходимо исключить из оценки.
    default_params : dict
        Словарь с параметрами по умолчанию для моделей.
    all_models : dict
        Словарь со всеми доступными моделями.
    custom_params : dict
        Словарь с пользовательскими параметрами для моделей.

    Returns
    -------
    available_models : dict
        Словарь с доступными моделями и их параметрами.
    """
    if selected_models and unselected_models:
        raise ValueError(
            "Переданы одновременно 'selected_models' и "
            "'unselected_models'. Выберите только одно из них.")

    available_models = {
        name: (all_models[name], selected_models.get(name, {}))
        for name in selected_models
    } if selected_models else {
        name: (model, default_params.get(name, {}))
        for name, model in all_models.items() if
        not unselected_models or name not in unselected_models
    }

    if custom_params:
        for model_name, custom_param in custom_params.items():
            if model_name in available_models:
                model_class, _ = available_models[model_name]
                available_models[model_name] = (model_class, custom_param)

    return available_models
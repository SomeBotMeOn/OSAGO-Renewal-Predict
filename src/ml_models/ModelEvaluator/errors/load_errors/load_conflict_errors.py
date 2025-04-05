from ..models_errors.conflict_errors.selected_unselected_custom_models_conflict import \
    raise_selected_unselected_custom_models_conflict


def load_all_conflict_errors(conflict_errors_list: list | None = None,
                             selected_models: dict | None = None,
                             unselected_models: dict | None = None) -> None:
    """
    Вызывает ошибки инициализации.

    Параметры:
    ----------
    initilization_errors_list : list
        Список ошибок инициализации, которые необходимо вызвать.
    target_column : str
        Название целевой переменной.
    data : pd.DataFrame
        Датафрейм с данными.

    Заметки:
    -----
    Если список ошибок пуст, то вызываются все ошибки инициализации.
    """
    ERRORS = {
        "raise_selected_unselected_custom_models_conflict": lambda:
        raise_selected_unselected_custom_models_conflict(
            selected_models=selected_models,
            unselected_models=unselected_models
        )
    }

    if conflict_errors_list:
        for error in conflict_errors_list:
            ERRORS[error]()
    else:
        for error in ERRORS:
            ERRORS[error]()

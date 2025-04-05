from ..models_errors.saving_errors.save_model_path_not_exists import \
    raise_save_model_path_not_exists


def load_all_saving_errors(saving_errors_list: list | None = None,
                           save_path: str = None) -> None:
    """
    Вызывает ошибки сохранения.

    Параметры:
    ----------
    saving_errors_list : list
        Список ошибок сохранения, которые необходимо вызвать.
    save_path : str
        Путь, по которому будет выполняться сохранение модели.

    Заметки:
    -----
    Если список ошибок сохранения пуст, то вызываются все ошибки сохранения.
    """
    if 'raise_save_model_path_not_exists' in saving_errors_list \
            or not saving_errors_list:
        raise_save_model_path_not_exists(save_path=save_path)

    ERRORS = {
        "raise_save_model_path_not_exists": lambda: \
            raise_save_model_path_not_exists(save_path=save_path)
    }

    if saving_errors_list:
        for error in saving_errors_list:
            ERRORS[error]()
    else:
        for error in ERRORS:
            ERRORS[error]()

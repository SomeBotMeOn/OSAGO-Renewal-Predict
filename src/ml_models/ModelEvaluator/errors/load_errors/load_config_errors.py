import pandas as pd

from src.ml_models.ModelEvaluator.errors.ensembles_errors.config_errors.validate_ensemble_config import \
    raise_validate_ensemble_config


def load_all_config_errors(config_errors_list: list | None = None,
                           ensemble_config: dict | None = None) -> None:
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
    Если список ошибок инициализации пуст, то вызываются все ошибки инициализации.
    """
    ERRORS = {
        'raise_validate_ensemble_config': lambda:
        raise_validate_ensemble_config(
            ensemble_config=ensemble_config
        )
    }

    if config_errors_list:
        for error in config_errors_list:
            ERRORS[error]()
    else:
        for error in ERRORS:
            ERRORS[error]()

import pandas as pd

from ..ensembles_errors.config_errors.validate_ensemble_config import \
    raise_validate_ensemble_config
from ..models_errors.initialization_errors.target_column_not_found_error import \
    raise_target_column_not_found_error


def load_all_initilization_errors(initilization_errors_list: list | None = None,
                                  target_column: str = None,
                                  data: pd.DataFrame = None,
                                  ensemble_config: dict = None) -> None:
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
        "raise_target_column_not_found_error": lambda:
        raise_target_column_not_found_error(
            target_column=target_column,
            data=data
        ),
        'raise_validate_ensemble_config': lambda:
        raise_validate_ensemble_config(
            ensemble_config=ensemble_config
        )
    }

    if initilization_errors_list:
        for error in initilization_errors_list:
            ERRORS[error]()
    else:
        for error in ERRORS:
            ERRORS[error]()

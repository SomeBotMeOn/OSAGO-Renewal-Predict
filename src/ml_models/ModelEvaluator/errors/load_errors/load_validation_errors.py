from ..general_errors.unsupported_validation_strategies import \
    raise_unsupported_validation_strategies
from ...errors.general_errors.unsupported_cv_method_error import \
    raise_unsupported_cv_method_error
from ...errors.ensembles_errors.config_errors.validate_ensemble_config import \
    raise_validate_ensemble_config


def load_all_validation_errors(validation_errors_list: list | None = None,
                               cv_method: str | None = None,
                               ensemble_config: dict | None = None) -> None:
    """
    Вызывает ошибки валидации.

    Параметры:
    ----------
    validation_errors_list : list
        Список ошибок валидации, которые необходимо вызвать.
    cv_method : str
        Метод кросс-валидации, для которого проверяется корректность.

    Заметки:
    -----
    Если список ошибок валидации пуст, то вызываются все ошибки валидации.
    """
    ERRORS = {
        "raise_unsupported_cv_method_error": lambda: \
            raise_unsupported_cv_method_error(cv_method=cv_method),
        "raise_unsupported_validation_strategies": lambda: \
            raise_unsupported_validation_strategies(cv_method=cv_method),
        'raise_validate_ensemble_config': lambda: \
            raise_validate_ensemble_config(ensemble_config=ensemble_config)
    }

    if validation_errors_list:
        for error in validation_errors_list:
            ERRORS[error]()
    else:
        for error in ERRORS:
            ERRORS[error]()

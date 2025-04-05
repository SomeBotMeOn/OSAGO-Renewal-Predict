from ..general_errors.unsupported_cv_method_error import \
    raise_unsupported_cv_method_error
from ..general_errors.unsupported_validation_strategies import \
    raise_unsupported_validation_strategies
from ..general_errors.validate_models_parameters import \
    raise_validate_models_parameters


def load_all_general_errors(
        general_errors_list: list | None = None,
        cv_method: str | None = None,
        model_params: dict | None = None,
) -> None:
    """
    Параметры:
    ----------
    general_errors_list : list
        Список ошибок, которые необходимо вызвать.
    cv_method : str
        Метод валидации.
    model_params : dict
        Параметры модели.

    Заметки:
    -----
    Если список ошибок пуст, то вызываются все ошибки инициализации.
    """
    ERRORS = {
        'raise_unsupported_cv_method_error': lambda:
            raise_unsupported_cv_method_error(cv_method=cv_method),
        'raise_unsupported_validation_strategies': lambda:
            raise_unsupported_validation_strategies(cv_method=cv_method),
        'raise_validate_models_parameters': lambda:
            raise_validate_models_parameters(model_params=model_params),
    }

    if general_errors_list:
        for error in general_errors_list:
            ERRORS[error]()
    else:
        for error in ERRORS:
            ERRORS[error]()

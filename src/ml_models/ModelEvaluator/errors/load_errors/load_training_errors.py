from ..models_errors.training_errors.model_training_failed import \
    raise_model_training_failed


def load_all_training_errors(training_errors_list: list | None = None,
                             fold_scores: list = None) -> None:
    """
    Вызывает ошибки тренировки.

    Параметры:
    ----------
    training_errors_list : list
        Список ошибок тренировки, которые необходимо вызвать.
    fold_scores : list
        Список оценок модели по фолдам.

    Заметки:
    -----
    Если список ошибок тренировки пуст, то вызываются все ошибки тренировки.
    """
    ERRORS = {
        "raise_model_training_failed": lambda: raise_model_training_failed(
            fold_scores=fold_scores)
    }

    if training_errors_list:
        for error in training_errors_list:
            ERRORS[error]()
    else:
        for error in ERRORS:
            ERRORS[error]()

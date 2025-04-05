import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, train_test_split

def get_cv_method(cv_method: str, cv_params: dict, X: pd.DataFrame,
                  y: pd.Series, random_state: int) -> tuple:
    """
    Получение разбитых данных для обучения и валидации.

    Parameters
    ----------
    cv_method : str
        Метод кросс-валидации. Возможные значения: 'KFold', 'Stratified', 'LeaveOneOut', 'train_test_split'.
    cv_params : dict
        Параметры метода кросс-валидации: 'n_splits' - количество разбиений для KFold и StratifiedKFold, 'test_size' - размер тестовой выборки для train_test_split.
    X : pd.DataFrame
        Данные для обучения моделей.
    y : pd.Series
        Целевая переменная.

    Returns
    -------
    kf : object
        Объект метода кросс-валидации.
    data : tuple
        Разбитые данные для обучения и валидации (только для метода 'train_test_split').

    Raises
    ------
    ValueError: Неподдерживаемый метод кросс-валидации!
    """
    if cv_method == 'KFold':
        n_splits = cv_params.get('n_splits', 5) if cv_params else 5
        return KFold(n_splits=n_splits, shuffle=True,
                     random_state=random_state), None
    elif cv_method == 'Stratified':
        n_splits = cv_params.get('n_splits', 5) if cv_params else 5
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state), None
    elif cv_method == 'LeaveOneOut':
        return LeaveOneOut(), None
    elif cv_method == 'train_test_split':
        test_size = cv_params.get('test_size', 0.2) if cv_params else 0.2
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)
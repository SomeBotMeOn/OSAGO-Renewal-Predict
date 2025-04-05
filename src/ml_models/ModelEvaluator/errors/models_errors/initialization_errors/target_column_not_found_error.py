import pandas as pd


def raise_target_column_not_found_error(target_column: str,
                                        data: pd.DataFrame) -> None:
    """
    Вызывает исключение, если целевая переменная отсутствует в переданных данных.

    Параметры:
    ----------
    target_column : str
        Название целевой переменной.
    data : pd.DataFrame
        Датафрейм, в котором ищется целевая переменная.

    Исключения:
    ------
    ValueError
        Если целевая переменная отсутствует в переданных данных.
    """
    if target_column not in data.columns:
        raise ValueError(
            f"Целевая переменная '{target_column}' не найдена в данных.")

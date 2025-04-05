import pandas as pd
import numpy as np


def generate_arithmetic(
        df: pd.DataFrame,
        operations: dict = None
) -> tuple[pd.DataFrame, dict]:
    """
    Выполняет арифметические операции между колонками DataFrame и
    добавляет метрики для каждой колонки.

    Args:
        df (pd.DataFrame): Исходный DataFrame с числовыми данными.

        operations (dict or set, optional): Словарь с операциями, которые
            будут выполнены между колонками. Доступные ключи:
            - 'addition': (сложение),
            - 'subtraction': (вычитание),
            - 'multiplication': (умножение),
            - 'div': (деление).

    Returns:
        tuple: DataFrame с результатами арифметических операций.
    """

    # Определяем стандартные операции и метрики, если они не переданы
    if operations is None:
        operations = {"addition": "+", "subtraction": "-",
                      "multiplication": "*", "div": "/"}
    elif operations == {0} or operations == [0]:
        operations = {}

    column_names = df.columns
    results = {}

    # Выполняем операции
    for col1 in column_names:
        for col2 in column_names:
            if col1 != col2:
                if "addition" in operations:
                    results[f'{col1}_addition_{col2}'] = \
                        df[col1] + df[col2]
                if "subtraction" in operations:
                    results[f'{col1}_subtraction_{col2}'] = \
                        df[col1] - df[col2]
                    results[f'{col2}_subtraction_{col1}'] = \
                        df[col2] - df[col1]
                if "multiplication" in operations:
                    results[f'{col1}_multiplication_{col2}'] = \
                        df[col1] * df[col2]
                if "div" in operations:
                    results[f'{col1}_div_{col2}'] = \
                        df[col1] / df[col2].replace(0, np.nan)
                    results[f'{col2}_div_{col1}'] = \
                        df[col2] / df[col1].replace(0, np.nan)

    # Если операций не было, создаем пустой result_df
    if not results:
        result_df = pd.DataFrame(index=df.index)
    else:
        result_df = pd.DataFrame(results)

    return result_df

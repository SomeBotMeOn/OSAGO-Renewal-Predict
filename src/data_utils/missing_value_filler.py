import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd


def preprocess_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция для предварительной обработки и заполнения пропущенных
    значений в DataFrame.

    Args:
        df (pd.DataFrame): Исходный DataFrame, который нужно обработать.

    Returns:
        pd.DataFrame: Обработанный DataFrame с заполненными пропущенными
                      значениями.

    Notes:
        1. Заменяет 'NONE' на np.nan.
        2. Определяет числовые колонки с пропущенными значениями.
        3. Нормализует колонки с пропущенными значениями.
        4. Заполняет пропущенные числовые значения с помощью KNN.

    Raises:
        ValueError: Если переданный объект не является pandas DataFrame.
        ValueError: Если в DataFrame нет числовых колонок с пропущенными
                    значениями.
    """

    # Проверка, что передан DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Переданный объект не является pandas DataFrame.")

    # 1. Заменить 'NONE' на np.nan
    df.replace('NONE', np.nan, inplace=True)

    # 2. Определим числовые колонки с пропущенными значениями
    numeric_columns_with_missing = \
        df.select_dtypes(include=[np.number]).columns

    if len(numeric_columns_with_missing) == 0:
        raise ValueError("В DataFrame нет числовых колонок с пропущенными "
                         "значениями.")

    # 3. Избегаем нормализации колонок с одинаковыми значениями
    columns_to_scale = [col for col in numeric_columns_with_missing if \
                        df[col].nunique() > 1]

    # Если колонки с несколькими уникальными значениями найдены,
    # нормализуем их
    if columns_to_scale:
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # 4. Заполнить пропущенные числовые значения с помощью KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    df[columns_to_scale] = knn_imputer.fit_transform(df[columns_to_scale])

    return df

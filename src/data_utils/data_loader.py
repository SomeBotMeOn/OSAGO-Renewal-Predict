import os
import pandas as pd


def load_and_set_index(file_path: str, index_name: str = None) -> pd.DataFrame:
    """
    Загрузка данных из CSV файла и установка индекса по имени столбца.

    Args:
        file_path (str): Путь к CSV файлу, содержащему данные.
        index_name (str, optional): Имя столбца, который будет использован
                                    в качестве индекса. Если параметр не
                                    передан, индексация не изменяется.
                                    По умолчанию None.

    Returns:
        pd.DataFrame: DataFrame с данными из CSV файла, где столбец с
                      именем `index_name` установлен в качестве индекса,
                      если оно указано.

    Notes:
        Если имя индекса не указано, то данные будут возвращены без
        изменения индекса.

    Raises:
        ValueError: Если файл не существует или путь не верен.
        ValueError: Если указанный столбец `index_name` не существует
                    в данных.
    """
    # Проверка на существование файла
    if not os.path.exists(file_path):
        raise ValueError(f"Файл по пути '{file_path}' не существует.")

    data = pd.read_csv(file_path)

    # Проверка на пустоту данных
    if data.empty:
        raise ValueError("CSV файл пустой или не содержит данных.")

    # Проверка, если индексное имя передано
    if index_name:
        # Проверка на наличие указанного столбца в данных
        if index_name not in data.columns:
            raise ValueError(f"Столбец '{index_name}' не найден в данных.")

        data.set_index(index_name, inplace=True)

    return data

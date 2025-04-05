import os


def raise_save_model_path_not_exists(save_path: str) -> None:
    """
    Вызывает исключение, если путь для сохранения модели не существует.

    Параметры:
    ----------
    save_path : str
        Путь, по которому предполагается сохранение модели.

    Исключения:
    ------
    ValueError
        Если указанный путь для сохранения не существует.
    """
    if not os.path.exists(save_path):
        raise ValueError(
            f"Путь для сохранения моделей не существует: {save_path}")

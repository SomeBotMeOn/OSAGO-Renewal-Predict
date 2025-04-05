import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pandas as pd


def qq_plots(data: pd.DataFrame, dist: str = "norm", sparams: tuple = (),
             n_cols: int = 5, **kwargs) -> None:
    """
    Функция для построения графиков QQ в строках с возможностью изменения
    распределения и других параметров.

    Args:
        data (pd.DataFrame): Данные для построения графиков QQ.
        dist (str, optional): Распределение для сравнения.
                              По умолчанию "norm".
        sparams (tuple, optional): Параметры для распределения.
                                   По умолчанию пустой кортеж.
        n_cols (int, optional): Количество столбцов на графиках.
                                По умолчанию 5.
        kwargs (dict, optional): Дополнительные параметры, передаваемые
                                 в `stats.probplot`.

    Returns:
        None: Функция отображает графики.

    Raises:
        ValueError: Если переданные данные не являются pandas DataFrame.
        ValueError: Если столбцы данных не числовые.

    Note:
        Функция строит графики QQ для всех числовых признаков в DataFrame,
        используя заданное распределение. Параметры `kwargs` могут быть
        использованы для настройки поведения `stats.probplot`.
    """
    # Проверка, что данные - это pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Переданные данные не являются pandas DataFrame.")

    # Проверка на наличие числовых столбцов
    if not all(
            pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
        raise ValueError("Все столбцы должны содержать числовые значения.")

    # Определяем количество строк, исходя из числа данных
    n_rows = math.ceil(len(data.columns) / n_cols)

    # Создаем фигуру с нужным количеством строк и столбцов
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    axes = axes.flatten()

    for i, (feature_name, values) in enumerate(data.items()):
        # Строим QQ-график с заданными параметрами
        stats.probplot(values, dist=dist, sparams=sparams, plot=axes[i],
                       **kwargs)
        axes[i].set_title(feature_name)

    # Убираем лишние графики
    for i in range(len(data.columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

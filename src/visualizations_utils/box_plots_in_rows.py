import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_boxplots(
        data: pd.DataFrame,
        column_types: tuple = None,
        n_cols: int = 5,
        color: str = "blue",
        box_kwargs: dict = {},
        title_kwargs: dict = {},
        figsize: tuple = None
) -> None:
    """
    Строит box plot (ящики с усами) для указанных типов данных в DataFrame.

    Args:
        data (pd.DataFrame): Датасет, для которого строятся box plot.
        column_types (tuple, optional): Типы данных, которые будут включены в графики.
                                        По умолчанию используются все колонки.
        n_cols (int, optional): Количество столбцов на графиках. По умолчанию 5.
        color (str, optional): Цвет box plot. По умолчанию "blue".
        box_kwargs (dict, optional): Дополнительные параметры для `sns.boxplot`.
        title_kwargs (dict, optional): Дополнительные параметры для заголовков графиков.
        figsize (tuple, optional): Размер изображения в формате (ширина, высота).

    Returns:
        None: Функция отображает графики.

    Raises:
        ValueError: Если входной `data` не является DataFrame.
        ValueError: Если DataFrame пуст.
        ValueError: Если `n_cols` меньше или равно 0.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Входной объект не является DataFrame.")

    if data.empty:
        raise ValueError("DataFrame пуст.")

    if n_cols <= 0:
        raise ValueError("Количество столбцов должно быть больше 0.")

    # Если column_types не указан, используем все колонки
    if column_types is None:
        selected_columns = list(data.columns)
    else:
        selected_columns = [col for col in data.columns if
                            data[col].dtype.name in column_types]

    if not selected_columns:
        raise ValueError("В DataFrame нет колонок с указанными типами данных.")

    n_rows = math.ceil(len(selected_columns) / n_cols)

    if figsize is None:
        figsize = (n_cols * 5, n_rows * 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(selected_columns):
        sns.boxplot(y=data[col], color=color, ax=axes[i],
                    orientation='vertical', **box_kwargs)
        axes[i].set_title(col, **title_kwargs)

    for i in range(len(selected_columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_histograms(
        data: pd.DataFrame,
        n_cols: int = 5,
        kde: bool = True,
        color: str = "blue",
        bins: int = 20,
        hist_kwargs: dict = {},
        kde_kwargs: dict = {},
        title_kwargs: dict = {},
        figsize: tuple = None
) -> None:
    """
    Строит гистограммы для всех числовых признаков в DataFrame по несколько штук в строку.

    Args:
        data (pd.DataFrame): Датасет, для которого строятся гистограммы.
        n_cols (int, optional): Количество столбцов на графиках.
                                По умолчанию 5.
        kde (bool, optional): Включение или отключение KDE (плотности
                               вероятности). По умолчанию True.
        color (str, optional): Цвет для гистограммы. По умолчанию "blue".
        bins (int, optional): Количество bins для гистограммы.
                              По умолчанию 20.
        hist_kwargs (dict, optional): Дополнительные параметры для
                                      `sns.histplot`,
                                      такие как `stat="density"`.
        kde_kwargs (dict, optional): Дополнительные параметры для
                                     линии KDE, например `color="red"`.
        title_kwargs (dict, optional): Дополнительные параметры для
                                       заголовков графиков.
        figsize (tuple, optional): Размер изображения в формате (ширина,
                                   высота).

    Returns:
        None: Функция отображает графики.

    Raises:
        ValueError: Если входной `data` не является DataFrame.
        ValueError: Если DataFrame пуст.
        ValueError: Если `n_cols` меньше или равно 0.
        ValueError: Если `bins` меньше или равно 0.

    Notes:
        Строит гистограммы для всех числовых признаков в DataFrame.
    """
    # Проверка на корректность входных данных
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Входной объект не является DataFrame.")

    if data.empty:
        raise ValueError("DataFrame пуст.")

    if n_cols <= 0:
        raise ValueError(
            "Количество столбцов для графиков должно быть больше 0.")

    if bins <= 0:
        raise ValueError(
            "Количество bins для гистограммы должно быть больше 0.")

    # Определяем количество строк, исходя из числа данных
    n_rows = math.ceil(len(data.columns) / n_cols)  # округление вверх

    # Если размер не передан, рассчитываем его автоматически
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 5)

    # Создаем фигуру с нужным количеством строк и столбцов
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Преобразуем axes в одномерный массив
    axes = axes.flatten()

    for i, (feature_name, values) in enumerate(data.items()):
        # Строим гистограмму с KDE, если это нужно
        sns.histplot(values, kde=kde, color=color, bins=bins, ax=axes[i],
                     **hist_kwargs)

        if kde:
            # Настройки для линии KDE, если она включена
            sns.kdeplot(values, ax=axes[i], **kde_kwargs)

        axes[i].set_title(feature_name, **title_kwargs)

    # Убираем лишние графики
    for i in range(len(data.columns), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

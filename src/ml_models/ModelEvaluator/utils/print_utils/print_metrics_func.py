def print_metrics(metrics: tuple[str, float, str | float, str | float, str | float],
                  indent: int = 4) -> None:
    """
    Функция для вывода метрик в удобном формате.

    Parameters
    ----------
    metrics : list
        Список с метриками.
    indent : int, optional, default = 4
        Отступ для вывода.

    Returns
    -------
    None
        Функция выводит метрики.
    """
    indent_space = ' ' * indent
    print(f"{indent_space}Среднее значение метрики:  {metrics[1]}")
    print(f"{indent_space}Минимальное значение:     {metrics[2]}")
    print(f"{indent_space}Максимальное значение:    {metrics[3]}")
    print(f"{indent_space}Стандартное отклонение:   {metrics[4]}")

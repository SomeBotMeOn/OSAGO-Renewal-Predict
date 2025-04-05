def print_header(title: str, width: int = 72) -> None:
    """
    Функция печати заголовка.

    Parameters
    ---------
    title : str
        Заголовок.
    width : int, optional, default=72

    Returns
    -------
    None
        Функция печатает заголовок.
    """
    print('\n' + '-' * width)
    print(f"{f' {title} ':.^{width}}")
    print('-' * width)
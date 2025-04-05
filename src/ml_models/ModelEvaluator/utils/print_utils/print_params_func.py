from typing import Any, Dict

def print_params(params: Dict[str, Any], indent: int = 4) -> None:
    """
    Форматированный вывод параметров в консоль.

    Parameters
    ----------
    params : Dict[str, Any]
        Словарь с параметрами для вывода (ключ-значение).
    indent : int, optional [default=4]
        Количество пробелов для отступа слева.

    Examples
    --------
    >>> params = {'learning_rate': 0.01, 'max_depth': 7, 'model': 'CatBoost'}
    >>> print_params(params)
        learning_rate:          0.01
        max_depth:              7
        model:                 CatBoost
    """
    for k, v in params.items():
        value_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{' ' * indent}{k + ':':<25} {value_str}")
def raise_validate_ensemble_config(ensemble_config: dict) -> None:
    """
    Проверяет корректность конфигурации ансамблевых методов.

    Параметры:
    ----------
    ensemble_config : dict
        Словарь с параметрами ансамблевых методов.

    Возвращает:
    ------
    ValueError
        При обнаружении ошибок в конфигурации.
    """
    method_params = {
        'Stacking': {
            'required': ['base_models', 'meta_model'],
            'allowed': ['base_models', 'meta_model', 'cv_method', 'cv_params',
                        'custom_name']
        },
        'Blending': {
            'required': ['base_models', 'meta_model'],
            'allowed': ['base_models', 'meta_model', 'test_size', 'n_splits',
                        'custom_name']
        },
        'Bagging': {
            'required': ['base_models'],
            'allowed': ['base_models', 'n_estimators', 'max_samples',
                        'max_features',
                        'bootstrap', 'bootstrap_features', 'custom_name']
        },
        'Voting': {
            'required': ['base_models'],
            'allowed': ['base_models', 'voting_type', 'cv_method', 'cv_params',
                        'custom_name']
        }
    }

    for method in ensemble_config:
        # Проверка наличия метода
        if method not in method_params:
            raise ValueError(
                f"Неизвестный метод: '{method}'. Допустимые методы: {list(method_params.keys())}")

        params = method_params[method]
        config = ensemble_config[method]

        # Проверка обязательных параметров
        missing = [p for p in params['required'] if p not in config]
        if missing:
            raise ValueError(
                f"Метод '{method}': отсутствуют обязательные параметры: {missing}")

        # Проверка на недопустимые параметры
        invalid = [p for p in config if p not in params['allowed']]
        if invalid:
            suggestions = {}
            for param in invalid:
                # Попытка предложить исправление для частых опечаток
                if param == 'base_models' and method == 'Bagging':
                    suggestions[param] = 'base_model'
                elif param == 'base_model' and method in ['Stacking',
                                                          'Blending',
                                                          'Voting']:
                    suggestions[param] = 'base_models'

            error_msg = f"Метод '{method}': недопустимые параметры: {invalid}."
            if suggestions:
                error_msg += " Возможные исправления: " + ", ".join(
                    [f"'{k}' -> '{v}'" for k, v in suggestions.items()]
                )
            else:
                error_msg += f" Допустимые параметры: {params['allowed']}"
            raise ValueError(error_msg)

        # Дополнительные проверки
        if method == 'Blending':
            test_size = config.get('test_size', 0.2)
            if not (0 < test_size < 1):
                raise ValueError(
                    f"Blending: test_size должен быть между 0 и 1. Получено: {test_size}")

        if method == 'Voting':
            voting_type = config.get('voting_type', 'hard')
            if voting_type not in ['hard', 'soft']:
                raise ValueError(
                    f"Voting: недопустимый voting_type '{voting_type}'. Допустимо: 'hard' или 'soft'")

        # Проверка структуры параметров
        if method in ['Stacking', 'Blending', 'Voting']:
            if not isinstance(config['base_models'], dict) or len(
                    config['base_models']) == 0:
                raise ValueError(
                    f"Метод '{method}': base_models должен быть непустым словарем")

        if method == 'Bagging':
            if not isinstance(config['base_model'], dict) or len(
                    config['base_model']) == 0:
                raise ValueError(
                    "Bagging: base_model должен быть непустым словарем")

        if method in ['Stacking', 'Blending']:
            if not isinstance(config['meta_model'], dict) or len(
                    config['meta_model']) == 0:
                raise ValueError(
                    f"Метод '{method}': meta_model должен быть непустым словарем")

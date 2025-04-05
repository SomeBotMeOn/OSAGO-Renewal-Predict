import gc
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from tqdm import tqdm
import optuna
from optuna.logging import set_verbosity, CRITICAL
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from ..errors.ensembles_errors.config_errors.validate_ensemble_config import \
    raise_validate_ensemble_config
from ..errors.general_errors.unsupported_cv_method_error import \
    raise_unsupported_cv_method_error
from ..errors.general_errors.unsupported_validation_strategies import \
    raise_unsupported_validation_strategies
from ..errors.general_errors.validate_models_parameters import \
    raise_validate_models_parameters
from ..errors.load_errors.load_initilization_errors import \
    load_all_initilization_errors
from ..errors.models_errors.conflict_errors.selected_unselected_custom_models_conflict import \
    raise_selected_unselected_custom_models_conflict
from ..errors.models_errors.saving_errors.save_model_path_not_exists import \
    raise_save_model_path_not_exists
from ..resources.constants.all_models import ALL_MODELS
from ..utils.ensembles_utils.bagging import bagging_func
from ..utils.ensembles_utils.blending import blending_func
from ..utils.ensembles_utils.stacking import stacking_func
from ..utils.ensembles_utils.voting import voting_func
from ..utils.general_utils.cross_validation import get_cv_method
from ..utils.print_utils.print_metrics_func import \
    print_metrics
from ..utils.models_utils.model_initializer import initialize_models
from ..utils.models_utils.model_trainer import train_model
from ..utils.models_utils.model_saver import save_model
from ..utils.optuna_utils.optuna_objective import objective_wrapper


class ModelEvaluator:
    """
    Класс для оценки качества моделей машинного обучения.

    Параметры:
    ----------
    data : pd.DataFrame
        Данные для обучения моделей.
    target_column : str
        Название целевой переменной.
    random_state : int, optional [default=42]
        Seed для воспроизводимости.

    Возвращает:
    -------
    pd.DataFrame
        Таблица с результатами оценки качества моделей.

    Заметка:
    -------
    Поддерживаемые модели:
        - CatBoost
        - XGBoost
        - LightGBM
        - LogisticRegression
        - RandomForest
        - DecisionTree
        - KNeighbors
        - SVC
        - GradientBoosting
        - AdaBoost
        - GaussianNB
        - LDA
        - ExtraTrees

    Поддерживаемые методы ансамблирования:
        - Blending
        - Bagging
        - Stacking
        - Voting

    Исключения:
    ------
    ValueError
        - Если целевая переменная не найдена в данных
        - При одновременной передаче selected_models и unselected_models
        - При указании неподдерживаемого метода кросс-валидации
        - При ошибках обучения моделей
        - При несуществующем пути для сохранения моделей
        - При неправильной конфигурации ансамблей
    """

    def __init__(self, data: pd.DataFrame, target_column: str,
                 random_state: int = 42) -> None:
        """
        Инициализация класса.

        Параметры:
        ----------
        data : pd.DataFrame
            Данные для обучения моделей.
        target_column : str
            Название целевой переменной. Должна присутствовать в `data`.
        random_state : int, optional
            Seed для воспроизводимости (по умолчанию 42).

        Возвращает:
        ------
        ValueError
            Если `target_column` отсутствует в данных.

        Пример:
        --------
        >>> data = pd.DataFrame({'feature': [1, 2], 'target': [0, 1]})
        >>> evaluator = ModelEvaluator(data, 'target', random_state=42)
        """
        self.data = data
        self.target_column = target_column
        self.random_state = random_state

        initilization_errors_list = ['raise_target_column_not_found_error']
        load_all_initilization_errors(
            initilization_errors_list=initilization_errors_list,
            target_column=target_column, data=data)

    def evaluate_models(self, selected_models: dict | None = None,
                        unselected_models: dict | None = None,
                        custom_params: dict[str, dict] | None = None,
                        cv_method: str = 'KFold',
                        cv_params: dict[str, float | int] | None = None,
                        save_models_to: str | None = None) -> pd.DataFrame:
        """
        Оценка качества моделей машинного обучения.

        Параметры
        ----------
        selected_models : dict, optional [default=None]
            Словарь с выбранными моделями и их параметрами.
        unselected_models : dict, optional [default=None]
            Список моделей, которые необходимо исключить из оценки.
        custom_params : dict, optional [default=None]
            Словарь с пользовательскими параметрами для моделей.
        cv_method : str, optional [default='KFold']
            Метод кросс-валидации.
                Возможные значения:
                    - 'KFold',
                    - 'Stratified',
                    - 'LeaveOneOut',
                    - 'train_test_split'.
        cv_params : dict, optional [default=None]
            Параметры метода кросс-валидации:
                Возможные значения:
                    - 'n_splits' - количество разбиений для KFold и StratifiedKFold,
                    - 'test_size' - размер тестовой выборки для train_test_split.
        save_models_to : str, optional [default=None]
            Путь для сохранения моделей.

        Возвращает
        -------
        data : pd.DataFrame
            Таблица с результатами оценки качества моделей.

        Заметка:
        -------
        Поддерживаемые модели:
            - CatBoost
            - XGBoost
            - LightGBM
            - LogisticRegression
            - RandomForest
            - DecisionTree
            - KNeighbors
            - SVC
            - GradientBoosting
            - AdaBoost
            - GaussianNB
            - LDA
            - ExtraTrees

        Пример:
        --------
        1) selected_models - с выбранными моделями и их параметрами:

        >>> import pandas as pd
        >>> data = pd.read_csv('data.csv')
        >>> evaluator = ModelEvaluator(data, 'target')
        >>> results = evaluator.evaluate_models(
        ...     selected_models={'RandomForest': {'n_estimators': 100}},
        ...     cv_method='KFold',
        ...     cv_params={'n_splits': 5}
        ... )
        >>> print(results.head())

        2) unselected_models - список моделей, которые необходимо исключить из оценки:

        >>> import pandas as pd
        >>> data = pd.read_csv('data.csv')
        >>> evaluator = ModelEvaluator(data, 'target')
        >>> results = evaluator.evaluate_models(
        ...     unselected_models={
        ...         'RandomForest': {},
        ...         'KNeighbors': {}
        ...     },
        ...     cv_method='KFold',
        ...     cv_params={'n_splits': 5}
        ... )
        >>> print(results.head())

        3) custom_params - словарь с пользовательскими параметрами для моделей:

        >>> import pandas as pd
        >>> data = pd.read_csv('data.csv')
        >>> evaluator = ModelEvaluator(data, 'target')
        >>> results = evaluator.evaluate_models(
        ...     custom_params={
        ...         'RandomForest': {'n_estimators': 100},
        ...         'KNeighbors': {'n_neighbors': 5}
        ...     },
        ...     cv_method='KFold',
        ...     cv_params={'n_splits': 5}
        ... )
        >>> print(results.head())
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        print('-' * 72)
        print(f"{'Проверка общих конфликтов и ошибок... ':^72}")
        # Проверка конфликтов и ошибок
        raise_save_model_path_not_exists(save_path=save_models_to)
        raise_selected_unselected_custom_models_conflict(
            selected_models=selected_models,
            unselected_models=unselected_models,
            custom_params=custom_params
        )
        raise_unsupported_validation_strategies(cv_method=cv_method)
        if selected_models:
            raise_validate_models_parameters(model_params=selected_models)
        elif unselected_models:
            raise_validate_models_parameters(model_params=unselected_models)
        elif custom_params:
            raise_validate_models_parameters(model_params=custom_params)
        print(f"{' Успешно! ':^72}")
        print('-' * 72)

        cross_val_res = get_cv_method(cv_method=cv_method, cv_params=cv_params,
                                      X=X, y=y, random_state=self.random_state)
        kf, X_train, X_val, y_train, y_val = None, None, None, None, None
        if len(cross_val_res) == 2:
            kf, _ = cross_val_res
        else:
            X_train, X_val, y_train, y_val = cross_val_res[0], cross_val_res[
                1], cross_val_res[2], cross_val_res[3]

        print('\n' + '-' * 72)
        print(
            f"I Метод кросс-валидации: {cv_method}: "
            f"{cv_params if cv_params else 'default'}")
        print('-' * 72)

        default_params = {
            model: {'random_state': self.random_state} if model != 'CatBoost'
            else {'random_seed': self.random_state}
            for model in ALL_MODELS
            if 'random_state' in ALL_MODELS[
                model]().get_params() or model == 'CatBoost'}

        available_models = initialize_models(selected_models,
                                             unselected_models,
                                             default_params, ALL_MODELS,
                                             custom_params)

        print("II Используемые модели:")
        for model_name, (model_class, params) in available_models.items():
            print(f"{model_name}: {params if params else 'default'}")
        print('-' * 72)

        model_results, models_to_save = [], {}
        for model_name, (model_class, params) in tqdm(
                available_models.items(), desc="III обучение моделей",
                unit="model"
        ):
            print(
                f"Обучение модели {model_name} с параметрами: "
                f"{params if params else 'default'}...")
            avg_score, min_score, max_score, std_score, model = train_model(
                model_class=model_class,
                params=params,
                X=X,
                y=y,
                kf=kf,
                name=model_name,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val
            )
            model_results.append(
                (model_name, avg_score, min_score, max_score, std_score))
            print(
                f"Среднее значение метрики: {avg_score}, "
                f"Минимальное значение: {min_score}, "
                f"Максимальное значение: {max_score}, "
                f"Стандартное отклонение: {std_score}")
            if not save_models_to:
                print('-' * 72)

            # Очистка кэша после каждой модели
            gc.collect()

            # Сохранение модели
            if save_models_to:
                save_model(model=model, model_name=model_name,
                           save_path=save_models_to)
                print(f"{model_name} успешно сохранена в: {save_models_to}")
                print('-' * 72)

        print('\n' + '-' * 72)
        print(f"{' Итоговые результаты ансамблирования ':^72}")
        print('-' * 72 + '\n')

        return pd.DataFrame(model_results,
                            columns=['Model', 'Avg_Score', 'Min_Score',
                                     'Max_Score', 'Std_Score'])

    def evaluate_ensembles(
            self,
            ensemble_config: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        """
        Применение методов ансамблирования: Stacking, Bagging, Blending, Voting.

        Параметры
        ----------
        base_models : dict
            Словарь с базовыми моделями и их параметрами.
        meta_model : BaseEstimator, optional [default=None]
            Мета-модель для Stacking и Blending.
        ensemble_type : str, optional [default='Stacking']
            Тип ансамбля. Возможные значения: 'Stacking', 'Bagging', 'Blending', 'Voting'.
        voting_type : str, optional [default='hard']
            Тип голосования для Voting: 'hard' или 'soft'.
        n_splits : int, optional [default=5]
            Количество фолдов для кросс-валидации.
        test_size : float, optional [default=0.2]
            Размер тестовой выборки для Blending.
        random_state : int, optional [default=42]
            Установка начального состояния генератора случайных чисел.

        Возвращает
        -------
        pd.DataFrame
            Таблица с результатами оценки качества ансамблевых моделей.

        Пример:
        --------
        >>> import pandas as pd
        >>> data = pd.read_csv('data.csv')
        >>> evaluator = ModelEvaluator(data, 'target')
        >>> ensemble_config = {
        ...     'Stacking': {
        ...         'base_models': {
        ...             'RandomForest': {'n_estimators': 100},
        ...             'LogisticRegression': {'C': 0.1}
        ...         },
        ...         'meta_model': {'DecisionTree': {'max_depth': 3}},
        ...         'cv_method': 'KFold'
        ...     }
        ... }
        >>> results = evaluator.evaluate_ensembles(ensemble_config)
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        print('-' * 72)
        print(f"{' Проверка конфликтов и ошибок... ':^72}")

        # Проверка конфликтов и ошибок
        raise_validate_ensemble_config(ensemble_config=ensemble_config)

        # Создадим словарь со всеми моделями,
        # чтобы проверить их параметры
        models = {}

        # Blending
        blending = ensemble_config.get('Blending', {})
        models.update(blending.get('base_models', {}))
        models.update(blending.get('meta_model', {}))

        # Stacking
        stacking = ensemble_config.get('Stacking', {})
        models.update(stacking.get('base_models', {}))
        models.update(stacking.get('meta_model', {}))

        # Bagging
        bagging = ensemble_config.get('Bagging', {})
        models.update(bagging.get('base_models', {}))

        # Voting
        voting = ensemble_config.get('Voting', {})
        models.update(voting.get('base_models', {}))

        raise_validate_models_parameters(model_params=models)
        print(f"{' Успешно! ':^72}")
        print('-' * 72 + '\n')

        print('-' * 72)
        print("Применение методов ансамблирования:".center(72))

        results = []  # Список результатов работы ансамблей
        if ensemble_config.get('Stacking'):
            print('-' * 72)
            print(f"{' Stacking... ':^72}")
            stacking_result = stacking_func(
                X=X,
                y=y,
                ensemble_config=ensemble_config,
                random_state=self.random_state
            )
            results.append(stacking_result)
            print_metrics(stacking_result)

        if ensemble_config.get('Blending'):
            print('-' * 72)
            print(f"{' Blending... ':^72}")
            blending_results = blending_func(
                X=X,
                y=y,
                ensemble_config=ensemble_config,
                random_state=self.random_state
            )
            results.append(blending_results)
            print_metrics(blending_results)

        if ensemble_config.get('Bagging'):
            print('-' * 72)
            print(f"{' Bagging... ':^72}")
            bagging_results = bagging_func(
                X=X,
                y=y,
                ensemble_config=ensemble_config,
                random_state=self.random_state
            )
            results.append(bagging_results)
            print_metrics(bagging_results)

        if ensemble_config.get('Voting'):
            print('-' * 72)
        print(f"{' Voting... ':^72}")
        voting_results = voting_func(
            X=X,
            y=y,
            ensemble_config=ensemble_config,
            random_state=self.random_state
        )
        results.append(voting_results)
        print_metrics(voting_results)

        print('\n' + '-' * 72)
        print(f"{' Итоговые результаты ансамблирования ':^72}")
        print('-' * 72 + '\n')

        return pd.DataFrame(
            results,
            columns=['Ensemble_Type', 'Avg_Score', 'Min_Score', 'Max_Score',
                     'Std_Score']
        )

    def tune_models_with_optuna(
            self,
            optuna_config: Dict[str, Any],
            n_trials: Optional[int] = None,
            timeout: Optional[int] = None,
            scoring: str = 'accuracy',
            cv_method: str = 'KFold',
            cv_params: Optional[Dict[str, Any]] = None,
            show_progress_bar: bool = True
    ) -> tuple[dict, pd.DataFrame]:
        """
        Оптимизирует гиперпараметры моделей с использованием Optuna.

        Параметры
        ----------
        optuna_config : Dict[str, Any]
            Конфигурация гиперпараметров для каждой модели в формате:
            {'Имя_модели': {гиперпараметр: распределение Optuna}}.
        n_trials : int, optional [default=None]
            Количество испытаний для Optuna. Если не задано, используется timeout.
        timeout : int, optional [default=None]
            Время (в секундах) на оптимизацию. Если не задано, по умолчанию 3600 при отсутствии n_trials.
        scoring : str, optional [default='accuracy']
            Метрика для оптимизации.
        cv_method : str, optional [default='KFold']
            Метод кросс-валидации.
        cv_params : Dict[str, Any], optional [default={'n_splits': 5}]
            Дополнительные параметры для метода кросс-валидации (например, n_splits, test_size).
        show_progress_bar : bool, optional [default=True]
            Показывать прогресс-бар во время оптимизации.

        Возвращает
        -------
        tuple[dict, pd.DataFrame]
            Кортеж содержащий:
            - Лучшие модели (dict): Словарь с оптимизированными экземплярами моделей.
            - Метрики (pd.DataFrame): DataFrame с агрегированными результатами испытаний (среднее, мин/макс значения).

        Пример
        -------
        >>> evaluator = ModelEvaluator(data, 'target')
        >>> optuna_config = {
        ...     'RandomForest': {
        ...         'n_estimators': optuna.distributions.IntDistribution(100, 200),
        ...         'max_depth': optuna.distributions.IntDistribution(3, 10)
        ...     },
        ...     'LogisticRegression': {
        ...         'C': optuna.distributions.FloatDistribution(0.1, 1.0, log=True)
        ...     }
        ... }
        >>> best_models, metrics_df = evaluator.tune_models_with_optuna(
        ...     optuna_config,
        ...     n_trials=50,
        ...     cv_method='KFold',
        ...     cv_params={'n_splits': 5}
        ... )
        >>> print(metrics_df)
        """
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        if cv_params is None:
            cv_params = {'n_splits': 5}

        print('-' * 72)
        print(f"{' Проверка конфликтов и ошибок... ':^72}")

        # Проверка конфликтов и ошибок
        raise_unsupported_cv_method_error(cv_method=cv_method)
        config_for_validate = {
            model_name: {param: 0 for param in params.keys()}
            for model_name, params in optuna_config.items()
        }
        raise_validate_models_parameters(model_params=config_for_validate)

        print(f"{' Успешно! ':^72}")
        print('-' * 72 + '\n')

        # Получение метода кросс-валидации
        cv = get_cv_method(cv_method, cv_params, X, y, self.random_state)

        results = []
        best_models = {}

        print('-' * 72)
        print(f"{' Оптимизация гиперпараметров моделей... ':^72}")
        print('-' * 72 + '\n')
        for model_name in optuna_config:
            print('-' * 72)
            print(f"{f' Модель: {model_name} ':^72}")
            for param, distribution in optuna_config[model_name].items():
                print(f"{param}: {distribution}")
            # Установка timeout по умолчанию, если не заданы n_trials и timeout
            current_n_trials = n_trials
            current_timeout = timeout
            if current_n_trials is None and current_timeout is None:
                current_timeout = 3600

            # Создание исследования Optuna
            study = optuna.create_study(direction="maximize")
            if not show_progress_bar:
                set_verbosity(CRITICAL)
            study.optimize(
                lambda trial: objective_wrapper(
                    trial, model_name, optuna_config, scoring, X, y, cv[0]
                ),
                n_trials=current_n_trials,
                timeout=current_timeout,
                show_progress_bar=show_progress_bar
            )

            # Сбор статистики по завершенным испытаниям
            completed_trials = [t for t in study.trials if
                                t.state == optuna.trial.TrialState.COMPLETE]
            values = [t.value for t in completed_trials if
                      t.value != float('-inf')]

            # Заполнение метрик
            metrics = {
                'Model': model_name,
                'Mean_Score': np.mean(values) if values else 'N/A',
                'Min_Score': min(values) if values else 'N/A',
                'Max_Score': max(values) if values else 'N/A',
                'Std_Score': np.std(values) if values else 'N/A'
            }
            results.append(metrics)

            print('\n')
            print(f'{f" Модель: {model_name} ":^72}')
            print(f"{' Итоги оптимизации ':^72}")
            print_metrics(metrics=(metrics['Model'], metrics['Mean_Score'],
                                   metrics['Min_Score'], metrics['Max_Score'],
                                   metrics['Std_Score']))

            # Сохранение лучшей модели
            model_class = ALL_MODELS[model_name]
            best_model = model_class(**study.best_params)
            best_model.fit(X, y)
            best_models[model_name] = best_model

            print('-' * 72)

        return best_models, pd.DataFrame(results).fillna('N/A')

    def evaluate_clustering(
            self,
            metrics: list | None = None
    ) -> pd.DataFrame:
        """
        Вычисляет метрики качества кластеризации и возвращает результаты в виде таблицы.

        Параметры
        ----------
        metrics : list, optional [default=None]
            Список метрик для расчета. Если None, используются все доступные:
            ['silhouette', 'calinski_harabasz', 'davies_bouldin'].

        Возвращает
        -------
        pd.DataFrame
            DataFrame с результатами расчета метрик.

        Пример
        -------
        >>> from sklearn.datasets import make_blobs
        >>> from sklearn.cluster import KMeans

        # Генерация тестовых данных
        >>> X, _ = make_blobs(n_samples=500, centers=3, random_state=42)
        >>> kmeans = KMeans(n_clusters=3, random_state=42)
        >>> labels = kmeans.fit_predict(X)

        # Расчет метрик
        >>> metrics_df = evaluate_clustering(X, labels)
        >>> print(metrics_df.round(3))
                              Value
        Metric
        Silhouette Score      0.734
        Calinski-Harabasz    565.411
        Davies-Bouldin         0.554
        """
        X = self.data.drop(columns=[self.target_column])
        labels = self.data[self.target_column]
        available_metrics = {
            'silhouette': silhouette_score,
            'calinski_harabasz': calinski_harabasz_score,
            'davies_bouldin': davies_bouldin_score
        }

        # Проверка и установка метрик по умолчанию
        if metrics is None:
            metrics = list(available_metrics.keys())
        else:
            metrics = [m.lower() for m in metrics]

        results = {}

        # Расчет выбранных метрик
        for metric in metrics:
            if metric not in available_metrics:
                raise ValueError(
                    f"Неизвестная метрика: {metric}. "
                    f"Возможные метрики: {list(available_metrics.keys())}")

            try:
                score = available_metrics[metric](X, labels)
                results[metric.replace('_', ' ').title()] = score
            except Exception as e:
                raise RuntimeError(
                    f"Ошибка рассчета результата для {metric}: {str(e)}")

        # Форматирование результатов в DataFrame
        results_df = pd.DataFrame(
            data={'Value': [round(val, 3) for val in results.values()]},
            index=pd.Index(results.keys(), name='Metric')
        )

        return results_df

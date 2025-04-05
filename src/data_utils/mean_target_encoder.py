import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool
import tqdm
from functools import partial


def apply_mean_target_encoder_encoding(X_train, X_test, y_train, cat_features,
                                       alpha=1, n_jobs=4):
    """
    Применяет Mean Target Encoding ко всем указанным категориальным признакам.

    Args:
        X_train (pd.DataFrame): Тренировочные данные
        X_test (pd.DataFrame): Тестовые данные
        y_train (pd.Series): Целевая переменная
        cat_features (list): Список категориальных признаков
        alpha (int): Коэффициент регуляризации
        n_jobs (int): Количество процессов

    Returns:
        pd.DataFrame, pd.DataFrame: Закодированные тренировочные и тестовые данные
    """
    with Pool(n_jobs) as pool:
        results = list(tqdm.tqdm(
            pool.imap(
                partial(
                    mean_target_encoder,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    alpha=alpha
                ),
                cat_features  # Передаем только категориальные признаки
            ),
            total=len(cat_features)
        ))

    # Добавляем новые признаки и удаляем старые
    for col_name, train_enc, test_enc in results:
        X_train[f'{col_name}_encoded'] = train_enc
        X_test[f'{col_name}_encoded'] = test_enc

    return X_train.drop(cat_features, axis=1), X_test.drop(cat_features,
                                                           axis=1)

def mean_target_encoder(
        cat_col,
        X_train,
        X_test,
        y_train,
        alpha=1
):
    """
    Применяет регуляризованное Mean Target Encoding с кросс-валидацией для категориальных признаков.

    Args:
        cat_col (str): Название категориального признака
        X_train (pd.DataFrame): Тренировочный набор данных
        X_test (pd.DataFrame): Тестовый набор данных
        y_train (pd.Series): Целевая переменная для тренировочного набора
        alpha (int, optional): Коэффициент регуляризации. По умолчанию 1.

    Returns:
        tuple: Кортеж (имя_колонки, закодированные_значения_трейна, закодированные_значения_теста)
    """
    # Валидация данных (оставить без изменений)
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test,
                                                               pd.DataFrame):
        raise ValueError("X_train и X_test должны быть pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise ValueError("y_train должен быть pandas Series")
    if alpha <= 0:
        raise ValueError("alpha должен быть положительным числом")
    if cat_col not in X_train.columns or cat_col not in X_test.columns:
        raise KeyError(f"Колонка {cat_col} отсутствует в данных")

    global_mean = y_train.mean()

    # Исправленная часть: группировка по данным из X_train с использованием y_train
    grouped = y_train.groupby(X_train[cat_col]).agg(['sum', 'count'])
    sums = grouped['sum'].to_dict()
    counts = grouped['count'].to_dict()

    # Кодирование тестовых данных (оставить без изменений)
    test_mapping = {
        cat: (sums[cat] + alpha * global_mean) / (counts[cat] + alpha)
        for cat in sums
    }
    test_encoded = X_test[cat_col].map(test_mapping).fillna(global_mean)

    # Кросс-валидация (исправленная часть)
    train_encoded = np.zeros(len(X_train))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_train, y_train):
        # Группировка только по тренировочной части фолда
        grouped_fold = y_train.iloc[train_idx].groupby(
            X_train.iloc[train_idx][cat_col]).agg(['sum', 'count'])
        sums_fold = grouped_fold['sum'].to_dict()
        counts_fold = grouped_fold['count'].to_dict()

        # Кодирование валидационной части
        fold_mapping = {
            cat: (sums_fold.get(cat, 0) + alpha * global_mean) / (
                        counts_fold.get(cat, 0) + alpha)
            for cat in sums_fold
        }
        train_encoded[val_idx] = X_train.iloc[val_idx][cat_col].map(
            fold_mapping).fillna(global_mean)

    return cat_col, train_encoded, test_encoded
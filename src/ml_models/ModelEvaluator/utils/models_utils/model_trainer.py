import pandas as pd
import numpy as np

from catboost import Pool
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import lightgbm as lgb

from ...errors.load_errors.load_training_errors import load_all_training_errors


def train_model(model_class: type, params: dict, X: pd.DataFrame,
                y: pd.Series, kf: object, name: str, X_train: pd.DataFrame,
                X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> tuple:
    """
    Обучение модели и оценка качества.

    Parameters
    ----------
    model_class : type
        Класс модели.
    params : dict
        Параметры модели.
    X : pd.DataFrame
        Данные для обучения модели.
    y : pd.Series
        Целевая переменная.
    kf : object
        Объект метода кросс-валидации.
    name : str
        Название модели.
    X_train : pd.DataFrame
        Данные для обучения модели (при использовании train_test_split).
    X_val : pd.DataFrame
        Данные для валидации модели (при использовании train_test_split).
    y_train : pd.Series
        Целевая переменная для обучения модели (при использовании train_test_split).
    y_val : pd.Series
        Целевая переменная для валидации модели (при использовании train_test_split).

    Returns
    -------
    tuple
        Среднее значение, минимальное значение, максимальное значение и стандартное отклонение метрики качества модели.
        При использовании train_test_split минимальное, максимальное и стандартное отклонение заменяются на 'N/A'.
    """
    model = model_class(**params)
    fold_scores = []

    if kf:
        for train_index, val_index in kf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            if name == 'CatBoost':
                train_data = Pool(X_train_fold, label=y_train_fold)
                val_data = Pool(X_val_fold, label=y_val_fold)
                model = model_class(**params)
                model.fit(train_data, eval_set=val_data)
                preds = model.predict(X_val_fold)
            elif name == 'XGBoost':
                dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
                dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
                if 'verbosity' in params and params['verbosity'] == 0:
                    params['verbosity'] = 0
                    params['disable_default_eval_metric'] = 1
                model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'eval')])
                preds_proba = model.predict(dval)
                preds = (preds_proba > 0.5).astype(int) if preds_proba.ndim == 1 else np.argmax(preds_proba, axis=1)
            elif name == 'LightGBM':
                train_data_lgb = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data_lgb = lgb.Dataset(X_val_fold, label=y_val_fold)
                model = lgb.train(params, train_set=train_data_lgb, valid_sets=[train_data_lgb, val_data_lgb], valid_names=['train', 'eval'])
                preds_proba = model.predict(X_val_fold)
                preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 and preds_proba.shape[1] > 1 else (preds_proba > 0.5).astype(int)
            else:
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict(X_val_fold)

            score = balanced_accuracy_score(y_val_fold, preds)
            fold_scores.append(score)
    else:
        if name == 'CatBoost':
            train_data = Pool(X_train, label=y_train)
            val_data = Pool(X_val, label=y_val)
            model = model_class(**params)
            model.fit(train_data, eval_set=val_data)
            preds = model.predict(X_val)
        elif name == 'XGBoost':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            if 'verbosity' in params and params['verbosity'] == 0:
                params['verbosity'] = 0
                params['disable_default_eval_metric'] = 1
            model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'eval')])
            preds_proba = model.predict(dval)
            preds = (preds_proba > 0.5).astype(int) if preds_proba.ndim == 1 else np.argmax(preds_proba, axis=1)
        elif name == 'LightGBM':
            train_data_lgb = lgb.Dataset(X_train, label=y_train)
            val_data_lgb = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(params, train_set=train_data_lgb, valid_sets=[train_data_lgb, val_data_lgb], valid_names=['train', 'eval'])
            preds_proba = model.predict(X_val)
            preds = np.argmax(preds_proba, axis=1) if preds_proba.ndim > 1 and preds_proba.shape[1] > 1 else (preds_proba > 0.5).astype(int)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

        score = balanced_accuracy_score(y_val, preds)
        fold_scores.append(score)

    # Проверка ошибок обучения
    training_errors_list = ['raise_model_training_failed']
    load_all_training_errors(training_errors_list=training_errors_list, fold_scores=fold_scores)

    # Определение возвращаемых значений
    if kf:
        mean_score = round(np.mean(fold_scores), 2)
        min_score = round(np.min(fold_scores), 2)
        max_score = round(np.max(fold_scores), 2)
        std_score = round(np.std(fold_scores), 2)
    else:
        mean_score = round(np.mean(fold_scores), 2)
        min_score = 'N/A'
        max_score = 'N/A'
        std_score = 'N/A'

    return mean_score, min_score, max_score, std_score, model
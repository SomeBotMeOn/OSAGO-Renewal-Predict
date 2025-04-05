from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb

# Словарь всех моделей
ALL_MODELS = {
    'CatBoost': CatBoostClassifier,
    'XGBoost': xgb.XGBClassifier,
    'LightGBM': lgb.LGBMClassifier,
    'LogisticRegression': LogisticRegression,
    'RandomForest': RandomForestClassifier,
    'DecisionTree': DecisionTreeClassifier,
    'KNeighbors': KNeighborsClassifier,
    'SVC': SVC,
    'GradientBoosting': GradientBoostingClassifier,
    'AdaBoost': AdaBoostClassifier,
    'GaussianNB': GaussianNB,
    'LDA': LinearDiscriminantAnalysis,
    'ExtraTrees': ExtraTreesClassifier,
}
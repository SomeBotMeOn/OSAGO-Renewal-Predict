import os
import pickle

def save_model(model, model_name: str, save_path: str) -> str:
    """
    Сохранение модели в файл.

    Parameters
    ----------
    model : object
        Обученная модель.
    model_name : str
        Название модели.
    save_path : str
        Путь для сохранения модели.

    Returns
    -------
    str
        Путь к сохраненной модели.

    Raises
    ------
    ValueError: Путь для сохранения моделей не существует!
    """
    model_file: str = os.path.join(save_path,
                                   f"{model_name}.cbm"
                                   if model_name == 'CatBoost'
                                   else f"{model_name}.pkl")
    if model_name == 'CatBoost':
        model.save_model(model_file)
    else:
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)
    return model_file
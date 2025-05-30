import os
import pandas as pd
from logging_config import logger
#from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import psutil

def load_and_merge(data_dir: str) -> pd.DataFrame:
    """Загружает и объединяет все CSV-файлы."""
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = []
    
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            dfs.append(df)
            logger.info(f"Успешно загружен {file} (строк: {len(df)})")
        except Exception as e:
            logger.error(f"Ошибка загрузки {file}: {str(e)}")
            continue
    
    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Объединенный датасет: {len(merged_df)} строк")
    return merged_df

#def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет пропуски и дубликаты."""
    threshold = len(df) * 0.5
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    
    df_cleaned = df_cleaned.drop_duplicates()
    logger.info(f"После очистки: {len(df_cleaned)} строк")
    return df_cleaned
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет пропуски, дубликаты и бесконечные значения."""
    # Удаление столбцов с >50% пропусков
    threshold = len(df) * 0.5
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    
    # Замена бесконечных значений на NaN и их удаление
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Удаление дубликатов
    df_cleaned = df_cleaned.drop_duplicates()
    
    logger.info(f"После очистки: {len(df_cleaned)} строк")
    return df_cleaned
    

#1.def balance_classes(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Балансирует классы с помощью SMOTE."""
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X = df.drop(columns=[target_col])
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df[target_col] = le.inverse_transform(y_res)
    logger.info(f"После балансировки: {len(balanced_df)} строк")
    return balanced_df
#2.def balance_classes(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Балансирует классы с помощью SMOTE."""
    # Проверка на бесконечные значения
    if np.isinf(df.drop(target_col, axis=1)).any().any():
        raise ValueError("Данные содержат бесконечные значения. Сначала очистите их!")
    
    # Проверка на очень большие значения
    if (df.drop(target_col, axis=1).abs() > 1e20).any().any():
        logger.warning("Обнаружены экстремально большие значения. Масштабируйте данные.")
    
    # Кодировка меток
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X = df.drop(columns=[target_col])
    
    # Масштабирование (если нужно)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Балансировка
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    # Собираем обратно
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df[target_col] = le.inverse_transform(y_res)
    return balanced_df

def balance_classes(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Быстрая балансировка через RandomOverSampler"""
    from imblearn.over_sampling import RandomOverSampler
    
    # Уменьшаем данные до 300k строк (если больше)
    if len(df) > 300_000:
        df = df.sample(n=300_000, random_state=42)
        logger.info(f"Данные уменьшены до {len(df)} строк для быстрой балансировки")
    
    # Разделяем на признаки и целевую переменную
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Балансировка (в 5-10 раз быстрее SMOTE)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    
    # Собираем обратно в DataFrame
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df[target_col] = y_res
    
    logger.info(f"Балансировка завершена. Итоговый размер: {len(balanced_df)} строк")
    return balanced_df

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.3):
    """Разделяет данные на train и test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} строк, Test: {len(X_test)} строк")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    # Конфигурация путей
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # 1. Загрузка и объединение
    df = load_and_merge(raw_data_dir)
    raw_output_path = os.path.join(processed_data_dir, "1_merged_raw.csv")
    df.to_csv(raw_output_path, index=False)

    # Проверка столбцов 
    print("\nВсе столбцы в данных:")
    print(df.columns.tolist())

    print("\nПервые 5 строк:")
    print(df.head())
    
    # 2. Очистка данных (Шаг 3)
    df_cleaned = clean_data(df)
    cleaned_output_path = os.path.join(processed_data_dir, "2_cleaned.csv")
    df_cleaned.to_csv(cleaned_output_path, index=False)

    # ==== ВСТАВКА 1: Контроль памяти и уменьшение данных ====
    mem = psutil.virtual_memory()
    logger.info(f"Доступно RAM: {mem.available / 1024**3:.1f} GB")
    
    # Уменьшаем до 300k строк (или меньше при нехватке памяти)
    sample_size = 300_000
    if mem.available < 4 * 1024**3:  # Если меньше 4GB RAM
        sample_size = 200_000
        logger.warning("Мало памяти! Берём только 200k строк")
    
    if len(df_cleaned) > sample_size:
        df_cleaned = df_cleaned.sample(n=sample_size, random_state=42)
        logger.info(f"Данные уменьшены до {len(df_cleaned)} строк")
    
    # 3. Балансировка классов (Шаг 4)
    target_column = " Label" #"Label"  # Замените на ваше название столбца!
    df_balanced = balance_classes(df_cleaned, target_column)
    balanced_output_path = os.path.join(processed_data_dir, "3_balanced.csv")
    df_balanced.to_csv(balanced_output_path, index=False)

    
    # ==== ВСТАВКА 2: Проверка распределения классов ====
    logger.info("\nИтоговое распределение классов:")
    logger.info(df_balanced[target_column].value_counts())
    # ===================================================
    
    # 4. Разделение на train/test (Шаг 5)
    X_train, X_test, y_train, y_test = split_data(df_balanced, target_column)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(processed_data_dir, "4_train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_data_dir, "5_test.csv"), index=False)

    logger.info("Обработка завершена успешно!")













# import os
# import pandas as pd
# from logging_config import logger
# #import sys
# #sys.path.append('..')  # Добавляем корень проекта в пути поиска
# #from logging_config import logger
# from sklearn.model_selection import train_test_split


# def load_and_merge(data_dir: str) -> pd.DataFrame:
#     """Загружает и объединяет все CSV-файлы."""
#     all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
#     dfs = []
    
#     for file in all_files:
#         file_path = os.path.join(data_dir, file)
#         try:
#             df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
#             dfs.append(df)
#             logger.info(f"Успешно загружен {file} (строк: {len(df)})")
#         except Exception as e:
#             logger.error(f"Ошибка загрузки {file}: {str(e)}")
#             continue
    
#     merged_df = pd.concat(dfs, ignore_index=True)
#     logger.info(f"Объединенный датасет: {len(merged_df)} строк")
#     return merged_df

# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """Удаляет пропуски и дубликаты."""
#     # Удаление столбцов с >50% пропусков
#     threshold = len(df) * 0.5
#     df_cleaned = df.dropna(thresh=threshold, axis=1)
    
#     # Заполнение оставшихся пропусков
#     numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
#     df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    
#     # Удаление дубликатов
#     df_cleaned = df_cleaned.drop_duplicates()
    
#     logger.info(f"После очистки: {len(df_cleaned)} строк")
#     return df_cleaned


# def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.3):
#     """Разделяет данные на train и test."""
#     X = df.drop(columns=[target_col])
#     y = df[target_col]
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=42, stratify=y
#     )
    
#     logger.info(f"Train: {len(X_train)} строк, Test: {len(X_test)} строк")
#     return X_train, X_test, y_train, y_test

# if __name__ == "__main__":
#     raw_data_dir = "data/raw"
#     processed_data_dir = "data/processed"
    
#     X_train, X_test, y_train, y_test = split_data(df_balanced, target_column)
#     train_df = pd.concat([X_train, y_train], axis=1)
#     test_df = pd.concat([X_test, y_test], axis=1)

#     train_df.to_csv(os.path.join(processed_data_dir, "4_train.csv"), index=False)
#     test_df.to_csv(os.path.join(processed_data_dir, "5_test.csv"), index=False)

#     # Создаем папку для обработанных данных
#     os.makedirs(processed_data_dir, exist_ok=True)
    
#     # Загрузка и объединение
#     df = load_and_merge(raw_data_dir)
    
#     # Сохранение сырых объединенных данных
#     raw_output_path = os.path.join(processed_data_dir, "1_merged_raw.csv")
#     df.to_csv(raw_output_path, index=False)
#     logger.info(f"Сохранено в {raw_output_path}")

#     df_cleaned = clean_data(df)
#     cleaned_output_path = os.path.join(processed_data_dir, "2_cleaned.csv")
#     df_cleaned.to_csv(cleaned_output_path, index=False)
    


    


    
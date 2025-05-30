import pandas as pd
import psutil

def check_memory():
    mem = psutil.virtual_memory()
    print(f"Используется {mem.percent}% RAM. Доступно: {mem.available / (1024**3):.2f} GB")


def load_data(path, sample_size=300000):
    check_memory()
    # Указываем типы столбцов вручную (пример для CIC-IDS2017)
    dtypes = {
        'Src Port': 'int32',  # Вместо int64
        'Dst Port': 'int32',
        'Protocol': 'int8',
        'Flow Duration': 'int32',
        'Label': 'category'  # Категории вместо строк
    }
    
    # Чтение файла частями (chunks)
    chunks = pd.read_csv(path, dtype=dtypes, chunksize=100000)
    df = pd.concat([chunk.sample(frac=0.3) for chunk in chunks])  # Сэмплируем 30% из каждого чанка
    
    return df.sample(n=min(sample_size, len(df)))  # Фиксируем размер
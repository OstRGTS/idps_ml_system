from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# 1. Загрузка данных
data = pd.read_csv("../data/processed_data.csv")  # Путь к предобработанным данным
X = data.drop("Label", axis=1)
y = data["Label"]

# 2. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Обучение модели
model = XGBClassifier(tree_method='hist')  # Оптимизация под память
model.fit(X_train, y_train)

# 4. Сохранение модели
os.makedirs("../models", exist_ok=True)  # Создать папку, если её нет
joblib.dump(model, "../models/xgboost_light.pkl")

print("Модель обучена и сохранена в models/xgboost_light.pkl")




























# from xgboost import XGBClassifier
# import joblib
# import numpy as np
# import os

# # Примерные данные для обучения (замените на ваши)
# X_train = np.random.rand(100, 10)
# y_train = np.random.randint(0, 2, size=100)

# # Обучение модели
# model = XGBClassifier()
# model.fit(X_train, y_train)

# # Путь сохранения модели
# model_dir = os.path.join(os.path.dirname(__file__), "models")
# os.makedirs(model_dir, exist_ok=True)
# model_path = os.path.join(model_dir, "xgboost_light.pkl")

# # Сохраняем модель
# joblib.dump(model, model_path)

# print(f"✅ Модель сохранена по пути: {model_path}")
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# загрузка данных
df = pd.read_excel("data/credit_data.xls", header=1)

# признаки и таргет
X = df.drop("default payment next month", axis=1)
y = df["default payment next month"]

# деление данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# сохраняем список колонок и модель
artifacts = {
    "model": model,
    "columns": X.columns.tolist()
}

with open("src/model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Model saved to src/model.pkl")
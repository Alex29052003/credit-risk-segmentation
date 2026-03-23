import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel("data/credit_data.xls", header=1)

conn = sqlite3.connect("database.db")
df.to_sql("clients", conn, if_exists="replace", index=False)

# Чтение данных из SQL
query = "SELECT * FROM clients"
df_sql = pd.read_sql(query, conn)

print("Default distribution:")
print(df_sql["default payment next month"].value_counts())

print("\nDefault rate:")
print(df_sql["default payment next month"].mean())

print("\nAverage age by default:")
print(df_sql.groupby("default payment next month")["AGE"].mean())

print("\nAverage credit limit by default:")
print(df_sql.groupby("default payment next month")["LIMIT_BAL"].mean())

print("\nAverage payment history (PAY_0) by default:")
print(df_sql.groupby("default payment next month")["PAY_0"].mean())

print("\nAverage first payment amount by default:")
print(df_sql.groupby("default payment next month")["PAY_AMT1"].mean())

# SQL-анализ
query_group = """
SELECT 
    "default payment next month",
    COUNT(*) AS client_count
FROM clients
GROUP BY "default payment next month"
"""
df_group = pd.read_sql(query_group, conn)

print("\nSQL grouping result:")
print(df_group)

X = df_sql.drop("default payment next month", axis=1)
y = df_sql["default payment next month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Работа с вероятностями и своим порогом
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_prob_rf > 0.3).astype(int)

print("\nRandom Forest with threshold 0.3:")
print(classification_report(y_test, y_pred_custom))

df_sql["default payment next month"].value_counts().plot(kind="bar")
plt.title("Default Distribution")
plt.xlabel("Default (0 = paid, 1 = default)")
plt.ylabel("Number of clients")
plt.tight_layout()
plt.show()

conn.close()
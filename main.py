# =========================
# PROJETO IA - PREÇO DE CASAS
# =========================

import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar dados
df = pd.read_csv("housing.csv")

print("📊 Dados carregados:")
print(df.head())

# 2. Limpeza
df = df.dropna()

# 3. Análise simples
print("\n📈 Estatísticas:")
print(df.describe())

# 4. Gráfico bonito
plt.figure(figsize=(8,5))
plt.scatter(df["median_income"], df["median_house_value"])
plt.xlabel("Renda média")
plt.ylabel("Preço da casa")
plt.title("Renda vs Preço das Casas")
plt.grid()
plt.show()

# 5. Modelo de IA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[["median_income"]]
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 6. Avaliação
score = modelo.score(X_test, y_test)
print("\n🤖 Precisão do modelo:", score)

# 7. Previsão
previsao = modelo.predict([[6]])
print("💰 Preço previsto para renda 6:", previsao)
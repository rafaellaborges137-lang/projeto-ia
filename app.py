import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic IA", layout="centered")

# Título
st.title("🚢 Dashboard IA - Titanic")

# Carregar dados
df = pd.read_csv("titanic.csv")

# Limpeza
df = df[["Survived", "Pclass", "Sex", "Age"]]
df = df.dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Sidebar
st.sidebar.header("Filtros")

classe_filtro = st.sidebar.selectbox("Classe", [1, 2, 3])
sexo_filtro = st.sidebar.selectbox("Sexo", ["male", "female"])

# Mostrar dados
st.subheader("📊 Dados filtrados")

df_filtrado = df[(df["Pclass"] == classe_filtro) & (df["Sex"] == (0 if sexo_filtro == "male" else 1))]
st.write(df_filtrado.head())

# Gráfico
st.subheader("📈 Distribuição de idade")

fig, ax = plt.subplots()
ax.hist(df["Age"], bins=20)
ax.set_xlabel("Idade")
ax.set_ylabel("Quantidade")

st.pyplot(fig)

# Modelo IA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

score = modelo.score(X_test, y_test)

st.subheader("🤖 Desempenho do modelo")
st.write(f"Acurácia: {score:.2f}")

# Previsão interativa
st.subheader("🔮 Previsão de sobrevivência")

classe = st.selectbox("Classe do passageiro", [1, 2, 3])
sexo = st.selectbox("Sexo", ["male", "female"])
idade = st.slider("Idade", 0, 80, 25)

sexo_num = 0 if sexo == "male" else 1

if st.button("Prever"):
    previsao = modelo.predict([[classe, sexo_num, idade]])

    if previsao[0] == 1:
        st.success("🟢 Sobreviveu")
    else:
        st.error("🔴 Não sobreviveu")
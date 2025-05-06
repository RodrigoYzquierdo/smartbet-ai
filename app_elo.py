
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuración
st.set_page_config(page_title="SmartBet AI ELO", layout="centered")
st.title("⚽ SmartBet AI – Predicción con Rating ELO")

@st.cache_data
def cargar_datos_y_modelo():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
    ]
    columnas = ["Date", "HomeTeam", "AwayTeam", "FTR"]
    df = pd.concat([pd.read_csv(url, usecols=columnas, encoding="ISO-8859-1") for url in urls])
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.sort_values("Date", inplace=True)

    elo_ratings = {}
    K = 30
    default_elo = 1500

    def get_elo(team):
        return elo_ratings.get(team, default_elo)

    def update_elo(winner, loser, draw=False):
        Ra = get_elo(winner)
        Rb = get_elo(loser)
        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Eb = 1 / (1 + 10 ** ((Ra - Rb) / 400))
        if draw:
            Sa, Sb = 0.5, 0.5
        else:
            Sa, Sb = 1.0, 0.0
        elo_ratings[winner] = Ra + K * (Sa - Ea)
        elo_ratings[loser] = Rb + K * (Sb - Eb)

    elos_home, elos_away = [], []
    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]

        home_elo = get_elo(home)
        away_elo = get_elo(away)
        elos_home.append(home_elo)
        elos_away.append(away_elo)

        if result == "H":
            update_elo(home, away)
        elif result == "A":
            update_elo(away, home)
        elif result == "D":
            update_elo(home, away, draw=True)

    df["elo_home"] = elos_home
    df["elo_away"] = elos_away

    X = df[["elo_home", "elo_away"]]
    y = df["FTR"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return df, model, elo_ratings

df, model, elo_ratings = cargar_datos_y_modelo()

st.subheader("🔎 Selecciona los equipos")
equipos = sorted(df["HomeTeam"].unique())
local = st.selectbox("Equipo local", equipos)
visitante = st.selectbox("Equipo visitante", equipos)

def predecir_con_elo(local, visitante):
    elo_local = elo_ratings.get(local, 1500)
    elo_visit = elo_ratings.get(visitante, 1500)
    entrada = [[elo_local, elo_visit]]
    pred = model.predict(entrada)[0]
    etiquetas = {'H': '🏠 Gana local', 'D': '🤝 Empate', 'A': '🚗 Gana visitante'}
    return etiquetas.get(pred, "Indefinido")

if st.button("Predecir resultado"):
    resultado = predecir_con_elo(local, visitante)
    st.success(f"Resultado predicho: {resultado}")

# Visualización
st.subheader("📊 Distribución de ELO Ratings")
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["elo_home"], bins=30, alpha=0.6, label="ELO Local")
ax.hist(df["elo_away"], bins=30, alpha=0.6, label="ELO Visitante")
ax.set_title("Distribución de ELO Ratings")
ax.legend()
st.pyplot(fig)

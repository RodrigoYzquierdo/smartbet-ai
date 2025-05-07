
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartBet AI", layout="wide")
st.title("⚽ SmartBet AI – Predicción y Simulación Automática")
st.markdown("---")

@st.cache_data
def cargar_modelo():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
    ]
    columnas = ["Date", "HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"]
    df = pd.concat([pd.read_csv(url, usecols=lambda c: c in columnas, encoding="ISO-8859-1") for url in urls])
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
        home, away, result = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        elos_home.append(get_elo(home))
        elos_away.append(get_elo(away))
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, elo_ratings

model, elo_ratings = cargar_modelo()

st.markdown("### 📥 Sube el archivo con los partidos de la próxima jornada")
archivo = st.file_uploader("Archivo CSV de partidos", type=["csv"])

if archivo:
    partidos = pd.read_csv(archivo)
    partidos["elo_home"] = partidos["HomeTeam"].map(lambda t: elo_ratings.get(t, 1500))
    partidos["elo_away"] = partidos["AwayTeam"].map(lambda t: elo_ratings.get(t, 1500))
    X = partidos[["elo_home", "elo_away"]]
    proba = model.predict_proba(X)
    predicciones = model.predict(X)

    tabla = []
    for i, row in partidos.iterrows():
        partido = f"{row['HomeTeam']} vs {row['AwayTeam']}"
        resultado_pred = predicciones[i]
        prob = dict(zip(model.classes_, proba[i]))
        cuota = row["B365H"] if resultado_pred == "H" else row["B365D"] if resultado_pred == "D" else row["B365A"]
        tabla.append({
            "Partido": partido,
            "Predicción": "Local" if resultado_pred == "H" else "Empate" if resultado_pred == "D" else "Visitante",
            "Probabilidad estimada": round(prob[resultado_pred], 2),
            "Cuota": cuota
        })

    df_pred = pd.DataFrame(tabla)
    st.markdown("### 🔮 Predicciones del modelo por partido")
    st.dataframe(df_pred, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💼 Simulación automática de ganancias")

    banca_inicial = st.number_input("Banca inicial ($)", min_value=10, value=1000)
    monto_apuesta = st.number_input("Monto fijo por apuesta ($)", min_value=1, value=50)

    if st.button("▶️ Simular"):
        banca = banca_inicial
        historial = [banca]
        resultados = []

        for i, row in df_pred.iterrows():
            cuota = row["Cuota"]
            prob = row["Probabilidad estimada"]
            gana = np.random.rand() < prob
            if gana:
                banca += monto_apuesta * (cuota - 1)
            else:
                banca -= monto_apuesta
            historial.append(banca)
            resultados.append({
                "Partido": row["Partido"],
                "Predicción": row["Predicción"],
                "Resultado": "✅ Ganada" if gana else "❌ Perdida",
                "Banca post-apuesta": round(banca, 2)
            })

        st.markdown("### 📊 Resultados de la simulación")
        st.dataframe(pd.DataFrame(resultados), use_container_width=True)
        st.success(f"Ganancia neta: ${round(banca - banca_inicial, 2)}")

        st.markdown("### 📈 Evolución del bankroll")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(historial, linewidth=2)
        ax.set_ylabel("Banca")
        ax.set_xlabel("N° Apuestas")
        st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartBet AI", layout="wide")
st.title("📈 SmartBet AI – Apuestas Manuales + Simulador")
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

st.markdown("### ⚽ Subir archivo de partidos futuros")
archivo = st.file_uploader("Sube tu archivo CSV con partidos", type=["csv"])

if archivo:
    partidos = pd.read_csv(archivo)
    partidos["elo_home"] = partidos["HomeTeam"].map(lambda t: elo_ratings.get(t, 1500))
    partidos["elo_away"] = partidos["AwayTeam"].map(lambda t: elo_ratings.get(t, 1500))
    X = partidos[["elo_home", "elo_away"]]
    probabilidades = model.predict_proba(X)

    st.markdown("### 🎯 Elige tu apuesta manualmente para cada partido")

    elecciones = []
    for i, row in partidos.iterrows():
        st.markdown(f"**{row['HomeTeam']} vs {row['AwayTeam']}**")
        probs = dict(zip(model.classes_, probabilidades[i]))
        cuotas = {
            "Local": row["B365H"],
            "Empate": row["B365D"],
            "Visitante": row["B365A"]
        }
        col1, col2 = st.columns(2)
        col1.write(f"Probabilidades estimadas: { {k: round(v, 2) for k, v in probs.items()} }")
        col2.write(f"Cuotas: {cuotas}")
        eleccion = st.radio(
            f"Tu apuesta para {row['HomeTeam']} vs {row['AwayTeam']}",
            options=["Local", "Empate", "Visitante"],
            key=f"apuesta_{i}"
        )
        elecciones.append({
            "Partido": f"{row['HomeTeam']} vs {row['AwayTeam']}",
            "Elección": eleccion,
            "Cuota": cuotas[eleccion],
            "Probabilidad": probs["H"] if eleccion == "Local" else probs["D"] if eleccion == "Empate" else probs["A"]
        })

    st.markdown("---")
    st.markdown("### 💼 Simulación de tus apuestas")

    banca_inicial = st.number_input("Banca inicial ($)", min_value=10, value=1000)
    tasa_acierto = st.slider("Tasa real de acierto (%)", 30, 90, 55)
    monto_apuesta = st.number_input("Monto fijo por apuesta ($)", min_value=1, value=50)

    simular = st.button("▶️ Simular")

    if simular:
        banca = banca_inicial
        historial = [banca]
        resultados = []

        for apuesta in elecciones:
            cuota = apuesta["Cuota"]
            gana = np.random.rand() < (tasa_acierto / 100)
            if gana:
                banca += monto_apuesta * (cuota - 1)
            else:
                banca -= monto_apuesta
            historial.append(banca)
            resultados.append({
                "Partido": apuesta["Partido"],
                "Tu Apuesta": apuesta["Elección"],
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

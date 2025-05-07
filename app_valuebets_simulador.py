
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# CONFIG
st.set_page_config(page_title="SmartBet AI", layout="wide")
st.title("📈 SmartBet AI – Value Bets + Simulación de Ganancias")
st.markdown("---")

# MODELO
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

# SUBIR PARTIDOS
st.markdown("### ⚽ Subir archivo de partidos futuros")
ejemplo = {
    "Date": ["2025-05-10", "2025-05-11"],
    "HomeTeam": ["Liverpool", "Man United"],
    "AwayTeam": ["Arsenal", "Chelsea"],
    "B365H": [2.1, 2.3],
    "B365D": [3.3, 3.4],
    "B365A": [3.5, 3.1]
}
st.dataframe(pd.DataFrame(ejemplo), use_container_width=True)

archivo = st.file_uploader("Sube tu archivo CSV con partidos", type=["csv"])

if archivo:
    partidos = pd.read_csv(archivo)
    partidos["elo_home"] = partidos["HomeTeam"].map(lambda t: elo_ratings.get(t, 1500))
    partidos["elo_away"] = partidos["AwayTeam"].map(lambda t: elo_ratings.get(t, 1500))

    X = partidos[["elo_home", "elo_away"]]
    probabilidades = model.predict_proba(X)

    value_bets = []
    for i, row in partidos.iterrows():
        prob = dict(zip(model.classes_, probabilidades[i]))
        ev_H = prob["H"] * row["B365H"]
        ev_D = prob["D"] * row["B365D"]
        ev_A = prob["A"] * row["B365A"]
        mejor = max([("Local", ev_H), ("Empate", ev_D), ("Visitante", ev_A)], key=lambda x: x[1])
        value_bets.append({
            "Partido": f'{row["HomeTeam"]} vs {row["AwayTeam"]}',
            "Mejor Apuesta": mejor[0],
            "Valor Esperado": round(mejor[1], 2),
            "Probabilidad estimada": round(max(prob.values()), 2),
            "Cuota": round(max(row["B365H"], row["B365D"], row["B365A"]), 2)
        })

    st.markdown("### ✅ Value Bets detectadas")
    df_bets = pd.DataFrame(value_bets)
    st.dataframe(df_bets.sort_values("Valor Esperado", ascending=False), use_container_width=True)

    st.markdown("---")
    st.markdown("### 💼 Simular ganancias con estrategia de apuestas")

    banca_inicial = st.number_input("Define tu banca inicial ($)", min_value=10, value=1000, step=10)
    monto_apuesta = st.number_input("Monto por apuesta ($)", min_value=1, value=50, step=1)
    tasa_acierto = st.slider("Tasa estimada de acierto del modelo (%)", 30, 90, 55)

    resultados = []
    banca = banca_inicial
    historial = [banca]

    for _, row in df_bets.iterrows():
        cuota = row["Cuota"]
        valor_esperado = row["Valor Esperado"]
        if valor_esperado < 1.05:
            continue
        gana = np.random.rand() < (tasa_acierto / 100)
        if gana:
            banca += monto_apuesta * (cuota - 1)
        else:
            banca -= monto_apuesta
        historial.append(banca)
        resultados.append({
            "Partido": row["Partido"],
            "Resultado": "✅ Ganada" if gana else "❌ Perdida",
            "Banca post-apuesta": round(banca, 2)
        })

    st.markdown("### 📊 Resultados simulados")
    st.dataframe(pd.DataFrame(resultados), use_container_width=True)
    st.success(f"Ganancia final: ${round(banca - banca_inicial, 2)}")

    st.markdown("### 📈 Evolución del bankroll")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(historial, linewidth=2)
    ax.set_title("Banca acumulada")
    ax.set_xlabel("Número de apuestas")
    ax.set_ylabel("Banca ($)")
    st.pyplot(fig)

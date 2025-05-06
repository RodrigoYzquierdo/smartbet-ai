
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartBet AI - Ranking ELO", layout="centered")
st.title("📊 Ranking ELO - SmartBet AI")

@st.cache_data
def calcular_elo():
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
    history = {}
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

    for _, row in df.iterrows():
        home, away, result = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        if result == "H":
            update_elo(home, away)
        elif result == "A":
            update_elo(away, home)
        elif result == "D":
            update_elo(home, away, draw=True)

        for team in [home, away]:
            history.setdefault(team, []).append(get_elo(team))

    final_elo = pd.DataFrame({
        "Equipo": list(elo_ratings.keys()),
        "ELO": [round(elo_ratings[k], 1) for k in elo_ratings]
    }).sort_values("ELO", ascending=False).reset_index(drop=True)
    return final_elo, history

ranking, historial = calcular_elo()

# Mostrar ranking
st.subheader("🏆 Tabla de clasificación por ELO")
st.dataframe(ranking.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

# Seleccionar equipo
st.subheader("📈 Evolución histórica del ELO")
equipo = st.selectbox("Selecciona un equipo", list(historial.keys()))
elo_trend = historial[equipo]
st.line_chart(elo_trend)

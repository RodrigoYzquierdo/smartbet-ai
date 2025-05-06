
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(page_title="SmartBet Value Bets", layout="centered")
st.title("💸 SmartBet AI – Detección de Value Bets")

@st.cache_data
def cargar_datos_y_modelo():
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df["proba_H"] = model.predict_proba(X)[:, list(model.classes_).index("H")]
    df["proba_D"] = model.predict_proba(X)[:, list(model.classes_).index("D")]
    df["proba_A"] = model.predict_proba(X)[:, list(model.classes_).index("A")]

    # Valor Esperado (EV) = Probabilidad estimada * cuota
    df["EV_H"] = df["proba_H"] * df["B365H"]
    df["EV_D"] = df["proba_D"] * df["B365D"]
    df["EV_A"] = df["proba_A"] * df["B365A"]

    return df

df = cargar_datos_y_modelo()

st.subheader("🎯 Value Bets detectadas")

# Mostrar partidos con EV > 1.05 en cualquier opción
value_bets = []

for _, row in df.iterrows():
    for res in ["H", "D", "A"]:
        ev = row[f"EV_{res}"]
        if ev > 1.05:
            value_bets.append({
                "Fecha": row["Date"].date(),
                "Local": row["HomeTeam"],
                "Visitante": row["AwayTeam"],
                "Resultado": {"H": "Local", "D": "Empate", "A": "Visitante"}[res],
                "Cuota": row[f"B365{res}"],
                "Probabilidad estimada": round(row[f"proba_{res}"], 2),
                "Valor Esperado (EV)": round(ev, 2)
            })

df_ev = pd.DataFrame(value_bets)
df_ev.sort_values("Valor Esperado (EV)", ascending=False, inplace=True)

st.dataframe(df_ev, use_container_width=True)

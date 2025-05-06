
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartBet AI", layout="wide")
st.title("⚽ SmartBet AI – Plataforma de Predicciones y Apuestas Inteligentes")

# ----------- CONFIGURACIÓN MODELO Y ELO -----------
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

        for team in [home, away]:
            history.setdefault(team, []).append(get_elo(team))

    df["elo_home"] = elos_home
    df["elo_away"] = elos_away

    X = df[["elo_home", "elo_away"]]
    y = df["FTR"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, elo_ratings, history

model, elo_ratings, history = cargar_datos_y_modelo()

# ----------- INTERFAZ PRINCIPAL -----------
opcion = st.sidebar.radio("Selecciona una funcionalidad", [
    "🔮 Predicción Manual",
    "📊 Ranking ELO + Evolución",
    "💸 Mejores Apuestas Próxima Jornada"
])

if opcion == "🔮 Predicción Manual":
    st.subheader("Haz una predicción entre dos equipos")

    equipos = sorted(list(elo_ratings.keys()))
    equipo_local = st.selectbox("Equipo Local", equipos, index=0)
    equipo_visitante = st.selectbox("Equipo Visitante", equipos, index=1)

    elo_local = elo_ratings.get(equipo_local, 1500)
    elo_visitante = elo_ratings.get(equipo_visitante, 1500)

    X_pred = pd.DataFrame([[elo_local, elo_visitante]], columns=["elo_home", "elo_away"])
    proba = model.predict_proba(X_pred)[0]
    pred = model.predict(X_pred)[0]

    st.success(f"Predicción del modelo: **{pred}**")
    st.write("Probabilidades:")
    st.write({c: round(p, 2) for c, p in zip(model.classes_, proba)})

elif opcion == "📊 Ranking ELO + Evolución":
    st.subheader("🏆 Tabla de clasificación por ELO")

    ranking = pd.DataFrame({
        "Equipo": list(elo_ratings.keys()),
        "ELO": [round(elo_ratings[k], 1) for k in elo_ratings]
    }).sort_values("ELO", ascending=False).reset_index(drop=True)

    st.dataframe(ranking.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

    st.subheader("📈 Evolución histórica del ELO")
    equipo = st.selectbox("Selecciona un equipo", list(history.keys()))
    plt.plot(history[equipo])
    plt.title(f"Evolución de ELO – {equipo}")
    st.pyplot(plt)

elif opcion == "💸 Mejores Apuestas Próxima Jornada":
    st.subheader("📁 Carga la plantilla con partidos y cuotas")

    ejemplo = {
        "Date": ["2025-05-10", "2025-05-11"],
        "HomeTeam": ["Liverpool", "Man United"],
        "AwayTeam": ["Arsenal", "Chelsea"],
        "B365H": [2.1, 2.3],
        "B365D": [3.3, 3.4],
        "B365A": [3.5, 3.1]
    }
    df_ejemplo = pd.DataFrame(ejemplo)
    with st.expander("Ver ejemplo de plantilla"):
        st.dataframe(df_ejemplo)

    archivo = st.file_uploader("Sube archivo CSV con partidos futuros", type=["csv"])

    if archivo:
        partidos = pd.read_csv(archivo)
        partidos["elo_home"] = partidos["HomeTeam"].map(lambda t: elo_ratings.get(t, 1500))
        partidos["elo_away"] = partidos["AwayTeam"].map(lambda t: elo_ratings.get(t, 1500))

        X = partidos[["elo_home", "elo_away"]]
        probabilidades = model.predict_proba(X)

        predicciones = []
        for i, row in partidos.iterrows():
            prob = dict(zip(model.classes_, probabilidades[i]))
            ev_H = prob["H"] * row["B365H"]
            ev_D = prob["D"] * row["B365D"]
            ev_A = prob["A"] * row["B365A"]
            mejores = {
                "Partido": f'{row["HomeTeam"]} vs {row["AwayTeam"]}',
                "Mejor Apuesta": max(
                    [("Local", ev_H), ("Empate", ev_D), ("Visitante", ev_A)],
                    key=lambda x: x[1]
                )[0],
                "Valor Esperado": round(max(ev_H, ev_D, ev_A), 2),
                "Probabilidad estimada": round(max(prob["H"], prob["D"], prob["A"]), 2),
                "Cuota": max(row["B365H"], row["B365D"], row["B365A"]),
            }
            predicciones.append(mejores)

        st.subheader("💰 Recomendaciones para apostar")
        st.dataframe(pd.DataFrame(predicciones).sort_values("Valor Esperado", ascending=False), use_container_width=True)

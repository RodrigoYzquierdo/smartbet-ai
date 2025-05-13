
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="MLB SmartBet", layout="wide")
st.title("⚾ MLB SmartBet – Predicción de Resultados y Simulación de Apuestas")
st.markdown("---")

# 1. Cargar datos ficticios (estructura simulada compatible con futura API)
@st.cache_data
def cargar_datos():
    data = {
        "Date": ["2024-07-10", "2024-07-11", "2024-07-12"],
        "HomeTeam": ["Yankees", "Dodgers", "Cubs"],
        "AwayTeam": ["Red Sox", "Giants", "Cardinals"],
        "HomeWin": [1, 0, 1],  # 1 si gana local, 0 si gana visitante
        "Odds_H": [1.8, 2.0, 1.9],
        "Odds_A": [2.1, 1.8, 2.2]
    }
    return pd.DataFrame(data)

df = cargar_datos()

# 2. Entrenar modelo simple con Random Forest
equipos = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel("K"))
equipo_id = {team: i for i, team in enumerate(equipos)}
df["HomeID"] = df["HomeTeam"].map(equipo_id)
df["AwayID"] = df["AwayTeam"].map(equipo_id)

X = df[["HomeID", "AwayID", "Odds_H", "Odds_A"]]
y = df["HomeWin"]
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# 3. Interfaz: carga de partidos futuros
st.markdown("### 📥 Subir archivo de partidos de la próxima jornada (CSV)")
ejemplo = pd.DataFrame({
    "Date": ["2024-07-13"],
    "HomeTeam": ["Yankees"],
    "AwayTeam": ["Astros"],
    "Odds_H": [1.95],
    "Odds_A": [1.90]
})
with st.expander("📌 Ejemplo de formato esperado"):
    st.dataframe(ejemplo)

archivo = st.file_uploader("Archivo CSV", type=["csv"])

if archivo:
    partidos = pd.read_csv(archivo)
    partidos["HomeID"] = partidos["HomeTeam"].map(equipo_id).fillna(-1)
    partidos["AwayID"] = partidos["AwayTeam"].map(equipo_id).fillna(-1)

    X_pred = partidos[["HomeID", "AwayID", "Odds_H", "Odds_A"]]
    prob = modelo.predict_proba(X_pred)

    resultados = []
    for i, row in partidos.iterrows():
        p_local = prob[i][1]
        pred = "Local" if p_local >= 0.5 else "Visitante"
        cuota = row["Odds_H"] if pred == "Local" else row["Odds_A"]
        value = p_local * cuota if pred == "Local" else (1 - p_local) * cuota
        resultados.append({
            "Partido": f"{row['HomeTeam']} vs {row['AwayTeam']}",
            "Predicción": pred,
            "Probabilidad": round(p_local if pred == "Local" else 1 - p_local, 2),
            "Cuota": cuota,
            "Value Bet": "✅ Sí" if value > 1 else "❌ No"
        })

    df_resultados = pd.DataFrame(resultados)
    st.markdown("### 🔮 Predicciones de la jornada")
    st.dataframe(df_resultados, use_container_width=True)

    # 4. Simulación
    st.markdown("### 💼 Simulación de Ganancias")
    banca_inicial = st.number_input("Banca inicial", min_value=10, value=1000)
    monto_apuesta = st.number_input("Monto por apuesta", min_value=1, value=50)

    if st.button("▶️ Simular apuestas"):
        banca = banca_inicial
        historial = [banca]
        sim = []

        for row in df_resultados.itertuples():
            prob = row.Probabilidad
            cuota = row.Cuota
            gana = np.random.rand() < prob
            ganancia = monto_apuesta * (cuota - 1) if gana else -monto_apuesta
            banca += ganancia
            historial.append(banca)
            sim.append({
                "Partido": row.Partido,
                "Resultado": "✅ Ganada" if gana else "❌ Perdida",
                "Ganancia": round(ganancia, 2)
            })

        st.dataframe(pd.DataFrame(sim), use_container_width=True)
        st.success(f"Ganancia total: ${round(banca - banca_inicial, 2)}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(historial, linewidth=2)
        ax.set_title("Evolución de la banca")
        ax.set_ylabel("Banca ($)")
        ax.set_xlabel("N° apuestas")
        st.pyplot(fig)

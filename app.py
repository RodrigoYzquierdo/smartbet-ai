
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Título
st.set_page_config(page_title="SmartBet AI", layout="centered")
st.title("⚽ SmartBet AI – Predicción Deportiva Inteligente")

# Cargar datos con cuotas
@st.cache_data
def cargar_datos():
    urls = [
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv"
    ]
    columnas = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
    df_all = pd.concat([pd.read_csv(url, usecols=lambda c: c in columnas, encoding="ISO-8859-1") for url in urls])
    df_all.dropna(inplace=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], dayfirst=True)
    df_all.sort_values("Date", inplace=True)
    teams = pd.unique(df_all[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    team_to_id = {team: i for i, team in enumerate(teams)}
    df_all["HomeID"] = df_all["HomeTeam"].map(team_to_id)
    df_all["AwayID"] = df_all["AwayTeam"].map(team_to_id)
    return df_all, team_to_id

df_all, team_to_id = cargar_datos()

# Forma reciente y goles
def calcular_forma_y_goles(df, equipo, fecha, local=True):
    if local:
        recientes = df[(df["HomeTeam"] == equipo) & (df["Date"] < fecha)].sort_values("Date", ascending=False).head(5)
        goles = recientes["FTHG"]
        resultado = recientes["FTR"] == "H"
    else:
        recientes = df[(df["AwayTeam"] == equipo) & (df["Date"] < fecha)].sort_values("Date", ascending=False).head(5)
        goles = recientes["FTAG"]
        resultado = recientes["FTR"] == "A"
    return resultado.sum(), goles.mean()

# Preprocesamiento
formas_local, formas_visit, goles_local, goles_visit = [], [], [], []

for _, row in df_all.iterrows():
    f_loc, g_loc = calcular_forma_y_goles(df_all, row["HomeTeam"], row["Date"], local=True)
    f_vis, g_vis = calcular_forma_y_goles(df_all, row["AwayTeam"], row["Date"], local=False)
    formas_local.append(f_loc)
    formas_visit.append(f_vis)
    goles_local.append(g_loc)
    goles_visit.append(g_vis)

df_all["forma_local"] = formas_local
df_all["forma_visit"] = formas_visit
df_all["goles_local"] = goles_local
df_all["goles_visit"] = goles_visit
df_all.dropna(inplace=True)

# Entrenar modelo
X = df_all[["HomeID", "AwayID", "forma_local", "forma_visit", "goles_local", "goles_visit"]]
y = df_all["FTR"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Interfaz
st.subheader("🎯 Haz tu predicción")
local = st.selectbox("Equipo local", sorted(team_to_id.keys()))
visitante = st.selectbox("Equipo visitante", sorted(team_to_id.keys()))

def predecir_resultado(local, visitante):
    h_id = team_to_id.get(local)
    a_id = team_to_id.get(visitante)
    ult_fecha = df_all["Date"].max()
    f_loc, g_loc = calcular_forma_y_goles(df_all, local, ult_fecha, local=True)
    f_vis, g_vis = calcular_forma_y_goles(df_all, visitante, ult_fecha, local=False)
    entrada = [[h_id, a_id, f_loc, f_vis, g_loc, g_vis]]
    pred = model.predict(entrada)[0]
    etiquetas = {'H': '🏠 Gana local', 'D': '🤝 Empate', 'A': '🚗 Gana visitante'}
    return etiquetas.get(pred)

def calcular_value_bet(local, visitante):
    partido = df_all[(df_all["HomeTeam"] == local) & (df_all["AwayTeam"] == visitante)].sort_values("Date", ascending=False).head(1)
    if partido.empty:
        return "Sin datos de cuotas"
    cuota_h, cuota_d, cuota_a = partido.iloc[0][["B365H", "B365D", "B365A"]]
    h_id = team_to_id[local]
    a_id = team_to_id[visitante]
    f_loc, g_loc = calcular_forma_y_goles(df_all, local, partido["Date"].values[0], local=True)
    f_vis, g_vis = calcular_forma_y_goles(df_all, visitante, partido["Date"].values[0], local=False)
    entrada = [[h_id, a_id, f_loc, f_vis, g_loc, g_vis]]
    proba = model.predict_proba(entrada)[0]
    ev = {
        "Local": round(proba[model.classes_ == "H"][0] * cuota_h, 2),
        "Empate": round(proba[model.classes_ == "D"][0] * cuota_d, 2),
        "Visitante": round(proba[model.classes_ == "A"][0] * cuota_a, 2)
    }
    return ev

if st.button("Predecir resultado"):
    resultado = predecir_resultado(local, visitante)
    st.success(f"Resultado predicho: {resultado}")
    st.subheader("💰 Evaluación Value Bet")
    value_bet = calcular_value_bet(local, visitante)
    st.write(value_bet)

# Visualización
st.subheader("📊 Goles recientes")
equipo_grafico = st.selectbox("Selecciona equipo para gráfico", sorted(team_to_id.keys()))
df_local = df_all[df_all["HomeTeam"] == equipo_grafico].sort_values("Date", ascending=False).head(5)
df_visit = df_all[df_all["AwayTeam"] == equipo_grafico].sort_values("Date", ascending=False).head(5)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_local["Date"], df_local["FTHG"], label="Local - Goles")
ax.plot(df_visit["Date"], df_visit["FTAG"], label="Visitante - Goles")
ax.set_title(f"Goles recientes de {equipo_grafico}")
ax.set_ylabel("Goles anotados")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Descargar datos de la Premier League
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)

# Seleccionar columnas relevantes
df = df[["HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"]].dropna()
df["Result"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})

# Codificar equipos
teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel("K"))
team_to_id = {team: i for i, team in enumerate(teams)}
df["HomeID"] = df["HomeTeam"].map(team_to_id)
df["AwayID"] = df["AwayTeam"].map(team_to_id)

# Definir variables de entrada
X = df[["HomeID", "AwayID", "B365H", "B365D", "B365A"]]
y = df["Result"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar modelos
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
}

# Entrenar y evaluar
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Modelo": name, "Precisión": round(acc, 3)})

# Mostrar resultados
results_df = pd.DataFrame(results)
print("\nComparación de modelos de predicción para fútbol:")
print(results_df.to_string(index=False))

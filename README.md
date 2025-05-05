# ⚽ SmartBet AI

**SmartBet AI** es una aplicación web construida con Python y Streamlit que predice el resultado de partidos de fútbol de la Premier League utilizando algoritmos de inteligencia artificial. El modelo considera el rendimiento reciente de los equipos, promedio de goles y analiza el valor esperado de las cuotas de apuestas (value bets).

---

## 🚀 Funcionalidades

- Predicción de resultados: 🏠 Gana local, 🤝 Empate, 🚗 Gana visitante
- Análisis de *forma reciente*: últimos 5 partidos y goles promedio
- Cálculo de *value bets* a partir de cuotas históricas de Bet365
- Visualización gráfica del rendimiento goleador reciente de cada equipo
- Interfaz web interactiva, desplegada en [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🛠 Tecnologías

- Python 3
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

---

## 🧠 Cómo funciona

El modelo utiliza un clasificador Random Forest entrenado con datos de las temporadas 2021–2024 de la Premier League. Los predictores incluyen:

- ID del equipo local y visitante
- Número de partidos ganados en los últimos 5 encuentros
- Promedio de goles anotados por cada equipo
- Cuotas históricas de apuestas (Bet365)

---

## 📦 Instalación local (opcional)

```bash
pip install -r requirements.txt
streamlit run app.py

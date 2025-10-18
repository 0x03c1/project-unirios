import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="EpimdemIA", layout="wide")


# Load data - para carregar meu dataset
def load_data():
    p = Path("data/sample_cases.csv")
    if p.exists():
        return pd.read_csv(p, parse_dates=["data"])
    return pd.read_csv("data/sample_cases.csv", parse_dates=["data"])


# Load model - para carregar meu modelo treinado
def ensure_data():
    try:
        clf = joblib.load('modelo/surtos.pkl')
    except Exception:
        clf = None
    return clf


st.title("EpimdemIA - Análises de surtos epidemilógicos")


tab1, tab2 = st.tabs(["Análise Exploratória", "Modelagem Preditiva"])


with tab1:
    st.header("Análise Exploratória de Dados")
    df = load_data().sort_values(["municipio", "data"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Casos na semana")
        fig = px.area(
            df,
            x="data",
            y="casos",
            color="municipio",
            title="Número de casos ao longo do tempo",
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )
    with c2:
        st.subheader("Casos vs Chuva(mm)")
        fig2 = px.scatter(
            df,
            x="chuva",
            y="casos",
            color="municipio",
            trendline="ols",
            title="Relação entre casos e precipitação"
        )
        st.plotly_chart(
            fig2,
            use_container_width=True
        )


with tab2:
    st.header("Modelagem Preditiva de surtos epidemilógicos")
    clf = ensure_data()
    if clf is None:
        st.error("Modelo preditivo não encontrado")
    c1, c2, c3, c4 = st.columns(4)
    casos_4s = c1.number_input("Casos nas últimas 4 semanas", 0, 10000, 40)
    chuva = c2.number_input("Chuva média (mm)", 0.0, 500.0, 20.0)
    temp = c3.number_input("Temperatura média (°C)", -5.0, 45.0, 25.0)
    umidade = c4.number_input("Umidade média (%)", 0.0, 100.0, 70.0)
    if clf:
        X = pd.DataFrame(
            [[casos_4s,
              chuva,
              temp,
              umidade]], columns=[
                  "casos_4s",
                  "chuva",
                  "temp",
                  "umidade"
            ])
        proba = clf.predict_proba(X)[0][1]
        st.metric("Probabilidade de surto2", f"{proba * 100:.2%}%")

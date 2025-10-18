import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="EpimdemIA", layout="wide")


def load_data():
    p = Path("data/sample_cases.csv")
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return pd.read_csv("data/sample_cases.csv", parse_dates=["date"])


st.title("EpimdemIA - Análises de surtos epidemilógicos")


tab1, tab2 = st.tabs(["Análise Exploratória", "Modelagem Preditiva"])

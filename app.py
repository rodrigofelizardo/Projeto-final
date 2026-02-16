import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Credit Scoring App")

st.title("💳 Sistema de Escoragem de Crédito")

# Carregar modelo
@st.cache_resource
def load_model():
    with open('model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Upload do CSV
uploaded_file = st.file_uploader("📂 Faça upload do arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dados carregados")
    st.write(df.head())

    # Fazer predição
    pred = model.predict(df)
    prob = model.predict_proba(df)[:,1]

    df['Score_Inadimplencia'] = prob
    df['Classificacao'] = pred

    st.subheader("📈 Resultado da Escoragem")
    st.write(df.head())

    st.download_button(
        label="📥 Baixar resultado",
        data=df.to_csv(index=False),
        file_name="resultado_score.csv",
        mime="text/csv"
    )

import pandas as pd
import streamlit as st

# chemin vers le dossier contenant les données CSV
datadir = "../"

# variable titre ; utile car on utilise le titre à plusieurs endroits
title="Projet Énergie"

# Dataframe chargé dès le début une seule fois si possible

@st.cache_data
def getDf():
    return pd.read_csv(datadir+"eco2mix-prepare-temperatures_indispo_prix.csv", sep=";", parse_dates=['datetime'], index_col=0)

@st.cache_data
def getDfx():
    return pd.read_csv(datadir+'eco2mix-regional-cons-def.csv', sep=";", index_col=0)

df = getDf()
dfx = getDfx()


liste_regions = (df['region'].unique())

import pandas as pd
import streamlit as st

# chemin vers le dossier contenant les données CSV
datadir = "../sources_donnees/"

# variable titre ; utile car on utilise le titre à plusieurs endroits
title="Projet Énergie"

# Dataframe chargé dès le début une seule fois si possible

@st.cache_data
def getDf():
    return pd.read_csv(datadir+"eco2mix-prepare-temperatures_indispo_prix.csv", sep=";", parse_dates=['datetime'], index_col=0)

@st.cache_data
def getDfx():
    #return pd.read_csv(datadir+'eco2mix-regional-cons-def.csv', sep=";", index_col=0)
    return pd.read_csv("https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-regional-cons-def/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B", sep=";")

df = getDf()
dfx = getDfx()


liste_regions = (df['region'].unique())

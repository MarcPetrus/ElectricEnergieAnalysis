import pandas as pd
import streamlit as st

# chemin vers le dossier contenant les données CSV
datadir = "sources_donnees/"

# variable titre ; utile car on utilise le titre à plusieurs endroits
title="Projet Énergie"

# Dataframe chargé dès le début une seule fois si possible

@st.cache_data
def getDf():
    return pd.read_csv(datadir+"eco2mix-prepare-temperatures_indispo_prix.csv", sep=";", parse_dates=['datetime'], index_col=0)

@st.cache_data
def getDfx():
    #return pd.read_csv("https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-regional-cons-def/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B", sep=";")
    df1 = pd.read_csv(datadir+'eco2mix-regional-cons-def_part1.csv')
    df2 = pd.read_csv(datadir+'eco2mix-regional-cons-def_part2.csv')
    df3 = pd.read_csv(datadir+'eco2mix-regional-cons-def_part3.csv')
    df_tot = pd.concat([df1, df2,df3], ignore_index=True).drop(columns=['Unnamed: 0'])
    return df_tot
    
df = getDf()
dfx = getDfx()

liste_regions = (df['region'].unique())

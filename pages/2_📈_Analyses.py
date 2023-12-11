import streamlit as st
import pandas as pd
import io
import numpy as np
import commun as cn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import spearmanr

df = cn.df
dfx = cn.dfx

st.header("📈 "+cn.title+ " - Analyses", divider='rainbow')

partie1 = "Exploration des données"
partie2 = "Analyses métier"
partie3 = "Corrélations entre variables"

pages=[":beginner: "+partie1, ":bar_chart: "+partie2, "🔢"+partie3]
page=st.sidebar.radio("Aller vers", pages)


if page == pages[0] : 
    st.subheader(partie1, divider='blue')    
    
    st.markdown("### Le **DataFrame brute issue de ORDE**")
    with st.expander("Forme du DataFrame"):
         st.write(" (nombre d'enregistrements, nombre de variables) = ", dfx.shape)
    
    with st.expander('Description statistiques du DataFrame'):
         st.dataframe(dfx.describe())   
    
    
    with st.expander("Analyse et traitement des valeurs manquantes") :
         st.subheader('Taux des valeurs manquantes par colonne')
         st.dataframe(round(dfx.isna().sum()/len(dfx)*100))
         st.write("""Cas des 100% de NA - Nous constatons que les colonnes :
         Stockage batterie, Déstockage batterie, Eolien terrestre,
         Eolien offshore et Column 30 
         ont 100% de valeurs manquantes. Nous décidons de supprimer ces colonnes pour le reste de notre analyse.""")
    
         st.write("""Cas des TCO et TCH (colonnes taux de couverture et de charge) - TCO et TCH présentent des taux\
         de valeurs manquantes supérieurs à 87%. 
         Cependant les valeurs des colonnes TCO et TCH sont toutes présentes pour l’année 2020.
         Nous les conservons dans un premier temps pour faire une prédiction avec ces features supplémentaires si \
         le temps nous le permet.""")
    
         st.write("""Cas de la production Nucléaire - 
         Nous avons noté que le taux de valeurs manquantes de la filière nucléaire est de 42%, cependant cette \
         filière est si importante dans le mix énergétique français que nous la gardons. Par ailleurs, nous \
         constatons que les valeurs manquantes sont présentes dans quelques régions seulement : Pays de la Loire,\
         Bourgogne-Franche-Comté, IDF, PACA, Bretagne""")
         st.write("""Cas de la filière Pompage - 
         Concernant la filière de pompage, les valeurs sont manquantes également pour quelques régions seulement ce qui peut \
         expliquer le fort taux de valeurs manquantes : Pays-de-la-Loire, IDF, Centre-Val-de-Loire, Normandie, \
         Hauts-de-France,Nouvelle-Aquitaine  """)
         st.write("Nous décidons de garder cette colonne qui représente une variable importante de la régulation \
         de l’offre et la demande en électricité.")
    
    st.divider()
    
    st.markdown("### Le **DataFrame de travail**")
    if st.checkbox("Afficher les 30 derniers enregistrements"):
        st.dataframe(df.tail(30), column_config={'_index':st.column_config.NumberColumn(format="%d"), 'annee':st.column_config.NumberColumn(format="%d")})
        
    st.markdown('''  
    Pour obtenir ce Dataframe, nous avons groupé les valeurs par jour, puis, nous y avons joint les données :  
    - températures quotidiennes régionales,  
    - les pannes,
    - les tarifs de bases de l'électricité des particuliers,  
    - la moyenne des tarifs heures creuses et heures pleines des particuliers''')                    
                
    st.divider()
    
    st.markdown("### *Informations* sur les **Colonnes**")
    with st.expander("$ \Large Afficher $") :
        buffer = io.StringIO()
        df.info(verbose=True, buf=buffer)
        st.text(buffer.getvalue())
    
    st.divider()

    st.markdown("### Statistiques des productions françaises (MWh)")
    with st.expander("$ \Large Afficher $") :    
        prod_cols = ['thermique', 'nucleaire', 'eolien', 'solaire',\
                     'hydraulique', 'pompage', 'bioenergies','ech.physiques']        
        st.dataframe(df[prod_cols].describe().round(0))
               
    st.markdown("### Statistiques de consommations françaises (MWh)")
    with st.expander("$ \Large Afficher $") :                          
        st.dataframe(df[['consommation']].describe().round(0))
             
    st.markdown('''**Remarque :** Les valeurs de consommations et de production dans la source de données d'origine s'expriment en puissance moyenne sur le pas de temps demi-heure en MW.  
                 Or nous avons simplement sommé ces valeurs pour chaque jour.  
                 Ainsi, pour obtenir les valeurs d'énergie en MWh il faut **multiplier** les valeurs de consommation et de production
                 de notre jeu de donnée par **0,5h** ''')    
       
if page == pages[1] : 
   st.subheader(partie2, divider='red')
   
   df_prod = df.groupby(by=['region']).agg({'thermique': sum,'nucleaire' : sum,'eolien' : sum,\
                                           'solaire': sum,'hydraulique' : sum,'pompage': sum,\
                                            'bioenergies' : sum,'ech.physiques':sum })
   df_prod_tot = df_prod.sum()
   df_prod_tot2 = df_prod_tot.drop("pompage",axis = 0)
   df_prod_tot3 = df_prod_tot2.drop("ech.physiques",axis = 0)
   chart_data = df_prod_tot3   
   with st.expander("Production (MW) par filière "): 
        st.bar_chart(chart_data) 
        st.write("La filière nucléaire est la source de production principale en France, suivie de l’hydraulique \
                  et du thermique.")   
     
    # Calcul de la production totale          
   dfx['Date']=pd.to_datetime(dfx['Date'])
   dfx['annee']=dfx['Date'].dt.year
   dfx['mois']= dfx['Date'].dt.month 
   prod_conso_mois = dfx.groupby("mois",'annee' == 2020).agg({'Consommation (MW)':'sum','Thermique (MW)' : "sum",\
                                                              'Nucléaire (MW)': "sum",'Eolien (MW)': "sum",'Solaire (MW)': "sum",\
                                                            'Hydraulique (MW)':"sum", 'Bioénergies (MW)':"sum" })           
   prod_tot = prod_conso_mois.drop('Consommation (MW)',axis = 1).sum(axis = 1)   
   df_prod_tot = pd.DataFrame(prod_tot) # Attention modifier nom de la colonne en vue du graphique      
   consol = prod_conso_mois.merge(df_prod_tot, on = 'mois', how = 'left')   
   consol = consol.rename({0 : 'Prod. tot.', 'Consommation (MW)' : "Conso.",
                           'Thermique (MW)': "Therm.",
                           'Nucléaire (MW)': 'Nucl.',
                          'Eolien (MW)': 'Eol.',
                          'Solaire (MW)':'Sol.',
                          'Hydraulique (MW)':'Hydrau.',
                         'Bioénergies (MW)': 'Bioén.'}, axis = 1)      
   with st.expander("Évolution mensuelle de la consommation et de la production en France (année 2020)"):        
        st.write("Nous constatons que la production et la consommation suivent une même tendance avec une production totale toujours légèrement supérieure à la consommation") 
        st.write(" L’écart entre le niveau de consommation et le niveau de production correspond au pompage \
               (stockage d'électricité) et aux échanges physiques (export de surplus de production)")
        lines_conso_fg = px.line(consol)
        lines_conso_fg.update_layout(yaxis_title='Energie élec (MW)')
        st.plotly_chart(lines_conso_fg)  
        st.markdown("*Notice : Cliquez sur les variables dans la légende pour faire disparaitre ou apparaitre la série correspondante sur le graphique.*")
           
   
   dfx_group_by = dfx.groupby(by = 'Région').agg({'Consommation (MW)': sum,
                                              'Thermique (MW)' : sum,
                                              'Nucléaire (MW)' : sum,
                                              'Eolien (MW)' : sum,
                                              'Solaire (MW)' : sum,
                                              'Hydraulique (MW)' : sum,'Bioénergies (MW)': sum})
   with st.expander("Consommation vs production par filière pour chaque région"):       
       st.bar_chart(dfx_group_by)
       st.write("Nous observons par ce graphique que bien que la filière nucléaire soit la première source \
                d’énergie en France, toutes les régions ne disposent cependant pas de centrales nucléaires, \
                    ou n’ont pas produit d’énergie nucléaire sur la période 2013-2020.")
       st.write("Les trois plus grands pourvoyeurs de cette filière énergétique sont les régions Auvergne-Rhône-\
                Alpes, Grand Est et Centre-Val de Loire")
       st.write("La région île-de-France consomme visiblement plus d’énergie que les autres régions,  même si \
                elle n’en produit quasiment pas; ")

       
   with st.expander("Implantation des énergies renouvelables"):     
       st.bar_chart(dfx_group_by[['Bioénergies (MW)', 'Solaire (MW)', 'Eolien (MW)']])
       st.write("Par ailleurs, en considérant les énergies renouvelables comme étant la bio énergie, l’énergie \
                solaire et l’éolien, on constate que l’énergie éolienne est la plus présente et essentiellement\
                    dans les régions Grand-Est et Hauts-de-France")
       st.write("L'énergie solaire arrive en deuxième position et est essentiellement produite en Nouvelle-Aquitaine, \
                en Occitanie et en Provence-Alpes-Côtes d’Azur.")

    
   df_filieres = df.iloc[:,10:19]
   df_num = df_filieres.select_dtypes('float')
   fig, ax = plt.subplots(figsize=(15,15))
   sns.heatmap(df_num.corr(), ax = ax, annot = True)    
   with st.expander("Corrélations entre la consommation et les filières de production"): 
         st.write(fig)
         st.write("Notons une forte corrélation négative entre la production nucléaire et les échanges physiques. \
                Nous pouvons en déduire que les échanges physiques viennent compenser les baisses de production \
                    nucléaire. La filière nucléaire est corrélée positivement avec les filières thermique \
                        et hydraulique ce aui est cohérent avec le poids de ces trois filières dans la production\
                            française.") 
      
if page == pages[2] : 
   st.subheader(partie3, divider='violet')
   
   st.markdown("#### Analyse des corrélations entre consommations et productions journalières par filière")
   st.markdown('''Cherchons la valeur des :green[coefficients de correlation (R)] par méthode statistique de :blue[**Spearman**].''')    
   with st.expander("$ \Large Afficher $") :        
       df_jours = df.groupby(by=[pd.Grouper(key='datetime', freq='D')]).agg({'consommation': 'sum', 'thermique': 'sum', 'nucleaire': 'sum', 'eolien': 'sum', 'solaire': 'sum', 'hydraulique': 'sum' })
       prod_cols = ['thermique', 'nucleaire', 'eolien', 'solaire','hydraulique']
       correlations = {}
       for col in prod_cols:
           r, p_value = spearmanr(df_jours.loc[:,'consommation'] , df_jours.loc[:,col] )
           correlations[col] = r
       correlations_df = pd.DataFrame.from_dict(data=correlations, orient='index', columns=['R'] ).reset_index()
       #st.dataframe(correlations_df) # affichage pour controle et debuggage
       figbar = px.bar(correlations_df, x='index', y='R', labels={'index':'filière de production', 'y':'R'},
                 color='index')
       st.plotly_chart(figbar, theme=None , use_container_width = True)
       st.markdown('''On peut dégager deux tendances : c’est avec la production nucléaire que la consommation
        journalière est la plus fortement corrélée, tandis que cette même consommation est
        faiblement corrélée négativement avec la production solaire.  
        On pourrait expliquer ce dernier résultat en se rappelant que la production solaire est la plus
        forte en milieu de journée et en été, périodes où les besoins en énergie domestiques et de
        chauffage sont les plus faibles.
        ''')         
   
   st.markdown("#### Analyse de la corrélation entre consommation mensuelle et mois de l'année")
   st.markdown('''Cherchons le :green[type de lien] entre ces deux variables par essais de régression.  
               *Il y a 12 valeurs par mois correspondant à chacune des 12 régions métropolitaines*.''')    
   with st.expander("$ \Large Afficher $") :    
       df_mois = df.groupby(by=[pd.Grouper(key='datetime', freq='M')]).agg({'num_mois': lambda x : x.unique()[0] , 'consommation': 'sum'})  
       fig = px.scatter(df_mois, x='num_mois', y='consommation', trendline="lowess")
       st.plotly_chart(fig, theme=None , use_container_width = True)
       st.markdown('''Nous trouvons qu'une régression :green[**polynomiale d’ordre 2**] modélise le mieux cette relation''')
   
   st.markdown("#### Analyse de la corrélation entre consommations et température, pour la région île de France")
   with st.expander("$ \Large Afficher $") :         
       fig = px.scatter(df.loc[df.code_region==11], x='TMin', y='consommation', trendline="ols")
       st.plotly_chart(fig, theme=None , use_container_width = True)
       st.markdown('''Nous obtenons une relation approximativement :green[**linéraire décroissante**] entre ces deux variables. 
                   Le coefficient de détermination :green[**R est de 0,8**] (R² = 0,658) sur l'ensembles des valeurs de températures.                     
                  Toutefois, l'on remarque que la relation est davantage linéaire depuis les temperatures négatives
                  jusqu'à environ 14°c. :red[Pour T°c > 14], la consommation remonte légérement et :red[n'est pas bien modélisée par cette droite]. On suppose une autre relation...''')        
       st.divider()
       st.markdown(''':rainbow[Modélisez avec d'autres régions :]''')
       region_selected = st.selectbox("", cn.liste_regions, index=0, placeholder="Selectionnez une région...")
       code_region_selected = df.loc[df['region']==region_selected, 'code_region'].unique()[0]
       user_fig = px.scatter(df.loc[df.code_region==code_region_selected], x='TMin', y='consommation', trendline="ols")
       st.plotly_chart(user_fig, theme=None , use_container_width = True)
                   
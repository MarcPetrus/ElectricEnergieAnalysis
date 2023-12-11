import streamlit as st
import pandas as pd
import numpy as np
import commun as cn

df = cn.df

st.header("🌇 "+cn.title+ " - Prévision des consommations", divider='rainbow')

partie1 = "Méthodologie"
partie2 = "Comparaison des modèles de régression"

pages=[":beginner: "+partie1, ":books: "+partie2]
page=st.sidebar.radio("Aller vers", pages)


if page == pages[0] : 
    st.subheader(partie1, divider='blue')   
    
    st.markdown("""Notre objectif est de prédire la consommation ou plutôt de prédire le risque de \
              black-out.
              Nous avons modélisé le risque de black-out de deux manières différentes : \
              
              1- Par régression : en utilisant la variable cible 'Consommation' \
              
              2- Par classification : en utilisation comme variable cible le solde Production - Consommation\
              
              Ces deux méthodes vont vous être présentées ci-après.""")
    st.subheader(' Modélisation par régression')
    
    st.write("Nous avons procédé par étapes en testant au départ toutes les variables de notre jeu de données.\
             Les scores de précision proches de 1 ont pu s’expliquer par le métier même de RTE qui vise à \
             équilibrer à chaque seconde l’électricité produite à l’électricité consommée. Les données de \
             production ne semblaient pas être pertinentes pour notre modélisation de recherche de black-out.\
             Nous avons poursuivi notre modélisation en enlevant toutes les données liées à la production, \
             puis en identifiant les variables les plus importantes.")
    st.write("Les modélisations ci-après ne concernent ques les variables suivantes :")
             
    st.write("- Pannes ou interruptions (defaut_energie_moy_jour)")
    st.write("- Mois de l'année (mois_cos)")
    st.write("- Température maximale (TMAX)")
    st.write("- Température moyenne (TMoy)")
    
    
if page == pages[1] : 
    st.subheader(partie2, divider='violet')   
    
    dfe = df
          
    dfe = dfe.drop("code_region", axis=1)         
    target = dfe["consommation"]
    feats = dfe[["defaut_energie_moy_jour","mois_cos","TMax","TMoy"]]
        
    from sklearn.model_selection import train_test_split    
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2)        
    col = ["TMax","TMoy"]    
    col_train = X_train[col]    
    col_test = X_test[col]
    
    # Remplacement des NANs des colonnes tco - tch et température par leurs médianes respectives
    from sklearn.impute import SimpleImputer    
    imputer = SimpleImputer(missing_values = np.nan, strategy = "median")    
    X_train.loc[:,col] = imputer.fit_transform(col_train)    
    X_test.loc[:,col] = imputer.transform(col_test)
    
    #Remplacement des NaN par 0 dans la colonne "defaut énergie moy jour"    
    X_train.defaut_energie_moy_jour = X_train.defaut_energie_moy_jour.fillna(0)    
    X_test.defaut_energie_moy_jour = X_test.defaut_energie_moy_jour.fillna(0)
    
    # Standardisation des variables numériques soit colonnes 3 à 30    
    col_num_train = X_train    
    col_num_test = X_test
    
    from sklearn.preprocessing import StandardScaler    
    scaler = StandardScaler()    
    col_num_train = scaler.fit_transform(col_num_train)    
    col_num_test = scaler.fit(col_num_test)    
    
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
        
    def prediction(regressor):
        if regressor == 'Random Forest':
            reg = RandomForestRegressor()
        elif regressor == 'Decision Tree':
            reg = DecisionTreeRegressor()
        elif regressor == 'GradientBoostingRegressor':
            reg = GradientBoostingRegressor()
        reg.fit(X_train, y_train)
        return reg
    
    def scores(reg, choice):
        if st.checkbox('Accuracy'):
            return reg.score(X_test, y_test)
           
    choix = ['GradientBoostingRegressor','Random Forest', 'Decision Tree']
    option = st.selectbox('Choisissez un modèle', choix)
    st.write('Le modèle choisi est :', option)
    
    reg = prediction(option)
    if st.checkbox('Affichage du score de précision'):
        st.write(reg.score(X_test, y_test))
          
    st.write("Variables par importance :")
    #reg.feature_importances_
    feat_imp = pd.DataFrame(reg.feature_importances_, index = X_train.columns, columns = ["Importance"]).sort_values(by = "Importance", ascending = False)
    st.bar_chart(feat_imp) 
        
    st.subheader("Métriques du modèle choisi :")
              
    y_pred = reg.predict(X_test)
    
    pred_train = reg.predict(X_train)
    
    mae_train = mean_absolute_error(y_train, pred_train)
    
    mse_train = mean_squared_error(y_train, pred_train)
    
    rmse_train = mean_squared_error(y_train, pred_train, squared = False)
    
    mae_test = mean_absolute_error(y_test, y_pred)
    
    mse_test = mean_squared_error(y_test, y_pred)
    
    rmse_test = mean_squared_error(y_test, y_pred, squared = False)
    
    data = {'MAE train':[mae_train],
           'MAE test' :[mae_test],
           'MSE train' :[mse_train],
           'MSE test' :[mse_test],
           'RMSE train':[rmse_train],
           'RMSE test':[rmse_test]}
    
    met = pd.DataFrame(data, index = [option])
        
    st.dataframe(met)     
          
          
    st.write("Parmi les différents modèles testés, le modèle de GradientBoostingRegressor a été le plus performant. \
             Il s'agit d'un modèle basé sur l'apprentissage séquentiel et les arbres de décision.")
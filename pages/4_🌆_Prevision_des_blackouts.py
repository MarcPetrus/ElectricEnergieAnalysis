import streamlit as st
import pandas as pd
import numpy as np
import commun as cn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

df = cn.df

st.header("🌆 "+cn.title+ " - Prévision des blackouts", divider='rainbow')

partie1 = "Préparation"
partie2 = "Comparaison de modèles de classification"
partie3 = "Optimisation du modèle *Regression Logisique*"

pages=[":beginner: "+partie1, ":books: "+partie2,  "🕹️ "+partie3]
page=st.sidebar.radio("Aller vers", pages)

@st.cache_data
def get_Df_consos_jour():
    return pd.read_csv(cn.datadir+"puissances_30min_nationale.csv", sep=";",  index_col=0)

df_consos_jour = get_Df_consos_jour()    
df_pic_consos_jour = pd.read_csv(cn.datadir+"max_puissance_jour_nationale.csv", sep=";",  index_col=0)    


if page == pages[0] : 
    st.subheader("Préparation pour comparaison des modèles de prédictions.", divider='blue')
    st.markdown('''**Variable cible à prédire :**  
                Nous avons choisi de répondre à la question du risque de blackout en prenant comme variable cible le solde énergétique.  
                Sous l’hypothèse qu‘ un blackout survient lorsque que la production énergétique disponible ne parvient pas répondre à la demande d’énergie, la consommation en l'occurrence, nous avons calculé le solde énergétique (à l'échelle mensuelle dans cette partie) par l’opération suivante:''')
    st.code('''Solde = production_brute - consommation   
            avec:  
                - production_brute : production nationale toutes filières énergétique sans échanges avec pays ni perte par pompage (barrages hydroélectriques)
                - consommation : toutes régions considérées ''')
    st.markdown('''**Interpretation :** Si solde positif, il n'y a pas blackout et le résulat prend 1 comme valeur. Si solde négatif, il y'a blackout et le résultat prend 0 comme valeur. ''')
       
    st.markdown('''**Choix des variables explicatives :**  
                Nous avons eu une première approche qui a consisté à retenir comme variables explicatives celles qui n’ont pas été considérées dans dans le calcul précédent. C'est-à-dire la  production des différentes filières énergétiques dont les valeurs ont déjà été considérées dans le calcul de la production agrégée. Les variables restantes considérées à ce niveau comme variables explicatives ont été:''')
    st.write("- le mois de l’année (à travers le calcul de son cosinus)")
    st.write("- la température moyenne")
    st.write("- le prix de base moyen de l'énergie")
    st.write("- la variable défaut d’énergie qui correspond à fréquence de pannes dans les centrales energétiques")
    st.write("- les échanges d’énergie qui correspond à l'importation ou à l’exportation d’énergie faite par la France")
    
    st.divider()
    
    st.subheader("Préparation pour les optimisations du modele de prediction *Regression Logistique*", divider='red')
    
    st.markdown('''**Sources de données successivement testées :**   
        - toutes les valeurs journalières nationale au pas de temps 30 minutes  
        - la demande de puissance maximale nationale de chaque jour ''')            
    
    if st.checkbox("Afficher les données de puissances nationales au pas de temps 30min "):
        st.dataframe(df_consos_jour, column_config={'annee':st.column_config.NumberColumn(format="%d")})
        
    if st.checkbox("Afficher les données puissance nationale maximum journalière"):
        st.dataframe(df_pic_consos_jour, column_config={'annee':st.column_config.NumberColumn(format="%d")})
    
    st.write('')# saut de ligne
    st.markdown('''**Variable cible à prédire :**  
                Soit le tableau source de données nommé *df_consos_jour* representant les données de puissances nationales (aggrégation de toutes les régions),  
                la variable cible à prédire est nommée *manquePuissance* representant un **blackout** a été définie ainsi :''')
    st.code('''df_consos_jour['deltaPuissance'] = df_consos_jour['production_brute'] - df_consos_jour['consommation']   
            df_consos_jour['manquePuissance'] = df_consos_jour['deltaPuissance'] < 0 ''' , language='python')
    
    st.markdown('''Ainsi le blackout est une valeur 0 ou 1 selon que la différence entre la puissance électrique demandée
                et la production brute sur la demi_heure est positive ou strictmement négative.  
                **:red[La production brute considérée pour les prédictions ne prend pas compte les echanges d'energies avec les pays frontaliers, il s'agit de la production brute nationale.]**''')                    
    st.markdown('''**A savoir :** Dans ce modèle, on considère qu'il y a un manque de puissance lorsque la production nationale est insuffisante par rapport à la demande nationale.''')
    
    
if page == pages[1] : 
    st.subheader(partie2, divider='blue')     
    
    df['solde_energie'] = df['production_brute'] - df['consommation']
    #fixation du jeu de données à considérer data_definitif
  
    data_definitif = df[['mois_cos', 'ech.physiques', 'solde_energie', 'defaut_energie_moy_jour', 'TMoy', 'prix_base_moyen_ttc']]
    #selection des variables cible et explicatives 

    target = data_definitif.solde_energie
   
    feats = data_definitif.drop('solde_energie', axis = 1)
    # Encodage des classes du solde
    targetx = pd.cut(x = target, bins = [-538450,0,584059], labels = ["blackout", "light"])
    # Création de deux nouvelles colonnes targ_blackout et targ_light
    dfxe = data_definitif
    dfxe = dfxe.join(pd.get_dummies(targetx, prefix="targ"))
    
    # Remplacement des valeurs booléenes par 1 ou 0
    dfxe.targ_blackout = list(map(int, dfxe.targ_blackout))
    dfxe.targ_light = list(map(int, dfxe.targ_light))
    
    # Décision de prendre la variable dfxe.targ_blackout comme variable cible et supprimer dfxe.targ_light
    target = dfxe.targ_blackout
    
    feats = dfxe.drop(['solde_energie', 'targ_light'], axis =1)
    #séparation des données entrainement et test avec 20%
   
    x_train, x_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2)
    #gestion de la valeur manquante de defaut energie

    x_train['defaut_energie_moy_jour'] = x_train['defaut_energie_moy_jour'].fillna(0)

    x_test['defaut_energie_moy_jour'] = x_test['defaut_energie_moy_jour'].fillna(0)
    #gestion des valeurs manquantes
    x_train = x_train.fillna(x_train.median())

    x_test = x_test.fillna(x_test.median())
    #standardisation des données
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)
    #Modélisation Regression Logistique 
   
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
  
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'Decision Tree':
            clf = tree.DecisionTreeClassifier()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(x_train, y_train)
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(x_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(x_test))
        
    choix = ['Random Forest', 'Decision Tree', 'Logistic Regression']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))
    
    st.subheader("Conclusion partielle sur cette première approche")
    st.write("Les résultats obtenus avec ces trois modèles laissent craindre un surapprentissage, même si l’analyse précédente des features importance ont montré que les variables explicatives utilisées semblent être les plus prédictives.")
    st.write("Au niveau des trois modèles entrainés, les matrix de confusion montrent de légères différences, cela dit avec une bonne prédiction des classes. Les scores (accuracy) ont tous pris la valeur de 1. La précision est parfaite, mais le surapprentissage est évident.")

if page == pages[2] : 
    st.subheader('🕹️ '+partie3, divider='red') 
    st.markdown('''Nous cherchons ici à optimiser le modèle ***Regression Logistique*** predisant les blackout, par :          
> **a.** le choix des données de consommation
 (cf. *puissance nationale maximum journalière* dans la partie *Préparation*)  
> **b.** la recherche des meilleures variables explicatives  
> **c.** le réequilibrage des classes de valeurs de la variable cible 'manque_puissance' (optionel)  
> **d**. la recherche des meilleurs paramètres du modèle (Optionel)''')
   
    st.divider()
    
    # initialisation des variables de travail
    major_feats_cols = ['saison', 'mois_sin', 'mois_cos', 'jour_sin', 'jour_cos','thermique', 'nucleaire', 'eolien',\
                        'solaire', 'hydraulique', 'pompage', 'bioenergies', 'ech.physiques','heure', 'TMoy']
    major_feats_cols_cible = major_feats_cols.copy()
    major_feats_cols_cible.append('manquePuissance_codee')  
    df_courant = df_consos_jour
   
    data = df_courant.drop(columns=['deltaPuissance','manquePuissance','manquePuissance_codee'], axis=1)
    target = df_courant['manquePuissance_codee']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)
    regression_logistique = LogisticRegression(max_iter = 500)
    predictions = None
    probabilitesPredites = None
       
    def updateDf(df_courant = df_consos_jour):   
        '''Met à jour les variables de travail en fonction des choix de réglages de l'utilisateur'''          
        df_courant = df_courant
        data = df_courant.drop(columns=['deltaPuissance','manquePuissance','manquePuissance_codee'], axis=1)
        target = df_courant['manquePuissance_codee']
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)
                                         
    st.markdown('''##### a. Choix des données de pics de consommations journalières seulement ''')
    st.markdown(''' **Info** : voir la partie Présentation''')
    if st.checkbox("Utiliser seulement les puissances maximum journalière"):
        updateDf( df_pic_consos_jour)
    else:
        updateDf( df_consos_jour)
           
    st.markdown('''##### b. Choix des variables explicatives ''') 
    with st.expander(''' **Étude des variables explicatives à utiliser** ''') :
        st.markdown('''Vérification des corrélations entre les features.   
                    **But :** Ne conserver qu'une famille de features libres (le moins corelées entre elles) mais les plus corrélées à la variable cible *manquePuissance_codee* ''')         
        fig_heat = plt.figure(figsize=(15, 15))
        sns.heatmap(df_courant[major_feats_cols_cible].corr(), annot=True, cmap='RdBu_r', center=0);
        st.pyplot(fig_heat)
        st.markdown('''Nous observons que les variables "echanges physiques" et "thermique" dont une moindre mesure, ont la plus forte corrélation statistique avec la variable cible "manque_puissance_codee". ''')
        st.image("Blackout_choix_features_pairplot.png")
        st.markdown('''Sur le graphique pairplot, on remarque que seule la variable 'nucleaire' présente un intervalle de valeurs pour lesquelles la variable cible est determinée : à partir de 80000, la variable cible reste à 0.  
                    Nous choisirons en priorité les features : **nucleaire et echanges physiques**  
                    Par ailleurs, on observe sur le dernier graph, un déséquilibre des classes de la variable cible.''')         
    features_selected = st.multiselect(":blue[Selectionnez les variables... ]", major_feats_cols, placeholder="Selectionnez les variables...")
   
    st.markdown('''##### c. Réequilibrage des classes (optionel)''')
    is_rebalanced_classes = False
    if st.checkbox("Réequilibrage des classes de la variable cible", disabled=len(features_selected)==0 ):
        is_rebalanced_classes = True  
        X_train_resampled, y_train_resampled = ADASYN().fit_resample(X_train[features_selected], y_train) 
        X_train = X_train_resampled
        y_train = y_train_resampled               
        st.markdown(':green[*Classes de valeurs rééquuilibrée !*]')        
    with st.expander(''' **Afficher la répartition des classes** de la variable cible 'manque_puissance' ''') :              
        fig_hist, ax = plt.subplots(figsize=(2, 2), layout='constrained')
        ax.hist(y_train)             
        plt.title('Répartition des valeurs de la variable cible', {'fontsize':'x-small'});
        plt.xlabel('valeur', {'fontsize':'x-small'})
        plt.ylabel("nombre",  {'fontsize':'x-small'});
        plt.yticks(ticks=np.arange(0, 110000, step=10000), fontsize='x-small');
        st.pyplot(fig_hist, use_container_width=False)    
        
    st.markdown('''##### d. Utilisation des meilleurs paramètres de regression logistique (optionel) ''') 
    is_best_params_search = False
    if st.checkbox("Rechercher puis utiliser les meilleurs paramètres de regression logistique", disabled=len(features_selected)==0):
        is_best_params_search = True      
        param_grid = [{'C': [0.25, 1, 10], 'solver' : ['newton-cg'], 'penalty' : ['l2', None]},\
            {'C': [0.25, 1, 10 ], 'solver' : ['lbfgs'], 'penalty' : ['l2', None]},\
            {'C': [0.25, 1, 10], 'solver' : ['saga'],  'penalty' : ['elasticnet', 'l1', 'l2', None]}]
        st.write("Paramètres proposés : " )
        st.write(param_grid)
        st.markdown("Documentation : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression")                                    
             
    st.divider()                   
    
    col1, col2 = st.columns([0.7, 0.3])
                
    with col1:
        if st.button(" ▶️ Tester le modèle", type="primary", disabled=(len(features_selected)==0 )) :
            # Cas utilisation du GridSearchCV par l'utilisateur
            if is_best_params_search:
                st.markdown('⏳  ...plusieurs minutes peuvent être necessaires')
                grid_lr = GridSearchCV(estimator=regression_logistique, param_grid=param_grid)
                grid_lr.fit(X_train[features_selected], y_train)
                predictions = grid_lr.predict(X_test[features_selected])
                
                # Affichage des résultats
                             
                st.markdown('##### :green[**Résultats obtenus avec les meilleurs paramètres :**]')
                st.write("Meilleurs paramètres trouvés:" )
                st.write( grid_lr.best_params_)
                st.write("Score associé: (accuracy = nb bonnes prédictions/nb prédictions)", grid_lr.best_score_)  
                st.markdown('Matrice de confusion (vrai positifs, faux positifs,...)')
                st.dataframe(pd.crosstab(y_test, predictions), column_config={'0':st.column_config.NumberColumn(format="%d"), '1':st.column_config.NumberColumn(format="%d")})
                st.divider()
                st.markdown('Rapport des metrics pour chaque classe prédites')
                st.text(classification_report(y_test,predictions))
            # Autres cas
            else:
                regression_logistique.fit(X_train[features_selected], y_train)
                predictions = regression_logistique.predict(X_test[features_selected])
                probabilitesPredites = regression_logistique.predict_proba(X_test[features_selected])
                
                # Affichage des résultats                
                st.markdown('##### :green[**Résultats du modèle de prédiction :**]') 
                st.markdown('Matrice de confusion (vrai positifs, faux positifs,...)')
                st.dataframe(pd.crosstab(y_test, predictions), column_config={'0':st.column_config.NumberColumn(format="%d"), '1':st.column_config.NumberColumn(format="%d")})
                st.divider()
                st.markdown('Rapport des metrics pour chaque classe prédites')
                st.text(classification_report(y_test,predictions))
            
    with col2:    
        if st.button("⏭️ Relancer la page"):
            st.rerun()
            
        if st.button("⏹️ Arreter l'execution"):
            st.stop()    
                    
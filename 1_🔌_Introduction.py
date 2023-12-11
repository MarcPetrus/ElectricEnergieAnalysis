import streamlit as st
import commun as cn

st.header("🔌 " + cn.title + " - Introduction", divider='rainbow')

partie1 = "Contexte"
partie2 = "Objectifs"
partie3 = "Sources de données"

pages=[":compass: "+partie1, "🎯 "+partie2, "🔎"+partie3]
page=st.sidebar.radio("Aller vers", pages)


if page == pages[0] : 
    st.subheader(partie1, divider='blue')

    st.markdown('''Les possibilités de stockage de l’électricité sont limitées, **ce qui est produit doit être consommé instantanément.**  
    Cela suppose de :  
        - surveiller en permanence le réseau,   
        - maîtriser les flux entre les régions et avec nos voisins européens,  
        - anticiper les évolutions de la consommation électrique à court, moyen et long terme.''')
        
    st.image("Reseau_elec.png")
    
    st.subheader("L'Énergie en France")
    st.write("""La gestion du réseau de transport d’électricité \
      est complexe, avec d’un côté, l’arrivée d’énergies renouvelables intermittentes \
      et une production d’électricité de plus en plus décentralisée ; de l’autre les modes \
      de consommation se transforment : voitures électriques, autoconsommation, etc.’\
      En France, si un déséquilibre temporaire apparaît entre l’offre et la demande d’électricité, le Centre \
      national d’exploitation du système (CNES) peut faire appel aux liaisons transfrontalières pour le corriger.\
      Ces échanges de rééquilibrage sont prioritaires par rapport aux transactions commerciales, ce qui permet \
      d’éviter tout risque de coupure de l’alimentation électrique de la France, même dans les cas extrêmes.
      La société RTE dispose de plusieurs outils éprouvés pour surveiller le réseau de transport d’électricité \
      et assurer en temps réel l’équilibre électrique.""")
    st.write('Source : https://www.rte-france.com/chaque-seconde-courant-passe/equilibrer-loffre-et-la-demande-delectricite')

    st.subheader("Carte des installations nucléaires en France métropolitaine")
    st.image("sites nucléaire.png", caption = 'Carte des sites nucléaires en France')
    st.write('Source ASN')

    st.subheader("Production régional des ENR et part dans la consommation française en 2019")
    st.image('production-regionale-electricite-CGDD.svg', caption= 'Production régionale ENR et part dans la consommation 2019')
    st.write("Source : https://www.statistiques.developpement-durable.gouv.fr/edition-numerique/chiffres-cles-energies-renouvelables-2021/1-les-energies-renouvelables-en-france")
      
    st.subheader("Production brute d'ENR par filière en 2020")
    st.image('production-brute-electricite-renou-CGDD.svg', caption= 'Production brute d’électricité renouvelable par filière en 2020')
    st.write("Source : https://www.statistiques.developpement-durable.gouv.fr/edition-numerique/chiffres-cles-energies-renouvelables-2021/1-les-energies-renouvelables-en-france")

if page == pages[1] : 
    st.subheader(partie2, divider='blue')
    
    st.markdown('''
             - Constater le phasage entre la consommation et la production énergétique au niveau
    national et au niveau régional    
    - Analyse par filière de production : énergie nucléaire / renouvelable ;     
    - Focus sur les énergies renouvelables (où sont- elles implantées ?  
      Quelle part représentent-elles par rapport à la production nationale totale ?  
      
        
    - Prédictions à l’échelle nationale des consommations et du risque de black-out.''')
    
    st.image("Equilibre_conso_prod.png")
    
if page == pages[2] : 
    st.subheader(partie3, divider='blue')
    st.markdown('''
    **L’ODRE (Open Data Réseaux Énergies)**  
    Open Data Énergie est une plateforme mutualisant les données transmises  
    par de nombreux fournisseurs d’énergie de gaz et d'électricité mais aussi par les services de
    météorologie tel Weather News, RTE, GRTgaz.  
    
    Ce jeu de données, rafraîchi une fois par jour, présente les données régionales consolidées depuis janvier 2021 et définitives (de janvier 2013 à décembre 2020) issues de l'application éCO2mix.  
    Elles sont élaborées à partir des comptages et complétées par des forfaits.  
    Les données sont dites consolidées lorsqu'elles ont été vérifiées et complétées. Elles deviennent définitives lorsque tous les partenaires ont transmis et vérifié l'ensemble des comptages.

    On y trouve au pas demi-heure:
    
    - La consommation réalisée.
    - La production selon les différentes filières composant le mix énergétique.
    - La consommation des pompes dans les Stations de Transfert d'Energie par Pompage (STEP).
    - Le solde des échanges avec les régions limitrophes.
    
     Définitions de TCO et TCH :
    
    TCO : le Taux de COuverture (TCO) d'une filière de production au sein d'une région représente la part de cette filière dans la consommation de cette région
    TCH : le Taux de CHarge (TCH) ou facteur de charge (FC) d'une filière représente son volume de production par rapport à la capacité de production installée et en service de cette filière
    ''')

    st.markdown('''https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature''')
    
    
    st.markdown("Les températures journalières régionales ont été jointes aux données. La source est la suivante: ")
    st.markdown("https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/?disjunctive.libelle_region&disjunctive.nature")
    
    st.markdown("La moyenne des prix tarifs de base et heures pleines - heures creuses a été rajoutée aux données. La source est la suivante :")
    st.markdown("https://www.data.gouv.fr/fr/datasets/historique-des-tarifs-reglementes-de-vente-delectricite-pour-les-consommateurs-residentiels/#/community-reuses")
    
    st.markdown("Les indisponibilités des moyens de production ont été rajoutées à notre jeu de données. Les sources sont les suivantes:")
    st.markdown("https://www.data.gouv.fr/fr/datasets/indisponibilites-des-moyens-de-production-de-edf-sa/")
    st.markdown("https://www.data.gouv.fr/fr/datasets/historique-des-indisponibilites-des-moyens-de-production-de-edf-sa-depuis-2015/")

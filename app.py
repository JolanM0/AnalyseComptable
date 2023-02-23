# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


### IMPORT ###
 
# Librarie
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# import os
from datetime import datetime
from time import time as t
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# Module
# import memoire as m

### FONCTION ###

def calcul_avec_TVA(df, classe, classeTVA, colAvecTVA, colBase, debit):
    for index, row in df.iterrows():
        if row.Compte.startswith(classe):
            TVA = False
            li_index_mouvement = df.index[df["Mouvement"] == row.Mouvement].tolist()
            for i in li_index_mouvement:
                if df.loc[i, "Compte"].startswith(classeTVA):
                    TVA = df.loc[i, colBase]
            if TVA:
                df.loc[row.name, colAvecTVA] = round(df.loc[row.name, colAvecTVA] + TVA, 2)
    return df

def complete_movement_dataframe(df, uploaded_file, format_file, df_plan, TABLE_CORRESPONDANCE, primary_keys_names=[]):
    # Format du fichier défini dans un CSV, les données de la compta arrive sous forme d'un champs texte par mouvement
    fmt = pd.read_csv(format_file)
    # on récupère ce champs texte et on formate la Serie en un dataframe 
    df_uploaded_file = pd.read_fwf(uploaded_file, header=None,
                 names=fmt['col'].tolist(),
                 widths=fmt['length'].tolist())
    #formatage de colonne
    df_uploaded_file['Date'] = pd.to_datetime(df_uploaded_file['Date'], format='%Y%m%d', errors='coerce')
    #Suppression du caractère de sep (to_csv(sep=";"))
    df_uploaded_file = df_uploaded_file.replace('\;', ',', regex=True)
    # Ajout de colonne :
    df_uploaded_file['AouK'] = ["A" if compte[-1]=="L" else "K" for compte in df_uploaded_file['Compte']]
    df_uploaded_file['NomCompte'] = df_uploaded_file['Compte'].map(df_plan.set_index("Compte")["Libelle"])
    df_uploaded_file['mois'] = df_uploaded_file['Date'].dt.month
    df_uploaded_file['annee'] = df_uploaded_file['Date'].dt.year
    df_uploaded_file['quartAnnee'] = [int((mois-1)/3+1) for mois in df_uploaded_file['mois']]
    saison = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1 }
    df_uploaded_file['saison'] = df_uploaded_file['mois'].map(saison)
    
    df_uploaded_file = df_uploaded_file.assign(compteFus = df_uploaded_file["NomCompte"])
    df_uploaded_file['compteFus'] = df_uploaded_file['Compte'].map(TABLE_CORRESPONDANCE.set_index("Compte")["compteFus"], 
                                           na_action='ignore')
    # Gestion des immobilisations : compte de classe 2 :
    #Report pour les Report à Nouveau, Immobilisation pour les achats à comptabiliser dans le bilan de dépense
    for an in df_uploaded_file['annee'].unique():
        df_uploaded_file.loc[(df_uploaded_file['Compte'].str.startswith('2')) &
                     (df_uploaded_file['annee']==an) &
                     (df_uploaded_file['Date'] != datetime(an,1,1)), 'compteFus'] = "IMMOBILISATIONS"
        df_uploaded_file.loc[(df_uploaded_file['Compte'].str.startswith('2')) &
                     (df_uploaded_file['annee']==an) &
                     (df_uploaded_file['Date'] == datetime(an,1,1)), 'compteFus'] = "REPORT"
    
    df_uploaded_file["creditAvecTVA"] = df_uploaded_file["Credit"]
    df_uploaded_file["debitAvecTVA"] = df_uploaded_file["Debit"]
    
    df_uploaded_file = calcul_avec_TVA(df_uploaded_file, classe='6', classeTVA='44566', 
              colAvecTVA="debitAvecTVA", colBase="Debit", debit=True)
    df_uploaded_file = calcul_avec_TVA(df_uploaded_file , classe='2', classeTVA='4456', 
              colAvecTVA="debitAvecTVA", colBase="Debit", debit=True)
    
    df_uploaded_file = calcul_avec_TVA(df_uploaded_file, classe='7', classeTVA='4457', 
              colAvecTVA="creditAvecTVA", colBase="Credit", debit=False)
    
    df_uploaded_file = df_uploaded_file.reindex(columns=['Mouvement', 'Compte', 'NomCompte', 'compteFus', 'Libelle', 'Date',  'Debit', 'debitAvecTVA',
                         'Credit', 'creditAvecTVA', 'N° Piece',
                         'Paiement', 'AouK', 'mois', 'annee', 'quartAnnee', 'saison', 'Journal'])
    
    
    col_to_formate_text = ['Libelle', 'compteFus', 'NomCompte', 'Compte']
    for col in col_to_formate_text:
        df_uploaded_file[col] = df_uploaded_file[col].str.upper() 
              
    # on ajoute les datas à la base à compléter
    df = pd.concat([df_uploaded_file, df])
    #Bug loyer dans la compta 2022:
    df = df.drop(df.loc[df["Mouvement"]==26965].index)
    # on supprime les doublons selon la clé primaire défini par les colonnes # on garde les dernières valeurs, qui sont surement plus à jour
    df.drop_duplicates(subset=primary_keys_names, inplace=True, keep="first")
    df.sort_values(by=["Date"], inplace=True, ascending=False)
    
    return df


def update_plan_from_file(df, uploaded_file, format_file, file_to_update, primary_keys_names):
    """
    Prend en entrée le fichier uploader de la commpta et complète les données existantes du fichier correspondant
    """
    # Format du fichier défini dans un CSV, les données de la compta arrive sous forme d'un champs texte par mouvement
    fmt = pd.read_csv(format_file)
    # on récupère ce champs texte et on formate la Serie en un dataframe 
    df_uploaded_file = pd.read_fwf(uploaded_file, header=None, encoding='latin-1',
                 names=fmt['col'].tolist(),
                 widths=fmt['length'].tolist())
    # on ajoute les datas à la base à compléter
    df = pd.concat([df_uploaded_file, df])
    # on supprime les doublons selon la clé primaire défini par les colonnes
    df.drop_duplicates(subset=primary_keys_names, inplace=True)
    df["Libelle"] = df["Libelle"].replace('\"', '', regex=True)
    return df


def clear_filter_form(li_years):
    st.session_state["f_libelle"] = ""
    st.session_state["f_compte_name"] = ""
    st.session_state["f_comptefus"] = ""
    st.session_state["f_classe"] = "Toutes"
    st.session_state["f_amount"] = 0
    st.session_state["f_movement"] = 0
    st.session_state["f_date"] = False
    st.session_state.f_year = li_years
    st.session_state.f_justif = False
    st.session_state.f_radio_amount = "Egal"

def reset_filter_plan():
    st.session_state["f_compte_plan"] = "Toutes"
    st.session_state["f_word_plan"] = ""


def create_resume_dataframe(df, classe, display_negative_delta=False, compare_from_today=True):
    """
    Parameters
    ----------
    df : dataframe
        dataframe des entrées comptables à synthétiser
    classe : int
        La classe représente la classe comptable : 6 pour les dépenses, 7 pour les recettes..
    display_negative_delta : Boolean
        indique si on considère les delta negatif ou non (simplicité de lecture du tableau) default : False
    compare_from_today : Boolean
        indique si on compare au données des années entières ou jusqu'à la date du jour. defult : True
    Returns
    -------
    df_synthese_debit : dataframe
        dataframe des entrées comptables synthétisé

    """
    
    if compare_from_today:
        df = df.loc[df["Date"].dt.dayofyear <= datetime.today().timetuple().tm_yday]
    # si la classe est 7=Recette, les recettes sont des credits (a sommer), les debits sont à soustraire
    if classe == 7:
        df_groupby_montant = df.loc[((df['Compte'].str.startswith('7')) |
                                   (df['Compte'].str.startswith('451010J')) |
                                   (df['Compte'].str.startswith('445830L'))|
                                   (df['Compte'].str.startswith('470000L'))) &
                                        (df['annee'].isin(year_to_display)),
                       ["compteFus", "debitAvecTVA", "creditAvecTVA", "annee"]].groupby(by=["compteFus", "annee"]).sum()
        # on permet au année de devenir des valeurs pour le pivot en colonne
        df_groupby_montant = df_groupby_montant.reset_index()
        # Pivot sur le dataframe des montant pour sortir les debit et les credit 
        df_synthese_sommer = df_groupby_montant.pivot(index="compteFus", columns='annee', values="creditAvecTVA")
        df_synthese_soustraire = df_groupby_montant.pivot(index="compteFus", columns='annee', values="debitAvecTVA")
    # si la classe est 6=Dépense, les dépenses sont des debits (a sommer), les crédits sont à soustraire
    elif classe ==  6:
        df_groupby_depense = df.loc[((df["compteFus"].str.startswith('IMMOBILISATIONS')) | (df["Compte"].str.startswith("6"))) &
                                        (df['annee'].isin(year_to_display)),
                       ["compteFus", "debitAvecTVA", "creditAvecTVA", "annee"]].groupby(by=["compteFus", "annee"]).sum().round(0)
        # on permet au année de devenir des valeurs pour le pivot en colonne
        df_groupby_depense = df_groupby_depense.reset_index()
        
        
        df_synthese_sommer = df_groupby_depense.pivot(index="compteFus", columns='annee', values="debitAvecTVA")
        df_synthese_soustraire = df_groupby_depense.pivot(index="compteFus", columns='annee', values="creditAvecTVA")
    #  on tri avant d'ajouter le total pour avoir le total en dernière ligne
    df_synthese_sommer.sort_values(by=[current_year], inplace=True, ascending=False)
    df_synthese_sommer.fillna(0, inplace=True)
    df_synthese_soustraire.fillna(0, inplace=True)
    # équilibrage des dépense avec les remboursement compté dans les crédits
    for col in df_synthese_sommer.columns:
        df_synthese_sommer[col] = df_synthese_sommer[col] - df_synthese_soustraire[col] 
    #Calcul des deltas entre la dernière année et les précédentes
    for y in year_to_display:
        if y != current_year:
            df_synthese_sommer[f"{current_year} - {y}"] = df_synthese_sommer[current_year]-df_synthese_sommer[y]
            if display_negative_delta == False:
                df_synthese_sommer[f"{current_year} - {y}"] = df_synthese_sommer[f"{current_year} - {y}"].apply(lambda x : x if x>0 else 0)
            
    # Calcul des totaux de chaque colonne
    index_total = len(df_synthese_sommer)
    for col in year_to_display:
        if col != current_year:
            df_synthese_sommer.loc[index_total, f"{current_year} - {col}"] = df_synthese_sommer[current_year].sum() - df_synthese_sommer[col].sum() 
            df_synthese_sommer.loc[index_total, col] = df_synthese_sommer[col].sum()
    df_synthese_sommer.loc[index_total, current_year] = df_synthese_sommer[current_year].sum()
    # Mise en forme : float -> int, reset index, et nommage Categorie, ajout de la mention ligne TOTAL
    for col in df_synthese_sommer.columns:
        df_synthese_sommer[col] = df_synthese_sommer[col].astype("int64")
    df_synthese_sommer = df_synthese_sommer.reset_index()
    df_synthese_sommer.rename(columns={'compteFus': 'Categories'}, inplace=True)
    df_synthese_sommer.loc[index_total, "Categories"] = 'TOTAL'
    
    df_synthese_sommer.replace(0, "", inplace=True)
    return df_synthese_sommer


@st.cache_resource
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep=";", index=False).encode('utf-8')

@st.cache_data  # 👈 Add the caching decorator
def load_data(csv):
    df = pd.read_csv(csv, parse_dates=['Date'], sep=";")
    df.sort_values(by=['Date'], inplace=True, ascending=False)
    return df


################################################################## VARIABLE ### 
movement_csv_file = "MvtCompta.csv"
format_movement_csv_file = "movement_format.csv"
plan_csv_file = "PlanComptable.csv"
format_plan_csv_file = "plan_format.csv"
table_compte_fus = 'TABLE_CORRESPONDANCE.csv'

movement_primary_key = ["Mouvement", "Compte"]
plan_primary_key = ["Compte", "Libelle"]




# movement_columns = ["Mouvement", "Compte", "Libelle", "Date", "N° Piece", \
#                     "Debit", "Credit", "Paiement", "Echeance", \
#                     "Pointage", "Autre1", 'Journal', 'Autre']





########################################################## CORPS DE LA PAGE ###
st.set_page_config(layout="wide")
# CSS to inject contained in a string

hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

st.title('Synthèse de la comptabilité 	:sunrise:')





df_mvt = load_data(movement_csv_file)#pd.read_csv(movement_csv_file, parse_dates=['Date'], sep=";")
df_plan = pd.read_csv(plan_csv_file, sep=";")
df_plan.sort_values(by=["Compte"], inplace=True, ascending=True)
df_table_compte_fus = pd.read_csv(table_compte_fus, sep=";")
df_table_compte_fus.sort_values(by=["Compte"], inplace=True, ascending=True)

# st.write(st.session_state) #debug help
if "file_movement_up" in st.session_state:
    if st.session_state.file_movement_up:
        df_mvt = complete_movement_dataframe(df_mvt, st.session_state.file_movement_up, format_movement_csv_file, df_plan, df_table_compte_fus, movement_primary_key)

if "file_plan_up" in st.session_state:
    if st.session_state.file_plan_up:
        df_plan = update_plan_from_file(df_plan, st.session_state.file_plan_up, format_plan_csv_file, plan_csv_file, plan_primary_key)



total_entree = df_mvt.loc[df_mvt["Compte"]=="512115J"].sum()["Debit"]
total_sortie = df_mvt.loc[df_mvt["Compte"]=="512115J"].sum()["Credit"]
total_epargne_entree = df_mvt.loc[df_mvt["Compte"]=="512116J"].sum()["Debit"]
total_epargne_sortie = df_mvt.loc[df_mvt["Compte"]=="512116J"].sum()["Credit"]


entree, sortie, solde, epargne = st.columns(4)
with entree:
    st.metric("Entrée :small_red_triangle: :", int(total_entree))
with sortie:
    st.metric("Sortie :small_red_triangle_down: :", int(total_sortie))
with solde:
    st.metric("Solde :moneybag: :", int(total_entree - total_sortie))
with epargne:
    st.metric("Epargne :large_orange_diamond: : ", int(total_epargne_entree - total_epargne_sortie))

tab_data, tab_tableau = st.tabs(["Données", "Tableau"])


years = np.sort(df_mvt['annee'].unique())
with tab_data:
    st.subheader("Liste des entrées comptables")


    
    with st.expander(label="FILTRE des mouvements comptables :", expanded=True):

        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            libelle_filter = st.text_input("Filtrer les libéllés :",
                                           key="f_libelle",
                                           help="Colonne Libelle",) 
            movement_filter = st.number_input("Filter sur les mouvements :", 
                                              min_value=0,
                                              max_value=max(df_mvt["Mouvement"]),
                                              help="Colonne Mouvement, Mettre 0 pour réinitialiser le filtre",
                                              key="f_movement")
            
        with col2:
            compte_name_filter = st.text_input("Filter sur les noms de compte :",
                                               help="Colonne NomCompte ou Compte",
                                               key="f_compte_name")
            comptefus_filter = st.text_input("Filter sur les noms de compte fusionnés :",
                                             help="Colonne 'compteFus'",
                                             key="f_comptefus")
            classe_filter = st.radio(label="Définir la classe à afficher (defaut 6) ", 
                                        help="Colonne Compte",
                                        options=["1", "2", "3", "4", "5", "6", "7", "Toutes"], 
                                        index=7,
                                        horizontal=True,
                                        key='f_classe')
            
        with col3:
            amount_filter = st.number_input("Chercher un montant :",
                                            help="Colonne Credit, CreditAvecTVA, Debit, debitAvecTVA, Mettre 0 pour réinitialiser le filtre",                                            
                                            key="f_amount")
            radio_amount_filter = st.radio(" ", options=["Supérieur", "Inférieur", "Egal"],
                                           index=2,
                                           horizontal=True,
                                           key="f_radio_amount",
                                           label_visibility="collapsed")
            date_filter = st.date_input("Filtrer sur une date : ",
                                        help="Le filtre de date doit être appliqué avec la chexkbox ci-dessous")
            checkbox_date_filter = st.checkbox("Appliquer le filtre de date ?",
                                               help="Oui c'est bien cette case qu'il faut cocher pour activer le filtre de date",
                                               key="f_date")
        with col4:
            year_filter = st.multiselect(options=years, label="Année", default=years, key="f_year")
            justif_manquant = st.checkbox("Afficher seulement les justificatifs manquants", 
                                          key="f_justif")
            st.button(label="Effacer les filtres", on_click=clear_filter_form, args=(list(years), ))

    df_filtred_query = f"(annee in {year_filter}) "
    if libelle_filter:
        df_filtred_query += f"& (Libelle.str.contains('{libelle_filter.upper()}')) "
    if movement_filter !=0:
        df_filtred_query += f"& (Mouvement == {movement_filter}) "
    if comptefus_filter:
        df_filtred_query += f"& (compteFus.str.contains('{comptefus_filter.upper()}')) "
    if compte_name_filter:
        df_filtred_query += f"& ((NomCompte.str.contains('{compte_name_filter.upper()}')) | \
            (Compte.str.contains('{compte_name_filter.upper()}')))"
    if classe_filter != "Toutes":
        df_filtred_query += f"& (Compte.str.startswith('{classe_filter}')) "
    if amount_filter != 0.0:
        if radio_amount_filter == "Supérieur":
            df_filtred_query += f"& ((Credit >= {amount_filter}) | \
            (creditAvecTVA >= {amount_filter}) | \
            (Debit >= {amount_filter}) | \
            (debitAvecTVA >= {amount_filter}) ) "
        elif radio_amount_filter == "Inférieur":
            
            df_filtred_query += f"& (((Credit <= {amount_filter}) & (Credit != 0)) | \
            ((creditAvecTVA <= {amount_filter}) & (creditAvecTVA != 0)) | \
            ((Debit <= {amount_filter}) & (Debit != 0)) | \
            ((debitAvecTVA <= {amount_filter}) & (debitAvecTVA != 0)) ) "
    
        else:
            df_filtred_query += f"& ((Credit == {amount_filter}) | \
            (creditAvecTVA == {amount_filter}) | \
            (Debit == {amount_filter}) | \
            (debitAvecTVA == {amount_filter}) ) "
    if checkbox_date_filter:
        df_filtred_query += f"& (Date == '{date_filter}') "
    if justif_manquant:
        df_filtred_query += "& (Libelle.str.contains('\*'))"
        
    
    
    


    list_col_sum_value = [ "Debit", "debitAvecTVA", "Credit", "creditAvecTVA"]
    
    if df_filtred_query:
        if df_filtred_query[0] == "&":
            df_filtred_query = df_filtred_query[1:]
        df_filtred = df_mvt.query(df_filtred_query)
    else: # n'arrive jamais à priori grâce au  filtre d'année
        df_filtred = df_mvt
    st.markdown("###### Tableau des mouvements avec les filtres")
    st.dataframe(df_filtred)
    
    
    
    df_sum_filtred = df_filtred.sum(axis=0)[list_col_sum_value]
    df_total = pd.DataFrame(columns=df_sum_filtred.index)  
    df_total.loc[len(df_total)] = df_sum_filtred
    st.markdown("###### Total du tableau filtré : ")
    st.table(df_total)
    
    # Extract en CSV des données filtrées
    csv_mouvement = convert_df(df_filtred)
    st.download_button(
        label="Télécharger la synthèse des mouvements au format CSV",
        data=csv_mouvement,
        file_name=f"Extract Compta {datetime.today().strftime('%d-%m-%Y')}.csv",
        mime='text/csv')   

    uploaded_file_movement = st.file_uploader("Importer des nouvelles donneés issues de la comptabilité (Mouvement)", type='txt', key="file_movement_up")
    if uploaded_file_movement is not None:
        st.write("df_mvt modifié")
        # Appel de la fonction qui met à jour le csv des mouvements avec les nouveaux mouvements. 
        df_mvt = complete_movement_dataframe(df_mvt, uploaded_file_movement, format_movement_csv_file, df_plan, df_table_compte_fus, movement_primary_key)  
        save_movement_file = st.button("Enregistrer les données ajoutées au fichiers des mouvements comptables ?")
        # on enregistre le résultat dans notre BDD (ici un csv, ce qui laisse l'usage à l'utilisateur en dehors du logiciel possible)
        if save_movement_file:
            df_mvt.to_csv(movement_csv_file, index=False, sep=";")

 
    
    st.subheader("Liste du plan comptables")
       
    
    col_plan1, col_plan2 = st.columns(2)
    with col_plan1:
        plan_classe_filter = st.radio(label="Définir la classe à afficher (defaut Toutes)",
                               options=["1", "2", "3", "4", "5", "6", "7", "Toutes"], 
                               index=7,
                               horizontal=True, 
                               key="f_compte_plan")
    with col_plan2:
        plan_filter = st.text_input(label='Recherche dans le plan comptable :',
                                    key="f_word_plan")
    with col_plan1:
           
        df_plan_query = ""
        if plan_filter:
            df_plan_query = f"( (Compte.str.contains('{plan_filter.upper()}')) | (Libelle.str.contains('{plan_filter.upper()}')) ) "
        if plan_classe_filter != "Toutes":
            df_plan_query += f"& (Compte.str.startswith('{plan_classe_filter}'))"    
        if df_plan_query:
            if df_plan_query[0] == "&":
                df_plan_query = df_plan_query[1:]
            st.dataframe(df_plan.query(df_plan_query))
        else:
            st.dataframe(df_plan)
        
    with col_plan2:
        st.button(label="Effacer les filtres", on_click=reset_filter_plan, key="button_filter_plan")
            
    
    
    

     
    
    uploaded_file_plan = st.file_uploader("Importer des nouvelles donneés issues de la comptabilité (Plan Comptable)", type='txt')
    if uploaded_file_plan is not None:
        # mettre à jour le plan comptables
        df_plan = update_plan_from_file(df_plan, uploaded_file_plan, format_plan_csv_file, plan_csv_file, plan_primary_key)
        # enregistrer le plan comptable, on passe pas un boutton, en cas d'erreur une actualisation du site et l'erreur est annulée.
        save_plan_file = st.button("Enregistrer les données ajoutées au plan comptables ?")
                # on enregistre le résultat dans notre BDD (ici un csv, ce qui laisse l'usage à l'utilisateur en dehors du logiciel possible)
        if save_plan_file:
            df_plan.to_csv(plan_csv_file, index=False, sep=";")

    st.subheader("Liste de la synthèse du plan comptable")
    st.write("La colonne compteFus correspond à la colonne de regroupement pour synthétiser les dépenses par poste.")
   
    table_compte_fus_classe_filter = st.radio(label="Définir la classe à afficher (defaut 6)", 
                                options=["1", "2", "3", "4", "5", "6", "7", "Toutes"], 
                                index=5,
                                horizontal=True,
                                key="f_compte_synthese_plan")
    table_compte_fus_filter = st.text_input(label="Rechercher dans le plan comptable :", 
                                      key="f_word_table_compte_fus")
    
    df_table_compte_fus_query = ""
    if table_compte_fus_filter:
        df_table_compte_fus_query = f"( (Compte.str.contains('{table_compte_fus_filter.upper()}')) |  \
                                        (NomCompte.str.contains('{table_compte_fus_filter.upper()}')) | \
                                        (compteFus.str.contains('{table_compte_fus_filter.upper()}')) ) "
    if table_compte_fus_classe_filter != "Toutes":
        df_table_compte_fus_query += f"& (Compte.str.startswith('{table_compte_fus_classe_filter}'))"    
    if df_table_compte_fus_query:
        if df_table_compte_fus_query[0] == "&":
            df_table_compte_fus_query = df_table_compte_fus_query[1:]
        st.dataframe(df_table_compte_fus.query(df_table_compte_fus_query))
    else:
        st.dataframe(df_table_compte_fus)
    
    
    
    if df_mvt.loc[df_mvt['compteFus'].isnull()].count().all() != 0:
        st.dataframe(df_mvt.loc[df_mvt['compteFus'].isnull()])
    


with tab_tableau:
 
    years = np.sort(df_mvt['annee'].unique())
    default_year_option = years[-3:]
    year_to_display = st.multiselect("Choisi les années à inclure :", options=years , default=default_year_option)
    current_year = max(year_to_display)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        checkbox_display_negative_delta = st.checkbox("Conserver les soustractions negatives ? ", value=False)
    with c2:
        checkbox_compare_from_today = st.checkbox("Comparer jusqu'à aujourd'hui uniquement", value=True)
    
    col_depense, col_recette = st.columns(2)    
    with col_depense:      
        if checkbox_compare_from_today:
            st.subheader(f"Tableau de synthèse des dépenses au {datetime.today().strftime('%d %B %Y')}")   
        else:
            st.subheader(f"Tableau de synthèse des dépenses au 31 décembre {datetime.today().year}") 
        df_synthese_debit = create_resume_dataframe(df_mvt, 6, 
                                                    checkbox_display_negative_delta,
                                                    checkbox_compare_from_today)
        st.dataframe(df_synthese_debit)
        csv_depense = convert_df(df_synthese_debit)
        
        st.download_button(
            label="Télécharger la synthèse des dépenses au format CSV",
            data=csv_depense,
            file_name='Dépenses.csv',
            mime='text/csv')
    

        
        
    with col_recette:
        if checkbox_compare_from_today:
            st.subheader(f"Tableau de synthèse des recettes au {datetime.today().strftime('%d %B %Y')}")   
        else:
            st.subheader(f"Tableau de synthèse des recettes au 31 décembre {datetime.today().year}")   
        df_synthese_credit = create_resume_dataframe(df_mvt, 7, 
                                    checkbox_display_negative_delta,
                                    checkbox_compare_from_today)
        st.dataframe(df_synthese_credit)
        csv_recette = convert_df(df_synthese_credit)
        
        st.download_button(
            label="Télécharger la synthèse des recettes au format CSV",
            data=csv_recette,
            file_name='Recettes.csv',
            mime='text/csv',
        )



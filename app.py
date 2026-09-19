# -*- coding: utf-8 -*-
"""
GALOP ANALYZER - APPLICATION STREAMLIT
----------------------------------------
Affiche les prédictions TOP 3 (sans cote PMU) du modèle XGBoost
entraîné par entrainer_xgboost_galop.py, pour les courses du jour
téléchargées depuis le Google Sheets.

Réutilise directement la logique déjà validée de generer_dataset_IA.py,
features_supplementaires.py et predire_top3_jour.py : aucune feature
n'est recalculée "à la main" ici, pour éviter toute divergence avec le
modèle entraîné.

Structure de dépôt attendue (chemins relatifs, compatibles GitHub /
Streamlit Community Cloud) :

  app.py
  generer_dataset_IA.py
  features_supplementaires.py
  entrainer_xgboost_galop.py
  predire_top3_jour.py
  requirements.txt
  Hippodromes francais et etrangers.xlsx
  hippodromes_galop_complet.csv
  data/
    dataset/
      dataset_galop_france.csv      <- historique déjà généré
    modele/
      modele_xgboost_top3.json      <- modèle déjà entraîné
      features_top3.json            <- métadonnées du modèle
"""

import io
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb

# ============================================================
# CHEMINS PORTABLES (fonctionnent en local ET sur Streamlit Cloud)
# ============================================================

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

FICHIER_HISTORIQUE = BASE_DIR / "data" / "dataset" / "dataset_galop_france.csv"
FICHIER_MODELE = BASE_DIR / "data" / "modele" / "modele_xgboost_top3.json"
FICHIER_FEATURES_JSON = BASE_DIR / "data" / "modele" / "features_top3.json"

# Même Google Sheets que predire_top3_jour.py, onglet "courses du jour".
URL_CSV_JOUR = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=1852089216&single=true&output=csv"
)
TIMEOUT = 45

st.set_page_config(page_title="🏇 Galop Analyzer", layout="wide", page_icon="🏇")

# ============================================================
# IMPORT DE LA LOGIQUE DEJA VALIDEE (pas de réécriture ici)
# ============================================================

from generer_dataset_IA import (
    preparer_colonnes,
    enrichir_surface,
    add_history_features,
    add_recent_form_features,
    add_context_features,
    add_jockey_continuite_features,
    charger_listes_hippodromes,
    normaliser_hippodrome,
)
from features_supplementaires import enrichir_features_supplementaires
from entrainer_xgboost_galop import PARTANTS_MIN, PARTANTS_MAX

COLONNES_BASE = [
    "Date_Course", "Hippodrome", "Hippodrome_Norm", "Hippodrome_Canonique",
    "Type_Hippodrome", "Numero_Course", "Course_ID",
    "Cheval", "Cheval_Norm", "Jockey", "Jockey_Norm",
    "Entraineur", "Entraineur_Norm", "Proprietaire",
    "Distance", "Nature_Piste", "Etat_Terrain", "Penetrometre",
    "Oeilleres", "Surface_Final", "Surface_Confiance", "Surface_Methode",
    "Classement",
    "Age", "Sexe", "Poids", "Supplement", "Inedit",
    "Place_Corde", "Corde_Piste", "Discipline", "Musique",
    "Gains_Carriere", "Gains_Victoires", "Gains_Place",
    "Gains_Annee_En_Cours", "Gains_Annee_Precedente",
    "Nb_Courses", "Nb_Victoires", "Nb_Places", "Nb_Places_2e", "Nb_Places_3e",
]


# ============================================================
# CHARGEMENT DU MODELE (mis en cache : lu une seule fois)
# ============================================================

@st.cache_resource
def charger_modele():
    if not FICHIER_MODELE.exists() or not FICHIER_FEATURES_JSON.exists():
        return None, None, None, None, (
            f"Fichier modèle introuvable. Attendu :\n"
            f"- {FICHIER_MODELE}\n- {FICHIER_FEATURES_JSON}\n\n"
            f"Lance d'abord entrainer_xgboost_galop.py, puis copie le "
            f"dossier data/modele dans le dépôt."
        )

    meta = json.loads(FICHIER_FEATURES_JSON.read_text(encoding="utf-8"))
    features = meta["features"]
    colonnes_categorielles = meta["colonnes_categorielles"]
    categories_par_colonne = meta.get("categories_par_colonne", {})

    modele = xgb.XGBClassifier(enable_categorical=True)
    modele.load_model(str(FICHIER_MODELE))
    # load_model() ne restaure pas les hyperparamètres du wrapper
    # scikit-learn (dont enable_categorical) : il faut le refixer.
    modele.enable_categorical = True

    return modele, features, colonnes_categorielles, categories_par_colonne, None


# ============================================================
# CHARGEMENT DE L'HISTORIQUE (mis en cache 1h)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def charger_historique_base():
    if not FICHIER_HISTORIQUE.exists():
        return None, f"Fichier historique introuvable : {FICHIER_HISTORIQUE}"

    df = pd.read_csv(FICHIER_HISTORIQUE, low_memory=False)
    df["Date_Course"] = pd.to_datetime(df["Date_Course"], errors="coerce")
    colonnes = [c for c in COLONNES_BASE if c in df.columns]
    return df[colonnes].copy(), None


# ============================================================
# TELECHARGEMENT DES COURSES DU JOUR (mis en cache 10 min)
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def telecharger_courses_du_jour():
    try:
        r = requests.get(
            URL_CSV_JOUR, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}
        )
    except Exception as e:
        return None, f"Impossible d'accéder au Google Sheets : {e}"

    if r.status_code == 404:
        return None, "Le lien Google Sheets renvoie une 404 (onglet non publié en CSV ?)."

    head = r.content[:300].lower()
    if b"<html" in head or b"<!doctype" in head:
        return None, "Le lien ne renvoie pas un CSV mais une page HTML."

    df = pd.read_csv(io.BytesIO(r.content), dtype=str)
    if df.empty:
        return None, "Le CSV des courses du jour est vide."

    return df, None


def filtrer_hippodromes_jour(df):
    fr_map, etr_map, inclus_etr = charger_listes_hippodromes()
    hip_norm = df["Hippodrome"].map(normaliser_hippodrome)
    is_fr = hip_norm.isin(fr_map)
    is_etr_proche = hip_norm.isin(inclus_etr)

    out = df.copy()
    out["Type_Hippodrome"] = np.where(
        is_fr, "FRANCE", np.where(is_etr_proche, "ETRANGER_PROCHE", "EXCLU")
    )
    out["Hippodrome_Canonique"] = hip_norm.map(
        lambda h: fr_map.get(h, etr_map.get(h, str(h).strip()))
    )
    return out[is_fr | is_etr_proche].copy()


def preparer_courses_du_jour(df_brut):
    df = preparer_colonnes(df_brut)
    df = filtrer_hippodromes_jour(df)
    df["Classement"] = np.nan  # anti-fuite : la course n'a pas eu lieu
    if df.empty:
        return None
    df = enrichir_surface(df)
    return df


# ============================================================
# RECALCUL DES FEATURES "AVANT COURSE" AVEC TOUT L'HISTORIQUE
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def calculer_features_jour(_hist, jour):
    """
    _hist : historique (préfixé _ pour indiquer à Streamlit de ne pas
    tenter de le hacher à chaque appel — objet stable via son propre
    cache pendant 1h).
    """
    ids_jour = set(jour["Course_ID"])

    total = pd.concat([_hist, jour], ignore_index=True, sort=False)
    total["_ordre_du_jour"] = total["Course_ID"].isin(ids_jour)
    total = total.sort_values(
        ["Date_Course", "_ordre_du_jour", "Course_ID"], kind="stable"
    ).drop(columns=["_ordre_du_jour"]).reset_index(drop=True)

    total = add_history_features(total)
    total = add_recent_form_features(total)
    total = add_jockey_continuite_features(total)
    total = add_context_features(total)
    total = enrichir_features_supplementaires(total)

    return total[total["Course_ID"].isin(ids_jour)].copy()


# ============================================================
# PREDICTION
# ============================================================

def preparer_X(df, features, colonnes_categorielles, categories_par_colonne):
    df = df.copy()

    for f in features:
        if f not in df.columns:
            df[f] = np.nan

    for c in colonnes_categorielles:
        if c not in df.columns:
            continue
        valeurs = df[c].astype(str).fillna("INCONNU")
        categories_connues = categories_par_colonne.get(c)
        if categories_connues:
            inconnues = ~valeurs.isin(categories_connues)
            if inconnues.any() and "INCONNU" in categories_connues:
                valeurs = valeurs.where(~inconnues, "INCONNU")
            df[c] = pd.Categorical(valeurs, categories=categories_connues)
        else:
            df[c] = valeurs.astype("category")

    for c in features:
        if c in colonnes_categorielles or c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    return df[features]


def formater_num_pmu(valeur):
    """Affiche le numéro PMU proprement (gère '1', '1.0', NaN, texte...)."""
    if pd.isna(valeur) or str(valeur).strip() in ("", "nan", "None"):
        return "?"
    try:
        return str(int(float(valeur)))
    except (ValueError, TypeError):
        return str(valeur).strip()


def predire(modele, df, features, colonnes_categorielles, categories_par_colonne):
    X = preparer_X(df, features, colonnes_categorielles, categories_par_colonne)
    df = df.copy()
    df["Proba_Podium"] = modele.predict_proba(X)[:, 1]
    return df.sort_values("Proba_Podium", ascending=False).reset_index(drop=True)


# ============================================================
# ANALYSE NARRATIVE (factuelle : compare le cheval au reste du peloton)
# ============================================================

def _joindre(clauses):
    """Joint une liste de fragments de phrase avec des virgules et un
    'et' avant le dernier, pour une lecture naturelle."""
    clauses = [c for c in clauses if c]
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + " et " + clauses[-1]


def construire_recit(row, df_course):
    """
    Construit un commentaire en langage naturel qui COMBINE les
    differents signaux (distance, jockey, hippodrome, forme, poids...)
    plutot que de les lister separement.
    """
    nom = row.get("Cheval", "Ce cheval")
    pronom = "Elle" if str(row.get("Sexe", "")).strip().upper().startswith("F") else "Il"

    clauses_contexte = []
    clauses_globales = []
    vigilances = []

    # ---------- Distance ----------
    nb_dist = row.get("Cheval_Distance_Courses_Avant")
    distance_m = row.get("Distance")
    distance_txt = f"{distance_m:.0f} m" if pd.notna(distance_m) else "cette distance"
    if pd.notna(nb_dist) and nb_dist > 0:
        victoires_dist = int(row.get("Cheval_Distance_Victoires_Avant") or 0)
        podiums_dist = int(row.get("Cheval_Distance_Podiums_Avant") or 0)
        places_dist = max(0, podiums_dist - victoires_dist)
        taux_dist = podiums_dist / nb_dist
        if taux_dist >= 0.35 or victoires_dist >= 1:
            clauses_contexte.append(
                f"court sur une distance qu'{'elle' if pronom=='Elle' else 'il'} semble apprecier "
                f"({victoires_dist} victoire(s) et {places_dist} place(s) en {int(nb_dist)} sortie(s) a {distance_txt})"
            )
        elif nb_dist >= 3 and podiums_dist == 0:
            vigilances.append(
                f"n'a jamais ete place{'e' if pronom=='Elle' else ''} a {distance_txt} en {int(nb_dist)} sorties"
            )
    else:
        vigilances.append(f"aborde la distance de {distance_txt} pour la premiere fois")

    # ---------- Continuite recente avec CE jockey (3 dernieres courses) ----------
    nb_meme_jockey_recent = row.get("Meme_Jockey_Nb_Recentes")
    nb_courses_analysees = row.get("Nb_Dernieres_Courses_Analysees")
    if pd.notna(nb_meme_jockey_recent) and pd.notna(nb_courses_analysees) and nb_courses_analysees > 0:
        placements_recents = int(row.get("Meme_Jockey_Nb_Recentes_Places") or 0)
        if nb_meme_jockey_recent >= 2:
            if nb_meme_jockey_recent == nb_courses_analysees:
                clauses_contexte.append(
                    f"a fait ses {int(nb_courses_analysees)} dernieres courses avec le meme "
                    f"jockey, place {placements_recents} fois"
                )
            else:
                clauses_contexte.append(
                    f"a couru {int(nb_meme_jockey_recent)} de ses {int(nb_courses_analysees)} "
                    f"dernieres courses avec le meme jockey, place {placements_recents} fois"
                )

    # ---------- Historique complet avec CE jockey ----------
    nb_couplage = row.get("Couplage_Courses_Avant")
    if pd.notna(nb_couplage) and nb_couplage > 0 and (pd.isna(nb_meme_jockey_recent) or nb_meme_jockey_recent < 2):
        victoires_couplage = int(row.get("Couplage_Victoires_Avant") or 0)
        podiums_couplage = int(row.get("Couplage_Podiums_Avant") or 0)
        places_couplage = max(0, podiums_couplage - victoires_couplage)
        if podiums_couplage > 0:
            clauses_contexte.append(
                f"a deja couru {int(nb_couplage)} fois avec ce jockey par le passe "
                f"(gagne {victoires_couplage} fois, place {places_couplage} fois)"
            )
        elif nb_couplage >= 3:
            vigilances.append(f"n'a jamais ete place avec ce jockey en {int(nb_couplage)} sorties ensemble")

    # ---------- Hippodrome ----------
    taux_podium = row.get("Cheval_Taux_Podium_Avant")
    nb_hippo = row.get("Cheval_Hippodrome_Courses_Avant")
    if pd.notna(nb_hippo) and nb_hippo >= 2:
        podiums_hippo = int(row.get("Cheval_Hippodrome_Podiums_Avant") or 0)
        taux_hippo = row.get("Cheval_Hippodrome_Taux_Podium_Avant")
        if pd.notna(taux_podium) and taux_hippo is not None and taux_hippo >= taux_podium + 0.15:
            clauses_contexte.append(
                f"revient sur un hippodrome qu'{'elle' if pronom=='Elle' else 'il'} affectionne "
                f"visiblement ({podiums_hippo} podium(s) en {int(nb_hippo)} course(s) ici, contre "
                f"{taux_podium*100:.0f}% de podiums en moyenne partout ailleurs)"
            )
        elif pd.notna(taux_podium) and taux_hippo is not None and taux_hippo <= taux_podium - 0.15:
            vigilances.append(
                f"reussit moins bien sur cet hippodrome que sa moyenne habituelle "
                f"({taux_hippo*100:.0f}% de podiums ici contre {taux_podium*100:.0f}% ailleurs)"
            )

    # ---------- Poids ----------
    poids = row.get("Poids_Num")
    if pd.notna(poids) and "Poids_Num" in df_course.columns:
        moyenne_poids = df_course["Poids_Num"].mean()
        ecart = poids - moyenne_poids
        if ecart <= -1.5:
            clauses_contexte.append(
                f"porte un poids inferieur de {abs(ecart):.1f} kg a la moyenne du peloton ({poids:.1f} kg)"
            )
        elif ecart >= 1.5:
            vigilances.append(f"porte un poids superieur de {ecart:.1f} kg a la moyenne du peloton")

    supplement = row.get("Supplement_Num")
    if pd.notna(supplement) and supplement > 0:
        vigilances.append(f"court avec un supplement de +{supplement:.1f} kg")

    # ---------- Forme et regularite globale ----------
    if pd.notna(taux_podium) and row.get("Cheval_Courses_Avant", 0) >= 3:
        nb_courses_avant = int(row.get("Cheval_Courses_Avant"))
        if taux_podium >= 0.35:
            clauses_globales.append(
                f"affiche un taux de podium solide de {taux_podium*100:.0f}% sur ses "
                f"{nb_courses_avant} dernieres courses connues"
            )
        elif taux_podium <= 0.10:
            vigilances.append(f"n'a ete place que {taux_podium*100:.0f}% du temps sur ses {nb_courses_avant} dernieres courses")
    elif row.get("Cheval_Courses_Avant", 0) == 0:
        vigilances.append("n'a aucun historique connu dans la base (cheval inedit ou peu couru)")

    forme = row.get("Cheval_Forme_3")
    if pd.notna(forme):
        rang_estime = 11 - forme
        if forme >= 7:
            clauses_globales.append(
                f"traverse une bonne dynamique recente (classement moyen d'environ "
                f"{rang_estime:.0f} sur ses 3 dernieres courses)"
            )
        elif forme <= 3:
            vigilances.append(
                f"connait un passage plus discret (classement moyen d'environ "
                f"{rang_estime:.0f} sur ses 3 dernieres courses)"
            )

    musique_taux = row.get("Musique_Taux_Podium")
    if pd.notna(musique_taux) and row.get("Musique_Nb_Perfs", 0) >= 3 and musique_taux >= 0.5:
        clauses_globales.append(f"sa musique recente est favorable ({musique_taux*100:.0f}% de places)")

    jockey_taux = row.get("Jockey_Taux_Victoire_Avant")
    if pd.notna(jockey_taux) and jockey_taux >= 0.15:
        clauses_globales.append(f"est confie a un jockey qui gagne {jockey_taux*100:.0f}% de ses courses")

    entraineur_taux = row.get("Entraineur_Taux_Victoire_Avant")
    if pd.notna(entraineur_taux) and entraineur_taux >= 0.15:
        clauses_globales.append(f"est entraine par une ecurie en reussite ({entraineur_taux*100:.0f}% de victoires)")

    gains_par_course = row.get("Gains_Par_Course")
    if pd.notna(gains_par_course) and "Gains_Par_Course" in df_course.columns:
        moyenne_gains = df_course["Gains_Par_Course"].mean()
        if pd.notna(moyenne_gains) and moyenne_gains > 0 and gains_par_course > moyenne_gains * 1.5:
            clauses_globales.append("ses gains par course sont nettement au-dessus de la moyenne du peloton")

    if row.get("Inedit_Flag") == 1:
        vigilances.append("n'a aucune course connue dans la base (Inedit)")

    # ---------- Assemblage du recit ----------
    phrases = []
    if clauses_contexte:
        phrases.append(f"{nom} " + _joindre(clauses_contexte) + ".")
    if clauses_globales:
        phrases.append(f"{pronom} " + _joindre(clauses_globales) + ".")
    if not phrases:
        phrases.append(f"Peu de signaux disponibles sur {nom} pour cette course.")
    if vigilances:
        phrases.append("A surveiller : " + _joindre(vigilances) + ".")

    return " ".join(phrases)




# ============================================================
# PIPELINE COMPLET (mis en cache pour l'ensemble des courses du jour)
# ============================================================

def charger_predictions_du_jour():
    hist, err_hist = charger_historique_base()
    if err_hist:
        return None, err_hist

    jour_brut, err_jour = telecharger_courses_du_jour()
    if err_jour:
        return None, err_jour

    jour = preparer_courses_du_jour(jour_brut)
    if jour is None:
        return None, "Aucune course du jour dans le périmètre France / étrangers proches."

    jour_features = calculer_features_jour(hist, jour)

    modele, features, colonnes_categorielles, categories_par_colonne, err_modele = charger_modele()
    if err_modele:
        return None, err_modele

    jour_predit = predire(modele, jour_features, features, colonnes_categorielles, categories_par_colonne)
    return jour_predit, None


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

def afficher_top3_medailles(df_c):
    medailles = ["🥇", "🥈", "🥉"]
    for rang, (_, row) in enumerate(df_c.head(3).iterrows()):
        num_pmu = formater_num_pmu(row.get("Num_PMU"))
        st.markdown(
            f"{medailles[rang]} N°{num_pmu} **{row['Cheval']}** "
            f"({row['Proba_Podium']*100:.0f}%)"
        )


def formater_jumele_reduit(df_c):
    """
    Combinaison 'Jumelé placé en champ réduit' pour les courses de 10 à
    14 partants : 3 lignes de 4 combinaisons chacune (12 combinaisons
    au total, 1€ la combinaison = 12€ de mise totale), construites à
    partir du classement de probabilité du modèle (Top1 à Top6).
    """
    df_c = df_c.sort_values("Proba_Podium", ascending=False).reset_index(drop=True)
    if len(df_c) < 6:
        return None

    tops = {}
    for i in range(6):
        row = df_c.iloc[i]
        tops[i + 1] = f"N°{formater_num_pmu(row.get('Num_PMU'))} {row['Cheval']}"

    lignes = [
        (1, [3, 4, 5, 6]),
        (1, [2, 4, 5, 6]),
        (2, [3, 4, 5, 6]),
    ]

    texte = ["**🎫 Jumelé placé (champ réduit)**"]
    for banquier, partenaires in lignes:
        partenaires_txt = " - ".join(tops[p] for p in partenaires)
        texte.append(f"- {tops[banquier]} / {partenaires_txt}")
    texte.append("*Total : 12 combinaisons à 1 € = 12 €*")

    return "\n\n".join(texte)


def main():
    st.title("🏇 Galop Analyzer")
    st.markdown(
        "### 🤖 Prédictions IA (XGBoost) du TOP 3 — **sans utiliser la cote PMU**"
    )

    with st.spinner("Chargement de l'historique et des courses du jour "
                     "(peut prendre plusieurs minutes la première fois)..."):
        df_predit, erreur = charger_predictions_du_jour()

    if erreur:
        st.error(erreur)
        st.stop()

    if df_predit is None or df_predit.empty:
        st.info("Aucune prédiction disponible pour le moment.")
        st.stop()

    with st.sidebar:
        st.header("📍 Sélection")
        uniquement_8_16 = st.checkbox(
            f"Uniquement {PARTANTS_MIN}-{PARTANTS_MAX} partants "
            f"(périmètre d'entraînement du modèle)",
            value=True,
        )

    df_affichable = df_predit.copy()
    if uniquement_8_16:
        df_affichable = df_affichable[
            (df_affichable["Nb_Partants"] >= PARTANTS_MIN)
            & (df_affichable["Nb_Partants"] <= PARTANTS_MAX)
        ]

    if df_affichable.empty:
        st.warning(
            f"Aucune course du jour n'a entre {PARTANTS_MIN} et {PARTANTS_MAX} "
            f"partants. Décoche le filtre dans la barre latérale pour tout voir."
        )
        st.stop()

    # Regroupement Réunion (si dispo) -> Hippodrome -> Course
    reunion_col = "Reunion" if "Reunion" in df_affichable.columns else None
    df_affichable["_cle_reunion"] = (
        df_affichable[reunion_col].astype(str) if reunion_col
        else df_affichable["Hippodrome_Canonique"]
    )

    with st.sidebar:
        reunions = sorted(df_affichable["_cle_reunion"].unique())
        reunion_choisie = st.selectbox("Réunion", reunions)

        df_reunion = df_affichable[df_affichable["_cle_reunion"] == reunion_choisie]
        hippodrome_actuel = df_reunion["Hippodrome_Canonique"].iloc[0]

        courses_dispo = sorted(
            df_reunion["Numero_Course"].unique(),
            key=lambda x: (len(str(x)), str(x)),
        )
        course_choisie = st.selectbox("Course", courses_dispo)

        st.markdown("---")
        st.info(f"📍 **{hippodrome_actuel}**")

        st.markdown("---")
        st.subheader("📋 Résumé des courses")
        for r in reunions:
            df_r = df_affichable[df_affichable["_cle_reunion"] == r]
            st.markdown(f"**{r}** — {df_r['Hippodrome_Canonique'].iloc[0]}")
            for c in sorted(df_r["Numero_Course"].unique(), key=lambda x: (len(str(x)), str(x))):
                df_c = df_r[df_r["Numero_Course"] == c].sort_values("Proba_Podium", ascending=False)
                nb_partants_c = len(df_c)
                with st.expander(f"Course {c} ({nb_partants_c} partants)", expanded=False):
                    afficher_top3_medailles(df_c)

    # ============================================================
    # AFFICHAGE DE LA COURSE SELECTIONNEE
    # ============================================================
    df_course = df_affichable[
        (df_affichable["_cle_reunion"] == reunion_choisie)
        & (df_affichable["Numero_Course"] == course_choisie)
    ].sort_values("Proba_Podium", ascending=False).reset_index(drop=True)

    st.markdown("---")
    st.subheader(f"📍 {reunion_choisie} — Course {course_choisie} | {hippodrome_actuel}")
    st.caption(f"{len(df_course)} partants — Distance : {df_course['Distance'].iloc[0]:.0f} m")

    df_course = df_course.copy()
    df_course["Num_PMU_Fmt"] = df_course.get("Num_PMU", pd.Series(dtype=object)).apply(formater_num_pmu)

    colonnes_affichage = {
        "Num_PMU_Fmt": "N° PMU",
        "Cheval": "Cheval",
        "Jockey": "Jockey",
        "Poids_Num": "Poids (kg)",
        "Place_Corde_Num": "Corde",
        "Proba_Podium": "Prob. Top 3 (%)",
    }
    colonnes_dispo = [c for c in colonnes_affichage if c in df_course.columns]
    df_affiche = df_course[colonnes_dispo].rename(columns=colonnes_affichage).copy()
    df_affiche["Prob. Top 3 (%)"] = (df_affiche["Prob. Top 3 (%)"] * 100).round(1)
    df_affiche.index = range(1, len(df_affiche) + 1)
    st.dataframe(df_affiche, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Analyse du TOP 3 prédit")

    medailles_titre = ["🥇 1er favori", "🥈 2ème favori", "🥉 3ème favori"]
    for i in range(min(3, len(df_course))):
        row = df_course.iloc[i]
        proba = row["Proba_Podium"] * 100
        with st.expander(
            f"{medailles_titre[i]} : {row['Cheval']} — {proba:.1f}% de finir dans le top 3",
            expanded=(i == 0),
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Prob. Top 3", f"{proba:.1f}%")
            c2.metric("Jockey", str(row.get("Jockey", "N/A")))
            c3.metric("Corde", str(row.get("Place_Corde_Num", "N/A")))

            st.markdown("**🔎 Pourquoi ce cheval ?**")
            st.markdown(construire_recit(row, df_course))

    st.markdown("---")
    st.caption(
        "🤖 Modèle XGBoost entraîné sans la cote PMU. Les probabilités "
        "sont indicatives et ne remplacent pas ton propre jugement."
    )


if __name__ == "__main__":
    main()

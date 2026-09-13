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

def expliquer_cheval(row, df_course):
    forces, vigilances = [], []

    taux_podium = row.get("Cheval_Taux_Podium_Avant")
    nb_courses_avant = row.get("Cheval_Courses_Avant")
    if pd.notna(taux_podium) and pd.notna(nb_courses_avant) and nb_courses_avant >= 3:
        if taux_podium >= 0.35:
            forces.append(
                f"🏆 Taux de podium de **{taux_podium*100:.0f}%** sur ses "
                f"{int(nb_courses_avant)} dernières courses connues."
            )
        elif taux_podium <= 0.10:
            vigilances.append(
                f"Taux de podium modeste : **{taux_podium*100:.0f}%** sur "
                f"{int(nb_courses_avant)} courses."
            )
    elif pd.isna(nb_courses_avant) or nb_courses_avant == 0:
        vigilances.append("Aucun historique connu dans la base (cheval inédit ou peu couru).")

    forme = row.get("Cheval_Forme_3")
    if pd.notna(forme):
        if forme <= 3:
            forces.append(f"📈 Bonne forme récente : classement moyen de **{forme:.1f}** sur ses 3 dernières courses.")
        elif forme >= 7:
            vigilances.append(f"Forme récente en retrait : classement moyen de **{forme:.1f}** sur ses 3 dernières courses.")

    musique_taux = row.get("Musique_Taux_Podium")
    if pd.notna(musique_taux) and row.get("Musique_Nb_Perfs", 0) >= 3:
        if musique_taux >= 0.5:
            forces.append(f"🎵 Musique favorable : {musique_taux*100:.0f}% de places dans le top 3 récemment.")

    jockey_taux = row.get("Jockey_Taux_Victoire_Avant")
    if pd.notna(jockey_taux) and jockey_taux >= 0.15:
        forces.append(f"🏇 Jockey performant : **{jockey_taux*100:.0f}%** de victoires sur ses montes passées.")

    entraineur_taux = row.get("Entraineur_Taux_Victoire_Avant")
    if pd.notna(entraineur_taux) and entraineur_taux >= 0.15:
        forces.append(f"👤 Entraîneur en réussite : **{entraineur_taux*100:.0f}%** de victoires.")

    gains_par_course = row.get("Gains_Par_Course")
    if pd.notna(gains_par_course) and "Gains_Par_Course" in df_course.columns:
        moyenne_peloton = df_course["Gains_Par_Course"].mean()
        if pd.notna(moyenne_peloton) and moyenne_peloton > 0 and gains_par_course > moyenne_peloton * 1.5:
            forces.append(f"💰 Gains par course nettement au-dessus de la moyenne du peloton.")

    poids = row.get("Poids_Num")
    if pd.notna(poids) and "Poids_Num" in df_course.columns:
        moyenne_poids = df_course["Poids_Num"].mean()
        ecart = poids - moyenne_poids
        if ecart <= -1.5:
            forces.append(f"⚖️ Avantage au poids : {poids:.1f} kg ({ecart:+.1f} kg vs moyenne du peloton).")
        elif ecart >= 1.5:
            vigilances.append(f"Poids pénalisant : {poids:.1f} kg ({ecart:+.1f} kg vs moyenne du peloton).")

    if row.get("Inedit_Flag") == 1:
        vigilances.append("Cheval sans course connue dans la base (Inédit).")

    return forces, vigilances


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
                with st.expander(f"Course {c} ({len(df_c)} partants)", expanded=False):
                    medailles = ["🥇", "🥈", "🥉"]
                    for rang, (_, row) in enumerate(df_c.head(3).iterrows()):
                        num_pmu = formater_num_pmu(row.get("Num_PMU"))
                        st.markdown(
                            f"{medailles[rang]} N°{num_pmu} **{row['Cheval']}** "
                            f"({row['Proba_Podium']*100:.0f}%)"
                        )

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

    colonnes_affichage = {
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

            forces, vigilances = expliquer_cheval(row, df_course)

            col_f, col_v = st.columns(2)
            with col_f:
                st.markdown("**🟢 Points forts détectés**")
                if forces:
                    for f in forces:
                        st.markdown(f"- {f}")
                else:
                    st.markdown("- Rien de particulièrement saillant.")
            with col_v:
                st.markdown("**🟠 Points de vigilance**")
                if vigilances:
                    for v in vigilances:
                        st.markdown(f"- {v}")
                else:
                    st.markdown("- Aucun signal négatif détecté.")

    st.markdown("---")
    st.caption(
        "🤖 Modèle XGBoost entraîné sans la cote PMU. Les probabilités "
        "sont indicatives et ne remplacent pas ton propre jugement."
    )


if __name__ == "__main__":
    main()

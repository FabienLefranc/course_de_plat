# -*- coding: utf-8 -*-
"""
FEATURES_SUPPLEMENTAIRES.PY
----------------------------
Ajoute au dataset (déjà produit par generer_dataset_IA.py) les colonnes
brutes de la source PMU qui ne sont PAS encore exploitées comme features
numériques : Age, Sexe, Poids, Corde, Musique, Inédit, Gains, historique
carrière fourni par la source (Nb_Courses / Nb_Victoires / ...).

Toutes ces colonnes sont disponibles AVANT la course (ce sont des infos
de "présentation" du cheval au départ), donc aucune fuite introduite.

Importé par :
  - entrainer_xgboost_galop.py
  - predire_top3_jour.py
"""

import re
import numpy as np
import pandas as pd


# ============================================================
# OUTILS
# ============================================================

def _to_num(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce"
    )


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    return out


# ============================================================
# MUSIQUE (forme longue, souvent au-delà de ce que capture
# Cheval_Forme_3 / Cheval_Forme_5 puisqu'elle inclut aussi les
# courses hors périmètre du dataset : trot, étranger exclu, avant
# le 02/09/2025, etc.)
# ============================================================

# Une performance PMU s'écrit : <place ou code><lettre discipline>
# ex: "3p3p1p0p" -> 3e (plat), 3e (plat), 1er (plat), non-placé (plat)
_MUSIQUE_TOKEN = re.compile(r"(\d{1,2}|[DTARN]{1,3})([a-zA-Z])")

_CODES_INCIDENT = {"D", "T", "A", "R", "RET", "NP", "DAI", "DA"}


def parse_musique(musique):
    """
    Retourne un dict de features numériques résumant la Musique.
    Ne lève jamais d'exception : musique vide/illisible -> tout NaN.
    """
    defaut = {
        "Musique_Nb_Perfs": 0,
        "Musique_Nb_Victoires": np.nan,
        "Musique_Nb_Podiums": np.nan,
        "Musique_Taux_Podium": np.nan,
        "Musique_Moyenne_Place": np.nan,
        "Musique_Derniere_Place": np.nan,
        "Musique_Taux_Incident": np.nan,
    }

    if pd.isna(musique):
        return defaut

    s = str(musique).strip().upper()
    if not s:
        return defaut

    tokens = _MUSIQUE_TOKEN.findall(s)
    if not tokens:
        return defaut

    places = []
    incidents = 0

    for code, _discipline in tokens:
        if code.isdigit():
            p = int(code)
            if p > 0:
                places.append(p)
            # "0" = non placé au-delà de la 9e / hors du cadre : ignoré
            # du calcul de moyenne mais compte dans le nombre de perfs.
        elif code in _CODES_INCIDENT or code[:1] in {"D", "T", "A", "R", "N"}:
            incidents += 1

    nb_perfs = len(tokens)
    nb_victoires = sum(1 for p in places if p == 1)
    nb_podiums = sum(1 for p in places if p <= 3)

    return {
        "Musique_Nb_Perfs": nb_perfs,
        "Musique_Nb_Victoires": nb_victoires,
        "Musique_Nb_Podiums": nb_podiums,
        "Musique_Taux_Podium": (nb_podiums / nb_perfs) if nb_perfs else np.nan,
        "Musique_Moyenne_Place": (float(np.mean(places)) if places else np.nan),
        "Musique_Derniere_Place": (places[0] if places else np.nan),  # 1er token = perf la + récente
        "Musique_Taux_Incident": (incidents / nb_perfs) if nb_perfs else np.nan,
    }


def add_musique_features(df, col="Musique"):
    df = df.copy()
    if col not in df.columns:
        df[col] = np.nan

    parsed = df[col].apply(parse_musique).apply(pd.Series)
    for c in parsed.columns:
        df[c] = parsed[c]

    return df


# ============================================================
# CARACTERISTIQUES DU CHEVAL AU DEPART
# ============================================================

def add_signalement_features(df):
    """Age, Sexe, Poids, Supplément, Œillères (numérisé), Inédit."""
    df = df.copy()

    df["Age_Num"] = _to_num(df["Age"]) if "Age" in df.columns else np.nan
    df["Poids_Num"] = _to_num(df["Poids"]) if "Poids" in df.columns else np.nan
    df["Supplement_Num"] = _to_num(df["Supplement"]) if "Supplement" in df.columns else 0.0
    df["Supplement_Num"] = df["Supplement_Num"].fillna(0.0)

    df["Sexe_Cat"] = (
        df["Sexe"].astype(str).str.strip().str.upper() if "Sexe" in df.columns else "INCONNU"
    )

    df["Inedit_Flag"] = (
        df["Inedit"].astype(str).str.strip().str.upper().eq("OUI").astype(int)
        if "Inedit" in df.columns else 0
    )

    return df


# ============================================================
# CORDE / DISCIPLINE / PISTE
# ============================================================

def add_corde_discipline_features(df):
    df = df.copy()

    df["Place_Corde_Num"] = (
        _to_num(df["Place_Corde"]) if "Place_Corde" in df.columns else np.nan
    )

    df["Corde_Piste_Cat"] = (
        df["Corde_Piste"].astype(str).str.strip().str.upper()
        if "Corde_Piste" in df.columns else "INCONNU"
    )

    df["Discipline_Cat"] = (
        df["Discipline"].astype(str).str.strip().str.upper()
        if "Discipline" in df.columns else "INCONNU"
    )

    return df


# ============================================================
# GAINS ET HISTORIQUE CARRIERE FOURNIS PAR LA SOURCE PMU
# (déjà "avant course" par construction du flux PMU -> pas de fuite)
# ============================================================

_GAINS_COLS = [
    "Gains_Carriere", "Gains_Victoires", "Gains_Place",
    "Gains_Annee_En_Cours", "Gains_Annee_Precedente",
]

_CARRIERE_COLS = [
    "Nb_Courses", "Nb_Victoires", "Nb_Places", "Nb_Places_2e", "Nb_Places_3e",
]


def add_gains_et_carriere_features(df):
    df = df.copy()

    for c in _GAINS_COLS:
        num_col = f"{c}_Num"
        df[num_col] = _to_num(df[c]) if c in df.columns else np.nan
        # Compression : les gains suivent une distribution très étalée.
        df[f"{c}_Log"] = np.log1p(df[num_col].clip(lower=0))

    for c in _CARRIERE_COLS:
        num_col = f"{c}_Num"
        df[num_col] = _to_num(df[c]) if c in df.columns else np.nan

    nb_courses = df.get("Nb_Courses_Num", pd.Series(np.nan, index=df.index))
    nb_victoires = df.get("Nb_Victoires_Num", pd.Series(np.nan, index=df.index))
    nb_places = df.get("Nb_Places_Num", pd.Series(np.nan, index=df.index))

    df["Carriere_Taux_Victoire"] = _safe_div(nb_victoires, nb_courses)
    df["Carriere_Taux_Podium"] = _safe_div(
        nb_victoires.fillna(0) + nb_places.fillna(0), nb_courses
    )

    if "Gains_Carriere_Num" in df.columns:
        df["Gains_Par_Course"] = _safe_div(df["Gains_Carriere_Num"], nb_courses)
    else:
        df["Gains_Par_Course"] = np.nan

    return df


# ============================================================
# TAILLE DE PELOTON (nombre de partants par course)
# ============================================================

def add_nb_partants(df, course_id_col="Course_ID"):
    df = df.copy()
    df["Nb_Partants"] = df.groupby(course_id_col)[course_id_col].transform("size")
    return df


# ============================================================
# PIPELINE COMPLET
# ============================================================

def enrichir_features_supplementaires(df):
    """Applique tous les enrichissements ci-dessus, dans l'ordre."""
    df = add_musique_features(df)
    df = add_signalement_features(df)
    df = add_corde_discipline_features(df)
    df = add_gains_et_carriere_features(df)
    df = add_nb_partants(df)
    return df


# Colonnes catégorielles ajoutées par ce module (dtype 'category' pour XGBoost)
COLONNES_CATEGORIELLES_SUPP = [
    "Sexe_Cat",
    "Corde_Piste_Cat",
    "Discipline_Cat",
]

# Colonnes numériques ajoutées par ce module
COLONNES_NUMERIQUES_SUPP = [
    "Musique_Nb_Perfs", "Musique_Nb_Victoires", "Musique_Nb_Podiums",
    "Musique_Taux_Podium", "Musique_Moyenne_Place", "Musique_Derniere_Place",
    "Musique_Taux_Incident",
    "Age_Num", "Poids_Num", "Supplement_Num", "Inedit_Flag",
    "Place_Corde_Num",
    "Gains_Carriere_Log", "Gains_Victoires_Log", "Gains_Place_Log",
    "Gains_Annee_En_Cours_Log", "Gains_Annee_Precedente_Log",
    "Nb_Courses_Num", "Nb_Victoires_Num", "Nb_Places_Num",
    "Nb_Places_2e_Num", "Nb_Places_3e_Num",
    "Carriere_Taux_Victoire", "Carriere_Taux_Podium", "Gains_Par_Course",
    "Nb_Partants",
]

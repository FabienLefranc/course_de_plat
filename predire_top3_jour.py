# -*- coding: utf-8 -*-
"""
PREDIRE_TOP3_JOUR.PY
----------------------
Prédit le TOP 3 des courses du jour, à l'aide du modèle entraîné par
entrainer_xgboost_galop.py.

Les courses du jour sont téléchargées directement depuis le Google
Sheets (même classeur que l'historique, mais un onglet différent,
publié en CSV) — plus besoin d'exporter un fichier Excel à la main.

Sans la cote PMU, sans fuite.

Principe
--------
Les features "historique cheval/jockey/entraîneur" et "forme récente"
sont calculées par des fonctions SEQUENTIELLES de generer_dataset_IA.py
(add_history_features, add_recent_form_features) : elles reconstruisent
les compteurs course après course, dans l'ordre chronologique.

Pour que les chevaux du jour aient un historique juste (basé sur TOUTES
leurs courses passées, y compris celles d'hier), on :
  1. reprend les données brutes historiques déjà calculées
     (dataset_galop_france.csv) ;
  2. télécharge les courses du jour et calcule leur surface (même
     logique que le dataset historique) ;
  3. concatène historique + jour, trié par date ;
  4. relance add_history_features / add_recent_form_features /
     add_context_features sur l'ensemble -> les compteurs "avant course"
     du jour tiennent alors compte de tout l'historique.
  5. ne garde que les lignes du jour pour la prédiction.

C'est plus lent qu'un simple appel au modèle, mais c'est la seule façon
de rester rigoureusement cohérent avec la logique anti-fuite du dataset
d'entraînement.

Sortie :
  predictions_top3_du_jour.csv
"""

import io
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

# Le script doit être dans le même dossier que generer_dataset_IA.py
sys.path.insert(0, str(Path(__file__).parent))

from generer_dataset_IA import (
    preparer_colonnes,
    enrichir_surface,
    add_history_features,
    add_recent_form_features,
    add_context_features,
    charger_listes_hippodromes,
    normaliser_hippodrome,
    FICHIER_CSV,
    TIMEOUT,
)

from features_supplementaires import (
    enrichir_features_supplementaires,
    COLONNES_CATEGORIELLES_SUPP,
)

from entrainer_xgboost_galop import (
    COLONNES_CATEGORIELLES_DATASET,
    PARTANTS_MIN,
    PARTANTS_MAX,
    FICHIER_MODELE,
    FICHIER_FEATURES,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Onglet "courses du jour" du même Google Sheets que l'historique,
# publié en CSV (Fichier > Partager > Publier sur le Web > CSV).
URL_CSV_JOUR = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=1852089216&single=true&output=csv"
)

FICHIER_SORTIE = Path(
    r"C:\Users\33662\OneDrive\Bureau\course_de_plat\data\dataset\predictions_top3_du_jour.csv"
)

# Colonnes "brutes / identité" à conserver de l'historique avant de
# relancer les fonctions séquentielles. On exclut volontairement les
# colonnes déjà calculées par un précédent passage (Cheval_Courses_Avant,
# Cheval_Forme_3, Mois_Sin, Target_*, ...) : elles seront recréées à
# l'identique par le recalcul, donc inutile de les transporter.
COLONNES_BASE = [
    "Date_Course", "Hippodrome", "Hippodrome_Norm", "Hippodrome_Canonique",
    "Type_Hippodrome", "Numero_Course", "Course_ID",
    "Cheval", "Cheval_Norm", "Jockey", "Jockey_Norm",
    "Entraineur", "Entraineur_Norm", "Proprietaire",
    "Distance", "Nature_Piste", "Etat_Terrain", "Penetrometre",
    "Oeilleres", "Surface_Final", "Surface_Confiance", "Surface_Methode",
    "Classement",
    # colonnes brutes utilisées par features_supplementaires.py
    "Age", "Sexe", "Poids", "Supplement", "Inedit",
    "Place_Corde", "Corde_Piste", "Discipline", "Musique",
    "Gains_Carriere", "Gains_Victoires", "Gains_Place",
    "Gains_Annee_En_Cours", "Gains_Annee_Precedente",
    "Nb_Courses", "Nb_Victoires", "Nb_Places", "Nb_Places_2e", "Nb_Places_3e",
]


# ============================================================
# CHARGEMENT DE L'HISTORIQUE (déjà calculé)
# ============================================================

def charger_historique_base():
    print(f"Lecture de l'historique : {FICHIER_CSV}")
    df = pd.read_csv(FICHIER_CSV, low_memory=False)
    df["Date_Course"] = pd.to_datetime(df["Date_Course"], errors="coerce")

    colonnes = [c for c in COLONNES_BASE if c in df.columns]
    df = df[colonnes].copy()

    print(f"  {len(df):,} lignes historiques / {df['Course_ID'].nunique():,} courses")
    return df


# ============================================================
# CHARGEMENT ET PREPARATION DES COURSES DU JOUR
# ============================================================

def filtrer_hippodromes_jour(df):
    """
    Même logique que filtrer_hippodromes() de generer_dataset_IA.py,
    mais SANS écraser le fichier des courses étrangères exclues
    (ce fichier appartient au dataset d'entraînement, pas au jour J).
    """
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

    avant = len(out)
    out = out[is_fr | is_etr_proche].copy()
    print(f"Périmètre géographique du jour : {len(out):,}/{avant:,} lignes conservées "
          f"(France + étrangers proches)")

    return out


def telecharger_courses_du_jour():
    print(f"\nTéléchargement des courses du jour (Google Sheets)...")

    try:
        r = requests.get(
            URL_CSV_JOUR,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    except Exception as e:
        raise RuntimeError(f"Impossible d'accéder au Google Sheets du jour : {e}")

    if r.status_code == 404:
        raise RuntimeError(
            "\nLe lien Google Sheets des courses du jour renvoie une 404.\n"
            "Vérifiez que l'onglet est bien publié en CSV "
            "(Fichier > Partager > Publier sur le Web > CSV) et que "
            "URL_CSV_JOUR est à jour en haut de predire_top3_jour.py.\n"
        )

    r.raise_for_status()
    content = r.content

    head = content[:300].lower()
    if b"<html" in head or b"<!doctype" in head:
        raise RuntimeError(
            "\nLe lien ne renvoie pas un CSV mais une page HTML.\n"
            "Vérifiez que l'onglet du jour est bien publié en CSV.\n"
        )

    df = pd.read_csv(io.BytesIO(content), dtype=str)

    if df.empty:
        raise RuntimeError("Le CSV des courses du jour est vide.")

    print(f"  {len(df):,} lignes / {len(df.columns)} colonnes téléchargées")
    return df


def charger_courses_du_jour():
    df = telecharger_courses_du_jour()

    df = preparer_colonnes(df)
    df = filtrer_hippodromes_jour(df)

    # Sécurité anti-fuite : la course n'a pas encore eu lieu, un
    # éventuel Classement présent dans le fichier est ignoré.
    df["Classement"] = np.nan

    if df.empty:
        raise RuntimeError(
            "Aucune course du jour dans le périmètre France / étrangers proches."
        )

    print("Inférence de la surface pour les courses du jour...")
    df = enrichir_surface(df)

    return df


# ============================================================
# RECALCUL DE L'HISTORIQUE JUSQU'A AUJOURD'HUI
# ============================================================

def recalculer_avec_historique(hist, jour):
    ids_jour = set(jour["Course_ID"])

    total = pd.concat([hist, jour], ignore_index=True, sort=False)
    total["_ordre_du_jour"] = total["Course_ID"].isin(ids_jour)

    # Tri chronologique strict ; en cas d'ex-aequo sur la date, les
    # lignes déjà présentes dans l'historique passent avant celles du
    # jour (elles sont par définition antérieures ou simultanées).
    total = total.sort_values(
        ["Date_Course", "_ordre_du_jour", "Course_ID"], kind="stable"
    ).drop(columns=["_ordre_du_jour"]).reset_index(drop=True)

    print(f"\nRecalcul de l'historique sur {len(total):,} lignes "
          f"({total['Course_ID'].nunique():,} courses) — peut prendre quelques minutes...")

    print("  Historique cheval / jockey / entraîneur...")
    total = add_history_features(total)

    print("  Forme récente...")
    total = add_recent_form_features(total)

    print("  Variables de contexte...")
    total = add_context_features(total)

    print("  Features supplémentaires (Musique, Age, Sexe, Poids, ...)...")
    total = enrichir_features_supplementaires(total)

    jour_final = total[total["Course_ID"].isin(ids_jour)].copy()
    print(f"\n{len(jour_final):,} lignes du jour, features à jour.")

    return jour_final


# ============================================================
# FILTRE METIER : 8 A 16 PARTANTS
# ============================================================

def filtrer_taille_peloton(df):
    avant_courses = df["Course_ID"].nunique()
    df = df[
        (df["Nb_Partants"] >= PARTANTS_MIN) & (df["Nb_Partants"] <= PARTANTS_MAX)
    ].copy()
    apres_courses = df["Course_ID"].nunique()

    print(f"\nFiltre {PARTANTS_MIN}-{PARTANTS_MAX} partants : "
          f"{apres_courses}/{avant_courses} courses conservées")

    if df.empty:
        raise RuntimeError(
            f"Aucune course du jour n'a entre {PARTANTS_MIN} et {PARTANTS_MAX} partants."
        )

    return df


# ============================================================
# CHARGEMENT DU MODELE ET PREDICTION
# ============================================================

def charger_modele():
    meta = json.loads(FICHIER_FEATURES.read_text(encoding="utf-8"))
    features = meta["features"]
    colonnes_categorielles = meta["colonnes_categorielles"]
    categories_par_colonne = meta.get("categories_par_colonne", {})

    modele = xgb.XGBClassifier(enable_categorical=True)
    modele.load_model(str(FICHIER_MODELE))
    # IMPORTANT : load_model() restaure les poids du modèle mais PAS les
    # hyperparamètres du wrapper scikit-learn (dont enable_categorical).
    # Sans cette ligne, predict_proba() rejette toute colonne 'category'.
    modele.enable_categorical = True

    print(f"\nModèle chargé : {FICHIER_MODELE}")
    print(f"  {len(features)} features attendues")

    return modele, features, colonnes_categorielles, categories_par_colonne


def preparer_X(df, features, colonnes_categorielles, categories_par_colonne):
    df = df.copy()

    manquantes = [f for f in features if f not in df.columns]
    for f in manquantes:
        df[f] = np.nan
    if manquantes:
        print(f"  ATTENTION : {len(manquantes)} features absentes des courses du "
              f"jour, remplies en NaN : {manquantes[:10]}"
              + (" ..." if len(manquantes) > 10 else ""))

    for c in colonnes_categorielles:
        if c not in df.columns:
            continue

        valeurs = df[c].astype(str).fillna("INCONNU")
        categories_connues = categories_par_colonne.get(c)

        if categories_connues:
            # Une valeur jamais vue à l'entraînement (nouvel hippodrome,
            # nouvelle combinaison, etc.) est basculée sur "INCONNU"
            # plutôt que de faire planter XGBoost (catégorie inconnue).
            inconnues = ~valeurs.isin(categories_connues)
            if inconnues.any() and "INCONNU" in categories_connues:
                valeurs = valeurs.where(~inconnues, "INCONNU")
            df[c] = pd.Categorical(valeurs, categories=categories_connues)
        else:
            df[c] = valeurs.astype("category")

    # Filet de sécurité : toute feature qui n'est PAS catégorielle doit
    # être numérique pour XGBoost. On ne se fie plus au typage
    # automatique de pandas (une colonne vide à l'entraînement mais
    # remplie de texte à la prédiction, ou l'inverse, ferait planter le
    # modèle sinon).
    for c in features:
        if c in colonnes_categorielles or c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    return df[features]


def predire(modele, df, features, colonnes_categorielles, categories_par_colonne):
    X = preparer_X(df, features, colonnes_categorielles, categories_par_colonne)
    proba = modele.predict_proba(X)[:, 1]
    df = df.copy()
    df["Proba_Podium"] = proba
    return df


# ============================================================
# MISE EN FORME DU RESULTAT
# ============================================================

def construire_top3(df):
    lignes = []

    for course_id, g in df.groupby("Course_ID"):
        g = g.sort_values("Proba_Podium", ascending=False).reset_index(drop=True)
        g["Rang_Predit"] = g.index + 1

        for _, row in g.head(3).iterrows():
            lignes.append({
                "Date_Course": row["Date_Course"].date() if pd.notna(row["Date_Course"]) else None,
                "Hippodrome": row["Hippodrome_Canonique"],
                "Numero_Course": row["Numero_Course"],
                "Nb_Partants": int(row["Nb_Partants"]),
                "Rang_Predit": int(row["Rang_Predit"]),
                "Cheval": row["Cheval"],
                "Jockey": row["Jockey"],
                "Proba_Podium": round(float(row["Proba_Podium"]), 4),
            })

    resultat = pd.DataFrame(lignes)
    resultat = resultat.sort_values(
        ["Date_Course", "Hippodrome", "Numero_Course", "Rang_Predit"]
    ).reset_index(drop=True)

    return resultat


def afficher(resultat):
    print("\n" + "=" * 70)
    print("TOP 3 PREDIT (sans utiliser la cote PMU)")
    print("=" * 70)

    cles = ["Date_Course", "Hippodrome", "Numero_Course"]
    for (date, hippo, num), g in resultat.groupby(cles, sort=False):
        nb_partants = g["Nb_Partants"].iloc[0]
        print(f"\n{date} - {hippo} - Course {num}  ({nb_partants} partants)")
        for _, row in g.iterrows():
            print(f"   {row['Rang_Predit']}. {row['Cheval']:25s} "
                  f"(jockey : {row['Jockey']:20s}) "
                  f"proba top3 = {row['Proba_Podium']:.1%}")


# ============================================================
# MAIN
# ============================================================

def main():
    hist = charger_historique_base()
    jour = charger_courses_du_jour()

    jour_avec_features = recalculer_avec_historique(hist, jour)
    jour_filtre = filtrer_taille_peloton(jour_avec_features)

    modele, features, colonnes_categorielles, categories_par_colonne = charger_modele()
    jour_predit = predire(modele, jour_filtre, features, colonnes_categorielles, categories_par_colonne)

    resultat = construire_top3(jour_predit)
    afficher(resultat)

    resultat.to_csv(FICHIER_SORTIE, index=False, encoding="utf-8-sig")
    print(f"\nRésultat exporté : {FICHIER_SORTIE}")


if __name__ == "__main__":
    main()

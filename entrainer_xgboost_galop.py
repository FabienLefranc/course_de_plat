# -*- coding: utf-8 -*-
"""
ENTRAINER_XGBOOST_GALOP.PY
----------------------------
Entraîne un modèle XGBoost qui estime, pour chaque cheval partant,
sa probabilité de finir dans le TOP 3 (Target_Podium), SANS jamais
utiliser la cote PMU.

Entrée :
  Le dataset produit par generer_dataset_IA.py
  (dataset_galop_france.csv), enrichi ici avec des features
  supplémentaires (features_supplementaires.py) qui ne sont pas
  encore calculées par ce script : Age, Sexe, Poids, Corde, Musique,
  Inédit, Gains, historique carrière fourni par la source PMU.

Principes
---------
1. Cote_Direct et toute variable dérivée de la cote : JAMAIS utilisée.
2. Découpage TEMPOREL train / validation / test (jamais aléatoire) :
   une course de septembre ne doit jamais "voir" un résultat d'avril.
3. Le filtre "8 à 16 partants" n'est PAS appliqué à l'entraînement
   (le modèle apprend sur toutes les tailles de peloton pour avoir un
   maximum de données), mais reste disponible en option et est de
   toute façon la responsabilité du script de prédiction du jour.
4. Le modèle est un classifieur binaire (Target_Podium), plus simple
   à entraîner/valider/déboguer qu'un ranker, mais évalué comme un
   problème de RANKING intra-course : pour chaque course, on trie les
   chevaux par probabilité décroissante et on regarde combien des 3
   chevaux ainsi désignés sont réellement arrivés dans le vrai TOP 3
   (métrique Precision@3), en plus de l'AUC et du log loss classiques.

Sortie :
  modele_xgboost_top3.json     -> modèle XGBoost
  features_top3.json           -> liste des features + colonnes catégorielles
  rapport_entrainement.txt     -> métriques
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb

from features_supplementaires import (
    enrichir_features_supplementaires,
    COLONNES_CATEGORIELLES_SUPP,
    COLONNES_NUMERIQUES_SUPP,
)

# ============================================================
# CONFIGURATION
# ============================================================

DOSSIER_DATASET = Path(
    r"C:\Users\33662\OneDrive\Bureau\course_de_plat\data\dataset"
)
FICHIER_DATASET = DOSSIER_DATASET / "dataset_galop_france.csv"

DOSSIER_MODELE = Path(
    r"C:\Users\33662\OneDrive\Bureau\course_de_plat\data\modele"
)
FICHIER_MODELE = DOSSIER_MODELE / "modele_xgboost_top3.json"
FICHIER_FEATURES = DOSSIER_MODELE / "features_top3.json"
FICHIER_RAPPORT = DOSSIER_MODELE / "rapport_entrainement.txt"

CIBLE = "Target_Podium"

# Fraction des courses (les plus récentes) réservées à la validation
# et au test. Découpage temporel strict.
FRACTION_TEST = 0.15
FRACTION_VALIDATION = 0.15

# Cible dérivée du filtre métier (utilisée seulement pour un rapport
# séparé, PAS pour restreindre l'entraînement).
PARTANTS_MIN = 8
PARTANTS_MAX = 16

# Colonnes explicitement interdites (fuite ou identifiants).
COLONNES_INTERDITES = {
    "Cote_Direct", "Cote", "Cote_num",
    "Classement", "Classement_num", "Arrivée", "Rang",
    "Target_Victoire", "Target_Podium", "Target_Top5",
    "Course_ID", "Cheval_Norm", "Jockey_Norm", "Entraineur_Norm",
    "Hippodrome_Norm", "Date_Course", "Hippodrome",
    "Cheval", "Jockey", "Entraineur", "Proprietaire", "Nom",
    "Driver_Jockey", "Musique", "Surface_Source",
    "Numero_Course", "Num_PMU", "Reunion", "Course",
    "Age", "Sexe", "Poids", "Supplement", "Inedit",
    "Place_Corde", "Corde_Piste", "Discipline",
    "Gains_Carriere", "Gains_Victoires", "Gains_Place",
    "Gains_Annee_En_Cours", "Gains_Annee_Precedente",
    "Nb_Courses", "Nb_Victoires", "Nb_Places", "Nb_Places_2e", "Nb_Places_3e",
    "Allure", "Date",
    "Feature_Count_V2_9", "Feature_Count_V2_8",  # colonne technique, valeur constante
}

# Colonnes catégorielles déjà produites par generer_dataset_IA.py
# (Nature_Piste / Etat_Terrain sont explicitement listées ici : sans ça,
# elles ne deviennent "numériques" que par accident quand elles sont
# vides sur tout le dataset d'entraînement, ce qui plante dès qu'une
# vraie valeur texte apparaît à la prédiction.)
COLONNES_CATEGORIELLES_DATASET = [
    "Hippodrome_Canonique", "Type_Hippodrome",
    "Surface_Final", "Surface_Methode",
    "Distance_Classe", "Surface_Distance", "Surface_Oeilleres",
    "Distance_Oeilleres", "Oeilleres",
    "Nature_Piste", "Etat_Terrain",
]


# ============================================================
# CHARGEMENT ET PREPARATION
# ============================================================

def charger_dataset():
    print(f"Lecture du dataset : {FICHIER_DATASET}")
    df = pd.read_csv(FICHIER_DATASET, low_memory=False)
    print(f"  {len(df):,} lignes / {len(df.columns)} colonnes")

    df["Date_Course"] = pd.to_datetime(df["Date_Course"], errors="coerce")

    if CIBLE not in df.columns:
        raise ValueError(f"Colonne cible '{CIBLE}' absente du dataset.")

    avant = len(df)
    df = df[df[CIBLE].notna()].copy()
    print(f"  Lignes avec résultat connu : {len(df):,} / {avant:,}")

    return df


def construire_features(df):
    print("Ajout des features supplémentaires (Musique, Age, Sexe, "
          "Poids, Corde, Gains, carrière PMU)...")
    df = enrichir_features_supplementaires(df)

    # Penetrometre est une vraie mesure numérique (ex: 3.5), mais selon
    # les exports elle peut être lue comme texte ou comme colonne
    # entièrement vide -> on la force explicitement en numérique pour
    # ne pas dépendre du hasard du typage pandas.
    if "Penetrometre" in df.columns:
        df["Penetrometre"] = pd.to_numeric(
            df["Penetrometre"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    colonnes_categorielles = [
        c for c in COLONNES_CATEGORIELLES_DATASET + COLONNES_CATEGORIELLES_SUPP
        if c in df.columns
    ]

    candidates = []
    for c in df.columns:
        if c in COLONNES_INTERDITES:
            continue
        if c in colonnes_categorielles:
            candidates.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)

    # On force le typage 'category' pour que XGBoost gère nativement
    # les variables catégorielles (enable_categorical=True).
    categories_par_colonne = {}
    for c in colonnes_categorielles:
        df[c] = df[c].astype(str).fillna("INCONNU").astype("category")
        categories_par_colonne[c] = df[c].cat.categories.tolist()

    print(f"  Features retenues : {len(candidates):,} "
          f"({len(colonnes_categorielles)} catégorielles, "
          f"{len(candidates) - len(colonnes_categorielles)} numériques)")

    if "Cote_Direct" in candidates:
        raise AssertionError("ERREUR : Cote_Direct s'est glissée dans les features.")

    return df, candidates, colonnes_categorielles, categories_par_colonne


# ============================================================
# DECOUPAGE TEMPOREL (par COURSE, jamais par ligne isolée)
# ============================================================

def decoupage_temporel(df):
    courses = (
        df[["Course_ID", "Date_Course"]]
        .drop_duplicates()
        .sort_values("Date_Course", kind="stable")
        .reset_index(drop=True)
    )

    n = len(courses)
    n_test = max(1, int(n * FRACTION_TEST))
    n_val = max(1, int(n * FRACTION_VALIDATION))
    n_train = n - n_test - n_val

    if n_train <= 0:
        raise ValueError(
            "Pas assez de courses pour un découpage temporel "
            "train/validation/test. Réduis FRACTION_TEST/FRACTION_VALIDATION."
        )

    courses_train = set(courses["Course_ID"].iloc[:n_train])
    courses_val = set(courses["Course_ID"].iloc[n_train:n_train + n_val])
    courses_test = set(courses["Course_ID"].iloc[n_train + n_val:])

    date_min = courses["Date_Course"].min()
    date_sep_val = courses["Date_Course"].iloc[n_train]
    date_sep_test = courses["Date_Course"].iloc[n_train + n_val]
    date_max = courses["Date_Course"].max()

    print("\nDécoupage temporel :")
    print(f"  Train      : {n_train:,} courses  ({date_min.date()} -> {date_sep_val.date()})")
    print(f"  Validation : {n_val:,} courses  (-> {date_sep_test.date()})")
    print(f"  Test       : {n_test:,} courses  (-> {date_max.date()})")

    train = df[df["Course_ID"].isin(courses_train)].copy()
    val = df[df["Course_ID"].isin(courses_val)].copy()
    test = df[df["Course_ID"].isin(courses_test)].copy()

    return train, val, test


# ============================================================
# ENTRAINEMENT
# ============================================================

def entrainer(train, val, features, cible=CIBLE):
    X_train, y_train = train[features], train[cible].astype(int)
    X_val, y_val = val[features], val[cible].astype(int)

    # Déséquilibre de classes : la classe "TOP3" est minoritaire dès
    # que le peloton dépasse 6 chevaux.
    taux_positifs = y_train.mean()
    scale_pos_weight = (1 - taux_positifs) / max(taux_positifs, 1e-6)
    print(f"\nTaux de TOP3 dans le train : {taux_positifs:.1%} "
          f"-> scale_pos_weight = {scale_pos_weight:.2f}")

    modele = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        reg_alpha=0.1,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=75,
        n_jobs=-1,
        random_state=42,
    )

    modele.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    print(f"\nMeilleure itération : {modele.best_iteration} "
          f"(sur {modele.n_estimators} max)")

    return modele


# ============================================================
# EVALUATION
# ============================================================

def precision_top3_par_course(df, proba_col="Proba_Podium", cible=CIBLE):
    """
    Pour chaque course : on prend les 3 chevaux avec la plus forte
    probabilité prédite, et on regarde combien sont réellement dans
    le vrai TOP 3 (cible=1).
    """
    resultats = []

    for course_id, g in df.groupby("Course_ID"):
        g = g.sort_values(proba_col, ascending=False)
        top3_predits = g.head(3)
        bons = int(top3_predits[cible].sum())
        resultats.append({
            "Course_ID": course_id,
            "Nb_Partants": len(g),
            "Bons_dans_top3_predit": bons,
        })

    res = pd.DataFrame(resultats)
    res["Precision_At_3"] = res["Bons_dans_top3_predit"] / 3.0

    return {
        "nb_courses": len(res),
        "precision_at_3_moyenne": float(res["Precision_At_3"].mean()),
        "taux_au_moins_1_bon": float((res["Bons_dans_top3_predit"] >= 1).mean()),
        "taux_au_moins_2_bons": float((res["Bons_dans_top3_predit"] >= 2).mean()),
        "taux_3_sur_3": float((res["Bons_dans_top3_predit"] == 3).mean()),
    }, res


def evaluer(modele, df, features, nom, cible=CIBLE):
    proba = modele.predict_proba(df[features])[:, 1]
    df = df.copy()
    df["Proba_Podium"] = proba

    y = df[cible].astype(int)
    auc = roc_auc_score(y, proba)
    ll = log_loss(y, proba)

    metier, _ = precision_top3_par_course(df, cible=cible)

    print(f"\n--- Évaluation [{nom}] ---")
    print(f"  AUC                         : {auc:.4f}")
    print(f"  Log loss                    : {ll:.4f}")
    print(f"  Courses évaluées            : {metier['nb_courses']:,}")
    print(f"  Precision@3 moyenne         : {metier['precision_at_3_moyenne']:.3f}  "
          f"(sur 3 chevaux prédits, combien sont réellement dans le vrai top3)")
    print(f"  Au moins 1 bon dans le top3 : {metier['taux_au_moins_1_bon']:.1%}")
    print(f"  Au moins 2 bons dans le top3: {metier['taux_au_moins_2_bons']:.1%}")
    print(f"  Top3 exact (3/3)            : {metier['taux_3_sur_3']:.1%}")

    # Rapport spécifique au filtre métier 8-16 partants.
    sous_ensemble = df[
        (df["Nb_Partants"] >= PARTANTS_MIN) & (df["Nb_Partants"] <= PARTANTS_MAX)
    ]
    if not sous_ensemble.empty:
        metier_filtre, _ = precision_top3_par_course(sous_ensemble, cible=cible)
        print(f"  -- Sur courses {PARTANTS_MIN}-{PARTANTS_MAX} partants "
              f"({metier_filtre['nb_courses']:,} courses) --")
        print(f"     Precision@3 : {metier_filtre['precision_at_3_moyenne']:.3f}  "
              f"| au moins 1 bon : {metier_filtre['taux_au_moins_1_bon']:.1%}")

    return {
        "nom": nom,
        "auc": auc,
        "log_loss": ll,
        **metier,
    }


# ============================================================
# IMPORTANCE DES FEATURES
# ============================================================

def afficher_importance(modele, features, top_n=25):
    importances = modele.feature_importances_
    ordre = np.argsort(importances)[::-1][:top_n]

    print(f"\nTop {top_n} features (gain) :")
    for i in ordre:
        print(f"  {features[i]:40s} {importances[i]:.4f}")


# ============================================================
# EXPORT
# ============================================================

def sauvegarder(modele, features, colonnes_categorielles, categories_par_colonne, metriques):
    DOSSIER_MODELE.mkdir(parents=True, exist_ok=True)

    modele.save_model(str(FICHIER_MODELE))

    meta = {
        "cible": CIBLE,
        "features": features,
        "colonnes_categorielles": colonnes_categorielles,
        "categories_par_colonne": categories_par_colonne,
        "partants_min": PARTANTS_MIN,
        "partants_max": PARTANTS_MAX,
    }
    FICHIER_FEATURES.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(FICHIER_RAPPORT, "w", encoding="utf-8") as f:
        for m in metriques:
            f.write(f"=== {m['nom']} ===\n")
            for k, v in m.items():
                if k == "nom":
                    continue
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"\nModèle sauvegardé  : {FICHIER_MODELE}")
    print(f"Métadonnées        : {FICHIER_FEATURES}")
    print(f"Rapport             : {FICHIER_RAPPORT}")


# ============================================================
# MAIN
# ============================================================

def main():
    df = charger_dataset()
    df, features, colonnes_categorielles, categories_par_colonne = construire_features(df)

    train, val, test = decoupage_temporel(df)
    print(f"\nLignes -> train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")

    modele = entrainer(train, val, features)

    metriques = [
        evaluer(modele, val, features, "VALIDATION"),
        evaluer(modele, test, features, "TEST (courses les plus récentes)"),
    ]

    afficher_importance(modele, features)

    sauvegarder(modele, features, colonnes_categorielles, categories_par_colonne, metriques)


if __name__ == "__main__":
    main()

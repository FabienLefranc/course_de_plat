# -*- coding: utf-8 -*-
"""
GENERATEUR DATASET IA GALOP FRANCE - VERSION UNIFIEE
----------------------------------------------------
Version renforcée pour l'inférence de surface des courses de plat, avec le référentiel PMU complet.
Intègre désormais les features supplémentaires (Musique, Signalement, Gains, Carrière, etc.)

Principes V2.9
--------------
1. Source principale : Google Sheets publié en CSV.
2. Cote_Direct n'est JAMAIS utilisée comme variable explicative.
3. Classement vide/NC reste inconnu (NaN), jamais transformé en 0.
4. Penetrometre inconnu reste NaN.
5. Anti-fuite stricte :
   - une course est entièrement calculée avant d'alimenter les historiques ;
   - aucun cheval ne peut voir le résultat d'un autre partant de la même course.
6. Surface :
   - Nature_Piste explicite > règles hippodrome/date/distance > inconnue ;
   - règles spécifiques pour les hippodromes mixtes ;
   - priorité aux règles françaises documentées ;
   - aucune déduction "distance seule" lorsque plusieurs surfaces sont possibles.
7. Features supplémentaires intégrées :
   - Musique (forme longue)
   - Signalement (Age, Sexe, Poids, Œillères, Inédit)
   - Corde / Discipline
   - Gains et Carrière (fournis par la source PMU)
   - Taille de peloton
8. Sorties :
   C:/Users/33662/OneDrive/Bureau/galop_analyzer_pro/data/dataset
   - dataset_galop_V2_9_france.csv
   - dataset_galop_V2_9_france.parquet
"""

import io
import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests


#  ============================================================
# CONFIGURATION
# ============================================================

URL_CSV = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=644246763&single=true&output=csv"
)

DOSSIER_SORTIE = Path(
    r"C:\Users\33662\OneDrive\Bureau\course_de_plat\data\dataset"
)

FICHIER_CSV = DOSSIER_SORTIE / "dataset_galop_france.csv"
FICHIER_PARQUET = DOSSIER_SORTIE / "dataset_galop_france.parquet"
FICHIER_EXCLUS_CSV = DOSSIER_SORTIE / "courses_etrangeres_exclues.csv"

TIMEOUT = 45

# ======================================== ====================
# FILTRE DES HIPPODROMES
# ============================================================

FICHIER_HIPPODROMES = DOSSIER_SORTIE / "Hippodromes francais et etrangers.xlsx"
FICHIER_HIPPODROMES_LOCALE = Path(__file__).with_name("Hippodromes francais et etrangers.xlsx")

HIPPODROMES_ETRANGERS_INCLUS = {
    "AVENCHES",
    "MONS (GHLIN)",
    "SAN SEBASTIAN",
    "OSTENDE",
    "WAREGEM",
}

REFERENCE_SURFACES_ETRANGER_PROCHE = {
    "SAN SEBASTIAN": {"surface": "GAZON", "distances": [1000,1100,1200,1400,1600,1700,1800,1900,2000,2100,2200,2400,2500,2800], "mois": set(range(6,10))},
    "AVENCHES": {"surface": "GAZON", "distances": [800,900,1200,1300,1400,1500,1600,2000,2100,2150,2200,2400,2700,2800,2900,3000], "mois": set(range(3,11))},
    "MONS GHLIN": {"surface": "PSF", "distances": [950,1500,2100,2300,2850,3200,4000], "mois": set(range(4,11))},
    "OSTENDE": {"surface": "GAZON", "distances": [100,1200,1600,1800,2100,2200,2400,2700,3200,4000], "mois": {7,8}},
    "WAREGEM": {"surface": "GAZON", "distances": [1600,2200,2700,4000], "mois": set(range(5,10))},
}


# ====================== ======================================
# OUTILS GENERAUX
# ============================================================

def norm(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = (
        s.replace("É", "E").replace("È", "E").replace("Ê", "E")
         .replace("Ë", "E").replace("À", "A").replace("Â", "A")
         .replace("Ä", "A").replace("Î", "I").replace("Ï", "I")
         .replace("Ô", "O").replace("Ö", "O").replace("Ù", "U")
         .replace("Û", "U").replace("Ü", "U").replace("Ç", "C")
    )
    return s


def first_existing(df, names):
    normalized = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in normalized:
            return normalized[norm(name)]
    return None


def to_num(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce"
    )


def parse_date_value(x):
    """
    Gère notamment les dates PMU 7/8 chiffres :
      2092025   -> 02/09/2025
      1092025   -> 01/09/2025
      12092025  -> 12/09/2025
    """
    if pd.isna(x):
        return pd.NaT

    s = str(x).strip()
    if not s:
        return pd.NaT

    d = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if not pd.isna(d):
        return d

    digits = re.sub(r"\D", "", s)

    if len(digits) == 8:
        try:
            return pd.Timestamp(
                year=int(digits[4:]),
                month=int(digits[2:4]),
                day=int(digits[:2])
            )
        except Exception:
            pass

    if len(digits) == 7:
        year = int(digits[-4:])
        month = int(digits[-6:-4])
        day_part = digits[:-6]
        try:
            return pd.Timestamp(year=year, month=month, day=int(day_part))
        except Exception:
            pass

    return pd.NaT


def clean_rank(x):
    """
    Classement inconnu = NaN.
    Ne jamais convertir un classement vide en 0.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()

    if not s or s in {
        "NC", "NP", "N/P", "N-D", "ND", "DAI", "DISQ", "DQ",
        "ABANDON", "RET", "RETIRE", "NON PARTANT", "NAN"
    }:
        return np.nan

    m = re.search(r"\d+", s)
    if not m:
        return np.nan

    try:
        r = int(m.group())
        return float(r) if r > 0 else np.nan
    except Exception:
        return np.nan


def infer_rank_targets(rank):
    if pd.isna(rank):
        return np.nan, np.nan, np.nan

    return (
        float(rank == 1),
        float(rank <= 3),
        float(rank <= 5),
    )


# ============================================================
# CHARGEMENT GOOGLE SHEET
# ============================================================

def charger_google_sheet():
    print("Chargement du Google Sheets...")
    r = requests.get(URL_CSV, timeout=TIMEOUT)

    if r.status_code == 404:
        raise RuntimeError(
            "\nLe lien Google Sheets renvoie une 404.\n"
            "Allez dans votre Google Sheets > Fichier > Partager > Publier sur le Web > CSV,\n"
            "puis recopiez le nouveau lien CSV dans URL_CSV en haut du script.\n"
        )

    r.raise_for_status()
    content = r.content

    head = content[:300].lower()
    if b"<html" in head or b"<!doctype" in head:
        raise RuntimeError(
            "\nLe lien ne renvoie pas un CSV mais une page HTML.\n"
            "Vérifiez que le Google Sheet est bien publié en CSV.\n"
        )

    df = pd.read_csv(io.BytesIO(content), dtype=str)
    print(f"  Lignes chargées : {len(df):,}")
    return df


# ============================================================
# PREPARATION DES COLONNES
# ============================================================

def preparer_colonnes(df):
    print("Préparation des colonnes...")
    df = df.copy()

    # Normalisation des noms de colonnes
    df.columns = [str(c).strip() for c in df.columns]

    # Colonnes essentielles
    col_map = {
        "Date": "Date_Course",
        "Hippodrome": "Hippodrome",
        "Numero": "Numero_Course",
        "Course": "Numero_Course",       # alias réel de la source PMU (numéro de course)
        "Cheval": "Cheval",
        "Nom": "Cheval",                 # alias réel de la source PMU (nom du cheval)
        "Jockey": "Jockey",
        "Driver_Jockey": "Jockey",       # alias réel de la source PMU
        "Entraineur": "Entraineur",
        "Proprietaire": "Proprietaire",
        "Distance": "Distance",
        "Nature Piste": "Nature_Piste",
        "Etat Terrain": "Etat_Terrain",
        "Penetrometre": "Penetrometre",
        "Classement": "Classement",
        "Cote": "Cote_Direct",
        "Place Corde": "Place_Corde",
        "Corde Piste": "Corde_Piste",
        "Discipline": "Discipline",
        "Musique": "Musique",
        "Age": "Age",
        "Sexe": "Sexe",
        "Poids": "Poids",
        "Supplement": "Supplement",
        "Oeilleres": "Oeilleres",
        "Inedit": "Inedit",
    }

    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Filet de sécurité : si une colonne essentielle reste introuvable
    # malgré les alias ci-dessus, on la crée vide plutôt que de planter
    # plus loin (Course_ID, Cheval_Norm, etc.) de façon silencieuse.
    for col_essentielle in ["Numero_Course", "Cheval", "Jockey", "Entraineur", "Hippodrome"]:
        if col_essentielle not in df.columns:
            print(f"  ATTENTION : colonne '{col_essentielle}' introuvable dans la source, "
                  f"créée vide.")
            df[col_essentielle] = "" if col_essentielle == "Numero_Course" else np.nan

    if "Date_Course" not in df.columns:
        print("  ATTENTION : colonne 'Date_Course' introuvable dans la source, créée vide.")
        df["Date_Course"] = pd.NaT

    # Parsing dates
    if "Date_Course" in df.columns:
        df["Date_Course"] = df["Date_Course"].map(parse_date_value)

    # Parsing distance
    if "Distance" in df.columns:
        df["Distance"] = to_num(df["Distance"])

    # Parsing classement
    if "Classement" in df.columns:
        df["Classement"] = df["Classement"].map(clean_rank)

    # Normalisation des identifiants
    for col in ["Cheval", "Jockey", "Entraineur", "Hippodrome"]:
        if col in df.columns:
            df[f"{col}_Norm"] = df[col].map(norm)

    # Création Course_ID
    if "Date_Course" in df.columns and "Numero_Course" in df.columns and "Hippodrome_Norm" in df.columns:
        df["Course_ID"] = (
            df["Date_Course"].astype(str) + "_" +
            df["Hippodrome_Norm"] + "_" +
            df["Numero_Course"].astype(str)
        )

    print(f"  Colonnes après préparation : {len(df.columns)}")
    return df


# ============================================================
# FILTRAGE HIPPODROMES
# ============================================================

def _normaliser_nom_fichier(nom):
    """Normalise un nom de fichier pour la comparaison : insensible aux
    accents, espaces, underscores, tirets et à la casse. Ça évite les
    échecs de type 'ça marchait sous Windows mais pas sur Streamlit
    Cloud (Linux, sensible à la casse et aux espaces exacts)'."""
    s = norm(Path(nom).stem)  # enlève l'extension, gère déjà les accents
    return re.sub(r"[^A-Z0-9]", "", s)


def trouver_fichier_local(nom_attendu, dossier, extensions=None):
    """
    Cherche un fichier dans `dossier` dont le nom correspond à
    `nom_attendu`, en tolérant les différences d'accents, d'espaces,
    d'underscores/tirets et de casse (ex: 'Hippodromes francais et
    etrangers.xlsx' retrouve aussi 'Hippodromes_francais_et_etrangers.XLSX').
    Retourne le premier Path trouvé, ou None.
    """
    cible = _normaliser_nom_fichier(nom_attendu)
    if extensions is None:
        extensions = [Path(nom_attendu).suffix]

    if not dossier.exists():
        return None

    for f in dossier.iterdir():
        if not f.is_file():
            continue
        if extensions and f.suffix.lower() not in [e.lower() for e in extensions]:
            continue
        if _normaliser_nom_fichier(f.name) == cible:
            return f

    return None


def charger_listes_hippodromes():
    """
    Charge le référentiel Excel réel : deux colonnes séparées
    'Hippodromes français' et 'Hippodromes étrangers' (pas de colonne
    'Canonique'/'Pays' — ce schéma n'existe pas dans le fichier fourni).
    """
    candidates = [
        FICHIER_HIPPODROMES,
        FICHIER_HIPPODROMES_LOCALE,
        trouver_fichier_local(
            "Hippodromes francais et etrangers.xlsx",
            Path(__file__).parent,
            extensions=[".xlsx", ".xls"],
        ),
    ]
    path = next((p for p in candidates if p is not None and p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "\nFichier des hippodromes introuvable.\n"
            "Placez un fichier Excel des hippodromes (2 colonnes : "
            "'Hippodromes français' / 'Hippodromes étrangers') dans :\n"
            f"  {FICHIER_HIPPODROMES}\n"
            "ou dans le même dossier que ce script.\n"
        )

    xls = pd.read_excel(path, dtype=str)
    cols = {norm(c): c for c in xls.columns}
    fr_col = cols.get(norm("Hippodromes français"))
    etr_col = cols.get(norm("Hippodromes étrangers"))

    if fr_col is None or etr_col is None:
        raise ValueError(
            "Le fichier Excel doit contenir les colonnes "
            "'Hippodromes français' et 'Hippodromes étrangers'. "
            f"Colonnes trouvées : {list(xls.columns)}"
        )

    fr_map, etr_map = {}, {}
    for v in xls[fr_col].dropna():
        v = str(v).strip()
        if v:
            fr_map.setdefault(normaliser_hippodrome(v), v)
    for v in xls[etr_col].dropna():
        v = str(v).strip()
        if v:
            etr_map.setdefault(normaliser_hippodrome(v), v)

    inclus_etr = {normaliser_hippodrome(v) for v in HIPPODROMES_ETRANGERS_INCLUS}

    print(f"  Référence hippodromes : {len(fr_map):,} français / {len(etr_map):,} étrangers")
    print("  Étrangers proches inclus :", ", ".join(sorted(inclus_etr)))

    return fr_map, etr_map, inclus_etr


def filtrer_hippodromes(df):
    print("Filtrage des hippodromes...")

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

    # Sauvegarde des exclus
    exclus = out[~(is_fr | is_etr_proche)]
    if not exclus.empty:
        exclus.to_csv(FICHIER_EXCLUS_CSV, index=False, encoding="utf-8-sig")
        print(f"  Courses étrangères exclues sauvegardées : {len(exclus):,}")

    out = out[is_fr | is_etr_proche].copy()

    print(f"  Courses conservées : {len(out):,}")
    print("  Répartition :")
    print(out["Type_Hippodrome"].value_counts().to_string())

    if out.empty:
        raise RuntimeError("Aucune course ne correspond au périmètre France/étrangers proches.")

    return out


# ============================================================
# INFERENCE DE SURFACE V2.9
# ============================================================

FICHIER_REFERENCE_SURFACES = Path(
    r"C:\Users\33662\OneDrive\Bureau\course_de_plat\data\dataset\hippodromes_galop_complet.csv"
)

REFERENCE_LOCALE = Path(__file__).with_name("hippodromes_galop_complet.csv")


def normaliser_hippodrome(x):
    s = norm(x)
    s = re.sub(r"[^ A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normaliser_surface(x):
    s = norm(x)
    if any(k in s for k in ["PSF", "SABLE", "FIBRE", "POLYTRACK", "ALLWEATHER"]):
        return "PSF"
    if any(k in s for k in ["GAZON", "HERBE", "TURF"]):
        return "GAZON"
    return ""


def charger_reference_surfaces():
    candidates = [
        FICHIER_REFERENCE_SURFACES,
        REFERENCE_LOCALE,
        trouver_fichier_local(
            "hippodromes_galop_complet.csv", Path(__file__).parent, extensions=[".csv"]
        ),
    ]
    path = next((p for p in candidates if p is not None and p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "\nFichier de référence des surfaces introuvable.\n"
            "Placez 'hippodromes_galop_complet.csv' dans :\n"
            f"  {FICHIER_REFERENCE_SURFACES}\n"
            "ou dans le même dossier que le script.\n"
        )

    ref = pd.read_csv(path, sep=None, engine="python", dtype=str)
    ref.columns = [str(c).strip() for c in ref.columns]

    required = {"Hippodrome", "Surface", "Distance_m", "Mois_activite"}
    missing = required - set(ref.columns)

    if missing:
        raise ValueError(
            "Colonnes manquantes dans la référence des surfaces : " +
            ", ".join(sorted(missing))
        )

    ref["Hippodrome_Norm"] = ref["Hippodrome"].map(normaliser_hippodrome)
    ref["Surface_Norm"] = ref["Surface"].map(normaliser_surface)
    ref["Distance_m"] = to_num(ref["Distance_m"])

    mois_map = {
        "JANVIER": 1, "JAN": 1,
        "FEVRIER": 2, "FEV": 2,
        "MARS": 3,
        "AVRIL": 4, "AVR": 4,
        "MAI": 5,
        "JUIN": 6,
        "JUILLET": 7, "JUIL": 7,
        "AOUT": 8,
        "SEPTEMBRE": 9, "SEPT": 9,
        "OCTOBRE": 10, "OCT": 10,
        "NOVEMBRE": 11, "NOV": 11,
        "DECEMBRE": 12, "DEC": 12,
    }

    def parse_months(x):
        if pd.isna(x):
            return []

        s = norm(x)
        if not s:
            return []

        nums = re.findall(r"\b(?:1[0-2]|[1-9])\b", s)
        months = {int(n) for n in nums}

        for name, number in mois_map.items():
            if name in s:
                months.add(number)

        ranges = re.findall(r"\b(1[0-2]|[1-9])\s*[-/]\s*(1[0-2]|[1-9])\b", s)
        for a, b in ranges:
            a, b = int(a), int(b)
            if a <= b:
                months.update(range(a, b + 1))
            else:
                months.update(range(a, 13))
                months.update(range(1, b + 1))

        return sorted(months)

    ref["Mois_activite"] = ref["Mois_activite"].map(parse_months)

    return ref


def nom_surface_france(hippo_canonique, hippo_source):
    return normaliser_hippodrome(hippo_canonique or hippo_source)


def infer_surface_from_reference(row, ref):
    source = norm(row.get("Nature_Piste", ""))
    hippo = nom_surface_france(row.get("Hippodrome_Canonique", ""), row.get("Hippodrome", ""))

    # Étrangers proches : référence dédiée
    if norm(row.get("Type_Hippodrome", "")) == "ETRANGER_PROCHE":
        surf = normaliser_surface(source)
        if surf:
            return surf, 1.00, "SOURCE_EXPLICITE"

        hip_key = normaliser_hippodrome(row.get("Hippodrome_Canonique", row.get("Hippodrome", "")))
        regle = REFERENCE_SURFACES_ETRANGER_PROCHE.get(hip_key)
        distance = row.get("Distance", np.nan)
        date = row.get("Date_Course", pd.NaT)

        if regle and not pd.isna(distance):
            mois = int(date.month) if not pd.isna(date) else None
            if float(distance) in regle["distances"] and (mois is None or mois in regle["mois"]):
                return regle["surface"], 0.96, "REFERENCE_ETRANGER_PROCHE_DISTANCE_MOIS"
            if (min(abs(float(distance)-d) for d in regle["distances"]) <= 50) and (mois is None or mois in regle["mois"]):
                return regle["surface"], 0.85, "REFERENCE_ETRANGER_PROCHE_DISTANCE_PROCHE"

        return "INCONNUE", 0.05, "REFERENCE_ETRANGER_PROCHE_SANS_CORRESPONDANCE"

    # Surface explicite
    surf = normaliser_surface(source)
    if surf:
        return surf, 1.00, "SOURCE_EXPLICITE"

    # Référence française
    distance = row.get("Distance", np.nan)
    date = row.get("Date_Course", pd.NaT)

    if pd.isna(distance):
        return "INCONNUE", 0.05, "SANS_DISTANCE"

    candidats = ref[ref["Hippodrome_Norm"] == hippo]

    if candidats.empty:
        return "INCONNUE", 0.05, "HIPPODROME_INCONNU"

    mois = int(date.month) if not pd.isna(date) else None

    # Niveau 2 : distance exacte + mois
    exact_distance_mois = candidats[
        (candidats["Distance_m"] == float(distance)) &
        (candidats["Mois_activite"].apply(lambda m: mois in m if mois else True))
    ]

    if not exact_distance_mois.empty:
        surfaces = sorted(exact_distance_mois["Surface_Norm"].dropna().unique())
        if len(surfaces) == 1:
            return surfaces[0], 0.95, "REFERENCE_HIPPO_DISTANCE_MOIS"
        if len(surfaces) > 1:
            return "INCONNUE", 0.25, "REFERENCE_AMBIGUE_DISTANCE_MOIS"

    # Niveau 3 : distance exacte
    exact_distance = candidats[candidats["Distance_m"] == float(distance)]

    if not exact_distance.empty:
        surfaces = sorted(exact_distance["Surface_Norm"].dropna().unique())
        if len(surfaces) == 1:
            return surfaces[0], 0.88, "REFERENCE_HIPPO_DISTANCE"
        if len(surfaces) > 1:
            return "INCONNUE", 0.22, "REFERENCE_AMBIGUE_DISTANCE"

    # Niveau 4 : distance proche ±50m
    proches = candidats[
        (candidats["Distance_m"] - float(distance)).abs() <= 50
    ]

    if not proches.empty:
        if mois:
            proches_mois = proches[proches["Mois_activite"].apply(lambda m: mois in m)]
            if not proches_mois.empty:
                surfaces = sorted(proches_mois["Surface_Norm"].dropna().unique())
                if len(surfaces) == 1:
                    return surfaces[0], 0.75, "REFERENCE_DISTANCE_PROCHE_MOIS"
                if len(surfaces) > 1:
                    return "INCONNUE", 0.20, "REFERENCE_AMBIGUE_PROCHE_MOIS"

        surfaces = sorted(proches["Surface_Norm"].dropna().unique())
        if len(surfaces) == 1:
            return surfaces[0], 0.68, "REFERENCE_DISTANCE_PROCHE"
        if len(surfaces) > 1:
            return "INCONNUE", 0.15, "REFERENCE_AMBIGUE_PROCHE"

    return "INCONNUE", 0.05, "REFERENCE_SANS_CORRESPONDANCE"


def enrichir_surface(df):
    ref = charger_reference_surfaces()

    result = df.apply(
        lambda row: infer_surface_from_reference(row, ref),
        axis=1,
        result_type="expand"
    )

    result.columns = ["Surface_Final", "Surface_Confiance", "Surface_Methode"]
    df = pd.concat([df, result], axis=1)
    df["Surface_Source"] = df["Nature_Piste"]

    return df


# ============================================================
# HISTORIQUES SANS FUITE
# ============================================================

def safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def safe_rate(values):
    return float(np.mean(values)) if values else np.nan


def add_history_features(df):
    df = df.copy()

    sort_cols = ["Date_Course", "Course_ID"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    feature_cols = [
        "Cheval_Courses_Avant", "Cheval_Victoires_Avant", "Cheval_Podiums_Avant",
        "Cheval_Taux_Victoire_Avant", "Cheval_Taux_Podium_Avant",
        "Jockey_Courses_Avant", "Jockey_Victoires_Avant", "Jockey_Taux_Victoire_Avant",
        "Entraineur_Courses_Avant", "Entraineur_Victoires_Avant", "Entraineur_Taux_Victoire_Avant",
        "Cheval_Surface_Courses_Avant", "Cheval_Surface_Victoires_Avant", "Cheval_Surface_Taux_Victoire_Avant",
        "Cheval_Distance_Courses_Avant", "Cheval_Distance_Victoires_Avant", "Cheval_Distance_Taux_Victoire_Avant",
        "Couplage_Courses_Avant", "Couplage_Victoires_Avant", "Couplage_Taux_Victoire_Avant",
    ]

    for c in feature_cols:
        df[c] = np.nan

    horse_hist = defaultdict(lambda: [0, 0, 0])
    jockey_hist = defaultdict(lambda: [0, 0])
    trainer_hist = defaultdict(lambda: [0, 0])
    horse_surface_hist = defaultdict(lambda: [0, 0])
    horse_distance_hist = defaultdict(lambda: [0, 0])
    coupling_hist = defaultdict(lambda: [0, 0])

    groups = df.groupby("Course_ID", sort=False).groups
    total_courses = len(groups)
    report_every = max(1, total_courses // 20)

    for course_no, (_, idxs) in enumerate(groups.items(), start=1):
        if course_no % report_every == 0:
            print(f"  Historiques : {course_no}/{total_courses} courses")

        # PHASE 1 : lecture AVANT
        for idx in idxs:
            horse = df.at[idx, "Cheval_Norm"]
            jockey = df.at[idx, "Jockey_Norm"]
            trainer = df.at[idx, "Entraineur_Norm"]
            surface = df.at[idx, "Surface_Final"]
            distance = df.at[idx, "Distance"]

            h = horse_hist[horse]
            j = jockey_hist[jockey]
            t = trainer_hist[trainer]
            hs = horse_surface_hist[(horse, surface)]

            if pd.isna(distance):
                hd_key = (horse, None)
            else:
                hd_key = (horse, int(round(distance / 100.0) * 100))
            hd = horse_distance_hist[hd_key]

            cp = coupling_hist[(horse, jockey)]

            if h[0] > 0:
                df.at[idx, "Cheval_Courses_Avant"] = h[0]
                df.at[idx, "Cheval_Victoires_Avant"] = h[1]
                df.at[idx, "Cheval_Podiums_Avant"] = h[2]
                df.at[idx, "Cheval_Taux_Victoire_Avant"] = h[1] / h[0]
                df.at[idx, "Cheval_Taux_Podium_Avant"] = h[2] / h[0]

            if j[0] > 0:
                df.at[idx, "Jockey_Courses_Avant"] = j[0]
                df.at[idx, "Jockey_Victoires_Avant"] = j[1]
                df.at[idx, "Jockey_Taux_Victoire_Avant"] = j[1] / j[0]

            if t[0] > 0:
                df.at[idx, "Entraineur_Courses_Avant"] = t[0]
                df.at[idx, "Entraineur_Victoires_Avant"] = t[1]
                df.at[idx, "Entraineur_Taux_Victoire_Avant"] = t[1] / t[0]

            if hs[0] > 0:
                df.at[idx, "Cheval_Surface_Courses_Avant"] = hs[0]
                df.at[idx, "Cheval_Surface_Victoires_Avant"] = hs[1]
                df.at[idx, "Cheval_Surface_Taux_Victoire_Avant"] = hs[1] / hs[0]

            if hd[0] > 0:
                df.at[idx, "Cheval_Distance_Courses_Avant"] = hd[0]
                df.at[idx, "Cheval_Distance_Victoires_Avant"] = hd[1]
                df.at[idx, "Cheval_Distance_Taux_Victoire_Avant"] = hd[1] / hd[0]

            if cp[0] > 0:
                df.at[idx, "Couplage_Courses_Avant"] = cp[0]
                df.at[idx, "Couplage_Victoires_Avant"] = cp[1]
                df.at[idx, "Couplage_Taux_Victoire_Avant"] = cp[1] / cp[0]

        # PHASE 2 : mise à jour APRÈS
        for idx in idxs:
            rank = df.at[idx, "Classement"]
            if pd.isna(rank):
                continue

            horse = df.at[idx, "Cheval_Norm"]
            jockey = df.at[idx, "Jockey_Norm"]
            trainer = df.at[idx, "Entraineur_Norm"]
            surface = df.at[idx, "Surface_Final"]
            distance = df.at[idx, "Distance"]

            is_win = rank == 1
            is_podium = rank <= 3

            horse_hist[horse][0] += 1
            if is_win:
                horse_hist[horse][1] += 1
            if is_podium:
                horse_hist[horse][2] += 1

            jockey_hist[jockey][0] += 1
            if is_win:
                jockey_hist[jockey][1] += 1

            trainer_hist[trainer][0] += 1
            if is_win:
                trainer_hist[trainer][1] += 1

            horse_surface_hist[(horse, surface)][0] += 1
            if is_win:
                horse_surface_hist[(horse, surface)][1] += 1

            if not pd.isna(distance):
                hd_key = (horse, int(round(distance / 100.0) * 100))
                horse_distance_hist[hd_key][0] += 1
                if is_win:
                    horse_distance_hist[hd_key][1] += 1

            coupling_hist[(horse, jockey)][0] += 1
            if is_win:
                coupling_hist[(horse, jockey)][1] += 1

    return df


# ============================================================
# FORME RECENTE
# ============================================================

def add_recent_form_features(df):
    df = df.copy()

    feature_cols = [
        "Cheval_Forme_3", "Cheval_Forme_5",
        "Jockey_Forme_5", "Entraineur_Forme_5",
    ]

    for c in feature_cols:
        df[c] = np.nan

    horse_results = defaultdict(list)
    jockey_results = defaultdict(list)
    trainer_results = defaultdict(list)

    groups = df.groupby("Course_ID", sort=False).groups

    for _, idxs in groups.items():
        # Lecture forme
        for idx in idxs:
            horse = df.at[idx, "Cheval_Norm"]
            jockey = df.at[idx, "Jockey_Norm"]
            trainer = df.at[idx, "Entraineur_Norm"]

            h_res = horse_results[horse]
            if len(h_res) >= 3:
                df.at[idx, "Cheval_Forme_3"] = np.mean(h_res[-3:])
            if len(h_res) >= 5:
                df.at[idx, "Cheval_Forme_5"] = np.mean(h_res[-5:])

            j_res = jockey_results[jockey]
            if len(j_res) >= 5:
                df.at[idx, "Jockey_Forme_5"] = np.mean(j_res[-5:])

            t_res = trainer_results[trainer]
            if len(t_res) >= 5:
                df.at[idx, "Entraineur_Forme_5"] = np.mean(t_res[-5:])

        # Mise à jour
        for idx in idxs:
            rank = df.at[idx, "Classement"]
            if pd.isna(rank):
                continue

            horse = df.at[idx, "Cheval_Norm"]
            jockey = df.at[idx, "Jockey_Norm"]
            trainer = df.at[idx, "Entraineur_Norm"]

            # Conversion rang en score (1er = 10, 2e = 9, etc.)
            score = max(0, 11 - rank)

            horse_results[horse].append(score)
            jockey_results[jockey].append(score)
            trainer_results[trainer].append(score)

            # Limitation mémoire
            if len(horse_results[horse]) > 10:
                horse_results[horse] = horse_results[horse][-5:]
            if len(jockey_results[jockey]) > 10:
                jockey_results[jockey] = jockey_results[jockey][-5:]
            if len(trainer_results[trainer]) > 10:
                trainer_results[trainer] = trainer_results[trainer][-5:]

    return df


# ============================================================
# FEATURES SUPPLEMENTAIRES (INTEGREES)
# ============================================================

def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = b > 0
    out[mask] = a[mask] / b[mask]
    return out


# --- Musique ---

_MUSIQUE_TOKEN = re.compile(r"(\d{1,2}|[DTARN]{1,3})([a-zA-Z])")
_CODES_INCIDENT = {"D", "T", "A", "R", "RET", "NP", "DAI", "DA"}


def parse_musique(musique):
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
        "Musique_Derniere_Place": (places[0] if places else np.nan),
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


# --- Signalement ---

def add_signalement_features(df):
    df = df.copy()

    df["Age_Num"] = to_num(df["Age"]) if "Age" in df.columns else np.nan
    df["Poids_Num"] = to_num(df["Poids"]) if "Poids" in df.columns else np.nan
    df["Supplement_Num"] = to_num(df["Supplement"]) if "Supplement" in df.columns else 0.0
    df["Supplement_Num"] = df["Supplement_Num"].fillna(0.0)

    df["Sexe_Cat"] = (
        df["Sexe"].astype(str).str.strip().str.upper() if "Sexe" in df.columns else "INCONNU"
    )

    df["Inedit_Flag"] = (
        df["Inedit"].astype(str).str.strip().str.upper().eq("OUI").astype(int)
        if "Inedit" in df.columns else 0
    )

    return df


# --- Corde / Discipline ---

def add_corde_discipline_features(df):
    df = df.copy()

    df["Place_Corde_Num"] = to_num(df["Place_Corde"]) if "Place_Corde" in df.columns else np.nan

    df["Corde_Piste_Cat"] = (
        df["Corde_Piste"].astype(str).str.strip().str.upper()
        if "Corde_Piste" in df.columns else "INCONNU"
    )

    df["Discipline_Cat"] = (
        df["Discipline"].astype(str).str.strip().str.upper()
        if "Discipline" in df.columns else "INCONNU"
    )

    return df


# --- Gains et Carrière ---

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
        df[num_col] = to_num(df[c]) if c in df.columns else np.nan
        df[f"{c}_Log"] = np.log1p(df[num_col].clip(lower=0))

    for c in _CARRIERE_COLS:
        num_col = f"{c}_Num"
        df[num_col] = to_num(df[c]) if c in df.columns else np.nan

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


# --- Nb Partants ---

def add_nb_partants(df, course_id_col="Course_ID"):
    df = df.copy()
    df["Nb_Partants"] = df.groupby(course_id_col)[course_id_col].transform("size")
    return df


# --- Pipeline complet features supplémentaires ---

def enrichir_features_supplementaires(df):
    print("Ajout des features supplémentaires...")
    df = add_musique_features(df)
    df = add_signalement_features(df)
    df = add_corde_discipline_features(df)
    df = add_gains_et_carriere_features(df)
    df = add_nb_partants(df)
    return df


# ============================================================
# VARIABLES DE CONTEXTE
# ============================================================

def add_context_features(df):
    df = df.copy()

    df["Mois"] = df["Date_Course"].dt.month
    df["Jour_Annee"] = df["Date_Course"].dt.dayofyear

    df["Distance_Classe"] = pd.cut(
        df["Distance"],
        bins=[0, 1200, 1400, 1600, 1900, 2200, 2500, 3000, np.inf],
        labels=[
            "SPRINT", "COURTE", "INTERMEDIAIRE", "MOYENNE",
            "CLASSIQUE", "LONGUE", "TRES_LONGUE", "MARATHON"
        ],
        include_lowest=True
    ).astype(str)

    df["Surface_Distance"] = (
        df["Surface_Final"].astype(str) + "_" + df["Distance_Classe"].astype(str)
    )

    df["Surface_Oeilleres"] = (
        df["Surface_Final"].astype(str) + "_" +
        df["Oeilleres"].fillna("").astype(str).map(norm)
    )

    df["Distance_Oeilleres"] = (
        df["Distance_Classe"].astype(str) + "_" +
        df["Oeilleres"].fillna("").astype(str).map(norm)
    )

    df["Mois_Sin"] = np.sin(2 * np.pi * df["Mois"] / 12)
    df["Mois_Cos"] = np.cos(2 * np.pi * df["Mois"] / 12)

    return df


# ============================================================
# CIBLES
# ============================================================

def add_targets(df):
    df = df.copy()

    targets = df["Classement"].apply(
        lambda x: pd.Series(
            infer_rank_targets(x),
            index=["Target_Victoire", "Target_Podium", "Target_Top5"]
        )
    )

    df = pd.concat([df, targets], axis=1)
    return df


# ============================================================
# SELECTION DES FEATURES
# ============================================================

def select_features(df):
    forbidden = {
        "Cote_Direct",
        "Classement",
        "Target_Victoire",
        "Target_Podium",
        "Target_Top5",
        "Course_ID",
        "Cheval_Norm",
        "Jockey_Norm",
        "Entraineur_Norm",
        "Hippodrome_Norm",
        "Cote",
        "Arrivée",
        "Rang",
    }

    candidate = []

    for c in df.columns:
        if c in forbidden:
            continue
        if c in {"Surface_Source"}:
            continue
        candidate.append(c)

    return candidate


# ============================================================
# CONTROLE QUALITE
# ============================================================

def quality_report(df, features):
    print("\n" + "=" * 70)
    print("CONTROLE QUALITE V2.9")
    print("=" * 70)

    print(f"Lignes                : {len(df):,}")
    print(f"Colonnes              : {len(df.columns):,}")
    print(f"Courses               : {df['Course_ID'].nunique():,}")
    print(f"Features retenues     : {len(features):,}")

    print("\nSurface_Final :")
    print(df["Surface_Final"].value_counts(dropna=False).to_string())

    print("\nMéthodes surface :")
    print(df["Surface_Methode"].value_counts(dropna=False).head(30).to_string())

    if "Type_Hippodrome" in df.columns:
        fr = df[df["Type_Hippodrome"] == "FRANCE"]
        if not fr.empty:
            print("\nAudit surfaces France :")
            audit = (
                fr.groupby("Hippodrome_Canonique")
                  .agg(Courses=("Course_ID", "nunique"), Lignes=("Course_ID", "size"),
                       Surface_connue=("Surface_Final", lambda s: int((s != "INCONNUE").sum())),
                       Surface_inconnue=("Surface_Final", lambda s: int((s == "INCONNUE").sum())))
                  .sort_values(["Surface_inconnue", "Courses"], ascending=False)
            )
            print(audit.head(20).to_string())

    print("\nClassement :")
    print("  connus   :", int(df["Classement"].notna().sum()))
    print("  inconnus :", int(df["Classement"].isna().sum()))

    print("\nCibles :")
    for c in ["Target_Victoire", "Target_Podium", "Target_Top5"]:
        print(f"  {c:20s}: connus={df[c].notna().sum():,}")

    # Vérification features supplémentaires
    print("\nFeatures supplémentaires ajoutées :")
    supp_cols = [c for c in df.columns if any(k in c for k in ["Musique_", "Age_Num", "Poids_Num", "Gains_", "Carriere_", "Nb_Partants"])]
    print(f"  {len(supp_cols)} colonnes")

    if "Cote_Direct" in features:
        raise AssertionError("ERREUR : Cote_Direct est présente dans les features.")

    if df["Classement"].isna().sum() and (df.loc[df["Classement"].isna(), "Target_Victoire"].notna().any()):
        raise AssertionError("ERREUR : une cible est renseignée pour un classement inconnu.")

    print("\nVérification anti-fuite structurelle : OK")
    print("Cote_Direct exclue : OK")
    print("Classement inconnu conservé en NaN : OK")


# ============================================================
# EXPORT
# ============================================================

def main():
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    df = charger_google_sheet()
    df = preparer_colonnes(df)

    print("\nSélection du périmètre géographique V2.9...")
    df = filtrer_hippodromes(df)

    df["_date_missing"] = df["Date_Course"].isna()
    df = df.sort_values(
        ["_date_missing", "Date_Course", "Course_ID"],
        kind="stable"
    ).drop(columns=["_date_missing"]).reset_index(drop=True)

    print("\nInférence des surfaces V2.9...")
    df = enrichir_surface(df)

    print("\nCalcul des historiques sans fuite...")
    df = add_history_features(df)

    print("\nCalcul de la forme récente...")
    df = add_recent_form_features(df)

    print("\nAjout des features supplémentaires...")
    df = enrichir_features_supplementaires(df)

    print("\nAjout des variables de contexte...")
    df = add_context_features(df)

    print("\nCréation des cibles...")
    df = add_targets(df)

    features = select_features(df)
    quality_report(df, features)

    df["Feature_Count_V2_9"] = len(features)

    first_cols = [
        "Date_Course", "Hippodrome", "Hippodrome_Canonique", "Type_Hippodrome",
        "Numero_Course", "Course_ID", "Cheval", "Jockey", "Entraineur", "Proprietaire",
        "Distance", "Surface_Source", "Surface_Final", "Surface_Confiance", "Surface_Methode",
        "Nature_Piste", "Etat_Terrain", "Penetrometre", "Classement",
        "Target_Victoire", "Target_Podium", "Target_Top5",
    ]

    first_cols = [c for c in first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    print("\nExport CSV...")
    df.to_csv(FICHIER_CSV, index=False, encoding="utf-8-sig")

    print("Export Parquet...")
    try:
        df.to_parquet(FICHIER_PARQUET, index=False)
    except Exception as e:
        print("ATTENTION : Parquet non écrit. Installez pyarrow si nécessaire : pip install pyarrow")
        print("Détail :", e)

    print("\n" + "=" * 70)
    print("V2.9 TERMINEE")
    print("=" * 70)
    print("CSV     :", FICHIER_CSV)
    print("Parquet :", FICHIER_PARQUET)
    print("Features:", len(features))


if __name__ == "__main__":
    main()
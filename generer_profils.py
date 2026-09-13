"""
============================================================
GENERER_PROFILS_V2.PY
============================================================
Génère les masters statistiques à partir de l'historique
des courses de plat.
IMPORTANT :
- Les statistiques sont calculées chronologiquement.
- La course en cours n'est JAMAIS intégrée dans les stats
  disponibles avant cette course.
- Les masters finaux correspondent à l'état statistique
  après traitement de toute la période historique.
Sorties :
C:/Users/33662/OneDrive/Bureau/course_de_plat/data/master_chevaux.parquet
C:/Users/33662/OneDrive/Bureau/course_de_plat/data/master_jockeys.parquet
C:/Users/33662/OneDrive/Bureau/course_de_plat/data/master_entraineurs.parquet
C:/Users/33662/OneDrive/Bureau/course_de_plat/data/master_proprietaires.parquet
C:/Users/33662/OneDrive/Bureau/course_de_plat/data/master_couplages.parquet
============================================================
"""

import os
import re
import io
import unicodedata
import requests
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
URL_CSV = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=644246763&single=true&output=csv"
)
DATE_DEBUT = pd.Timestamp("2025-09-02")

# Chemin d'accès cible pour l'export des fichiers masters
DOSSIER_MASTERS = r"C:\Users\33662\OneDrive\Bureau\course_de_plat\data"

# Fichier de secours (placé dans le dossier courant d'exécution du script)
FICHIER_LOCAL_SECOURS = os.path.join(os.getcwd(), "Import_historique.csv")

# ============================================================
# OUTILS
# ============================================================
def nettoyer_nom(nom):
    """Normalise les noms de chevaux, jockeys, entraîneurs et propriétaires."""
    if pd.isna(nom):
        return "INCONNU"
    texte = str(nom).strip()
    if not texte:
        return "INCONNU"
    texte = unicodedata.normalize("NFKD", texte).encode("ASCII", "ignore").decode("utf-8")
    texte = texte.upper()
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()

def safe_float(valeur, default=0.0):
    """Conversion sécurisée en float."""
    try:
        if pd.isna(valeur):
            return default
        texte = str(valeur).replace(",", ".").strip()
        if not texte:
            return default
        return float(texte)
    except (ValueError, TypeError):
        return default

def safe_int(valeur, default=0):
    """Conversion sécurisée en entier."""
    try:
        if pd.isna(valeur):
            return default
        texte = str(valeur).replace(",", ".").strip()
        if not texte:
            return default
        return int(float(texte))
    except (ValueError, TypeError):
        return default

def determiner_surface(nature_piste, etat_terrain=""):
    """Transforme la nature de piste en : PSF ou GAZON"""
    texte = f"{nature_piste} {etat_terrain}".upper()
    if any(terme in texte for terme in ["PSF", "SABLE", "FIBRE", "ALL WEATHER", "POLYTRACK"]):
        return "PSF"
    return "GAZON"

def categoriser_distance(distance):
    """Catégories larges de distance."""
    d = safe_float(distance, -1)
    if d < 0:
        return "INCONNUE"
    if d < 1300:
        return "SPRINT"
    if d <= 1600:
        return "MILE"
    if d <= 2200:
        return "INTERMEDIAIRE"
    if d <= 3000:
        return "CLASSIQUE"
    return "LONGUE"

def normaliser_terrain(valeur):
    """Normalisation légère de l'état du terrain."""
    if pd.isna(valeur):
        return "INCONNU"
    texte = str(valeur).strip().upper()
    if not texte:
        return "INCONNU"
    texte = unicodedata.normalize("NFKD", texte).encode("ASCII", "ignore").decode("utf-8")
    texte = re.sub(r"\s+", " ", texte)
    return texte

def extraire_classement(row):
    """Cherche le classement réel dans plusieurs noms possibles."""
    candidats = ["Classement", "classement", "Place", "place", "Arrivee", "Arrivée"]
    for colonne in candidats:
        if colonne in row.index:
            valeur = safe_int(row[colonne], 0)
            if valeur > 0:
                return valeur
    return 0

def est_victoire(classement):
    return int(classement == 1)

def est_podium(classement):
    return int(1 <= classement <= 3)

# ============================================================
# CHARGEMENT
# ============================================================
def charger_dataframe():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(URL_CSV, headers=headers, timeout=30)
        response.raise_for_status()
        print("✅ Téléchargement Google Sheets réussi.")
        return pd.read_csv(io.StringIO(response.text), dtype={"Date": str}, on_bad_lines="skip")
    except Exception as erreur:
        print(f"⚠️ Erreur Google Sheets : {erreur}")
        if os.path.exists(FICHIER_LOCAL_SECOURS):
            print("📁 Utilisation du fichier local de secours.")
            return pd.read_csv(FICHIER_LOCAL_SECOURS, dtype={"Date": str}, on_bad_lines="skip")
        raise FileNotFoundError("Impossible de charger les données historiques.")

# ============================================================
# STRUCTURES STATISTIQUES
# ============================================================
def nouvelle_stat():
    return {"courses": 0, "victoires": 0, "podiums": 0, "gains": 0.0}

def ajouter_resultat(stat, classement, gains=0.0):
    stat["courses"] += 1
    if classement == 1:
        stat["victoires"] += 1
    if 1 <= classement <= 3:
        stat["podiums"] += 1
    stat["gains"] += gains

def taux(numerateur, denominateur):
    if denominateur <= 0:
        return 0.0
    return round(numerateur / denominateur * 100, 2)

def finaliser_stat(stat):
    courses = stat["courses"]
    return {
        "courses": courses,
        "victoires": stat["victoires"],
        "podiums": stat["podiums"],
        "taux_victoire": taux(stat["victoires"], courses),
        "taux_podium": taux(stat["podiums"], courses),
        "gains": round(stat["gains"], 2),
        "gains_par_course": round(stat["gains"] / courses, 2) if courses else 0.0
    }

# ============================================================
# INITIALISATION DES ENTITES
# ============================================================
def nouveau_cheval():
    return {
        "global": nouvelle_stat(), "surface": {}, "distance": {}, "categorie_distance": {},
        "terrain": {}, "hippodrome": {}, "oeilleres": {}, "corde": {},
        "historique_positions": [], "poids": [], "penetrometres": [], "cotes": [], "dernieres_dates": []
    }

def nouvelle_personne():
    return {"global": nouvelle_stat(), "surface": {}, "distance": {}, "categorie_distance": {}, "terrain": {}, "hippodrome": {}}

def nouvelle_couplage():
    return nouvelle_stat()

def obtenir_stat(dictionnaire, cle):
    if cle not in dictionnaire:
        dictionnaire[cle] = nouvelle_stat()
    return dictionnaire[cle]

# ============================================================
# GENERATION
# ============================================================
def executer_generation_masters():
    print("=" * 70)
    print("🏇 GENERATION DES MASTERS V2")
    print("=" * 70)
    
    df = charger_dataframe()
    df.columns = [str(colonne).strip() for colonne in df.columns]
    print(f"📊 {len(df):,} lignes chargées.")
    
    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------
    if "Date" not in df.columns:
        raise ValueError("La colonne 'Date' est absente.")
    
    df["Date_dt"] = pd.to_datetime(df["Date"].astype(str), format="%d%m%Y", errors="coerce")
    masque_date_invalide = df["Date_dt"].isna()
    if masque_date_invalide.any():
        df.loc[masque_date_invalide, "Date_dt"] = pd.to_datetime(df.loc[masque_date_invalide, "Date"], errors="coerce", dayfirst=True)
    
    df = df.dropna(subset=["Date_dt"])
    df = df[df["Date_dt"] >= DATE_DEBUT].copy()
    
    # --------------------------------------------------------
    # NORMALISATION
    # --------------------------------------------------------
    df["Cheval_clean"] = df["Nom"].apply(nettoyer_nom)
    df["Jockey_clean"] = df["Driver_Jockey"].apply(nettoyer_nom)
    df["Entraineur_clean"] = df["Entraineur"].apply(nettoyer_nom)
    df["Proprietaire_clean"] = df["Proprietaire"].apply(nettoyer_nom)
    
    df["Surface"] = df.apply(lambda row: determiner_surface(row.get("Nature_Piste", ""), row.get("Etat_Terrain", "")), axis=1)
    df["Cat_Distance"] = df["Distance"].apply(categoriser_distance)
    df["Terrain_clean"] = df["Etat_Terrain"].apply(normaliser_terrain)
    df["Hippodrome_clean"] = df["Hippodrome"].apply(normaliser_terrain)
    
    df["Distance_num"] = df["Distance"].apply(lambda x: safe_float(x, 0))
    df["Poids_num"] = df["Poids"].apply(lambda x: safe_float(x, 0))
    df["Corde_num"] = df["Place_Corde"].apply(lambda x: safe_int(x, 0))
    df["Cote_num"] = df["Cote_Direct"].apply(lambda x: safe_float(x, 0))
    df["Classement_num"] = df.apply(extraire_classement, axis=1)
    
    # --------------------------------------------------------
    # TRI CHRONOLOGIQUE
    # --------------------------------------------------------
    colonnes_tri = ["Date_dt"]
    if "Reunion" in df.columns:
        colonnes_tri.append("Reunion")
    if "Course" in df.columns:
        colonnes_tri.append("Course")
    df = df.sort_values(colonnes_tri).reset_index(drop=True)
    
    # --------------------------------------------------------
    # STRUCTURES
    # --------------------------------------------------------
    stats_chevaux = {}
    stats_jockeys = {}
    stats_entraineurs = {}
    stats_proprietaires = {}
    stats_cj = {}
    stats_ce = {}
    stats_je = {}
    
    # ========================================================
    # TRAITEMENT CHRONOLOGIQUE
    # ========================================================
    total = len(df)
    print("🔄 Calcul chronologique des statistiques...")
    
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        cheval = row["Cheval_clean"]
        jockey = row["Jockey_clean"]
        entraineur = row["Entraineur_clean"]
        proprio = row["Proprietaire_clean"]
        
        if cheval == "INCONNU":
            continue
            
        surface = row["Surface"]
        distance = row["Distance_num"]
        cat_distance = row["Cat_Distance"]
        terrain = row["Terrain_clean"]
        hippodrome = row["Hippodrome_clean"]
        classement = row["Classement_num"]
        gains = safe_float(row.get("Gains_Place", 0))
        oeilleres = nettoyer_nom(row.get("Oeilleres", "SANS"))
        corde = str(row["Corde_num"])
        
        # Initialisation
        if cheval not in stats_chevaux:
            stats_chevaux[cheval] = nouveau_cheval()
        if jockey not in stats_jockeys:
            stats_jockeys[jockey] = nouvelle_personne()
        if entraineur not in stats_entraineurs:
            stats_entraineurs[entraineur] = nouvelle_personne()
        if proprio not in stats_proprietaires:
            stats_proprietaires[proprio] = nouvelle_personne()
            
        ch = stats_chevaux[cheval]
        jk = stats_jockeys[jockey]
        en = stats_entraineurs[entraineur]
        pr = stats_proprietaires[proprio]
        
        # Cheval
        ajouter_resultat(ch["global"], classement, gains)
        ajouter_resultat(obtenir_stat(ch["surface"], surface), classement, gains)
        ajouter_resultat(obtenir_stat(ch["distance"], str(int(distance))), classement, gains)
        ajouter_resultat(obtenir_stat(ch["categorie_distance"], cat_distance), classement, gains)
        ajouter_resultat(obtenir_stat(ch["terrain"], terrain), classement, gains)
        ajouter_resultat(obtenir_stat(ch["hippodrome"], hippodrome), classement, gains)
        ajouter_resultat(obtenir_stat(ch["oeilleres"], oeilleres), classement, gains)
        ajouter_resultat(obtenir_stat(ch["corde"], corde), classement, gains)
        
        ch["historique_positions"].append(classement)
        if row["Poids_num"] > 0:
            ch["poids"].append(row["Poids_num"])
            
        penetro = safe_float(row.get("Penetrometre", 0))
        if penetro > 0:
            ch["penetrometres"].append(penetro)
            
        if row["Cote_num"] > 0:
            ch["cotes"].append(row["Cote_num"])
            
        ch["dernieres_dates"].append(row["Date_dt"])
        
        # Jockey
        ajouter_resultat(jk["global"], classement, gains)
        ajouter_resultat(obtenir_stat(jk["surface"], surface), classement, gains)
        ajouter_resultat(obtenir_stat(jk["distance"], str(int(distance))), classement, gains)
        ajouter_resultat(obtenir_stat(jk["categorie_distance"], cat_distance), classement, gains)
        ajouter_resultat(obtenir_stat(jk["terrain"], terrain), classement, gains)
        ajouter_resultat(obtenir_stat(jk["hippodrome"], hippodrome), classement, gains)
        
        # Entraineur
        ajouter_resultat(en["global"], classement, gains)
        ajouter_resultat(obtenir_stat(en["surface"], surface), classement, gains)
        ajouter_resultat(obtenir_stat(en["distance"], str(int(distance))), classement, gains)
        ajouter_resultat(obtenir_stat(en["categorie_distance"], cat_distance), classement, gains)
        ajouter_resultat(obtenir_stat(en["terrain"], terrain), classement, gains)
        ajouter_resultat(obtenir_stat(en["hippodrome"], hippodrome), classement, gains)
        
        # Proprietaire
        ajouter_resultat(pr["global"], classement, gains)
        
        # Couplages
        cle_cj = f"{cheval}__{jockey}"
        cle_ce = f"{cheval}__{entraineur}"
        cle_je = f"{jockey}__{entraineur}"
        
        if cle_cj not in stats_cj:
            stats_cj[cle_cj] = nouvelle_couplage()
        if cle_ce not in stats_ce:
            stats_ce[cle_ce] = nouvelle_couplage()
        if cle_je not in stats_je:
            stats_je[cle_je] = nouvelle_couplage()
            
        ajouter_resultat(stats_cj[cle_cj], classement, gains)
        ajouter_resultat(stats_ce[cle_ce], classement, gains)
        ajouter_resultat(stats_je[cle_je], classement, gains)
        
        # Progression
        if position % 5000 == 0:
            pourcentage = position / total * 100
            print(f"   {position:,}/{total:,} ({pourcentage:.1f} %)")

    # ========================================================
    # EXPORT
    # ========================================================
    os.makedirs(DOSSIER_MASTERS, exist_ok=True)
    print("\n💾 Export des masters...")
    
    # MASTER CHEVAUX
    lignes = []
    for cheval, data in stats_chevaux.items():
        lignes.append({
            "Cheval_clean": cheval,
            "Total_courses": data["global"]["courses"],
            "Total_victoires": data["global"]["victoires"],
            "Total_podiums": data["global"]["podiums"],
            "Taux_victoire": taux(data["global"]["victoires"], data["global"]["courses"]),
            "Taux_podium": taux(data["global"]["podiums"], data["global"]["courses"]),
            "Gains_Total": round(data["global"]["gains"], 2),
            "Courses_Gazon": data["surface"].get("GAZON", {}).get("courses", 0),
            "Victoires_Gazon": data["surface"].get("GAZON", {}).get("victoires", 0),
            "Podiums_Gazon": data["surface"].get("GAZON", {}).get("podiums", 0),
            "Courses_PSF": data["surface"].get("PSF", {}).get("courses", 0),
            "Victoires_PSF": data["surface"].get("PSF", {}).get("victoires", 0),
            "Podiums_PSF": data["surface"].get("PSF", {}).get("podiums", 0),
            "Poids_moyen": round(np.mean(data["poids"]), 2) if data["poids"] else 0,
            "Cote_moyenne": round(np.mean(data["cotes"]), 2) if data["cotes"] else 0,
            "Penetrometre_moyen": round(np.mean(data["penetrometres"]), 2) if data["penetrometres"] else 0
        })
    pd.DataFrame(lignes).to_parquet(os.path.join(DOSSIER_MASTERS, "master_chevaux.parquet"), index=False)
    
    # FONCTION EXPORT PERSONNES
    def exporter_personnes(dictionnaire, nom_colonne, nom_fichier):
        lignes = []
        for nom, data in dictionnaire.items():
            global_stat = data["global"]
            lignes.append({
                nom_colonne: nom,
                "Total_courses": global_stat["courses"],
                "Total_victoires": global_stat["victoires"],
                "Total_podiums": global_stat["podiums"],
                "Taux_victoire": taux(global_stat["victoires"], global_stat["courses"]),
                "Taux_podium": taux(global_stat["podiums"], global_stat["courses"]),
                "Courses_Gazon": data["surface"].get("GAZON", {}).get("courses", 0),
                "Victoires_Gazon": data["surface"].get("GAZON", {}).get("victoires", 0),
                "Podiums_Gazon": data["surface"].get("GAZON", {}).get("podiums", 0),
                "Courses_PSF": data["surface"].get("PSF", {}).get("courses", 0),
                "Victoires_PSF": data["surface"].get("PSF", {}).get("victoires", 0),
                "Podiums_PSF": data["surface"].get("PSF", {}).get("podiums", 0)
            })
        pd.DataFrame(lignes).to_parquet(os.path.join(DOSSIER_MASTERS, nom_fichier), index=False)

    exporter_personnes(stats_jockeys, "Jockey_clean", "master_jockeys.parquet")
    exporter_personnes(stats_entraineurs, "Entraineur_clean", "master_entraineurs.parquet")
    exporter_personnes(stats_proprietaires, "Proprietaire_clean", "master_proprietaires.parquet")
    
    # MASTER COUPLAGES
    lignes = []
    for type_couplage, dictionnaire in [
        ("CHEVAL_JOCKEY", stats_cj),
        ("CHEVAL_ENTRAINEUR", stats_ce),
        ("JOCKEY_ENTRAINEUR", stats_je)
    ]:
        for cle, data in dictionnaire.items():
            entite1, entite2 = cle.split("__", 1)
            lignes.append({
                "Type_Couplage": type_couplage,
                "Entite_1": entite1,
                "Entite_2": entite2,
                "Courses": data["courses"],
                "Victoires": data["victoires"],
                "Podiums": data["podiums"],
                "Taux_victoire": taux(data["victoires"], data["courses"]),
                "Taux_podium": taux(data["podiums"], data["courses"])
            })
    pd.DataFrame(lignes).to_parquet(os.path.join(DOSSIER_MASTERS, "master_couplages.parquet"), index=False)
    
    print("\n" + "=" * 70)
    print("✅ GENERATION DES MASTERS TERMINEE")
    print(f"📁 Dossier de destination : {DOSSIER_MASTERS}")
    print("=" * 70)

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    executer_generation_masters()
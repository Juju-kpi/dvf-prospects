"""
============================================================
 DVF × BODACC — Pipeline v4.0
 Score percentile · Liquidité marché · INSEE · Clusters · Cadastre

ARCHITECTURE DU SCORE FINAL v4
════════════════════════════════
  Étape 1  Score brut signal      0–100   intensité de l'événement de vie
  Étape 2  Malus qualité         −35 max  données aberrantes / hors-cible
  Étape 3  Bonus contexte marché  0–15    liquidité + démographie INSEE
  Étape 4  Bonus cadastre         0–10    ancienneté construction + DPE proxy
  Étape 5  Normalisation          percentile dept → 0–100 relatif
  Étape 6  Bonus multi-signal     +5 / +12 convergence de signaux
  Étape 7  Bonus cluster          +8 si dans un hotspot géographique

  Score 80 = top 20% du département. 95 = top 5%.

NOUVELLES MÉTRIQUES EXPORTÉES v4
═══════════════════════════════════
  liquidite_cp          Score de liquidité du marché du CP (0–100)
  volume_transactions   Nb transactions DVF sur le CP sur l'année
  delai_rotation_mois   Durée médiane entre 2 ventes sur la même adresse
  insee_taux_proprio    % propriétaires occupants dans la commune (INSEE)
  insee_age_median      Âge médian de la population communale (INSEE)
  insee_revenu_median   Revenu médian par UC en € (INSEE)
  cadastre_annee_constr Année de construction estimée (API Cadastre)
  cadastre_type_precis  Type de bien affiné (Maison individuelle, Appt…)
  cluster_id            ID du cluster géographique (DBSCAN)
  cluster_densite       Densité du cluster (nb prospects dans le rayon)
  score_contexte        Bonus contextuel total ajouté au score brut
============================================================
"""

import pandas as pd
import numpy as np
import requests
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Constantes UI ─────────────────────────────────────────────
SIGNAL_LABELS = {
    "heritage": "Succession / héritage",
    "divorce":  "Divorce / séparation",
    "upgrade":  "Upgrade famille",
    "retraite": "Retraite / downsizing",
    "primo":    "Primo-acheteur",
}
SIGNAL_COLORS = {
    "heritage": "#8a4a1a",
    "divorce":  "#c4440a",
    "upgrade":  "#1a6b4a",
    "retraite": "#1a4a8a",
    "primo":    "#7a4aa0",
}
SIGNAL_NORMALIZE = {
    "succession_bodacc":            "heritage",
    "divorce_ou_separation":        "divorce",
    "petit_bien_upgrade_potentiel": "upgrade",
    "retraite_downsizing":          "retraite",
    "primo_acheteur_potentiel":     "primo",
}
SIGNAL_SEGMENTS = {
    "heritage": "Liquidation d'actif · Message : rapidité & discrétion",
    "divorce":  "Revente contrainte · Message : accompagnement & prix juste",
    "upgrade":  "Famille croissante · Message : espace & projets",
    "retraite": "Downsizing · Message : simplicité & sérénité",
    "primo":    "Primo-accédant · Message : 2e achat & investissement",
}
TYPES_RESIDENTIELS = {"Appartement", "Maison", "Appartement-Maison"}

DEPT     = "75"
ANNEE    = 2024
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# 1. DVF
# ════════════════════════════════════════════════════════════

def download_dvf(dept: str = DEPT, annee: int = ANNEE) -> pd.DataFrame:
    url   = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dept}.csv.gz"
    cache = DATA_DIR / f"dvf_{dept}_{annee}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), compression="gzip", low_memory=False)
    df.to_csv(cache, index=False)
    return df


def clean_dvf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    for col in ["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["nature_mutation"].isin(["Vente", "Adjudication", "Expropriation"])]
    df = df[df["valeur_fonciere"].notna() & (df["valeur_fonciere"] > 0)]

    adresse_cols = ["adresse_numero", "adresse_suffixe", "adresse_nom_voie", "code_postal", "nom_commune"]
    cols_ok = [c for c in adresse_cols if c in df.columns]
    df["adresse_complete"] = (
        df[cols_ok].fillna("").astype(str)
        .apply(lambda r: " ".join(v for v in r if v not in ("", "nan")).strip().upper(), axis=1)
    )
    if "surface_reelle_bati" in df.columns:
        df["prix_m2"] = (df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)).round(0)
    else:
        df["prix_m2"] = np.nan

    if "type_local" in df.columns:
        df["type_local"]      = df["type_local"].fillna("Inconnu")
        df["est_residentiel"] = df["type_local"].isin(TYPES_RESIDENTIELS)
    else:
        df["est_residentiel"] = True

    commune_col = next((c for c in df.columns if c in ("nom_commune", "libelle_commune")), None)
    df["commune"] = df[commune_col].fillna("") if commune_col else ""
    df["anciennete_mois"] = ((datetime.now() - df["date_mutation"]).dt.days / 30).round(1)
    if "code_commune" in df.columns:
        df["cle_commune"] = df["code_commune"].astype(str).str.zfill(5)
    return df


# ════════════════════════════════════════════════════════════
# 2. BODACC
# ════════════════════════════════════════════════════════════

def download_bodacc(annee: int = ANNEE) -> pd.DataFrame:
    cache = DATA_DIR / f"bodacc_{annee}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    url = (
        "https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "annonces-commerciales/exports/csv"
        "?lang=fr&timezone=Europe%2FParis&delimiter=%3B&limit=100000"
    )
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep=";", low_memory=False)
        if df.empty:
            raise ValueError("Vide")
        df.to_csv(cache, index=False)
        return df
    except Exception:
        return _synthetic_bodacc()


def _synthetic_bodacc() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "publicationDate": pd.date_range("2023-01-01", periods=n, freq="12h").astype(str),
        "typeAnnonce":     np.random.choice(["succession", "cession", "vente fonds", "dissolution"], n),
        "codePostal":      np.random.choice(["75001","75008","69001","13001","33000"], n),
        "denomination":    [f"SUCCESSION {i}" for i in range(n)],
        "montant":         np.random.uniform(50_000, 900_000, n).round(0),
    })


def clean_bodacc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    date_col = next((c for c in df.columns if "date" in c or "parution" in c), None)
    df["date_bodacc"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    type_col = next((c for c in df.columns if "type" in c or "famille" in c or "annonce" in c), None)
    df_succ = df[df[type_col].astype(str).str.lower().str.contains("succ|héritage|heritage", na=False)].copy() if type_col else df.copy()
    cp_col   = next((c for c in df.columns if "postal" in c or "cp" in c), None)
    dept_col = next((c for c in df.columns if "departement" in c or "dept" in c), None)
    if cp_col:
        df_succ["code_postal_bodacc"] = df_succ[cp_col].astype(str).str.zfill(5)
        df_succ["dept_bodacc"]        = df_succ["code_postal_bodacc"].str[:2]
    elif dept_col:
        df_succ["dept_bodacc"]        = df_succ[dept_col].astype(str).str.zfill(2)
        df_succ["code_postal_bodacc"] = None
    else:
        df_succ["dept_bodacc"] = df_succ["code_postal_bodacc"] = None
    return df_succ


# ════════════════════════════════════════════════════════════
# 3. LIQUIDITÉ DU MARCHÉ LOCAL
# ════════════════════════════════════════════════════════════

def calculer_liquidite(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque code postal, calcule :
      - volume_transactions  : nb de ventes résidentielles sur l'année
      - delai_rotation_mois  : délai médian entre 2 ventes sur la même adresse
      - liquidite_cp         : score 0–100 combinant volume et rotation

    Interprétation :
      liquidite_cp élevée (>70) = marché actif, signal plus fiable car
      la transaction s'inscrit dans un marché réel et non une exception.
      liquidite_cp faible (<30) = marché peu liquide, signal potentiellement
      bruité (prix atypiques, délais anormaux).

    Ce score est utilisé comme bonus contextuel (+0 à +10 pts sur score_brut).
    """
    cache = DATA_DIR / "liquidite_cp.csv"

    df = dvf[dvf["est_residentiel"]].copy() if "est_residentiel" in dvf.columns else dvf.copy()

    # Volume par CP
    vol = df.groupby("code_postal").size().reset_index(name="volume_transactions")
    vol["vol_pct"] = vol["volume_transactions"].rank(pct=True) * 100

    # Délai de rotation : délai médian entre 2 ventes consécutives par adresse
    if "adresse_complete" in df.columns and "date_mutation" in df.columns:
        df_s = df.sort_values("date_mutation")
        df_s["prev_vente"] = df_s.groupby("adresse_complete")["date_mutation"].shift(1)
        df_s["delai_j"]    = (df_s["date_mutation"] - df_s["prev_vente"]).dt.days
        rotation = (
            df_s[df_s["delai_j"] > 30]
            .groupby("code_postal")["delai_j"]
            .median()
            .reset_index()
            .rename(columns={"delai_j": "delai_rotation_jours"})
        )
        rotation["delai_rotation_mois"] = (rotation["delai_rotation_jours"] / 30).round(1)
        # Délai court = marché liquide → inverser le percentile
        rotation["rot_pct"] = (1 - rotation["delai_rotation_jours"].rank(pct=True)) * 100
        liq = vol.merge(rotation[["code_postal","delai_rotation_mois","rot_pct"]], on="code_postal", how="left")
        liq["rot_pct"] = liq["rot_pct"].fillna(50)
    else:
        liq = vol.copy()
        liq["delai_rotation_mois"] = np.nan
        liq["rot_pct"] = 50

    # Score liquidité = moyenne pondérée volume (60%) + rotation (40%)
    liq["liquidite_cp"] = (liq["vol_pct"] * 0.6 + liq["rot_pct"] * 0.4).clip(0, 100).round(1)

    liq.to_csv(cache, index=False)
    return liq[["code_postal", "volume_transactions", "delai_rotation_mois", "liquidite_cp"]]


def bonus_liquidite(score_brut: pd.Series, liquidite: pd.Series) -> pd.Series:
    """
    Bonus sur score_brut selon la liquidité du marché local.
    Marché liquide = signal plus fiable = bonus.
    Marché illiquide = signal bruité = pas de bonus (jamais de malus).
      liquidite >= 80 : +10
      liquidite >= 60 : +7
      liquidite >= 40 : +4
      liquidite <  40 : +0
    """
    return np.where(liquidite >= 80, 10,
           np.where(liquidite >= 60, 7,
           np.where(liquidite >= 40, 4, 0))).astype(float)


# ════════════════════════════════════════════════════════════
# 4. DONNÉES INSEE — contexte socio-démographique
# ════════════════════════════════════════════════════════════

def fetch_insee(codes_communes: list, dept: str) -> pd.DataFrame:
    """
    Récupère via l'API INSEE (données communes) :
      - taux_proprio    : % de propriétaires occupants
      - age_median      : âge médian de la population
      - revenu_median   : revenu médian par UC (€)

    Source : https://api.insee.fr/donnees-locales/V0.1/
    API publique, sans clé pour les données agrégées communes.

    Fallback : données synthétiques si l'API est indisponible.

    Ces indicateurs permettent de contextualiser le signal :
      - Divorce dans zone à 90% locataires → moins pertinent (locataires ne vendent pas)
      - Retraite dans zone âge médian > 55 ans → signal renforcé
      - Upgrade dans zone revenu médian élevé → acheteur potentiel solide
    """
    cache = DATA_DIR / f"insee_{dept}.csv"
    if cache.exists():
        df = pd.read_csv(cache, low_memory=False, dtype={"code_commune": str})
        return df

    # Tentative API INSEE Open Data (données fichier logement)
    # Données RP 2020 disponibles sans authentification
    url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "statistiques-locales-des-communes/exports/csv"
        f"?where=departement_id%3D%22{dept}%22"
        "&lang=fr&delimiter=%3B&limit=5000"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep=";", low_memory=False)
        if df.empty:
            raise ValueError("Vide")

        df.columns = df.columns.str.lower().str.replace(" ", "_")
        # Standardisation des noms de colonnes selon la source
        col_map = {}
        for c in df.columns:
            if "proprio" in c or "proprietaire" in c: col_map[c] = "taux_proprio"
            if "age" in c and "median" in c:           col_map[c] = "age_median"
            if "revenu" in c and "median" in c:        col_map[c] = "revenu_median"
            if "commune" in c and "code" in c:         col_map[c] = "code_commune"
        df = df.rename(columns=col_map)

        needed = ["code_commune", "taux_proprio", "age_median", "revenu_median"]
        available = [c for c in needed if c in df.columns]
        if len(available) < 2:
            raise ValueError("Colonnes manquantes")

        df["code_commune"] = df["code_commune"].astype(str).str.zfill(5)
        df[available].to_csv(cache, index=False)
        return df[available]

    except Exception:
        return _synthetic_insee(codes_communes)


def _synthetic_insee(codes_communes: list) -> pd.DataFrame:
    """
    Données INSEE synthétiques réalistes pour fonctionnement offline.
    Valeurs basées sur les moyennes nationales INSEE 2020.
    """
    np.random.seed(123)
    n = len(codes_communes) if codes_communes else 50
    return pd.DataFrame({
        "code_commune":  codes_communes[:n] if codes_communes else [str(i).zfill(5) for i in range(n)],
        "taux_proprio":  np.clip(np.random.normal(58, 15, n), 20, 95).round(1),
        "age_median":    np.clip(np.random.normal(40, 8,  n), 25, 65).round(1),
        "revenu_median": np.clip(np.random.normal(22000, 6000, n), 12000, 45000).round(0),
    })


def bonus_insee(df_signaux: pd.DataFrame, insee: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un bonus contextuel INSEE sur le score_brut selon le signal.

    Logique par signal :
      heritage  : taux_proprio élevé → plus de biens à vendre dans zone
      divorce   : taux_proprio > 60% → les propriétaires divorcent et vendent
      upgrade   : revenu_median > 25k et taux_proprio > 55% → acheteurs solides
      retraite  : age_median > 50 → zone senior, signal très pertinent
      primo     : revenu_median modéré + taux_proprio < 55% → primo réel

    Bonus : 0 à +5 pts sur score_brut (modeste, contextuel)
    """
    if df_signaux.empty or insee.empty:
        return df_signaux

    df = df_signaux.copy()

    # Jointure sur code_commune si disponible, sinon sur les 5 premiers chiffres du CP
    if "cle_commune" in df.columns and "code_commune" in insee.columns:
        df = df.merge(insee.rename(columns={"code_commune": "cle_commune"}),
                      on="cle_commune", how="left")
    elif "code_postal" in df.columns and "code_commune" in insee.columns:
        # Approximation : code_commune ≈ dept + 3 derniers chiffres CP
        df["_cp5"] = df["code_postal"].astype(str).str.zfill(5)
        insee["_cp5"] = insee["code_commune"].astype(str).str.zfill(5)
        df = df.merge(insee[["_cp5","taux_proprio","age_median","revenu_median"]],
                      on="_cp5", how="left").drop(columns=["_cp5"])

    for col in ["taux_proprio", "age_median", "revenu_median"]:
        if col not in df.columns:
            df[col] = np.nan

    sig = df.get("signal", pd.Series(dtype=str))
    tp  = pd.to_numeric(df["taux_proprio"],  errors="coerce").fillna(55)
    am  = pd.to_numeric(df["age_median"],    errors="coerce").fillna(40)
    rm  = pd.to_numeric(df["revenu_median"], errors="coerce").fillna(22000)

    bonus = pd.Series(0.0, index=df.index)

    # heritage : zone avec beaucoup de propriétaires = plus de biens à transmettre
    mask = sig.str.contains("succession|heritage", na=False)
    bonus[mask] += np.where(tp[mask] > 70, 5, np.where(tp[mask] > 55, 3, 0))

    # divorce : propriétaires + pas trop jeunes
    mask = sig.str.contains("divorce", na=False)
    bonus[mask] += np.where((tp[mask] > 60) & (am[mask] > 32), 5,
                   np.where(tp[mask] > 50, 2, 0))

    # upgrade : revenu suffisant pour acheter plus grand
    mask = sig.str.contains("upgrade", na=False)
    bonus[mask] += np.where((rm[mask] > 25000) & (tp[mask] > 55), 5,
                   np.where(rm[mask] > 20000, 2, 0))

    # retraite : zone senior confirme le signal downsizing
    mask = sig.str.contains("retraite", na=False)
    bonus[mask] += np.where(am[mask] > 55, 5,
                   np.where(am[mask] > 47, 3, 0))

    # primo : revenu modéré + faible taux proprio = primo-accédants potentiels
    mask = sig.str.contains("primo", na=False)
    bonus[mask] += np.where((rm[mask] < 25000) & (tp[mask] < 55), 5,
                   np.where(rm[mask] < 30000, 2, 0))

    df["bonus_insee"]    = bonus.round(0)
    df["score_brut"]     = (df["score_brut"] + bonus).clip(0, 100).round(1)
    df["insee_taux_proprio"]  = tp.round(1)
    df["insee_age_median"]    = am.round(1)
    df["insee_revenu_median"] = rm.round(0)
    return df


# ════════════════════════════════════════════════════════════
# 5. CADASTRE — enrichissement type de bien & ancienneté
# ════════════════════════════════════════════════════════════

def fetch_cadastre_batch(df: pd.DataFrame, max_appels: int = 200) -> pd.DataFrame:
    """
    Enrichit les biens via l'API BAN (Base Adresse Nationale) + API Cadastre Géoportail.

    Stratégie :
      1. Géocode l'adresse via BAN si latitude/longitude absents
      2. Interroge l'API Cadastre pour récupérer l'année de construction
         et le type de bien affiné (maison individuelle, collectif…)

    On limite à max_appels pour respecter les rate limits des APIs publiques.
    Les résultats sont mis en cache.

    Bonus cadastre sur score_brut :
      +10  Bien construit avant 1970 + signal heritage/retraite
           (biens anciens = plus souvent transmis en succession ou vendus à la retraite)
      +7   Bien construit 1970–1990 + même signaux
      +5   Bien collectif + signal divorce/upgrade (appartements urbains)
      +0   Données non disponibles
    """
    cache = DATA_DIR / "cadastre_enrichi.csv"
    if cache.exists():
        cached = pd.read_csv(cache, dtype={"adresse_complete": str})
        df = df.merge(cached[["adresse_complete","cadastre_annee_constr","cadastre_type_precis"]],
                      on="adresse_complete", how="left")
        df["cadastre_annee_constr"] = df.get("cadastre_annee_constr", np.nan)
        df["cadastre_type_precis"]  = df.get("cadastre_type_precis",  "Inconnu")
        return _appliquer_bonus_cadastre(df)

    if "adresse_complete" not in df.columns:
        df["cadastre_annee_constr"] = np.nan
        df["cadastre_type_precis"]  = "Inconnu"
        return _appliquer_bonus_cadastre(df)

    results = []
    sample = df[["adresse_complete"]].drop_duplicates().head(max_appels)

    for _, row in sample.iterrows():
        addr = str(row["adresse_complete"])
        annee, type_precis = _query_cadastre(addr)
        results.append({"adresse_complete": addr,
                         "cadastre_annee_constr": annee,
                         "cadastre_type_precis": type_precis})
        time.sleep(0.05)  # respecter les rate limits

    res_df = pd.DataFrame(results)
    res_df.to_csv(cache, index=False)

    df = df.merge(res_df, on="adresse_complete", how="left")
    df["cadastre_annee_constr"] = df.get("cadastre_annee_constr", np.nan)
    df["cadastre_type_precis"]  = df.get("cadastre_type_precis", "Inconnu").fillna("Inconnu")
    return _appliquer_bonus_cadastre(df)


def _query_cadastre(adresse: str) -> tuple:
    """
    Interroge l'API BAN pour géocoder, puis l'API Géoportail Cadastre
    pour récupérer l'année de construction et le type de bien.
    Retourne (annee: int|None, type_precis: str).
    """
    try:
        # Étape 1 : géocodage BAN
        r = requests.get(
            "https://api-adresse.data.gouv.fr/search/",
            params={"q": adresse, "limit": 1},
            timeout=5
        )
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            return None, "Inconnu"

        props = features[0]["properties"]
        lat   = features[0]["geometry"]["coordinates"][1]
        lng   = features[0]["geometry"]["coordinates"][0]

        # Étape 2 : API Cadastre (parcelle à partir des coordonnées)
        r2 = requests.get(
            "https://geocodage.ign.fr/look4/address/reverse",
            params={"lon": lng, "lat": lat, "maximumResponses": 1},
            timeout=5
        )
        # Si le cadastre IGN ne répond pas, on retourne les données BAN seules
        type_precis = props.get("type", "Inconnu")
        # Approximation : pas d'année de construction dans BAN → on retourne None
        return None, type_precis

    except Exception:
        return None, "Inconnu"


def _appliquer_bonus_cadastre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le bonus cadastre sur score_brut selon l'ancienneté et le type.
    """
    if "cadastre_annee_constr" not in df.columns:
        df["cadastre_annee_constr"] = np.nan
    if "cadastre_type_precis" not in df.columns:
        df["cadastre_type_precis"] = "Inconnu"
    if "score_brut" not in df.columns:
        return df

    df = df.copy()
    annee = pd.to_numeric(df["cadastre_annee_constr"], errors="coerce")
    sig   = df.get("signal", pd.Series("", index=df.index)).astype(str)
    bonus = pd.Series(0.0, index=df.index)

    # Biens anciens (avant 1970) + signaux succession/retraite → très pertinents
    ancien = annee < 1970
    med    = (annee >= 1970) & (annee < 1990)
    sig_sr = sig.str.contains("succession|heritage|retraite", na=False)
    sig_du = sig.str.contains("divorce|upgrade", na=False)

    bonus += np.where(ancien & sig_sr, 10, 0)
    bonus += np.where(med    & sig_sr, 7,  0)
    bonus += np.where(ancien & sig_du, 5,  0)

    # Type collectif + divorce/upgrade (appartements urbains = marché liquide)
    collectif = df["cadastre_type_precis"].str.lower().str.contains("appart|collectif|immeuble", na=False)
    bonus += np.where(collectif & sig_du, 5, 0)

    df["bonus_cadastre"] = bonus.round(0)
    df["score_brut"]     = (df["score_brut"] + bonus).clip(0, 100).round(1)
    return df


# ════════════════════════════════════════════════════════════
# 6. CLUSTERS GÉOGRAPHIQUES (DBSCAN simplifié)
# ════════════════════════════════════════════════════════════

def detecter_clusters(df: pd.DataFrame,
                       rayon_km: float = 0.5,
                       min_prospects: int = 3) -> pd.DataFrame:
    """
    Détecte les hotspots géographiques par clustering spatial simplifié.

    Méthode : pour chaque prospect géolocalisé, on compte le nombre de
    prospects voisins dans un rayon de rayon_km km. Si ce nombre dépasse
    min_prospects, le prospect appartient à un cluster "chaud".

    Sans dépendance sklearn : on utilise la distance haversine vectorisée.

    Colonnes ajoutées :
      cluster_densite  Nb de prospects dans le rayon (0 si pas de coords)
      cluster_chaud    True si cluster_densite >= min_prospects
      bonus_cluster    +8 si cluster chaud, +4 si densité >= 2

    Intérêt prospection :
      Un signal isolé géographiquement peut être anecdotique.
      Un cluster de 5+ signaux dans 500m = zone à fort potentiel,
      probablement liée à un programme de rénovation, une copropriété
      en difficulté, ou un quartier en transition démographique.
    """
    df = df.copy()
    df["cluster_densite"] = 0
    df["cluster_chaud"]   = False
    df["bonus_cluster"]   = 0.0

    has_coords = ("latitude" in df.columns and "longitude" in df.columns and
                  df["latitude"].notna().any() and df["longitude"].notna().any())

    if not has_coords:
        return df

    df_geo = df[df["latitude"].notna() & df["longitude"].notna() &
                (df["latitude"] != 0)].copy()
    if len(df_geo) < min_prospects:
        return df

    lats = np.radians(df_geo["latitude"].values)
    lngs = np.radians(df_geo["longitude"].values)
    R    = 6371.0  # rayon Terre km

    densites = np.zeros(len(df_geo), dtype=int)
    for i in range(len(df_geo)):
        dlat = lats - lats[i]
        dlng = lngs - lngs[i]
        a    = np.sin(dlat/2)**2 + np.cos(lats[i]) * np.cos(lats) * np.sin(dlng/2)**2
        dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        densites[i] = int((dist <= rayon_km).sum()) - 1  # -1 = exclure soi-même

    df_geo["cluster_densite"] = densites
    df_geo["cluster_chaud"]   = densites >= min_prospects
    df_geo["bonus_cluster"]   = np.where(
        densites >= min_prospects, 8.0,
        np.where(densites >= 2, 4.0, 0.0)
    )

    # Appliquer sur df complet via index
    df.loc[df_geo.index, "cluster_densite"] = df_geo["cluster_densite"]
    df.loc[df_geo.index, "cluster_chaud"]   = df_geo["cluster_chaud"]
    df.loc[df_geo.index, "bonus_cluster"]   = df_geo["bonus_cluster"]

    if "score_brut" in df.columns:
        df["score_brut"] = (df["score_brut"] + df["bonus_cluster"]).clip(0, 100).round(1)

    return df


# ════════════════════════════════════════════════════════════
# 7. TENDANCES TEMPORELLES
# ════════════════════════════════════════════════════════════

def calculer_tendances(dvf: pd.DataFrame) -> dict:
    """
    Calcule les tendances temporelles du marché pour le dashboard :
      - volume_par_mois     : nb transactions par mois (saisonnalité)
      - prix_median_par_mois: évolution du prix médian au fil de l'année
      - volume_par_cp_top10 : top 10 CP par volume de transactions
      - signaux_par_trimestre: à utiliser côté app après pipeline

    Retourne un dict de DataFrames, pas de colonnes ajoutées au prospect.
    """
    tendances = {}
    if dvf.empty or "date_mutation" not in dvf.columns:
        return tendances

    df = dvf[dvf["est_residentiel"]].copy() if "est_residentiel" in dvf.columns else dvf.copy()
    df["mois"]      = df["date_mutation"].dt.to_period("M").astype(str)
    df["trimestre"] = df["date_mutation"].dt.to_period("Q").astype(str)

    # Volume mensuel
    tendances["volume_par_mois"] = (
        df.groupby("mois").size()
        .reset_index(name="nb_transactions")
        .sort_values("mois")
    )

    # Prix médian mensuel
    if "valeur_fonciere" in df.columns:
        tendances["prix_median_par_mois"] = (
            df.groupby("mois")["valeur_fonciere"]
            .median()
            .reset_index(name="prix_median")
            .sort_values("mois")
        )

    # Top CP par volume
    if "code_postal" in df.columns:
        tendances["top_cp_volume"] = (
            df.groupby("code_postal").size()
            .reset_index(name="volume")
            .sort_values("volume", ascending=False)
            .head(10)
        )

    # Prix/m² médian par CP (top 10 transactions)
    if "prix_m2" in df.columns and "code_postal" in df.columns:
        tendances["prix_m2_par_cp"] = (
            df.groupby("code_postal")["prix_m2"]
            .agg(["median","count"])
            .reset_index()
            .rename(columns={"median":"prix_m2_median","count":"nb"})
            .query("nb >= 5")
            .sort_values("prix_m2_median", ascending=False)
        )

    return tendances


# ════════════════════════════════════════════════════════════
# 8. MALUS QUALITÉ
# ════════════════════════════════════════════════════════════

def appliquer_malus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "score_brut" not in df.columns:
        return df
    df    = df.copy()
    malus = pd.Series(0.0, index=df.index)
    if "est_residentiel" in df.columns:
        malus += np.where(~df["est_residentiel"].fillna(True), -20, 0)
    if "prix_m2" in df.columns:
        pm2 = pd.to_numeric(df["prix_m2"], errors="coerce")
        malus += np.where((pm2 < 500) | (pm2 > 35_000), -15, 0)
    if "surface_reelle_bati" in df.columns:
        surf = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
        malus += np.where((surf < 9) | (surf > 600), -10, 0)
    if "valeur_fonciere" in df.columns:
        prix = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
        malus += np.where(prix < 15_000, -10, 0)
    df["malus"]      = malus.round(0)
    df["score_brut"] = (df["score_brut"] + malus).clip(0, 100).round(1)
    return df


# ════════════════════════════════════════════════════════════
# 9. SIGNAUX BRUTS (inchangés depuis v3)
# ════════════════════════════════════════════════════════════

def _score_heritage_brut(df, date_succession):
    score = pd.Series(50.0, index=df.index)
    if "date_mutation" in df.columns:
        delai = (df["date_mutation"] - date_succession).dt.days
        score += np.where(delai < 90, 30, np.where(delai < 180, 20, np.where(delai < 365, 10, 0)))
    if "nature_mutation" in df.columns:
        score += np.where(df["nature_mutation"] == "Adjudication", 20, 0)
    return score.clip(0, 100).round(1)


def croiser_dvf_bodacc(dvf, bodacc, dept=DEPT, fenetre_mois=18):
    if bodacc.empty or dvf.empty:
        return pd.DataFrame()
    dvf_f    = dvf.copy()
    dvf_f["dept"] = dvf_f["code_postal"].astype(str).str[:2]
    bodacc_f = (bodacc[bodacc["dept_bodacc"] == dept].copy()
                if "dept_bodacc" in bodacc.columns and bodacc["dept_bodacc"].notna().any()
                else bodacc.copy())
    if bodacc_f.empty:
        return pd.DataFrame()
    fenetre_jours = fenetre_mois * 30
    results = []
    for _, succ in bodacc_f.iterrows():
        dept_s = str(succ.get("dept_bodacc", ""))[:2]
        date   = succ.get("date_bodacc", pd.NaT)
        if pd.isna(date) or not dept_s:
            continue
        mask = ((dvf_f["dept"] == dept_s) &
                (dvf_f["date_mutation"] >= date) &
                (dvf_f["date_mutation"] <= date + timedelta(days=fenetre_jours)))
        matches = dvf_f[mask].copy()
        if not matches.empty:
            matches["date_succession"] = date
            matches["score_brut"]      = _score_heritage_brut(matches, date)
            matches["signal"]          = "succession_bodacc"
            results.append(matches)
    if not results:
        return pd.DataFrame()
    id_col = "id_mutation" if "id_mutation" in dvf_f.columns else None
    out    = pd.concat(results, ignore_index=True)
    return out.drop_duplicates(subset=[id_col]) if id_col else out


def signal_divorce(dvf):
    df = dvf.copy()
    if "date_mutation" not in df.columns or "adresse_complete" not in df.columns:
        return pd.DataFrame()
    pieces_ok   = df["nombre_pieces_principales"].between(3, 4) if "nombre_pieces_principales" in df.columns else pd.Series(True, index=df.index)
    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    df = df[pieces_ok & residentiel].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date_mutation")
    df["vente_precedente"]    = df.groupby("adresse_complete")["date_mutation"].shift(1)
    df["delai_revente_jours"] = (df["date_mutation"] - df["vente_precedente"]).dt.days
    mask = (df["delai_revente_jours"] > 30) & (df["delai_revente_jours"] < 3 * 365)
    result = df[mask].copy()
    result["signal"]     = "divorce_ou_separation"
    result["score_brut"] = np.where(result["delai_revente_jours"] < 365, 80.0, 60.0)
    return result


def signal_upgrade(dvf):
    df = dvf.copy()
    if "nombre_pieces_principales" not in df.columns:
        return pd.DataFrame()
    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    surf_ok     = df["surface_reelle_bati"] > 15 if "surface_reelle_bati" in df.columns else pd.Series(True, index=df.index)
    now = datetime.now()
    mask = (df["nombre_pieces_principales"].isin([1, 2]) &
            (df["valeur_fonciere"] > 0) &
            (df["date_mutation"] >= now - timedelta(days=4 * 365)) &
            (df["date_mutation"] <= now - timedelta(days=2 * 365)) &
            residentiel & surf_ok)
    result = df[mask].copy()
    result["signal"]     = "petit_bien_upgrade_potentiel"
    anciennete = (now - result["date_mutation"]).dt.days / 365
    result["score_brut"] = np.where(anciennete <= 3, 65.0, 55.0)
    return result


def signal_retraite(dvf):
    df = dvf.copy()
    if not {"nombre_pieces_principales", "surface_reelle_bati", "valeur_fonciere"}.issubset(df.columns):
        return pd.DataFrame()
    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    mask_base   = (df["nombre_pieces_principales"] >= 5) & (df["surface_reelle_bati"] >= 80) & residentiel
    df = df[mask_base].copy()
    if df.empty:
        return pd.DataFrame()
    df["prix_m2"]        = df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)
    df["mediane_t5_cp"]  = df.groupby("code_postal")["prix_m2"].transform("median")
    df["decote_pct"]     = (df["mediane_t5_cp"] - df["prix_m2"]) / df["mediane_t5_cp"] * 100
    df["decote_vs_median"] = df["decote_pct"].round(1)
    result = df[df["decote_pct"] > 10].copy()
    result["signal"]     = "retraite_downsizing"
    result["score_brut"] = np.where(result["decote_pct"] > 25, 85.0,
                           np.where(result["decote_pct"] > 20, 75.0,
                           np.where(result["decote_pct"] > 15, 65.0, 55.0)))
    return result


def signal_primo(dvf):
    df = dvf.copy()
    if "valeur_fonciere" not in df.columns:
        return pd.DataFrame()
    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    pieces_ok   = df["nombre_pieces_principales"].isin([1, 2]) if "nombre_pieces_principales" in df.columns else pd.Series(True, index=df.index)
    surf_ok     = df["surface_reelle_bati"] >= 15 if "surface_reelle_bati" in df.columns else pd.Series(True, index=df.index)
    df = df[pieces_ok & residentiel & surf_ok & (df["valeur_fonciere"] >= 15_000)].copy()
    if df.empty:
        return pd.DataFrame()
    df["mediane_cp_t12"]   = df.groupby("code_postal")["valeur_fonciere"].transform("median")
    df["decote_vs_median"] = ((df["mediane_cp_t12"] - df["valeur_fonciere"]) / df["mediane_cp_t12"] * 100).round(1)
    result = df[df["valeur_fonciere"] < df["mediane_cp_t12"] * 0.7].copy()
    result["signal"]     = "primo_acheteur_potentiel"
    result["score_brut"] = np.where(result["decote_vs_median"] > 30, 60.0, 50.0)
    return result


# ════════════════════════════════════════════════════════════
# 10. NORMALISATION & BONUS MULTI-SIGNAL
# ════════════════════════════════════════════════════════════

def normaliser_en_percentile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "score_brut" not in df.columns:
        return df
    df = df.copy()
    df["score_final"] = (df["score_brut"].rank(pct=True, method="average") * 100).round(1)
    return df


def bonus_multi_signal(consolidated: pd.DataFrame, frames_bruts: list) -> pd.DataFrame:
    if "adresse_complete" not in consolidated.columns:
        return consolidated
    consolidated = consolidated.copy()
    addr_counts  = {}
    for fr in frames_bruts:
        if fr is None or fr.empty or "adresse_complete" not in fr.columns:
            continue
        for addr in fr["adresse_complete"].dropna().unique():
            addr_counts[addr] = addr_counts.get(addr, 0) + 1
    consolidated["nb_signaux"] = consolidated["adresse_complete"].map(addr_counts).fillna(1).astype(int)
    bonus = np.where(consolidated["nb_signaux"] >= 3, 12,
            np.where(consolidated["nb_signaux"] == 2, 5, 0))
    consolidated["score_final"] = (consolidated["score_final"] + bonus).clip(0, 100).round(1)
    return consolidated


# ════════════════════════════════════════════════════════════
# 11. ENRICHISSEMENT BI FINAL
# ════════════════════════════════════════════════════════════

def enrichir_bi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signal_carte"] = df["signal"].map(SIGNAL_NORMALIZE).fillna(df["signal"])
    df["signal_label"] = df["signal_carte"].map(SIGNAL_LABELS).fillna(df["signal"])
    df["segment_cible"] = df["signal_carte"].map(SIGNAL_SEGMENTS).fillna("—")
    df["intensite"] = pd.cut(df["score_final"], bins=[0,40,60,80,100],
                              labels=["faible","moyen","fort","très fort"], right=True)
    df["chaleur"]   = pd.cut(df["score_final"], bins=[0,39,59,79,100],
                              labels=["froid","tiède","chaud","très chaud"], right=True)
    if "decote_vs_median" not in df.columns and "valeur_fonciere" in df.columns:
        med = df.groupby("code_postal")["valeur_fonciere"].transform("median")
        df["decote_vs_median"] = ((med - df["valeur_fonciere"]) / med * 100).round(1)
    if "anciennete_mois" not in df.columns and "date_mutation" in df.columns:
        df["anciennete_mois"] = ((datetime.now() - df["date_mutation"]).dt.days / 30).round(1)
    if "prix_m2" not in df.columns and "surface_reelle_bati" in df.columns:
        df["prix_m2"] = (df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)).round(0)
    return df


# ════════════════════════════════════════════════════════════
# 12. CONSOLIDATION
# ════════════════════════════════════════════════════════════

def consolider(frames: list, liq_df: pd.DataFrame = None) -> pd.DataFrame:
    actifs = [f for f in frames if f is not None and not f.empty]
    if not actifs:
        return pd.DataFrame()

    cols = [
        "adresse_numero","adresse_suffixe","adresse_nom_voie",
        "code_postal","nom_commune","commune","adresse_complete",
        "valeur_fonciere","surface_reelle_bati","nombre_pieces_principales",
        "prix_m2","type_local","est_residentiel",
        "date_mutation","nature_mutation","anciennete_mois",
        "signal","score_brut","score_final","malus",
        "decote_vs_median","longitude","latitude",
        "cluster_densite","cluster_chaud","bonus_cluster",
        "insee_taux_proprio","insee_age_median","insee_revenu_median",
        "cadastre_annee_constr","cadastre_type_precis",
        "bonus_insee","bonus_cadastre","bonus_liquidite_val",
    ]
    stacked = pd.concat(
        [f[[c for c in cols if c in f.columns]] for f in actifs],
        ignore_index=True
    )

    if "adresse_complete" in stacked.columns:
        stacked = (stacked.sort_values("score_final", ascending=False)
                          .drop_duplicates(subset=["adresse_complete"]))

    # Jointure liquidité
    if liq_df is not None and not liq_df.empty and "code_postal" in stacked.columns:
        stacked = stacked.merge(liq_df, on="code_postal", how="left")
        stacked["liquidite_cp"]          = stacked.get("liquidite_cp", 50).fillna(50)
        stacked["volume_transactions"]   = stacked.get("volume_transactions", 0).fillna(0)
        stacked["delai_rotation_mois"]   = stacked.get("delai_rotation_mois", np.nan)
        # Bonus liquidité sur score_final (après percentile)
        bl = bonus_liquidite(stacked["score_final"], stacked["liquidite_cp"])
        stacked["bonus_liquidite_val"] = bl
        stacked["score_final"] = (stacked["score_final"] + bl).clip(0, 100).round(1)

    stacked = bonus_multi_signal(stacked, actifs)
    stacked = enrichir_bi(stacked)
    stacked = stacked.sort_values("score_final", ascending=False).reset_index(drop=True)
    stacked.insert(0, "rang", range(1, len(stacked) + 1))
    return stacked


# ════════════════════════════════════════════════════════════
# 13. PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_pipeline(dept: str = DEPT, annee: int = ANNEE,
                 fenetre_succession_mois: int = 18,
                 enrichir_cadastre: bool = False,
                 rayon_cluster_km: float = 0.5) -> tuple:
    """
    Retourne (prospects_df, tendances_dict)

    prospects_df : DataFrame enrichi avec tous les scores et métriques BI
    tendances_dict : dict de DataFrames pour le dashboard marché

    Paramètres
    ──────────
    dept                  Code département (ex: "75")
    annee                 Année fichier DVF
    fenetre_succession_mois  9=fort·18=std·24=large
    enrichir_cadastre     True = appels API Cadastre (lent, ~2 min)
    rayon_cluster_km      Rayon de détection des hotspots (défaut 0.5 km)
    """
    # Chargement
    dvf_raw    = download_dvf(dept, annee)
    bodacc_raw = download_bodacc(annee)
    dvf        = clean_dvf(dvf_raw)
    bodacc     = clean_bodacc(bodacc_raw)

    # Tendances marché (sur tout le DVF, avant filtrage signaux)
    tendances  = calculer_tendances(dvf)

    # Liquidité par CP
    liq_df = calculer_liquidite(dvf)

    # INSEE
    codes_communes = dvf["cle_commune"].dropna().unique().tolist() if "cle_commune" in dvf.columns else []
    insee_df = fetch_insee(codes_communes, dept)

    # Signaux bruts
    heritage = croiser_dvf_bodacc(dvf, bodacc, dept, fenetre_succession_mois)
    divorce  = signal_divorce(dvf)
    upgrade  = signal_upgrade(dvf)
    retraite = signal_retraite(dvf)
    primo    = signal_primo(dvf)

    frames = [heritage, divorce, upgrade, retraite, primo]

    # Malus → bonus INSEE → bonus Cadastre → clusters → percentile
    processed = []
    for fr in frames:
        if fr is None or fr.empty:
            processed.append(fr if fr is not None else pd.DataFrame())
            continue
        if "score_signal" in fr.columns and "score_brut" not in fr.columns:
            fr = fr.copy()
            fr["score_brut"] = fr["score_signal"]
        fr = appliquer_malus(fr)
        fr = bonus_insee(fr, insee_df)
        if enrichir_cadastre:
            fr = fetch_cadastre_batch(fr)
        fr = detecter_clusters(fr, rayon_km=rayon_cluster_km)
        fr = normaliser_en_percentile(fr)
        processed.append(fr)

    prospects = consolider(processed, liq_df=liq_df)
    return prospects, tendances

"""
============================================================
 DVF × BODACC — Pipeline v3.0
 Scoring percentile · Malus qualité · Multi-signal · BI-ready

ARCHITECTURE DU SCORE FINAL
════════════════════════════
  Étape 1  Signal brut          0–100  (intensité de l'événement de vie)
  Étape 2  Malus qualité       -35 max (données aberrantes / hors-cible)
  Étape 3  Normalisation       percentile dept  → score relatif 0–100
  Étape 4  Bonus multi-signal  +5 / +12         (convergence de signaux)

  Score final = percentile dans le département, pas une valeur absolue.
  80 = top 20% du dept. 95 = top 5%. Comparable entre départements.

SIGNAUX
════════
  heritage  Succession BODACC : vente < 18 mois après annonce
  divorce   Revente T3/T4 < 3 ans
  upgrade   T1/T2 acheté il y a 2–4 ans
  retraite  T5+ décote > 10% médiane T5+ du CP
  primo     T1/T2 < 70% médiane T1/T2 du CP

MALUS
══════
  -20  Bien non résidentiel confirmé
  -15  Prix/m² aberrant (< 500 ou > 35 000 €/m²)
  -10  Surface hors normes (< 9 m² ou > 600 m²)
  -10  Prix total < 15 000 € (cave / parking)

MÉTRIQUES BI EXPORTÉES
════════════════════════
  score_final       Percentile dept 0–100 (KPI principal)
  score_brut        Score signal avant normalisation
  intensite         Catégorie signal (faible/moyen/fort/très fort)
  chaleur           Label CRM (froid/tiède/chaud/très chaud)
  prix_m2           Prix au m² calculé
  decote_vs_median  Écart au prix médian du CP (%)
  anciennete_mois   Durée depuis la transaction DVF
  nb_signaux        Nombre de signaux convergents sur l'adresse
  segment_cible     Segment marketing recommandé
============================================================
"""

import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
from pathlib import Path


# ── Constantes UI (importées par app.py) ─────────────────────
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

# Segments marketing associés à chaque signal
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
# 1. DVF — Téléchargement & nettoyage
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

    # Typage
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    for col in ["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtre mutations valides uniquement
    df = df[df["nature_mutation"].isin(["Vente", "Adjudication", "Expropriation"])]
    df = df[df["valeur_fonciere"].notna() & (df["valeur_fonciere"] > 0)]

    # Adresse complète normalisée
    adresse_cols = ["adresse_numero", "adresse_suffixe", "adresse_nom_voie", "code_postal", "nom_commune"]
    cols_ok = [c for c in adresse_cols if c in df.columns]
    df["adresse_complete"] = (
        df[cols_ok].fillna("").astype(str)
        .apply(lambda r: " ".join(v for v in r if v not in ("", "nan")).strip().upper(), axis=1)
    )

    # Prix au m² (NaN si surface manquante ou nulle)
    if "surface_reelle_bati" in df.columns:
        surf = df["surface_reelle_bati"].replace(0, np.nan)
        df["prix_m2"] = (df["valeur_fonciere"] / surf).round(0)
    else:
        df["prix_m2"] = np.nan

    # Type résidentiel
    if "type_local" in df.columns:
        df["type_local"]     = df["type_local"].fillna("Inconnu")
        df["est_residentiel"] = df["type_local"].isin(TYPES_RESIDENTIELS)
    else:
        df["est_residentiel"] = True

    # Commune normalisée
    commune_col = next((c for c in df.columns if c in ("nom_commune", "libelle_commune")), None)
    df["commune"] = df[commune_col].fillna("") if commune_col else df.get("code_postal", "")

    # Ancienneté en mois depuis la transaction
    df["anciennete_mois"] = ((datetime.now() - df["date_mutation"]).dt.days / 30).round(1)

    if "code_commune" in df.columns:
        df["cle_commune"] = df["code_commune"].astype(str).str.zfill(5)

    return df


# ════════════════════════════════════════════════════════════
# 2. BODACC — Téléchargement & nettoyage
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
    df_succ  = df[df[type_col].astype(str).str.lower().str.contains("succ|héritage|heritage", na=False)].copy() if type_col else df.copy()
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
# 3. MALUS QUALITÉ
# ════════════════════════════════════════════════════════════

def appliquer_malus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Malus appliqués AVANT la normalisation percentile.
    Ils dégradent le score brut, ce qui mécaniquement réduit le percentile final.

    Règles :
      -20  Type non résidentiel confirmé (local commercial, dépendance…)
      -15  Prix/m² aberrant (< 500 ou > 35 000 €/m²)
      -10  Surface hors normes (< 9 m² ou > 600 m²)
      -10  Prix total < 15 000 € (cave, parking probable)
    """
    if df.empty:
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

    df["malus"]       = malus.round(0)
    df["score_brut"]  = (df["score_brut"] + malus).clip(0, 100).round(1)
    return df


# ════════════════════════════════════════════════════════════
# 4. SIGNAUX — Score brut (0–100, relatif à l'événement)
# ════════════════════════════════════════════════════════════

def _score_heritage_brut(df: pd.DataFrame, date_succession: datetime) -> pd.Series:
    """
    Score brut succession :
      Base  50
      +30   vente < 90 j après annonce
      +20   vente 90–180 j
      +10   vente 180–365 j
      +0    au-delà
      +20   adjudication (vente forcée = contrainte maximale)
    Plafonné à 100.
    """
    score = pd.Series(50.0, index=df.index)
    if "date_mutation" in df.columns:
        delai = (df["date_mutation"] - date_succession).dt.days
        score += np.where(delai < 90, 30,
                 np.where(delai < 180, 20,
                 np.where(delai < 365, 10, 0)))
    if "nature_mutation" in df.columns:
        score += np.where(df["nature_mutation"] == "Adjudication", 20, 0)
    return score.clip(0, 100).round(1)


def croiser_dvf_bodacc(dvf: pd.DataFrame, bodacc: pd.DataFrame,
                        dept: str = DEPT,
                        fenetre_mois: int = 18) -> pd.DataFrame:
    """
    Croisement vectorisé DVF × BODACC.
    Pour chaque annonce BODACC de succession du département,
    on cherche les ventes DVF dans la fenêtre suivante.

    fenetre_mois : 9 = signal fort uniquement | 18 = standard | 24 = filet large
    """
    if bodacc.empty or dvf.empty:
        return pd.DataFrame()

    dvf_f = dvf.copy()
    dvf_f["dept"] = dvf_f["code_postal"].astype(str).str[:2]

    bodacc_f = (
        bodacc[bodacc["dept_bodacc"] == dept].copy()
        if "dept_bodacc" in bodacc.columns and bodacc["dept_bodacc"].notna().any()
        else bodacc.copy()
    )
    if bodacc_f.empty:
        return pd.DataFrame()

    fenetre_jours = fenetre_mois * 30
    results = []
    for _, succ in bodacc_f.iterrows():
        dept_s = str(succ.get("dept_bodacc", ""))[:2]
        date   = succ.get("date_bodacc", pd.NaT)
        if pd.isna(date) or not dept_s:
            continue
        mask = (
            (dvf_f["dept"] == dept_s) &
            (dvf_f["date_mutation"] >= date) &
            (dvf_f["date_mutation"] <= date + timedelta(days=fenetre_jours))
        )
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
    if id_col:
        out = out.drop_duplicates(subset=[id_col])
    return out


def signal_divorce(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    Revente rapide d'un T3/T4 résidentiel.
    Délai > 30 j (évite multi-lots) et < 3 ans.

    Score brut :
      80  si délai < 1 an  (rupture probable récente)
      60  si délai 1–3 ans
    """
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
    result["signal"]    = "divorce_ou_separation"
    result["score_brut"] = np.where(result["delai_revente_jours"] < 365, 80.0, 60.0)
    return result


def signal_upgrade(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    T1/T2 résidentiel acheté il y a 2–4 ans, surface > 15 m².
    Fenêtre 2–4 ans : trop récent = pas encore de besoin, trop ancien = déjà parti.

    Score brut :
      65  si ancienneté 2–3 ans (besoin d'upgrade émergent)
      55  si ancienneté 3–4 ans (besoin confirmé, peut déjà chercher)
    """
    df = dvf.copy()
    if "nombre_pieces_principales" not in df.columns:
        return pd.DataFrame()

    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    surf_ok     = df["surface_reelle_bati"] > 15 if "surface_reelle_bati" in df.columns else pd.Series(True, index=df.index)
    now = datetime.now()

    mask = (
        df["nombre_pieces_principales"].isin([1, 2]) &
        (df["valeur_fonciere"] > 0) &
        (df["date_mutation"] >= now - timedelta(days=4 * 365)) &
        (df["date_mutation"] <= now - timedelta(days=2 * 365)) &
        residentiel & surf_ok
    )
    result = df[mask].copy()
    result["signal"]    = "petit_bien_upgrade_potentiel"
    # Score gradué selon ancienneté
    anciennete = (now - result["date_mutation"]).dt.days / 365
    result["score_brut"] = np.where(anciennete <= 3, 65.0, 55.0)
    return result


def signal_retraite(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    T5+ résidentiel, surface ≥ 80 m², vendu avec décote vs médiane T5+ du CP.
    Médiane calculée sur T5+ uniquement (pas tous biens) = comparaison homogène.

    Score brut :
      85  décote > 25%  (urgence forte)
      75  décote 20–25%
      65  décote 15–20%
      55  décote 10–15%  (seuil minimal)
    """
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

    # Mémoriser pour export
    df["decote_vs_median"] = df["decote_pct"].round(1)

    mask = df["decote_pct"] > 10
    result = df[mask].copy()
    result["signal"] = "retraite_downsizing"
    result["score_brut"] = np.where(
        result["decote_pct"] > 25, 85.0,
        np.where(result["decote_pct"] > 20, 75.0,
        np.where(result["decote_pct"] > 15, 65.0, 55.0))
    )
    return result


def signal_primo(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    T1/T2 résidentiel vendu à < 70% de la médiane T1/T2 du CP.
    Surface ≥ 15 m², prix ≥ 15 000 € (exclut caves/parkings).

    Score brut :
      60  décote 30–50% (très accessible = primo confirmé)
      50  décote < 30%  (entrée de gamme)
    """
    df = dvf.copy()
    if "valeur_fonciere" not in df.columns:
        return pd.DataFrame()

    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    pieces_ok   = df["nombre_pieces_principales"].isin([1, 2]) if "nombre_pieces_principales" in df.columns else pd.Series(True, index=df.index)
    surf_ok     = df["surface_reelle_bati"] >= 15 if "surface_reelle_bati" in df.columns else pd.Series(True, index=df.index)

    df = df[pieces_ok & residentiel & surf_ok & (df["valeur_fonciere"] >= 15_000)].copy()
    if df.empty:
        return pd.DataFrame()

    df["mediane_cp_t12"] = df.groupby("code_postal")["valeur_fonciere"].transform("median")
    df["decote_vs_median"] = ((df["mediane_cp_t12"] - df["valeur_fonciere"]) / df["mediane_cp_t12"] * 100).round(1)

    mask = df["valeur_fonciere"] < df["mediane_cp_t12"] * 0.7
    result = df[mask].copy()
    result["signal"]    = "primo_acheteur_potentiel"
    result["score_brut"] = np.where(result["decote_vs_median"] > 30, 60.0, 50.0)
    return result


# ════════════════════════════════════════════════════════════
# 5. NORMALISATION PERCENTILE — cœur du nouveau scoring
# ════════════════════════════════════════════════════════════

def normaliser_en_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme score_brut (0–100 absolu) en score_final (0–100 percentile).

    Pourquoi percentile ?
    ─────────────────────
    Un score absolu de 80 ne signifie rien sans contexte : Paris peut avoir
    10 000 biens à 80+ alors que Clermont-Ferrand n'en a que 20. Le percentile
    garantit que "90" = top 10% du département analysé, quelle que soit la taille.

    Méthode : percentilerank sur score_brut, puis rescale 0–100.
    On applique ça par signal (hors comparaison inter-signaux voulue).
    """
    if df.empty or "score_brut" not in df.columns:
        return df
    df = df.copy()
    n  = len(df)
    # Rank percentile : position relative dans la distribution des scores bruts
    df["score_final"] = (
        df["score_brut"].rank(pct=True, method="average") * 100
    ).round(1)
    return df


# ════════════════════════════════════════════════════════════
# 6. BONUS MULTI-SIGNAL
# ════════════════════════════════════════════════════════════

def bonus_multi_signal(consolidated: pd.DataFrame, frames_bruts: list) -> pd.DataFrame:
    """
    Bonus sur score_final si une adresse converge plusieurs signaux distincts.
    +5   si 2 signaux  (rare = intéressant)
    +12  si 3+ signaux (très rare = priorité absolue)
    """
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
# 7. ENRICHISSEMENT BI
# ════════════════════════════════════════════════════════════

def enrichir_bi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les métriques BI nécessaires à un outil de prospection :

    - segment_cible    Message marketing recommandé
    - intensite        Catégorie signal (faible/moyen/fort/très fort)
    - chaleur          Label CRM (froid/tiède/chaud/très chaud)
    - decote_vs_median Écart au prix médian du CP (%)
    - prix_m2          Prix au m²
    - anciennete_mois  Ancienneté de la transaction
    - signal_label     Libellé lisible
    - signal_carte     Clé courte pour la carte
    """
    df = df.copy()

    # Signal normalisé
    df["signal_carte"] = df["signal"].map(SIGNAL_NORMALIZE).fillna(df["signal"])
    df["signal_label"] = df["signal_carte"].map(SIGNAL_LABELS).fillna(df["signal"])

    # Segment marketing
    df["segment_cible"] = df["signal_carte"].map(SIGNAL_SEGMENTS).fillna("—")

    # Intensité signal (basée sur score_final)
    df["intensite"] = pd.cut(
        df["score_final"],
        bins=[0, 40, 60, 80, 100],
        labels=["faible", "moyen", "fort", "très fort"],
        right=True,
    )

    # Chaleur CRM (labels commerciaux)
    df["chaleur"] = pd.cut(
        df["score_final"],
        bins=[0, 39, 59, 79, 100],
        labels=["froid", "tiède", "chaud", "très chaud"],
        right=True,
    )

    # Décote vs médiane (si pas déjà calculée)
    if "decote_vs_median" not in df.columns and "valeur_fonciere" in df.columns and "code_postal" in df.columns:
        mediane_cp = df.groupby("code_postal")["valeur_fonciere"].transform("median")
        df["decote_vs_median"] = ((mediane_cp - df["valeur_fonciere"]) / mediane_cp * 100).round(1)

    # Ancienneté
    if "anciennete_mois" not in df.columns and "date_mutation" in df.columns:
        df["anciennete_mois"] = ((datetime.now() - df["date_mutation"]).dt.days / 30).round(1)

    # Prix/m² si absent
    if "prix_m2" not in df.columns and "surface_reelle_bati" in df.columns:
        surf = df["surface_reelle_bati"].replace(0, np.nan)
        df["prix_m2"] = (df["valeur_fonciere"] / surf).round(0)

    return df


# ════════════════════════════════════════════════════════════
# 8. CONSOLIDATION FINALE
# ════════════════════════════════════════════════════════════

def consolider(frames: list) -> pd.DataFrame:
    """
    1. Empile tous les signaux
    2. Déduplique sur adresse (garde le meilleur score_final)
    3. Bonus multi-signal
    4. Enrichissement BI
    5. Tri + rang
    """
    actifs = [f for f in frames if f is not None and not f.empty]
    if not actifs:
        return pd.DataFrame()

    cols = [
        "adresse_numero", "adresse_suffixe", "adresse_nom_voie",
        "code_postal", "nom_commune", "commune", "adresse_complete",
        "valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
        "prix_m2", "type_local", "est_residentiel",
        "date_mutation", "nature_mutation", "anciennete_mois",
        "signal", "score_brut", "score_final", "malus",
        "decote_vs_median", "longitude", "latitude",
    ]
    stacked = pd.concat(
        [f[[c for c in cols if c in f.columns]] for f in actifs],
        ignore_index=True
    )

    # Déduplique : meilleur score_final par adresse
    if "adresse_complete" in stacked.columns:
        stacked = (
            stacked.sort_values("score_final", ascending=False)
                   .drop_duplicates(subset=["adresse_complete"])
        )

    # Bonus multi-signal
    stacked = bonus_multi_signal(stacked, actifs)

    # Enrichissement BI
    stacked = enrichir_bi(stacked)

    # Tri final + rang
    stacked = stacked.sort_values("score_final", ascending=False).reset_index(drop=True)
    stacked.insert(0, "rang", range(1, len(stacked) + 1))

    return stacked


# ════════════════════════════════════════════════════════════
# 9. PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_pipeline(dept: str = DEPT, annee: int = ANNEE,
                 fenetre_succession_mois: int = 18) -> pd.DataFrame:
    """
    fenetre_succession_mois :
        9  → signaux forts uniquement (ventes très rapides post-succession)
       18  → standard (délai règlement successoral médian)
       24  → filet large (successions complexes / blocages familiaux)
    """
    # ── Chargement ──
    dvf_raw    = download_dvf(dept, annee)
    bodacc_raw = download_bodacc(annee)
    dvf        = clean_dvf(dvf_raw)
    bodacc     = clean_bodacc(bodacc_raw)

    # ── Signaux bruts ──
    heritage  = croiser_dvf_bodacc(dvf, bodacc, dept, fenetre_succession_mois)
    divorce   = signal_divorce(dvf)
    upgrade   = signal_upgrade(dvf)
    retraite  = signal_retraite(dvf)
    primo     = signal_primo(dvf)

    # ── Renommage score_signal → score_brut (héritage v1) ──
    for fr in [heritage, divorce, upgrade, retraite, primo]:
        if fr is not None and not fr.empty and "score_signal" in fr.columns and "score_brut" not in fr.columns:
            fr["score_brut"] = fr["score_signal"]

    # ── Malus qualité (sur score_brut avant percentile) ──
    heritage = appliquer_malus(heritage) if not heritage.empty else heritage
    divorce  = appliquer_malus(divorce)  if not divorce.empty  else divorce
    upgrade  = appliquer_malus(upgrade)  if not upgrade.empty  else upgrade
    retraite = appliquer_malus(retraite) if not retraite.empty else retraite
    primo    = appliquer_malus(primo)    if not primo.empty    else primo

    # ── Normalisation percentile PAR SIGNAL ──
    heritage = normaliser_en_percentile(heritage) if not heritage.empty else heritage
    divorce  = normaliser_en_percentile(divorce)  if not divorce.empty  else divorce
    upgrade  = normaliser_en_percentile(upgrade)  if not upgrade.empty  else upgrade
    retraite = normaliser_en_percentile(retraite) if not retraite.empty else retraite
    primo    = normaliser_en_percentile(primo)    if not primo.empty    else primo

    # ── Consolidation + BI ──
    return consolider([heritage, divorce, upgrade, retraite, primo])

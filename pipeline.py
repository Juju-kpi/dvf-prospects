"""
DVF × BODACC — Pipeline de détection de prospects immobiliers
Module importé par l'application Streamlit (app.py)

GRILLE DE SCORING
═════════════════
  succession_bodacc          : Base 50 + délai vente + adjudication  → 50–100
  divorce_ou_separation      : Revente T3/T4 < 3 ans                 → 60 | 80
  petit_bien_upgrade_potentiel: Achat T1/T2 < 4 ans                  → 55
  retraite_downsizing        : T5+ décote > 10 % médiane commune     → 65 | 80
  primo_acheteur_potentiel   : T1/T2 < 70 % médiane CP              → 50
  Seuil "chaud"  : ≥ 70
  Seuil "très chaud" : ≥ 80
"""

import pandas as pd
import numpy as np
import requests
import io
import os
from datetime import datetime, timedelta
from pathlib import Path

# ── Constantes UI (utilisées dans app.py) ────────────────────
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
# Mapping valeurs longues → clés courtes
SIGNAL_NORMALIZE = {
    "succession_bodacc":           "heritage",
    "divorce_ou_separation":       "divorce",
    "petit_bien_upgrade_potentiel":"upgrade",
    "retraite_downsizing":         "retraite",
    "primo_acheteur_potentiel":    "primo",
}

# ── Config ────────────────────────────────────────────────────
DEPT          = "75"
ANNEE         = 2024
MOIS_LOOKBACK = 24
DATA_DIR      = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# 1. DVF
# ════════════════════════════════════════════════════════════

def download_dvf(dept: str = DEPT, annee: int = ANNEE) -> pd.DataFrame:
    url   = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dept}.csv.gz"
    cache = DATA_DIR / f"dvf_{dept}_{annee}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    r = requests.get(url, timeout=60)
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
    df = df[df["valeur_fonciere"] > 0]

    # Adresse postale complète
    adresse_cols = ["adresse_numero", "adresse_suffixe", "adresse_nom_voie", "code_postal", "nom_commune"]
    cols_dispo   = [c for c in adresse_cols if c in df.columns]
    df["adresse_complete"] = (
        df[cols_dispo].fillna("").astype(str)
        .apply(lambda r: " ".join(v for v in r if v not in ("", "nan")).strip().upper(), axis=1)
    )
    df["cle_commune_section"] = df["code_commune"].astype(str).str.zfill(5)
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
            raise ValueError("Fichier vide")
        df.to_csv(cache, index=False)
        return df
    except Exception:
        return _synthetic_bodacc()


def _synthetic_bodacc() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "publicationDate": pd.date_range("2023-01-01", periods=n, freq="12h").astype(str),
        "typeAnnonce":     np.random.choice(["succession","cession","vente fonds","dissolution"], n),
        "ville":           np.random.choice(["PARIS","LYON","MARSEILLE","BORDEAUX","TOULOUSE"], n),
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
    if type_col:
        mask = df[type_col].astype(str).str.lower().str.contains("succ|héritage|heritage", na=False)
        df_succ = df[mask].copy()
    else:
        df_succ = df.copy()
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
# 3. CROISEMENT DVF × BODACC
# ════════════════════════════════════════════════════════════

def _score_heritage(df: pd.DataFrame, date_succession: datetime) -> pd.Series:
    """
    Base 50 pts
    +30 si vente < 90 j  · +20 si < 180 j  · +10 si < 365 j
    +20 si adjudication
    """
    score = pd.Series(50, index=df.index, dtype=float)
    if "date_mutation" in df.columns:
        delai = (df["date_mutation"] - date_succession).dt.days
        score += np.where(delai < 90, 30, np.where(delai < 180, 20, np.where(delai < 365, 10, 0)))
    if "nature_mutation" in df.columns:
        score += np.where(df["nature_mutation"] == "Adjudication", 20, 0)
    return score.clip(0, 100).round(1)


def croiser_dvf_bodacc(dvf, bodacc, dept=DEPT, lookback_mois=MOIS_LOOKBACK):
    cutoff = datetime.now() - timedelta(days=lookback_mois * 30)
    dvf_f  = dvf[dvf["date_mutation"] >= cutoff].copy()
    bodacc_f = bodacc[bodacc["dept_bodacc"] == dept].copy() if "dept_bodacc" in bodacc.columns and bodacc["dept_bodacc"].notna().any() else bodacc.copy()
    if bodacc_f.empty or dvf_f.empty:
        return pd.DataFrame()
    dvf_f["dept"] = dvf_f["code_postal"].astype(str).str[:2]
    results = []
    for _, succ in bodacc_f.iterrows():
        dept_s = str(succ.get("dept_bodacc", ""))[:2]
        date   = succ.get("date_bodacc", pd.NaT)
        if pd.isna(date) or not dept_s:
            continue
        mask = (
            (dvf_f["dept"] == dept_s) &
            (dvf_f["date_mutation"] >= date) &
            (dvf_f["date_mutation"] <= date + timedelta(days=18*30))
        )
        matches = dvf_f[mask].copy()
        if not matches.empty:
            matches["date_succession"] = date
            matches["score_heritage"]  = _score_heritage(matches, date)
            matches["score_signal"]    = matches["score_heritage"]
            matches["signal"]          = "succession_bodacc"
            results.append(matches)
    if not results:
        return pd.DataFrame()
    id_col = "id_mutation" if "id_mutation" in dvf_f.columns else None
    out = pd.concat(results, ignore_index=True).drop_duplicates(subset=[id_col] if id_col else None)
    return out.sort_values("score_heritage", ascending=False)


# ════════════════════════════════════════════════════════════
# 4. SIGNAUX ÉVÉNEMENTS DE VIE
# ════════════════════════════════════════════════════════════

def signal_divorce_downsizing(dvf):
    """T3/T4 revendu < 3 ans → score 80 si <1 an | 60 si 1–3 ans"""
    df = dvf.copy()
    if "date_mutation" not in df.columns or "adresse_complete" not in df.columns:
        return pd.DataFrame()
    df_s = df.sort_values("date_mutation")
    df_s["vente_precedente"]     = df_s.groupby("adresse_complete")["date_mutation"].shift(1)
    df_s["delai_revente_jours"]  = (df_s["date_mutation"] - df_s["vente_precedente"]).dt.days
    pieces_ok = df_s["nombre_pieces_principales"].between(3, 4) if "nombre_pieces_principales" in df_s.columns else pd.Series(True, index=df_s.index)
    mask = (df_s["delai_revente_jours"] > 0) & (df_s["delai_revente_jours"] < 3*365) & pieces_ok
    result = df_s[mask].copy()
    result["signal"]      = "divorce_ou_separation"
    result["commune"]     = result["nom_commune"] if "nom_commune" in result.columns else ""
    result["score_signal"] = np.where(result["delai_revente_jours"] < 365, 80, 60)
    return result


def signal_naissance_upgrade(dvf):
    """T1/T2 acheté < 4 ans → score fixe 55"""
    df = dvf.copy()
    if "nombre_pieces_principales" not in df.columns:
        return pd.DataFrame()
    mask = (
        df["nombre_pieces_principales"].isin([1, 2]) &
        (df["valeur_fonciere"] > 0) &
        (df["date_mutation"] >= datetime.now() - timedelta(days=4*365))
    )
    result = df[mask].copy()
    result["signal"]       = "petit_bien_upgrade_potentiel"
    result["score_signal"] = 55
    return result


def signal_retraite_downsizing(dvf):
    """T5+ décote > 10% médiane commune → 65 | si > 20% → 80"""
    df = dvf.copy()
    if not {"nombre_pieces_principales","surface_reelle_bati","valeur_fonciere"}.issubset(df.columns):
        return pd.DataFrame()
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)
    commune_col = next((c for c in df.columns if c in ("commune","nom_commune","libelle_commune")), "code_postal")
    if commune_col != "commune":
        df["commune"] = df[commune_col]
    df["mediane_commune"] = df.groupby("commune")["prix_m2"].transform("median")
    df["decote_pct"]      = (df["mediane_commune"] - df["prix_m2"]) / df["mediane_commune"] * 100
    mask   = (df["nombre_pieces_principales"] >= 5) & (df["decote_pct"] > 10)
    result = df[mask].copy()
    result["signal"]       = "retraite_downsizing"
    result["score_signal"] = np.where(result["decote_pct"] > 20, 80, 65)
    return result


def signal_primo_acheteurs(dvf):
    """T1/T2 vendu à < 70% médiane CP → score fixe 50"""
    df = dvf.copy()
    if "valeur_fonciere" not in df.columns:
        return pd.DataFrame()
    df["mediane_cp"] = df.groupby("code_postal")["valeur_fonciere"].transform("median")
    pieces_ok = df["nombre_pieces_principales"].isin([1,2]) if "nombre_pieces_principales" in df.columns else pd.Series(True, index=df.index)
    mask   = (df["valeur_fonciere"] < df["mediane_cp"] * 0.7) & pieces_ok
    result = df[mask].copy()
    result["signal"]       = "primo_acheteur_potentiel"
    result["score_signal"] = 50
    return result


# ════════════════════════════════════════════════════════════
# 5. CONSOLIDATION & EXPORT
# ════════════════════════════════════════════════════════════

def consolider_signaux(*dfs) -> pd.DataFrame:
    frames = [df for df in dfs if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    cols_communes = [
        "adresse_numero", "adresse_suffixe", "adresse_nom_voie",
        "code_postal", "nom_commune", "adresse_complete",
        "valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
        "date_mutation", "nature_mutation", "commune",
        "signal", "score_signal", "longitude", "latitude",
    ]
    result_frames = [df[[c for c in cols_communes if c in df.columns]] for df in frames]
    consolidated  = pd.concat(result_frames, ignore_index=True)

    if "adresse_complete" in consolidated.columns:
        consolidated = (
            consolidated.sort_values("score_signal", ascending=False)
            .drop_duplicates(subset=["adresse_complete"])
        )

    consolidated = consolidated.sort_values("score_signal", ascending=False).reset_index(drop=True)
    consolidated.insert(0, "rang", range(1, len(consolidated) + 1))

    # Clé courte pour la carte
    consolidated["signal_carte"] = consolidated["signal"].map(SIGNAL_NORMALIZE).fillna(consolidated["signal"])

    # Étiquette lisible
    consolidated["signal_label"] = consolidated["signal_carte"].map(SIGNAL_LABELS).fillna(consolidated["signal"])

    # Catégorie chaleur
    consolidated["chaleur"] = pd.cut(
        consolidated["score_signal"],
        bins=[0, 54, 69, 79, 100],
        labels=["froid (≤54)", "tiède (55–69)", "chaud (70–79)", "très chaud (≥80)"],
        right=True,
    )
    return consolidated


# ════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_pipeline(dept: str = DEPT, annee: int = ANNEE) -> pd.DataFrame:
    dvf_raw    = download_dvf(dept, annee)
    bodacc_raw = download_bodacc(annee)
    dvf        = clean_dvf(dvf_raw)
    bodacc     = clean_bodacc(bodacc_raw)

    heritage  = croiser_dvf_bodacc(dvf, bodacc, dept)
    divorce   = signal_divorce_downsizing(dvf)
    naissance = signal_naissance_upgrade(dvf)
    retraite  = signal_retraite_downsizing(dvf)
    primo     = signal_primo_acheteurs(dvf)

    if not heritage.empty and "score_heritage" in heritage.columns:
        heritage["score_signal"] = heritage["score_heritage"]

    return consolider_signaux(heritage, divorce, naissance, retraite, primo)
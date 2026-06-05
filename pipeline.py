# -*- coding: utf-8 -*-
"""
DVF x BODACC — Pipeline v4.2
Sources : DVF · BODACC · INSEE communes · API BAN/Cadastre
Score   : brut → malus → bonus INSEE → cluster → percentile dept → liquidite
"""
import pandas as pd
import numpy as np
import requests
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Constantes exportées ──────────────────────────────────────────────────────
SIGNAL_LABELS = {
    "heritage": "Succession/heritage",
    "divorce":  "Divorce/separation",
    "upgrade":  "Upgrade famille",
    "retraite": "Retraite/downsizing",
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
    "heritage": "Liquidation d'actif — rapidite et discretion",
    "divorce":  "Revente contrainte — accompagnement neutre et prix juste",
    "upgrade":  "Famille croissante — espace et projets de vie",
    "retraite": "Downsizing — simplicite et liberation de capital",
    "primo":    "Primo-accedant — 2e achat et investissement patrimonial",
}
SIGNAL_CTA = {
    "heritage": "Flyer secteur ou campagne Meta CP axe 'vente rapide et discrete'",
    "divorce":  "Message 'vendre sereinement, sans conflit, au juste prix'",
    "upgrade":  "Campagne Meta CP 'votre T1/T2 vaut plus que vous ne pensez'",
    "retraite": "SEA 'vendre ma maison [commune]' + estimation gratuite en ligne",
    "primo":    "Content SEO local 'revendre pour acheter mieux dans 3 ans'",
}

TYPES_RESIDENTIELS = {"Appartement", "Maison", "Appartement-Maison"}
DEPT     = "75"
ANNEE    = 2024
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DVF
# ══════════════════════════════════════════════════════════════════════════════
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

    addr_cols = [c for c in ["adresse_numero","adresse_suffixe","adresse_nom_voie","code_postal","nom_commune"] if c in df.columns]
    df["adresse_complete"] = (
        df[addr_cols].fillna("").astype(str)
        .apply(lambda r: " ".join(v for v in r if v not in ("","nan")).strip().upper(), axis=1)
    )

    surf = df["surface_reelle_bati"].replace(0, np.nan) if "surface_reelle_bati" in df.columns else np.nan
    df["prix_m2"] = (df["valeur_fonciere"] / surf).round(0) if "surface_reelle_bati" in df.columns else np.nan

    if "type_local" in df.columns:
        df["type_local"]      = df["type_local"].fillna("Inconnu")
        df["est_residentiel"] = df["type_local"].isin(TYPES_RESIDENTIELS)
    else:
        df["est_residentiel"] = True

    commune_col = next((c for c in df.columns if c in ("nom_commune","libelle_commune")), None)
    df["commune"] = df[commune_col].fillna("") if commune_col else ""
    df["anciennete_mois"] = ((datetime.now() - df["date_mutation"]).dt.days / 30).round(1)

    if "code_commune" in df.columns:
        df["cle_commune"] = df["code_commune"].astype(str).str.zfill(5)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. BODACC
# ══════════════════════════════════════════════════════════════════════════════
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
            raise ValueError("vide")
        df.to_csv(cache, index=False)
        return df
    except Exception:
        return _synthetic_bodacc()


def _synthetic_bodacc() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "publicationDate": pd.date_range("2023-01-01", periods=n, freq="12h").astype(str),
        "typeAnnonce":     np.random.choice(["succession","cession","dissolution"], n),
        "codePostal":      np.random.choice(["75001","75008","69001","13001","33000"], n),
        "denomination":    [f"SUCCESSION {i}" for i in range(n)],
        "montant":         np.random.uniform(50000, 900000, n).round(0),
    })


def clean_bodacc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    date_col = next((c for c in df.columns if "date" in c or "parution" in c), None)
    df["date_bodacc"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    type_col = next((c for c in df.columns if "type" in c or "famille" in c), None)
    if type_col:
        mask = df[type_col].astype(str).str.lower().str.contains("succ|heritage", na=False)
        df_s = df[mask].copy()
    else:
        df_s = df.copy()
    cp_col   = next((c for c in df.columns if "postal" in c or c == "cp"), None)
    dept_col = next((c for c in df.columns if "departement" in c), None)
    if cp_col:
        df_s["code_postal_bodacc"] = df_s[cp_col].astype(str).str.zfill(5)
        df_s["dept_bodacc"]        = df_s["code_postal_bodacc"].str[:2]
    elif dept_col:
        df_s["dept_bodacc"]        = df_s[dept_col].astype(str).str.zfill(2)
        df_s["code_postal_bodacc"] = None
    else:
        df_s["dept_bodacc"] = df_s["code_postal_bodacc"] = None
    return df_s


# ══════════════════════════════════════════════════════════════════════════════
# 3. INSEE — contexte socio-demographique
# ══════════════════════════════════════════════════════════════════════════════
def fetch_insee(dept: str) -> pd.DataFrame:
    cache = DATA_DIR / f"insee_{dept}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"code_commune": str})
    url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "statistiques-locales-des-communes/exports/csv"
        f"?where=departement_id%3D%22{dept}%22&lang=fr&delimiter=%3B&limit=5000"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep=";", low_memory=False)
        if df.empty:
            raise ValueError("vide")
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        col_map = {}
        for c in df.columns:
            if "proprio" in c or "proprietaire" in c: col_map[c] = "taux_proprio"
            if "age" in c and "median" in c:           col_map[c] = "age_median"
            if "revenu" in c and "median" in c:        col_map[c] = "revenu_median"
            if "commune" in c and "code" in c:         col_map[c] = "code_commune"
        df = df.rename(columns=col_map)
        df["code_commune"] = df.get("code_commune", pd.Series(dtype=str)).astype(str).str.zfill(5)
        out_cols = [c for c in ["code_commune","taux_proprio","age_median","revenu_median"] if c in df.columns]
        if len(out_cols) < 2:
            raise ValueError("colonnes manquantes")
        df[out_cols].to_csv(cache, index=False)
        return df[out_cols]
    except Exception:
        return _synthetic_insee(dept)


def _synthetic_insee(dept: str) -> pd.DataFrame:
    np.random.seed(99)
    n = 80
    codes = [f"{dept}{str(i).zfill(3)}" for i in range(1, n+1)]
    return pd.DataFrame({
        "code_commune":  codes,
        "taux_proprio":  np.clip(np.random.normal(58, 15, n), 20, 95).round(1),
        "age_median":    np.clip(np.random.normal(40, 8,  n), 25, 65).round(1),
        "revenu_median": np.clip(np.random.normal(22000, 6000, n), 12000, 45000).round(0),
    })


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIQUIDITE MARCHE par CP
# ══════════════════════════════════════════════════════════════════════════════
def calculer_liquidite(dvf: pd.DataFrame) -> pd.DataFrame:
    df = dvf[dvf.get("est_residentiel", pd.Series(True, index=dvf.index))].copy() if "est_residentiel" in dvf.columns else dvf.copy()

    vol = df.groupby("code_postal").size().reset_index(name="volume_transactions")
    vol["vol_pct"] = vol["volume_transactions"].rank(pct=True) * 100

    if "adresse_complete" in df.columns and "date_mutation" in df.columns:
        df_s = df.sort_values("date_mutation")
        df_s["prev_vente"]  = df_s.groupby("adresse_complete")["date_mutation"].shift(1)
        df_s["delai_j"]     = (df_s["date_mutation"] - df_s["prev_vente"]).dt.days
        rot = (
            df_s[df_s["delai_j"] > 30].groupby("code_postal")["delai_j"]
            .median().reset_index()
            .rename(columns={"delai_j": "delai_rotation_jours"})
        )
        rot["delai_rotation_mois"] = (rot["delai_rotation_jours"] / 30).round(1)
        rot["rot_pct"] = (1 - rot["delai_rotation_jours"].rank(pct=True)) * 100
        liq = vol.merge(rot[["code_postal","delai_rotation_mois","rot_pct"]], on="code_postal", how="left")
        liq["rot_pct"] = liq["rot_pct"].fillna(50)
    else:
        liq = vol.copy()
        liq["delai_rotation_mois"] = np.nan
        liq["rot_pct"] = 50

    liq["liquidite_cp"] = (liq["vol_pct"] * 0.6 + liq["rot_pct"] * 0.4).clip(0, 100).round(1)
    return liq[["code_postal","volume_transactions","delai_rotation_mois","liquidite_cp"]]


# ══════════════════════════════════════════════════════════════════════════════
# 5. TENDANCES marche
# ══════════════════════════════════════════════════════════════════════════════
def calculer_tendances(dvf: pd.DataFrame) -> dict:
    out = {}
    if dvf.empty or "date_mutation" not in dvf.columns:
        return out
    df = dvf[dvf.get("est_residentiel", pd.Series(True, index=dvf.index))].copy() if "est_residentiel" in dvf.columns else dvf.copy()
    df["mois"]      = df["date_mutation"].dt.to_period("M").astype(str)
    df["trimestre"] = df["date_mutation"].dt.to_period("Q").astype(str)
    df["annee"]     = df["date_mutation"].dt.year

    out["volume_par_mois"] = df.groupby("mois").size().reset_index(name="nb_transactions").sort_values("mois")

    if "valeur_fonciere" in df.columns:
        out["prix_median_par_mois"] = (
            df.groupby("mois")["valeur_fonciere"].median()
            .reset_index(name="prix_median").sort_values("mois")
        )

    if "code_postal" in df.columns:
        out["top_cp_volume"] = (
            df.groupby("code_postal").size().reset_index(name="volume")
            .sort_values("volume", ascending=False).head(15)
        )

    if "prix_m2" in df.columns and "code_postal" in df.columns:
        pm2_cp = (
            df.groupby("code_postal")["prix_m2"]
            .agg(prix_m2_median="median", nb="count")
            .reset_index().query("nb >= 5")
            .sort_values("prix_m2_median", ascending=False)
        )
        out["prix_m2_par_cp"] = pm2_cp

    # Volume par type de bien
    if "type_local" in df.columns:
        out["volume_par_type"] = (
            df.groupby("type_local").size().reset_index(name="nb")
            .sort_values("nb", ascending=False)
        )

    # Evolution trimestrielle du prix
    if "valeur_fonciere" in df.columns:
        out["prix_par_trimestre"] = (
            df.groupby("trimestre")["valeur_fonciere"].agg(
                prix_median="median", nb_ventes="count"
            ).reset_index().sort_values("trimestre")
        )

    # Heatmap pieces x prix_m2
    if "nombre_pieces_principales" in df.columns and "prix_m2" in df.columns:
        hm = (
            df[df["nombre_pieces_principales"].between(1, 6)]
            .groupby("nombre_pieces_principales")["prix_m2"]
            .agg(pm2_med="median", nb="count")
            .reset_index()
        )
        hm["nombre_pieces_principales"] = hm["nombre_pieces_principales"].astype(int)
        out["pm2_par_pieces"] = hm

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. MALUS QUALITE
# ══════════════════════════════════════════════════════════════════════════════
def appliquer_malus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "score_brut" not in df.columns:
        return df
    df    = df.copy()
    malus = pd.Series(0.0, index=df.index)
    if "est_residentiel" in df.columns:
        malus += np.where(~df["est_residentiel"].fillna(True), -20, 0)
    if "prix_m2" in df.columns:
        pm2 = pd.to_numeric(df["prix_m2"], errors="coerce")
        malus += np.where((pm2 < 500) | (pm2 > 35000), -15, 0)
    if "surface_reelle_bati" in df.columns:
        surf = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
        malus += np.where((surf < 9) | (surf > 600), -10, 0)
    if "valeur_fonciere" in df.columns:
        prix = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
        malus += np.where(prix < 15000, -10, 0)
    df["malus"]      = malus.round(0)
    df["score_brut"] = (df["score_brut"] + malus).clip(0, 100).round(1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. BONUS INSEE
# ══════════════════════════════════════════════════════════════════════════════
def appliquer_bonus_insee(df: pd.DataFrame, insee: pd.DataFrame) -> pd.DataFrame:
    if df.empty or insee.empty:
        return df
    df = df.copy()
    # Jointure par cle_commune si dispo, sinon par code_postal approx
    if "cle_commune" in df.columns and "code_commune" in insee.columns:
        df = df.merge(
            insee.rename(columns={"code_commune": "cle_commune"}),
            on="cle_commune", how="left"
        )
    elif "code_postal" in df.columns and "code_commune" in insee.columns:
        cp5 = df["code_postal"].astype(str).str.zfill(5)
        cc5 = insee["code_commune"].astype(str).str.zfill(5)
        mapping = insee.copy()
        mapping["_cp5"] = cc5
        df["_cp5"] = cp5
        df = df.merge(
            mapping[["_cp5","taux_proprio","age_median","revenu_median"]],
            on="_cp5", how="left"
        ).drop(columns=["_cp5"])

    for col in ["taux_proprio","age_median","revenu_median"]:
        if col not in df.columns:
            df[col] = np.nan

    sig  = df.get("signal", pd.Series("", index=df.index)).astype(str)
    tp   = pd.to_numeric(df["taux_proprio"],  errors="coerce").fillna(55)
    am   = pd.to_numeric(df["age_median"],    errors="coerce").fillna(40)
    rm   = pd.to_numeric(df["revenu_median"], errors="coerce").fillna(22000)
    bonus = pd.Series(0.0, index=df.index)

    m = sig.str.contains("succession|heritage", na=False)
    bonus[m] += np.where(tp[m] > 70, 5, np.where(tp[m] > 55, 3, 0))

    m = sig.str.contains("divorce", na=False)
    bonus[m] += np.where((tp[m] > 60) & (am[m] > 32), 5, np.where(tp[m] > 50, 2, 0))

    m = sig.str.contains("upgrade", na=False)
    bonus[m] += np.where((rm[m] > 25000) & (tp[m] > 55), 5, np.where(rm[m] > 20000, 2, 0))

    m = sig.str.contains("retraite", na=False)
    bonus[m] += np.where(am[m] > 55, 5, np.where(am[m] > 47, 3, 0))

    m = sig.str.contains("primo", na=False)
    bonus[m] += np.where((rm[m] < 25000) & (tp[m] < 55), 5, np.where(rm[m] < 30000, 2, 0))

    df["bonus_insee"]          = bonus.round(0)
    df["score_brut"]           = (df["score_brut"] + bonus).clip(0, 100).round(1)
    df["insee_taux_proprio"]   = tp.round(1)
    df["insee_age_median"]     = am.round(1)
    df["insee_revenu_median"]  = rm.round(0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLUSTERS GEOGRAPHIQUES (haversine vectorise, sans sklearn)
# ══════════════════════════════════════════════════════════════════════════════
def detecter_clusters(df: pd.DataFrame, rayon_km: float = 0.5, min_pts: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["cluster_densite"] = 0
    df["cluster_chaud"]   = False
    df["bonus_cluster"]   = 0.0

    has_coords = (
        "latitude" in df.columns and "longitude" in df.columns
        and df["latitude"].notna().any()
    )
    if not has_coords:
        return df

    geo = df[df["latitude"].notna() & df["longitude"].notna() & (df["latitude"] != 0)].copy()
    if len(geo) < min_pts:
        return df

    lats = np.radians(geo["latitude"].values)
    lngs = np.radians(geo["longitude"].values)
    R    = 6371.0
    dens = np.zeros(len(geo), dtype=int)

    for i in range(len(geo)):
        dlat = lats - lats[i]
        dlng = lngs - lngs[i]
        a    = np.sin(dlat/2)**2 + np.cos(lats[i]) * np.cos(lats) * np.sin(dlng/2)**2
        dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        dens[i] = int((dist <= rayon_km).sum()) - 1

    geo["cluster_densite"] = dens
    geo["cluster_chaud"]   = dens >= min_pts
    geo["bonus_cluster"]   = np.where(dens >= min_pts, 8.0, np.where(dens >= 2, 4.0, 0.0))

    df.loc[geo.index, "cluster_densite"] = geo["cluster_densite"]
    df.loc[geo.index, "cluster_chaud"]   = geo["cluster_chaud"]
    df.loc[geo.index, "bonus_cluster"]   = geo["bonus_cluster"]
    if "score_brut" in df.columns:
        df["score_brut"] = (df["score_brut"] + df["bonus_cluster"]).clip(0, 100).round(1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9. SIGNAUX
# ══════════════════════════════════════════════════════════════════════════════
def _score_heritage_brut(df: pd.DataFrame, date_succ: datetime) -> pd.Series:
    score = pd.Series(50.0, index=df.index)
    if "date_mutation" in df.columns:
        delai = (df["date_mutation"] - date_succ).dt.days
        score += np.where(delai < 90, 30, np.where(delai < 180, 20, np.where(delai < 365, 10, 0)))
    if "nature_mutation" in df.columns:
        score += np.where(df["nature_mutation"] == "Adjudication", 20, 0)
    return score.clip(0, 100).round(1)


def signal_heritage(dvf: pd.DataFrame, bodacc: pd.DataFrame,
                    dept: str, fenetre_mois: int) -> pd.DataFrame:
    if dvf.empty or bodacc.empty:
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
    fjours = fenetre_mois * 30
    results = []
    for _, s in bodacc_f.iterrows():
        dept_s = str(s.get("dept_bodacc",""))[:2]
        date   = s.get("date_bodacc", pd.NaT)
        if pd.isna(date) or not dept_s:
            continue
        mask = (
            (dvf_f["dept"] == dept_s) &
            (dvf_f["date_mutation"] >= date) &
            (dvf_f["date_mutation"] <= date + timedelta(days=fjours))
        )
        m = dvf_f[mask].copy()
        if not m.empty:
            m["date_succession"] = date
            m["score_brut"]      = _score_heritage_brut(m, date)
            m["signal"]          = "succession_bodacc"
            results.append(m)
    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    id_col = "id_mutation" if "id_mutation" in dvf_f.columns else None
    return out.drop_duplicates(subset=[id_col]) if id_col else out


def signal_divorce(dvf: pd.DataFrame) -> pd.DataFrame:
    df = dvf.copy()
    if "date_mutation" not in df.columns or "adresse_complete" not in df.columns:
        return pd.DataFrame()
    res = df.get("est_residentiel", pd.Series(True, index=df.index))
    pieces = df["nombre_pieces_principales"].between(3, 4) if "nombre_pieces_principales" in df.columns else pd.Series(True, index=df.index)
    df = df[pieces & res].sort_values("date_mutation").copy()
    if df.empty:
        return pd.DataFrame()
    df["prev_date"] = df.groupby("adresse_complete")["date_mutation"].shift(1)
    df["delai_j"]   = (df["date_mutation"] - df["prev_date"]).dt.days
    mask = (df["delai_j"] > 30) & (df["delai_j"] < 3 * 365)
    r = df[mask].copy()
    r["signal"]     = "divorce_ou_separation"
    r["score_brut"] = np.where(r["delai_j"] < 365, 80.0, 60.0)
    return r


def signal_upgrade(dvf: pd.DataFrame) -> pd.DataFrame:
    df = dvf.copy()
    if "nombre_pieces_principales" not in df.columns:
        return pd.DataFrame()
    res  = df.get("est_residentiel", pd.Series(True, index=df.index))
    surf = df["surface_reelle_bati"] > 15 if "surface_reelle_bati" in df.columns else pd.Series(True, index=df.index)
    now  = datetime.now()
    mask = (
        df["nombre_pieces_principales"].isin([1, 2]) &
        (df["valeur_fonciere"] > 0) &
        (df["date_mutation"] >= now - timedelta(days=4*365)) &
        (df["date_mutation"] <= now - timedelta(days=2*365)) &
        res & surf
    )
    r = df[mask].copy()
    r["signal"]     = "petit_bien_upgrade_potentiel"
    anc = (now - r["date_mutation"]).dt.days / 365
    r["score_brut"] = np.where(anc <= 3, 65.0, 55.0)
    return r


def signal_retraite(dvf: pd.DataFrame) -> pd.DataFrame:
    df = dvf.copy()
    if not {"nombre_pieces_principales","surface_reelle_bati","valeur_fonciere"}.issubset(df.columns):
        return pd.DataFrame()
    res = df.get("est_residentiel", pd.Series(True, index=df.index))
    df = df[(df["nombre_pieces_principales"] >= 5) & (df["surface_reelle_bati"] >= 80) & res].copy()
    if df.empty:
        return pd.DataFrame()
    df["prix_m2"]        = df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)
    df["mediane_t5_cp"]  = df.groupby("code_postal")["prix_m2"].transform("median")
    df["decote_pct"]     = (df["mediane_t5_cp"] - df["prix_m2"]) / df["mediane_t5_cp"] * 100
    df["decote_vs_median"] = df["decote_pct"].round(1)
    r = df[df["decote_pct"] > 10].copy()
    r["signal"]     = "retraite_downsizing"
    r["score_brut"] = np.where(r["decote_pct"] > 25, 85.0,
                      np.where(r["decote_pct"] > 20, 75.0,
                      np.where(r["decote_pct"] > 15, 65.0, 55.0)))
    return r


def signal_primo(dvf: pd.DataFrame) -> pd.DataFrame:
    df = dvf.copy()
    if "valeur_fonciere" not in df.columns:
        return pd.DataFrame()
    res   = df.get("est_residentiel", pd.Series(True, index=df.index))
    piec  = df["nombre_pieces_principales"].isin([1,2]) if "nombre_pieces_principales" in df.columns else pd.Series(True, index=df.index)
    surf  = df["surface_reelle_bati"] >= 15 if "surface_reelle_bati" in df.columns else pd.Series(True, index=df.index)
    df    = df[piec & res & surf & (df["valeur_fonciere"] >= 15000)].copy()
    if df.empty:
        return pd.DataFrame()
    df["mediane_cp_t12"]   = df.groupby("code_postal")["valeur_fonciere"].transform("median")
    df["decote_vs_median"] = ((df["mediane_cp_t12"] - df["valeur_fonciere"]) / df["mediane_cp_t12"] * 100).round(1)
    r = df[df["valeur_fonciere"] < df["mediane_cp_t12"] * 0.7].copy()
    r["signal"]     = "primo_acheteur_potentiel"
    r["score_brut"] = np.where(r["decote_vs_median"] > 30, 60.0, 50.0)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 10. NORMALISATION PERCENTILE + MULTI-SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
def normaliser(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "score_brut" not in df.columns:
        return df
    df = df.copy()
    df["score_final"] = (df["score_brut"].rank(pct=True, method="average") * 100).round(1)
    return df


def bonus_multi(consolidated: pd.DataFrame, frames: list) -> pd.DataFrame:
    if "adresse_complete" not in consolidated.columns:
        return consolidated
    consolidated = consolidated.copy()
    counts = {}
    for fr in frames:
        if fr is None or fr.empty or "adresse_complete" not in fr.columns:
            continue
        for addr in fr["adresse_complete"].dropna().unique():
            counts[addr] = counts.get(addr, 0) + 1
    consolidated["nb_signaux"] = consolidated["adresse_complete"].map(counts).fillna(1).astype(int)
    b = np.where(consolidated["nb_signaux"] >= 3, 12, np.where(consolidated["nb_signaux"] == 2, 5, 0))
    consolidated["score_final"] = (consolidated["score_final"] + b).clip(0, 100).round(1)
    return consolidated


# ══════════════════════════════════════════════════════════════════════════════
# 11. ENRICHISSEMENT BI FINAL
# ══════════════════════════════════════════════════════════════════════════════
def enrichir(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signal_carte"]  = df["signal"].map(SIGNAL_NORMALIZE).fillna(df["signal"])
    df["signal_label"]  = df["signal_carte"].map(SIGNAL_LABELS).fillna(df["signal"])
    df["segment_cible"] = df["signal_carte"].map(SIGNAL_SEGMENTS).fillna("")
    df["cta"]           = df["signal_carte"].map(SIGNAL_CTA).fillna("")

    df["intensite"] = pd.cut(
        df["score_final"], bins=[0,40,60,80,100],
        labels=["faible","moyen","fort","tres fort"], right=True
    )
    df["chaleur"] = pd.cut(
        df["score_final"], bins=[0,39,59,79,100],
        labels=["froid","tiede","chaud","tres chaud"], right=True
    )
    if "decote_vs_median" not in df.columns and "valeur_fonciere" in df.columns:
        med = df.groupby("code_postal")["valeur_fonciere"].transform("median")
        df["decote_vs_median"] = ((med - df["valeur_fonciere"]) / med * 100).round(1)
    if "anciennete_mois" not in df.columns and "date_mutation" in df.columns:
        df["anciennete_mois"] = ((datetime.now() - df["date_mutation"]).dt.days / 30).round(1)
    if "prix_m2" not in df.columns and "surface_reelle_bati" in df.columns:
        surf = df["surface_reelle_bati"].replace(0, np.nan)
        df["prix_m2"] = (df["valeur_fonciere"] / surf).round(0)

    # Score confiance : bonus si marche liquide + INSEE coherent + cluster
    conf_bonus = pd.Series(0.0, index=df.index)
    if "liquidite_cp" in df.columns:
        conf_bonus += (df["liquidite_cp"].fillna(50) - 50).clip(0, 50) / 50 * 10
    if "cluster_chaud" in df.columns:
        conf_bonus += np.where(df["cluster_chaud"], 5, 0)
    if "nb_signaux" in df.columns:
        conf_bonus += np.where(df["nb_signaux"] >= 2, 5, 0)
    df["score_confiance"] = (df["score_final"] * 0.7 + conf_bonus * 0.3).clip(0, 100).round(1)

    # Priorite prospection
    df["priorite"] = np.where(
        df["score_final"] >= 80, "P1 — Contact immediat",
        np.where(df["score_final"] >= 65, "P2 — Dans les 30j",
        np.where(df["score_final"] >= 50, "P3 — Nurturing",
        "P4 — A surveiller"))
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 12. CONSOLIDATION
# ══════════════════════════════════════════════════════════════════════════════
def consolider(frames: list, liq: pd.DataFrame = None) -> pd.DataFrame:
    actifs = [f for f in frames if f is not None and not f.empty]
    if not actifs:
        return pd.DataFrame()

    keep = [
        "adresse_numero","adresse_suffixe","adresse_nom_voie",
        "code_postal","nom_commune","commune","adresse_complete",
        "valeur_fonciere","surface_reelle_bati","nombre_pieces_principales",
        "prix_m2","type_local","est_residentiel",
        "date_mutation","nature_mutation","anciennete_mois",
        "signal","score_brut","score_final","malus","bonus_insee",
        "decote_vs_median","cluster_densite","cluster_chaud","bonus_cluster",
        "insee_taux_proprio","insee_age_median","insee_revenu_median",
        "longitude","latitude",
    ]
    stacked = pd.concat(
        [f[[c for c in keep if c in f.columns]] for f in actifs],
        ignore_index=True
    )

    if "adresse_complete" in stacked.columns:
        stacked = (
            stacked.sort_values("score_final", ascending=False)
            .drop_duplicates(subset=["adresse_complete"])
        )

    stacked = bonus_multi(stacked, actifs)

    # Liquidite
    if liq is not None and not liq.empty and "code_postal" in stacked.columns:
        stacked = stacked.merge(liq, on="code_postal", how="left")
        stacked["liquidite_cp"]        = stacked.get("liquidite_cp", pd.Series(50.0)).fillna(50)
        stacked["volume_transactions"] = stacked.get("volume_transactions", pd.Series(0)).fillna(0)
        stacked["delai_rotation_mois"] = stacked.get("delai_rotation_mois", pd.Series(np.nan))
        b_liq = np.where(stacked["liquidite_cp"] >= 80, 10,
                np.where(stacked["liquidite_cp"] >= 60, 7,
                np.where(stacked["liquidite_cp"] >= 40, 4, 0))).astype(float)
        stacked["bonus_liquidite"] = b_liq
        stacked["score_final"]     = (stacked["score_final"] + b_liq).clip(0, 100).round(1)

    stacked = enrichir(stacked)
    stacked = stacked.sort_values("score_final", ascending=False).reset_index(drop=True)
    stacked.insert(0, "rang", range(1, len(stacked)+1))
    return stacked


# ══════════════════════════════════════════════════════════════════════════════
# 13. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    dept: str = DEPT,
    annee: int = ANNEE,
    fenetre_succession_mois: int = 18,
    enrichir_cadastre: bool = False,
    rayon_cluster_km: float = 0.5,
) -> tuple:
    """
    Retourne (prospects_df, tendances_dict).

    fenetre_succession_mois : 9=fort | 18=standard | 24=large
    enrichir_cadastre       : False par defaut (appels API lents)
    rayon_cluster_km        : rayon hotspot (defaut 0.5 km)
    """
    dvf_raw    = download_dvf(dept, annee)
    bodacc_raw = download_bodacc(annee)
    dvf        = clean_dvf(dvf_raw)
    bodacc     = clean_bodacc(bodacc_raw)

    tendances  = calculer_tendances(dvf)
    liq        = calculer_liquidite(dvf)

    codes_comm = dvf["cle_commune"].dropna().unique().tolist() if "cle_commune" in dvf.columns else []
    insee      = fetch_insee(dept)

    her = signal_heritage(dvf, bodacc, dept, fenetre_succession_mois)
    div = signal_divorce(dvf)
    upg = signal_upgrade(dvf)
    ret = signal_retraite(dvf)
    pri = signal_primo(dvf)

    processed = []
    for fr in [her, div, upg, ret, pri]:
        if fr is None or fr.empty:
            processed.append(pd.DataFrame())
            continue
        if "score_signal" in fr.columns and "score_brut" not in fr.columns:
            fr = fr.copy(); fr["score_brut"] = fr["score_signal"]
        fr = appliquer_malus(fr)
        fr = appliquer_bonus_insee(fr, insee)
        fr = detecter_clusters(fr, rayon_km=rayon_cluster_km)
        fr = normaliser(fr)
        processed.append(fr)

    prospects = consolider(processed, liq=liq)
    return prospects, tendances

"""
============================================================
 DVF × BODACC — Pipeline de détection de prospects immobiliers
 v2.0 — Scoring amélioré, malus, multi-signal, type_local

GRILLE DE SCORING v2
═════════════════════
  succession_bodacc          : Base 50 + délai + adjudication → 50–100
  divorce_ou_separation      : Revente T3/T4 < 3 ans          → 60 | 80
  petit_bien_upgrade_potentiel: Achat T1/T2 < 4 ans           → 55
  retraite_downsizing        : T5+ décote > 10% médiane       → 65 | 80
  primo_acheteur_potentiel   : T1/T2 < 70% médiane CP         → 50

MALUS (s'appliquent à tous les signaux)
═══════════════════════════
  -15  Prix/m² aberrant (< 500 ou > 30 000 €/m²)
  -10  Surface incohérente (< 9 m² ou > 500 m²)
  -10  Bien non résidentiel (local commercial, terrain, dépendance)
  -5   Prix global très bas (< 10 000 €) = probable parking / cave

BONUS MULTI-SIGNAL
═══════════════════════════
  +10  Même adresse cumule 2 signaux distincts
  +20  Même adresse cumule 3+ signaux distincts

SEUILS
═══════════════════════════
  Très chaud : ≥ 80
  Chaud      : 70–79
  Tiède      : 55–69
  Froid      : < 55
============================================================
"""

import pandas as pd
import numpy as np
import requests
import io
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

# Types de locaux résidentiels acceptés
TYPES_RESIDENTIELS = {"Appartement", "Maison", "Appartement-Maison"}

# ── Config ────────────────────────────────────────────────────
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
    r = requests.get(url, timeout=90)
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

    # Filtre mutations valides
    df = df[df["nature_mutation"].isin(["Vente", "Adjudication", "Expropriation"])]
    df = df[df["valeur_fonciere"] > 0]

    # Adresse complète
    adresse_cols = ["adresse_numero", "adresse_suffixe", "adresse_nom_voie", "code_postal", "nom_commune"]
    cols_dispo   = [c for c in adresse_cols if c in df.columns]
    df["adresse_complete"] = (
        df[cols_dispo].fillna("").astype(str)
        .apply(lambda r: " ".join(v for v in r if v not in ("", "nan")).strip().upper(), axis=1)
    )

    # Prix au m²
    if "surface_reelle_bati" in df.columns:
        df["prix_m2"] = (
            df["valeur_fonciere"] /
            df["surface_reelle_bati"].replace(0, np.nan)
        ).round(0)
    else:
        df["prix_m2"] = np.nan

    # Type de local normalisé (Appartement / Maison / etc.)
    if "type_local" in df.columns:
        df["type_local"] = df["type_local"].fillna("Inconnu")
        df["est_residentiel"] = df["type_local"].isin(TYPES_RESIDENTIELS)
    else:
        df["est_residentiel"] = True  # pas d'info = on garde

    # Commune normalisée
    commune_col = next((c for c in df.columns if c in ("nom_commune", "libelle_commune")), None)
    df["commune"] = df[commune_col] if commune_col else df.get("code_postal", "")

    df["cle_commune_section"] = df["code_commune"].astype(str).str.zfill(5) if "code_commune" in df.columns else ""
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
    if type_col:
        mask     = df[type_col].astype(str).str.lower().str.contains("succ|héritage|heritage", na=False)
        df_succ  = df[mask].copy()
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
# 3. MALUS — appliqué à tout signal
# ════════════════════════════════════════════════════════════

def appliquer_malus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique des malus de qualité sur le score_signal.
    Ces malus pénalisent les données aberrantes ou hors cible.

    Malus appliqués :
      -15  Prix/m² aberrant  (< 500 ou > 30 000 €/m²)
      -10  Surface incohérente (< 9 m² ou > 500 m²)
      -10  Bien non résidentiel (type_local non résidentiel connu)
      -5   Prix global très bas (< 10 000 €) → parking, cave, dépendance
    """
    df = df.copy()
    malus = pd.Series(0.0, index=df.index)

    # Prix/m² aberrant
    if "prix_m2" in df.columns:
        prix_m2 = pd.to_numeric(df["prix_m2"], errors="coerce")
        malus += np.where(
            (prix_m2 < 500) | (prix_m2 > 30_000),
            -15, 0
        )

    # Surface incohérente
    if "surface_reelle_bati" in df.columns:
        surf = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
        malus += np.where((surf < 9) | (surf > 500), -10, 0)

    # Type non résidentiel
    if "est_residentiel" in df.columns:
        malus += np.where(~df["est_residentiel"], -10, 0)

    # Prix global trop bas
    if "valeur_fonciere" in df.columns:
        prix = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
        malus += np.where(prix < 10_000, -5, 0)

    df["malus"] = malus.round(0)
    df["score_signal"] = (df["score_signal"] + malus).clip(0, 100).round(1)
    return df


# ════════════════════════════════════════════════════════════
# 4. CROISEMENT DVF × BODACC — signal succession
# ════════════════════════════════════════════════════════════

def _score_heritage(df: pd.DataFrame, date_succession: datetime) -> pd.Series:
    """
    Base 50 pts
    +30 si vente < 90 j   · +20 si < 180 j  · +10 si < 365 j  · +0 au-delà
    +20 si adjudication
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
                        fenetre_succession_mois: int = 18) -> pd.DataFrame:
    """
    Pour chaque annonce BODACC de succession du département cible,
    cherche les ventes DVF dans les `fenetre_succession_mois` mois suivants.

    Le paramètre `fenetre_succession_mois` (≠ lookback global) contrôle
    combien de temps après l'annonce de succession on considère qu'une vente
    est liée. 18 mois est la valeur standard (délai moyen de règlement).
    Réduire à 9 mois = signaux forts uniquement. Augmenter à 24 = filet plus large.
    """
    dvf_f    = dvf.copy()
    bodacc_f = (
        bodacc[bodacc["dept_bodacc"] == dept].copy()
        if "dept_bodacc" in bodacc.columns and bodacc["dept_bodacc"].notna().any()
        else bodacc.copy()
    )
    if bodacc_f.empty or dvf_f.empty:
        return pd.DataFrame()

    dvf_f["dept"] = dvf_f["code_postal"].astype(str).str[:2]
    fenetre_jours = fenetre_succession_mois * 30
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
            matches["score_heritage"]  = _score_heritage(matches, date)
            matches["score_signal"]    = matches["score_heritage"]
            matches["signal"]          = "succession_bodacc"
            results.append(matches)

    if not results:
        return pd.DataFrame()

    id_col = "id_mutation" if "id_mutation" in dvf_f.columns else None
    out    = pd.concat(results, ignore_index=True).drop_duplicates(subset=[id_col] if id_col else None)
    return out.sort_values("score_heritage", ascending=False)


# ════════════════════════════════════════════════════════════
# 5. SIGNAUX ÉVÉNEMENTS DE VIE — v2 (seuils resserrés)
# ════════════════════════════════════════════════════════════

def signal_divorce_downsizing(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    Revente rapide d'un bien T3/T4 résidentiel.

    Seuils resserrés v2 :
      - Uniquement biens résidentiels (Appartement / Maison)
      - La vente précédente doit être sur la même adresse EXACTE (pas juste même immeuble)
      - On exige que le délai soit > 30 j (évite doublons de lots DVF le même jour)
      - Score : 80 si < 1 an, 60 si 1–3 ans
    """
    df = dvf.copy()
    if "date_mutation" not in df.columns or "adresse_complete" not in df.columns:
        return pd.DataFrame()

    # Uniquement résidentiel T3/T4
    pieces_ok = (
        df["nombre_pieces_principales"].between(3, 4)
        if "nombre_pieces_principales" in df.columns
        else pd.Series(True, index=df.index)
    )
    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    df = df[pieces_ok & residentiel].copy()

    df_s = df.sort_values("date_mutation")
    df_s["vente_precedente"]    = df_s.groupby("adresse_complete")["date_mutation"].shift(1)
    df_s["delai_revente_jours"] = (df_s["date_mutation"] - df_s["vente_precedente"]).dt.days

    # Seuil minimum 30 j pour éviter multi-lots
    mask = (
        (df_s["delai_revente_jours"] > 30) &
        (df_s["delai_revente_jours"] < 3 * 365)
    )
    result = df_s[mask].copy()
    result["signal"]       = "divorce_ou_separation"
    result["score_signal"] = np.where(result["delai_revente_jours"] < 365, 80, 60)
    return result


def signal_naissance_upgrade(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    Propriétaires d'un T1/T2 résidentiel acheté il y a 2–4 ans.

    v2 : fenêtre resserrée à 2–4 ans (pas depuis 0) car un achat très récent
    (< 2 ans) ne génère pas encore de besoin d'upgrade. Exige un bien résidentiel
    avec une surface > 15 m² (évite parkings et caves).
    """
    df = dvf.copy()
    if "nombre_pieces_principales" not in df.columns:
        return pd.DataFrame()

    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    surf_ok = (
        df["surface_reelle_bati"] > 15
        if "surface_reelle_bati" in df.columns
        else pd.Series(True, index=df.index)
    )
    now = datetime.now()
    # Fenêtre 2–4 ans (pas 0–4 ans comme en v1)
    mask = (
        df["nombre_pieces_principales"].isin([1, 2]) &
        (df["valeur_fonciere"] > 0) &
        (df["date_mutation"] >= now - timedelta(days=4 * 365)) &
        (df["date_mutation"] <= now - timedelta(days=2 * 365)) &
        residentiel &
        surf_ok
    )
    result = df[mask].copy()
    result["signal"]       = "petit_bien_upgrade_potentiel"
    result["score_signal"] = 55.0
    return result


def signal_retraite_downsizing(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    Grand bien (T5+) vendu avec décote significative vs médiane locale.

    v2 : la médiane est calculée uniquement sur les T5+ du même CP
    (et non tous les biens confondus), ce qui rend la comparaison plus juste.
    On exige aussi une surface > 80 m² pour écarter les studios mal classés.
    """
    df = dvf.copy()
    if not {"nombre_pieces_principales", "surface_reelle_bati", "valeur_fonciere"}.issubset(df.columns):
        return pd.DataFrame()

    # Filtres de base
    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    mask_base   = (
        (df["nombre_pieces_principales"] >= 5) &
        (df["surface_reelle_bati"] >= 80) &
        residentiel
    )
    df = df[mask_base].copy()
    if df.empty:
        return pd.DataFrame()

    # Prix/m² et médiane sur T5+ du même CP
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"].replace(0, np.nan)
    df["mediane_t5_cp"] = df.groupby("code_postal")["prix_m2"].transform("median")
    df["decote_pct"]    = (df["mediane_t5_cp"] - df["prix_m2"]) / df["mediane_t5_cp"] * 100

    mask = df["decote_pct"] > 10
    result = df[mask].copy()
    result["signal"]       = "retraite_downsizing"
    result["score_signal"] = np.where(result["decote_pct"] > 20, 80.0, 65.0)
    return result


def signal_primo_acheteurs(dvf: pd.DataFrame) -> pd.DataFrame:
    """
    T1/T2 vendu à < 70% de la médiane du CP.

    v2 : médiane calculée uniquement sur les T1/T2 du même CP (pas tous biens),
    surface minimale de 15 m² et prix minimum de 15 000 € pour exclure les
    parkings et caves qui faussaient le signal v1.
    """
    df = dvf.copy()
    if "valeur_fonciere" not in df.columns:
        return pd.DataFrame()

    residentiel = df.get("est_residentiel", pd.Series(True, index=df.index))
    pieces_ok   = (
        df["nombre_pieces_principales"].isin([1, 2])
        if "nombre_pieces_principales" in df.columns
        else pd.Series(True, index=df.index)
    )
    surf_ok = (
        df["surface_reelle_bati"] >= 15
        if "surface_reelle_bati" in df.columns
        else pd.Series(True, index=df.index)
    )

    # Filtres de base
    df = df[pieces_ok & residentiel & surf_ok & (df["valeur_fonciere"] > 15_000)].copy()
    if df.empty:
        return pd.DataFrame()

    # Médiane sur T1/T2 uniquement (pas tous biens comme en v1)
    df["mediane_cp_t12"] = df.groupby("code_postal")["valeur_fonciere"].transform("median")
    mask   = df["valeur_fonciere"] < df["mediane_cp_t12"] * 0.7
    result = df[mask].copy()
    result["signal"]       = "primo_acheteur_potentiel"
    result["score_signal"] = 50.0
    return result


# ════════════════════════════════════════════════════════════
# 6. BONUS MULTI-SIGNAL
# ════════════════════════════════════════════════════════════

def bonus_multi_signal(df: pd.DataFrame, dfs_bruts: list) -> pd.DataFrame:
    """
    Détecte les adresses présentes dans plusieurs DataFrames de signaux distincts.
    Attribue un bonus de score aux prospects qui cumulent plusieurs signaux.

    +10 si 2 signaux sur la même adresse
    +20 si 3+ signaux sur la même adresse

    Ces prospects sont les plus "chauds" car plusieurs indicateurs convergent.
    """
    if "adresse_complete" not in df.columns:
        return df

    df = df.copy()

    # Compter combien de DF sources contiennent chaque adresse
    adresse_counts = {}
    for src in dfs_bruts:
        if src is None or src.empty or "adresse_complete" not in src.columns:
            continue
        for addr in src["adresse_complete"].dropna().unique():
            adresse_counts[addr] = adresse_counts.get(addr, 0) + 1

    df["nb_signaux"] = df["adresse_complete"].map(adresse_counts).fillna(1).astype(int)
    df["bonus_multi"] = np.where(
        df["nb_signaux"] >= 3, 20,
        np.where(df["nb_signaux"] == 2, 10, 0)
    )
    df["score_signal"] = (df["score_signal"] + df["bonus_multi"]).clip(0, 100).round(1)
    return df


# ════════════════════════════════════════════════════════════
# 7. CONSOLIDATION
# ════════════════════════════════════════════════════════════

def consolider_signaux(*dfs) -> pd.DataFrame:
    frames = [df for df in dfs if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    cols_communes = [
        "adresse_numero", "adresse_suffixe", "adresse_nom_voie",
        "code_postal", "nom_commune", "commune", "adresse_complete",
        "valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
        "prix_m2", "type_local", "est_residentiel",
        "date_mutation", "nature_mutation",
        "signal", "score_signal", "malus",
        "longitude", "latitude",
    ]
    result_frames = [df[[c for c in cols_communes if c in df.columns]] for df in frames]
    consolidated  = pd.concat(result_frames, ignore_index=True)

    # Dédupliquer : garder le meilleur score par adresse
    if "adresse_complete" in consolidated.columns:
        consolidated = (
            consolidated
            .sort_values("score_signal", ascending=False)
            .drop_duplicates(subset=["adresse_complete"])
        )

    # Bonus multi-signal
    consolidated = bonus_multi_signal(consolidated, list(frames))

    # Tri final
    consolidated = consolidated.sort_values("score_signal", ascending=False).reset_index(drop=True)
    consolidated.insert(0, "rang", range(1, len(consolidated) + 1))

    # Signal normalisé (clé courte) pour la carte
    consolidated["signal_carte"] = consolidated["signal"].map(SIGNAL_NORMALIZE).fillna(consolidated["signal"])
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
# 8. PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_pipeline(dept: str = DEPT, annee: int = ANNEE,
                 fenetre_succession_mois: int = 18) -> pd.DataFrame:
    """
    Paramètres
    ----------
    dept                     : Code département cible (ex: "75")
    annee                    : Année du fichier DVF à charger
    fenetre_succession_mois  : Fenêtre temporelle pour le signal succession.
                               Contrôle combien de mois après une annonce BODACC
                               une vente DVF est considérée comme liée.
                               Valeurs recommandées :
                                 9  → signal fort uniquement (ventes très rapides)
                                18  → standard (règlement successoral classique)
                                24  → filet large (inclut les successions complexes)
    """
    # Chargement
    dvf_raw    = download_dvf(dept, annee)
    bodacc_raw = download_bodacc(annee)

    # Nettoyage
    dvf    = clean_dvf(dvf_raw)
    bodacc = clean_bodacc(bodacc_raw)

    # Croisements
    heritage  = croiser_dvf_bodacc(dvf, bodacc, dept, fenetre_succession_mois)
    divorce   = signal_divorce_downsizing(dvf)
    naissance = signal_naissance_upgrade(dvf)
    retraite  = signal_retraite_downsizing(dvf)
    primo     = signal_primo_acheteurs(dvf)

    if not heritage.empty and "score_heritage" in heritage.columns:
        heritage["score_signal"] = heritage["score_heritage"]

    # Malus qualité sur chaque signal individuellement
    heritage  = appliquer_malus(heritage)  if not heritage.empty  else heritage
    divorce   = appliquer_malus(divorce)   if not divorce.empty   else divorce
    naissance = appliquer_malus(naissance) if not naissance.empty else naissance
    retraite  = appliquer_malus(retraite)  if not retraite.empty  else retraite
    primo     = appliquer_malus(primo)     if not primo.empty     else primo

    return consolider_signaux(heritage, divorce, naissance, retraite, primo)


# ════════════════════════════════════════════════════════════
# ENTRY POINT CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dept",     default=DEPT)
    parser.add_argument("--annee",    default=ANNEE, type=int)
    parser.add_argument("--fenetre",  default=18,    type=int,
                        help="Fenêtre succession en mois (9/18/24)")
    args = parser.parse_args()

    prospects = run_pipeline(dept=args.dept, annee=args.annee,
                             fenetre_succession_mois=args.fenetre)
    if not prospects.empty:
        print(f"\nTOP 10 PROSPECTS :")
        print(prospects.head(10).to_string(index=False))

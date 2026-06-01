"""
DVF - BODACC — Application Streamlit
Déployable sur Streamlit Cloud (https://streamlit.io/cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
from pathlib import Path

# ── Import pipeline ──────────────────────────────────────────
from pipeline import run_pipeline, SIGNAL_LABELS, SIGNAL_COLORS

# ── Config page ──────────────────────────────────────────────
st.set_page_config(
    page_title="DVF BODACC — Prospects immobiliers",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:wght@300;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; }

.stApp { background: #f5f2ed; }

/* Métriques */
[data-testid="metric-container"] {
    background: #ede9e2;
    border: 1px solid rgba(26,24,20,.1);
    border-radius: 10px;
    padding: .75rem 1rem;
}
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; font-size: 1.6rem !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #1a1814 !important; }
[data-testid="stSidebar"] * { color: #f5f2ed !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: rgba(245,242,237,.55) !important; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="stSidebar"] .stButton button {
    background: #f5a05a; color: #1a1814; border: none;
    font-family: 'DM Mono', monospace; font-weight: 500;
    width: 100%; border-radius: 6px;
}
[data-testid="stSidebar"] .stButton button:hover { opacity: .85; }

/* Tableau */
[data-testid="stDataFrame"] { border-radius: 10px; }
.stDataFrame td { font-size: .75rem !important; }

/* Spinner */
.stSpinner > div { border-top-color: #f5a05a !important; }

/* Info box */
.info-box {
    background: #ede9e2; border: 1px solid rgba(26,24,20,.1);
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem;
    font-size: .78rem; line-height: 1.7;
}
.info-box b { color: #1a1814; }

/* Score badges */
.badge {
    display: inline-block; padding: .15rem .55rem;
    border-radius: 4px; font-size: .68rem; font-weight: 500;
    color: white; margin-right: .3rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
if "prospects" not in st.session_state:
    st.session_state.prospects = None
if "run_dept" not in st.session_state:
    st.session_state.run_dept = None


# ════════════════════════════════════════════════════════════
# SIDEBAR — Paramètres & lancement
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 DVF × BODACC")
    st.markdown("<div style='font-size:.65rem;opacity:.45;letter-spacing:.08em;text-transform:uppercase;margin-bottom:1.5rem'>Prospects immobiliers</div>", unsafe_allow_html=True)

    dept = st.selectbox("Département", options=[
        "75","69","13","33","31","06","59","67","44","34","76","38","92","93","94"
    ], format_func=lambda x: {
        "75":"75 — Paris","69":"69 — Rhône","13":"13 — Bouches-du-Rhône",
        "33":"33 — Gironde","31":"31 — Haute-Garonne","06":"06 — Alpes-Maritimes",
        "59":"59 — Nord","67":"67 — Bas-Rhin","44":"44 — Loire-Atlantique",
        "34":"34 — Hérault","76":"76 — Seine-Maritime","38":"38 — Isère",
        "92":"92 — Hauts-de-Seine","93":"93 — Seine-Saint-Denis","94":"94 — Val-de-Marne",
    }[x])

    annee = st.selectbox("Année DVF", [2024, 2023, 2022, 2021], index=0)

    st.markdown("---")

    score_min = st.slider("Score minimum", 0, 90, 70, 5,
                          help="Filtrer les prospects sous ce seuil (0 = tous)")

    signaux_choix = st.multiselect(
        "Signaux actifs",
        options=list(SIGNAL_LABELS.keys()),
        default=list(SIGNAL_LABELS.keys()),
        format_func=lambda x: SIGNAL_LABELS[x],
    )

    st.markdown("---")

    run_clicked = st.button("→ Lancer l'analyse", type="primary")

    st.markdown("---")
    st.markdown("""
<div style='font-size:.62rem; opacity:.4; line-height:1.7'>
Sources : DVF data.gouv.fr<br>
BODACC OpenDataSoft<br><br>
⚠ Vérification RGPD requise<br>
avant toute utilisation.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# LANCEMENT PIPELINE
# ════════════════════════════════════════════════════════════
if run_clicked:
    with st.spinner(f"Analyse en cours — Dept. {dept} / {annee}…"):
        try:
            df = run_pipeline(dept=dept, annee=annee)
            st.session_state.prospects = df
            st.session_state.run_dept  = dept
            st.success(f"✓ {len(df):,} prospects identifiés")
        except Exception as e:
            st.error(f"Erreur pipeline : {e}")
            st.stop()


# ════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ════════════════════════════════════════════════════════════
df_raw = st.session_state.prospects

if df_raw is None:
    # ── État vide ──
    st.markdown("## Prospects immobiliers DVF × BODACC")
    st.markdown("""
<div class="info-box">
<b>Comment ça marche ?</b><br>
1. Choisissez un <b>département</b> et une <b>année</b> dans la sidebar.<br>
2. Cliquez <b>→ Lancer l'analyse</b> — le pipeline télécharge DVF + BODACC, croise les données et score chaque prospect.<br>
3. La carte et le tableau se remplissent automatiquement. Vous pouvez filtrer par signal et par score.<br>
4. Téléchargez le CSV pour un usage externe.
</div>
""", unsafe_allow_html=True)

    # Grille de scoring
    st.markdown("### Grille de scoring")
    scoring_data = {
        "Signal": ["Succession BODACC", "Divorce / séparation", "Upgrade famille", "Retraite / downsizing", "Primo-acheteur"],
        "Critère": [
            "Vente dans les 18 mois après une succession BODACC",
            "Bien T3/T4 revendu < 3 ans après l'achat précédent",
            "Achat T1/T2 dans les 4 dernières années",
            "T5+ vendu > 10 % sous la médiane commune",
            "T1/T2 vendu à < 70 % du prix médian du CP",
        ],
        "Score": ["50–100", "60 ou 80", "55", "65 ou 80", "50"],
        "Détail bonus": [
            "+30 si <90j / +20 si <180j / +10 si <365j / +20 adjudication",
            "+80 si délai <1 an | +60 si 1–3 ans",
            "Score fixe",
            "+80 si décote >20% | +65 si 10–20%",
            "Score fixe",
        ],
    }
    st.dataframe(pd.DataFrame(scoring_data), use_container_width=True, hide_index=True)
    st.stop()


# ── Filtrage ─────────────────────────────────────────────────
df = df_raw.copy()
if "signal_carte" in df.columns:
    sig_col = "signal_carte"
elif "signal" in df.columns:
    sig_col = "signal"
    # Normaliser les valeurs longues
    NORM = {
        "succession_bodacc":"heritage","divorce_ou_separation":"divorce",
        "petit_bien_upgrade_potentiel":"upgrade","retraite_downsizing":"retraite",
        "primo_acheteur_potentiel":"primo",
    }
    df[sig_col] = df[sig_col].map(lambda x: NORM.get(x, x))
else:
    sig_col = None

if sig_col:
    df = df[df[sig_col].isin(signaux_choix)]
if "score_signal" in df.columns:
    df = df[df["score_signal"] >= score_min]

total = len(df)

# ════════════════════════════════════════════════════════════
# MÉTRIQUES
# ════════════════════════════════════════════════════════════
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Prospects", f"{total:,}")
with col2:
    avg = df["score_signal"].mean() if "score_signal" in df.columns and total else 0
    st.metric("Score moyen", f"{avg:.1f}/100")
with col3:
    hot = (df["score_signal"] >= 80).sum() if "score_signal" in df.columns else 0
    st.metric("Très chauds ≥80", f"{hot:,}")
with col4:
    chaud = ((df["score_signal"] >= 70) & (df["score_signal"] < 80)).sum() if "score_signal" in df.columns else 0
    st.metric("Chauds 70–79", f"{chaud:,}")
with col5:
    zones = df["nom_commune"].nunique() if "nom_commune" in df.columns else (df["commune"].nunique() if "commune" in df.columns else 0)
    st.metric("Communes", f"{zones:,}")

st.markdown("---")

# ════════════════════════════════════════════════════════════
# CARTE LEAFLET (composant HTML)
# ════════════════════════════════════════════════════════════
def build_leaflet_html(df: pd.DataFrame, sig_col: str) -> str:
    """Génère le HTML Leaflet avec les données injectées."""
    SIGNAL_COLORS_JS = json.dumps(SIGNAL_COLORS)
    SIGNAL_LABELS_JS = json.dumps(SIGNAL_LABELS)

    # Préparer les points
    pts = []
    has_coords = "latitude" in df.columns and "longitude" in df.columns
    for _, row in df.iterrows():
        lat = float(row.get("latitude", 0) or 0)
        lng = float(row.get("longitude", 0) or 0)
        if not has_coords or (lat == 0 and lng == 0):
            continue
        sig   = str(row.get(sig_col, "") or "")
        score = float(row.get("score_signal", 50) or 50)
        addr  = str(row.get("adresse_complete", "") or "")
        cp    = str(row.get("code_postal", "") or "")
        comm  = str(row.get("nom_commune", row.get("commune", "")) or "")
        prix  = float(row.get("valeur_fonciere", 0) or 0)
        surf  = float(row.get("surface_reelle_bati", 0) or 0)
        piec  = float(row.get("nombre_pieces_principales", 0) or 0)
        date  = str(row.get("date_mutation", "") or "")[:10]
        pts.append({"lat":lat,"lng":lng,"signal":sig,"score":score,
                    "adresse":addr,"cp":cp,"commune":comm,
                    "prix":prix,"surface":surf,"pieces":piec,"date":date})

    data_json = json.dumps(pts)
    center_lat = df["latitude"].median() if has_coords and len(df) else 46.5
    center_lng = df["longitude"].median() if has_coords and len(df) else 2.3
    zoom = 11 if len(pts) > 0 else 6

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family: monospace; background:#f5f2ed; }}
  #map {{ width:100%; height:520px; }}
  .info-popup {{ font-family:monospace; font-size:.78rem; line-height:1.6; min-width:220px; }}
  .info-popup .sig-badge {{ display:inline-block; padding:.12rem .4rem; border-radius:3px; color:#fff; font-size:.65rem; font-weight:600; margin-bottom:.4rem; }}
  .info-popup .score-big {{ font-size:1.4rem; font-weight:700; line-height:1; margin-bottom:.3rem; }}
  .info-popup .addr {{ font-size:.75rem; font-weight:600; margin-bottom:.4rem; color:#1a1814; }}
  .info-popup table {{ width:100%; border-collapse:collapse; font-size:.7rem; }}
  .info-popup td {{ padding:.15rem 0; border-bottom:1px solid rgba(0,0,0,.07); }}
  .info-popup td:first-child {{ color:#6b6860; }}
  .info-popup td:last-child {{ font-weight:500; text-align:right; }}
  .info-popup .breakdown {{ background:#ede9e2; border-radius:5px; padding:.4rem .5rem; margin-top:.5rem; font-size:.65rem; }}
  .info-popup .bd-title {{ color:#b8b5ae; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.25rem; font-size:.6rem; }}
  .info-popup .bd-row {{ display:flex; justify-content:space-between; }}
  .info-popup .bd-plus {{ color:#1a6b4a; font-weight:600; }}
  #no-coords {{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    background:rgba(245,242,237,.95); padding:1.5rem 2rem; border-radius:10px;
    border:1px solid rgba(0,0,0,.1); font-size:.85rem; color:#6b6860; text-align:center;
    display:none; z-index:1000; }}
  .map-wrap {{ position:relative; }}
</style>
</head><body>
<div class="map-wrap">
  <div id="map"></div>
  <div id="no-coords">⚠ Coordonnées GPS absentes dans ce CSV.<br>Les points ne peuvent pas être affichés.</div>
</div>
<script>
const DATA   = {data_json};
const COLORS = {SIGNAL_COLORS_JS};
const LABELS = {SIGNAL_LABELS_JS};

const BREAKDOWN = {{
  heritage: p => [['Base','+50 pts'],[p.score>=80?'Adjudication':p.score>=80?'':p.score>=70?'Vente <180j':'Vente <365j', p.score>=80?'+20 pts':p.score>=70?'+20 pts':'+10 pts'],['Délai succession','calculé']],
  divorce:  p => [['Bien T3/T4 revendu <3 ans',''], [p.score>=75?'Délai <1 an':'Délai 1–3 ans', p.score>=75?'+80 pts':'+60 pts']],
  upgrade:  () => [['T1/T2 acheté <4 ans',''],['Candidat upgrade','+55 pts']],
  retraite: p => [['T5+ sous médiane commune',''], [p.score>=78?'Décote >20%':'Décote 10–20%', p.score>=78?'+80 pts':'+65 pts']],
  primo:    () => [['T1/T2 <70% médiane CP',''],['Futur 2e achat','+50 pts']],
}};

const map = L.map('map').setView([{center_lat},{center_lng}], {zoom});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap contributors', maxZoom:18
}}).addTo(map);

if (DATA.length === 0) {{
  document.getElementById('no-coords').style.display = 'block';
}} else {{
  const layer = L.layerGroup().addTo(map);
  DATA.forEach(pt => {{
    const color  = COLORS[pt.signal] || '#888';
    const radius = 5 + (pt.score - 50) / 8;
    const prix   = pt.prix ? pt.prix.toLocaleString('fr-FR') + ' €' : '—';
    const surf   = pt.surface ? pt.surface + ' m²' : '—';
    const piec   = pt.pieces  ? pt.pieces + 'p' : '—';
    const bd     = (BREAKDOWN[pt.signal]||(() => []))(pt);
    const bdHtml = bd.map(([l,v])=>`<div class="bd-row"><span>${{l}}</span><span class="bd-plus">${{v}}</span></div>`).join('');
    const scoreColor = pt.score>=80?'#c4440a':pt.score>=70?'#d4850a':'#1a1814';
    const popup = `<div class="info-popup">
      <div class="sig-badge" style="background:${{color}}">${{LABELS[pt.signal]||pt.signal}}</div>
      <div class="score-big" style="color:${{scoreColor}}">${{pt.score}}<span style="font-size:.7rem;color:#6b6860">/100</span></div>
      <div class="addr">${{pt.adresse||'—'}}</div>
      <table>
        <tr><td>Commune</td><td>${{pt.commune||'—'}}</td></tr>
        <tr><td>Code postal</td><td>${{pt.cp||'—'}}</td></tr>
        <tr><td>Prix</td><td>${{prix}}</td></tr>
        <tr><td>Surface</td><td>${{surf}} · ${{piec}}</td></tr>
        <tr><td>Date mutation</td><td>${{pt.date||'—'}}</td></tr>
      </table>
      <div class="breakdown"><div class="bd-title">Détail scoring</div>${{bdHtml}}</div>
    </div>`;
    L.circleMarker([pt.lat, pt.lng], {{
      radius, color, fillColor:color, fillOpacity:.65, weight:1.2, opacity:.9
    }}).bindPopup(popup).bindTooltip(
      `${{LABELS[pt.signal]||pt.signal}} · Score ${{pt.score}}`,
      {{direction:'top', offset:[0,-4]}}
    ).addTo(layer);
  }});

  if (DATA.length > 0) {{
    const lats = DATA.map(d=>d.lat), lngs = DATA.map(d=>d.lng);
    map.fitBounds([[Math.min(...lats),Math.min(...lngs)],[Math.max(...lats),Math.max(...lngs)]], {{padding:[30,30]}});
  }}
}}
</script>
</body></html>"""


# Rendu carte
st.markdown("### 🗺 Carte des prospects")

has_coords = "latitude" in df.columns and "longitude" in df.columns
df_map = df[df["latitude"].notna() & df["longitude"].notna() & (df["latitude"] != 0)] if has_coords else pd.DataFrame()

if not has_coords or df_map.empty:
    st.info("⚠ Coordonnées GPS absentes — les données DVF ne contiennent pas toujours latitude/longitude. Vérifiez que votre département est couvert par le fichier geo-dvf.")
else:
    map_html = build_leaflet_html(df_map, sig_col)
    st.components.v1.html(map_html, height=530, scrolling=False)


# ════════════════════════════════════════════════════════════
# TABLEAU + RANKING
# ════════════════════════════════════════════════════════════
st.markdown("---")
tab1, tab2 = st.tabs(["📋 Tableau prospects", "🏆 Ranking par zone"])

with tab1:
    st.markdown(f"**{total:,} prospects** — Score ≥ {score_min}, signaux : {', '.join(SIGNAL_LABELS.get(s,s) for s in signaux_choix)}")

    # Colonnes à afficher
    cols_show = [c for c in [
        "rang", "adresse_complete", "code_postal", "nom_commune",
        "signal_label", "score_signal", "chaleur",
        "valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales",
        "date_mutation", "nature_mutation",
    ] if c in df.columns]

    rename_map = {
        "rang":"#", "adresse_complete":"Adresse", "code_postal":"CP",
        "nom_commune":"Commune", "signal_label":"Signal", "score_signal":"Score",
        "chaleur":"Chaleur", "valeur_fonciere":"Prix (€)",
        "surface_reelle_bati":"Surface (m²)", "nombre_pieces_principales":"Pièces",
        "date_mutation":"Date", "nature_mutation":"Nature",
    }

    df_display = df[cols_show].rename(columns=rename_map).head(500)
    st.dataframe(df_display, use_container_width=True, hide_index=True,
                 column_config={
                     "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                     "Prix (€)": st.column_config.NumberColumn("Prix (€)", format="%.0f €"),
                 })
    if total > 500:
        st.caption(f"Affichage limité à 500 lignes sur {total:,}. Téléchargez le CSV pour tout voir.")

    # Export CSV
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇ Télécharger le CSV complet",
        data=csv_bytes,
        file_name=f"prospects_{st.session_state.run_dept}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with tab2:
    st.markdown("**Top zones par nombre de prospects**")
    if "nom_commune" in df.columns:
        zone_col = "nom_commune"
    elif "commune" in df.columns:
        zone_col = "commune"
    elif "code_postal" in df.columns:
        zone_col = "code_postal"
    else:
        zone_col = None

    if zone_col and sig_col:
        ranking = (
            df.groupby(zone_col)
            .agg(
                Prospects=(sig_col, "count"),
                Score_moy=("score_signal", "mean"),
                Tres_chauds=("score_signal", lambda x: (x >= 80).sum()),
                Chauds=("score_signal", lambda x: ((x >= 70) & (x < 80)).sum()),
                Prix_moy=("valeur_fonciere", "mean") if "valeur_fonciere" in df.columns else ("score_signal", "count"),
            )
            .sort_values("Prospects", ascending=False)
            .reset_index()
        )
        ranking.columns = [zone_col.replace("_", " ").title(), "Prospects", "Score moy.", "≥80", "70–79", "Prix moy. (€)"]
        ranking["Score moy."] = ranking["Score moy."].round(1)
        ranking["Prix moy. (€)"] = ranking["Prix moy. (€)"].round(0)

        st.dataframe(ranking.head(30), use_container_width=True, hide_index=True,
                     column_config={
                         "Score moy.": st.column_config.ProgressColumn("Score moy.", min_value=0, max_value=100, format="%.1f"),
                         "Prospects": st.column_config.NumberColumn("Prospects"),
                     })
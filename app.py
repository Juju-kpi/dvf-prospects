"""
DVF × BODACC — Application Streamlit v3.0
Outil BI de prospection immobilière
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from pipeline import run_pipeline, SIGNAL_LABELS, SIGNAL_COLORS, SIGNAL_SEGMENTS

# ── Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="DVF BODACC — Prospection immobilière",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,500;0,700;1,300&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
:root {
  --ink:#1a1814;--ink-light:#6b6860;--ink-faint:#b8b5ae;
  --paper:#f5f2ed;--paper-2:#ede9e2;--paper-3:#e0dbd2;
  --accent:#f5a05a;--accent-red:#c4440a;--accent-green:#1a6b4a;
  --accent-blue:#1a4a8a;--accent-purple:#7a4aa0;
  --border:rgba(26,24,20,.12);--radius:10px;--sidebar:#1a1814;
}
html,body,[class*="css"]{font-family:'DM Mono',monospace!important;color:var(--ink);}
h1,h2,h3{font-family:'Fraunces',serif!important;letter-spacing:-.02em;}
.stApp{background:var(--paper)!important;}
.stApp>header{background:var(--ink)!important;}
.main .block-container{padding-top:0!important;padding-bottom:3rem!important;max-width:1500px!important;}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--sidebar)!important;border-right:none!important;}
[data-testid="stSidebar"]>div:first-child{background:var(--sidebar)!important;padding:1.25rem 1rem!important;}
[data-testid="stSidebar"] *{color:#f5f2ed!important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stRadio label{color:rgba(245,242,237,.5)!important;font-size:.63rem!important;text-transform:uppercase!important;letter-spacing:.1em!important;}
[data-testid="stSidebar"] .stSelectbox>div>div,
[data-testid="stSidebar"] .stMultiSelect>div>div{background:rgba(255,255,255,.08)!important;border:1px solid rgba(255,255,255,.15)!important;border-radius:6px!important;}
[data-testid="stSidebar"] .stSelectbox svg{fill:rgba(245,242,237,.5)!important;}
[data-testid="stSidebar"] hr{border-color:rgba(245,242,237,.1)!important;margin:.65rem 0!important;}
[data-testid="stSidebar"] .stButton>button{background:var(--accent)!important;color:var(--ink)!important;border:none!important;font-family:'DM Mono',monospace!important;font-weight:500!important;font-size:.75rem!important;width:100%!important;border-radius:6px!important;padding:.55rem 1rem!important;}
[data-testid="stSidebar"] .stButton>button:hover{opacity:.85!important;}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;gap:0!important;}
[data-testid="stTabs"] [data-baseweb="tab"]{background:transparent!important;border:none!important;color:var(--ink-light)!important;font-family:'DM Mono',monospace!important;font-size:.7rem!important;letter-spacing:.05em!important;text-transform:uppercase!important;padding:.65rem 1.1rem!important;border-bottom:2px solid transparent!important;}
[data-testid="stTabs"] [data-baseweb="tab"]:hover{color:var(--ink)!important;}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--ink)!important;border-bottom:2px solid var(--accent-red)!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none!important;}
[data-testid="stTabs"] [data-baseweb="tab-panel"]{padding-top:1.5rem!important;}

/* Metrics */
[data-testid="metric-container"]{background:var(--paper-2)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;padding:.85rem 1rem!important;}
[data-testid="stMetricLabel"]{font-size:.6rem!important;text-transform:uppercase!important;letter-spacing:.08em!important;color:var(--ink-light)!important;}
[data-testid="stMetricValue"]{font-family:'Fraunces',serif!important;font-size:1.55rem!important;font-weight:700!important;color:var(--ink)!important;line-height:1.1!important;}
[data-testid="stMetricDelta"] svg{display:none!important;}

/* DataFrame */
[data-testid="stDataFrame"]{border-radius:var(--radius)!important;border:1px solid var(--border)!important;overflow:hidden!important;}
.stDataFrame td,.stDataFrame th{font-size:.72rem!important;font-family:'DM Mono',monospace!important;}

/* Alerts */
.stSuccess{background:rgba(26,107,74,.08)!important;border:1px solid rgba(26,107,74,.2)!important;border-radius:var(--radius)!important;}
.stError{background:rgba(196,68,10,.08)!important;border:1px solid rgba(196,68,10,.2)!important;border-radius:var(--radius)!important;}
.stInfo{background:var(--paper-2)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;}
.stWarning{background:rgba(245,160,90,.08)!important;border:1px solid rgba(245,160,90,.3)!important;border-radius:var(--radius)!important;}
.stSpinner>div{border-top-color:var(--accent)!important;}

/* Download */
[data-testid="stDownloadButton"]>button{background:var(--ink)!important;color:var(--paper)!important;border:none!important;font-family:'DM Mono',monospace!important;font-size:.72rem!important;border-radius:6px!important;padding:.45rem 1rem!important;}
[data-testid="stDownloadButton"]>button:hover{opacity:.85!important;}

/* Custom */
.page-header{background:var(--ink);color:var(--paper);margin:0 -3rem 1.75rem -3rem;padding:1.4rem 3rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;}
.page-header .logo{font-family:'Fraunces',serif;font-size:1.4rem;font-weight:700;letter-spacing:-.03em;color:var(--paper);}
.page-header .logo span{color:#f5a05a;}
.page-header .tagline{font-size:.62rem;color:rgba(245,242,237,.35);letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem;}
.page-header .badge{font-size:.62rem;background:rgba(245,160,90,.15);color:#f5a05a;border:1px solid rgba(245,160,90,.3);padding:.3rem .75rem;border-radius:20px;letter-spacing:.06em;white-space:nowrap;}

.sh{font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-faint);padding-bottom:.45rem;border-bottom:1px solid var(--border);margin-bottom:1rem;}
.sh-dark{font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:rgba(245,242,237,.3);padding-bottom:.45rem;border-bottom:1px solid rgba(245,242,237,.1);margin-bottom:.85rem;}

/* Score pill */
.score-pill{display:inline-flex;align-items:center;gap:.4rem;padding:.25rem .65rem;border-radius:20px;font-family:'DM Mono',monospace;font-size:.7rem;font-weight:600;}
.sp-hot{background:rgba(196,68,10,.12);color:#c4440a;border:1px solid rgba(196,68,10,.2);}
.sp-warm{background:rgba(245,160,90,.12);color:#b07030;border:1px solid rgba(245,160,90,.2);}
.sp-cool{background:rgba(26,74,138,.1);color:#1a4a8a;border:1px solid rgba(26,74,138,.15);}
.sp-cold{background:var(--paper-3);color:var(--ink-faint);border:1px solid var(--border);}

/* Insight card */
.insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.85rem;margin-bottom:1.25rem;}
.insight-card{background:var(--paper);border:1px solid var(--border);border-radius:var(--radius);padding:1.1rem 1.25rem;position:relative;overflow:hidden;}
.insight-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.ic-heritage::before{background:#8a4a1a;}
.ic-divorce::before{background:#c4440a;}
.ic-upgrade::before{background:#1a6b4a;}
.ic-retraite::before{background:#1a4a8a;}
.ic-primo::before{background:#7a4aa0;}
.ic-multi::before{background:var(--accent);}
.insight-card .ic-label{font-size:.6rem;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-faint);margin-bottom:.35rem;}
.insight-card .ic-val{font-family:'Fraunces',serif;font-size:1.7rem;font-weight:700;line-height:1;color:var(--ink);margin-bottom:.2rem;}
.insight-card .ic-sub{font-size:.68rem;color:var(--ink-light);line-height:1.5;}

/* Prospect card (liste) */
.prospect-row{display:flex;align-items:flex-start;gap:1rem;padding:1rem;border:1px solid var(--border);border-radius:var(--radius);margin-bottom:.6rem;background:var(--paper);transition:box-shadow .15s;}
.prospect-row:hover{box-shadow:0 2px 12px rgba(26,24,20,.07);}
.pr-score{text-align:center;min-width:52px;}
.pr-score .val{font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;line-height:1;}
.pr-score .lbl{font-size:.55rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-faint);}
.pr-body{flex:1;min-width:0;}
.pr-addr{font-weight:500;font-size:.82rem;color:var(--ink);margin-bottom:.2rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pr-meta{font-size:.7rem;color:var(--ink-light);line-height:1.7;}
.pr-meta span{margin-right:.75rem;}
.pr-tags{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.4rem;}
.pr-tag{font-size:.6rem;padding:.12rem .4rem;border-radius:4px;font-weight:500;}
.tag-signal{background:var(--paper-2);border:1px solid var(--border);color:var(--ink-light);}
.tag-chaud{background:rgba(196,68,10,.08);border:1px solid rgba(196,68,10,.15);color:#c4440a;}
.tag-tiede{background:rgba(245,160,90,.1);border:1px solid rgba(245,160,90,.2);color:#b07030;}
.tag-froid{background:var(--paper-3);border:1px solid var(--border);color:var(--ink-faint);}
.tag-multi{background:rgba(245,160,90,.12);border:1px solid rgba(245,160,90,.25);color:#c47820;}
.pr-right{text-align:right;min-width:90px;}
.pr-prix{font-family:'Fraunces',serif;font-size:1rem;font-weight:600;color:var(--ink);}
.pr-surf{font-size:.68rem;color:var(--ink-faint);margin-top:.15rem;}
.pr-decote{font-size:.65rem;color:#c4440a;font-weight:600;}

/* Zone card */
.zone-card{background:var(--paper);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.25rem;margin-bottom:.5rem;}
.zone-card .zc-head{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem;}
.zone-card .zc-name{font-weight:600;font-size:.85rem;color:var(--ink);}
.zone-card .zc-cp{font-size:.68rem;color:var(--ink-faint);}
.zone-bar{height:4px;background:var(--paper-3);border-radius:2px;margin:.5rem 0;}
.zone-bar-fill{height:100%;border-radius:2px;background:var(--accent-red);}
.zone-stats{display:flex;gap:1.5rem;font-size:.68rem;color:var(--ink-light);}
.zone-stats b{color:var(--ink);}

/* Segment card */
.seg-card{background:var(--paper);border:1px solid var(--border);border-radius:var(--radius);padding:1.1rem 1.25rem;margin-bottom:.65rem;}
.seg-card .sc-head{display:flex;align-items:center;gap:.65rem;margin-bottom:.5rem;}
.seg-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.seg-card .sc-name{font-weight:600;font-size:.82rem;color:var(--ink);}
.seg-card .sc-count{font-size:.68rem;color:var(--ink-faint);margin-left:auto;}
.seg-card .sc-msg{font-family:'Lora',serif;font-size:.82rem;color:var(--ink-light);line-height:1.65;margin-bottom:.5rem;}
.seg-card .sc-stats{display:flex;gap:1.5rem;font-size:.68rem;color:var(--ink-light);}
.seg-card .sc-stats b{color:var(--ink);}

/* Distribution bar */
.distrib-bar{display:flex;height:22px;border-radius:6px;overflow:hidden;margin:.75rem 0;}
.db-seg{display:flex;align-items:center;justify-content:center;font-size:.62rem;font-weight:600;color:#fff;transition:flex .3s;}

/* Info box */
.info-box{background:var(--paper-2);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:var(--radius);padding:1rem 1.25rem;margin-bottom:1.25rem;font-size:.78rem;line-height:1.8;color:var(--ink-light);}
.info-box b{color:var(--ink);}
.sn{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;background:var(--ink);color:var(--paper);border-radius:50%;font-size:.6rem;font-weight:700;margin-right:.3rem;}

/* Scoring table */
.sc-table{width:100%;border-collapse:collapse;font-size:.74rem;font-family:'DM Mono',monospace;}
.sc-table th{background:var(--ink);color:var(--paper);padding:.55rem .85rem;text-align:left;font-size:.6rem;letter-spacing:.08em;text-transform:uppercase;}
.sc-table tr:nth-child(even) td{background:var(--paper-2);}
.sc-table td{padding:.55rem .85rem;border-bottom:1px solid var(--border);color:var(--ink-light);vertical-align:top;line-height:1.6;}
.sc-table td:first-child{color:var(--ink);font-weight:500;}
.sc-table td:nth-child(3){color:var(--accent-red);font-weight:600;}

hr{border:none!important;border-top:1px solid var(--border)!important;margin:1.1rem 0!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for k, v in {"prospects": None, "run_dept": None, "run_annee": None, "run_params": {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

DEPT_LABELS = {
    "75":"75 — Paris","69":"69 — Rhône","13":"13 — Bouches-du-Rhône",
    "33":"33 — Gironde","31":"31 — Haute-Garonne","06":"06 — Alpes-Maritimes",
    "59":"59 — Nord","67":"67 — Bas-Rhin","44":"44 — Loire-Atlantique",
    "34":"34 — Hérault","76":"76 — Seine-Maritime","38":"38 — Isère",
    "92":"92 — Hauts-de-Seine","93":"93 — Seine-Saint-Denis","94":"94 — Val-de-Marne",
}

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style='margin-bottom:1.4rem'>
  <div style='font-family:"Fraunces",serif;font-size:1.15rem;font-weight:700;letter-spacing:-.02em;color:#f5f2ed'>
    DVF <span style='color:#f5a05a'>×</span> BODACC
  </div>
  <div style='font-size:.58rem;color:rgba(245,242,237,.3);text-transform:uppercase;letter-spacing:.1em;margin-top:.15rem'>
    Outil de prospection immobilière v3
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='sh-dark'>Paramètres</div>", unsafe_allow_html=True)
    dept  = st.selectbox("Département", options=list(DEPT_LABELS.keys()), format_func=lambda x: DEPT_LABELS[x])
    annee = st.selectbox("Année DVF", [2024, 2023, 2022, 2021], index=0)
    fenetre = st.select_slider(
        "Fenêtre succession",
        options=[9, 12, 18, 24],
        value=18,
        help="Mois après annonce BODACC pendant lesquels on cherche une vente DVF liée.\n9 = signaux très forts · 18 = standard · 24 = filet large"
    )

    st.markdown("---")
    st.markdown("<div class='sh-dark'>Filtres résultats</div>", unsafe_allow_html=True)
    score_min = st.slider("Score minimum (percentile)", 0, 90, 60, 5,
                          help="0 = tous · 60 = top 40% · 80 = top 20% · 90 = top 10%")
    signaux_choix = st.multiselect(
        "Signaux actifs",
        options=list(SIGNAL_LABELS.keys()),
        default=list(SIGNAL_LABELS.keys()),
        format_func=lambda x: SIGNAL_LABELS[x],
    )
    chaleurs_choix = st.multiselect(
        "Chaleur CRM",
        options=["très chaud", "chaud", "tiède", "froid"],
        default=["très chaud", "chaud", "tiède"],
    )

    st.markdown("---")
    run_clicked = st.button("→ Lancer l'analyse", type="primary")

    st.markdown("""
<div style='margin-top:1.5rem;font-size:.58rem;color:rgba(245,242,237,.25);line-height:2;font-family:"DM Mono",monospace'>
  Sources · DVF data.gouv.fr<br>
  BODACC OpenDataSoft<br><br>
  Score = percentile dans le dept.<br>
  80 = top 20% · 95 = top 5%<br><br>
  ⚠ Vérification RGPD requise.
</div>
""", unsafe_allow_html=True)

# ── En-tête ───────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div>
    <div class="logo">DVF <span>×</span> BODACC</div>
    <div class="tagline">Prospection immobilière par signaux de vie · Score percentile département</div>
  </div>
  <div class="badge">Pipeline v3.0 · Scoring percentile</div>
</div>
""", unsafe_allow_html=True)

# ── Lancement pipeline ────────────────────────────────────────
if run_clicked:
    with st.spinner(f"Analyse — {DEPT_LABELS[dept]} / {annee} · fenêtre {fenetre} mois…"):
        try:
            df = run_pipeline(dept=dept, annee=annee, fenetre_succession_mois=fenetre)
            st.session_state.prospects   = df
            st.session_state.run_dept    = dept
            st.session_state.run_annee   = annee
            st.session_state.run_params  = {"fenetre": fenetre}
            hot = (df["score_final"] >= 80).sum() if "score_final" in df.columns else 0
            st.success(f"✓ {len(df):,} prospects identifiés — {hot:,} dans le top 20%")
        except Exception as e:
            st.error(f"Erreur pipeline : {e}")
            st.stop()

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab_dash, tab_prospects, tab_zones, tab_segments, tab_carte, tab_doc = st.tabs([
    "📊 Dashboard", "🎯 Prospects", "🗺 Zones", "🏷 Segments", "📍 Carte", "📖 Doc"
])

df_raw = st.session_state.prospects

# ─── État vide ───────────────────────────────────────────────
if df_raw is None:
    with tab_dash:
        st.markdown("""
<div class="info-box">
  <b>Bienvenue dans l'outil de prospection DVF × BODACC v3</b><br><br>
  <span class="sn">1</span> Sélectionnez un <b>département</b> et une <b>année</b> dans la sidebar.<br>
  <span class="sn">2</span> Ajustez la <b>fenêtre succession</b> (18 mois = standard).<br>
  <span class="sn">3</span> Cliquez <b>→ Lancer l'analyse</b> — le pipeline croise DVF + BODACC et calcule un <b>score percentile</b> par rapport à tous les biens du département.<br>
  <span class="sn">4</span> Explorez les 5 onglets : Dashboard · Prospects · Zones · Segments · Carte.<br>
  <span class="sn">5</span> Téléchargez le CSV enrichi pour votre CRM ou vos campagnes publicitaires.
</div>
""", unsafe_allow_html=True)
        st.markdown("<div class='sh'>Grille de scoring v3 — score percentile</div>", unsafe_allow_html=True)
        rows = [
            ("Succession BODACC", "Vente dans la fenêtre après annonce", "Base 50 + délai + adjud.", "Percentile dept"),
            ("Divorce / séparation", "T3/T4 revendu 30j–3 ans après achat", "80 (< 1 an) · 60 (1–3 ans)", "Percentile dept"),
            ("Upgrade famille", "T1/T2 résidentiel acheté 2–4 ans avant", "65 (2–3 ans) · 55 (3–4 ans)", "Percentile dept"),
            ("Retraite / downsizing", "T5+ ≥ 80m² décote > 10% médiane T5+", "55–85 selon décote", "Percentile dept"),
            ("Primo-acheteur", "T1/T2 < 70% médiane T1/T2 du CP", "60 (décote > 30%) · 50", "Percentile dept"),
        ]
        st.markdown(
            '<table class="sc-table"><thead><tr><th>Signal</th><th>Critère</th><th>Score brut</th><th>Score final</th></tr></thead><tbody>' +
            "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>" for r in rows) +
            "</tbody></table>", unsafe_allow_html=True
        )
        st.markdown("""<div style='font-size:.72rem;color:var(--ink-light);margin-top:.85rem;line-height:1.8;font-family:"Lora",serif;padding:.85rem 1rem;background:var(--paper-2);border-radius:var(--radius);border:1px solid var(--border)'>
<b style='color:var(--ink)'>Pourquoi un score percentile ?</b><br>
Un score absolu de 80 peut représenter 5 000 biens à Paris et seulement 30 à Périgueux. Le percentile garantit que <b>score 80 = top 20% du département analysé</b>, quelle que soit sa taille. Les malus qualité (bien non résidentiel, prix/m² aberrant, surface hors normes) dégradent le score brut avant la normalisation, ce qui permet aux biens réellement intéressants de se démarquer.
</div>""", unsafe_allow_html=True)
    st.stop()

# ── Filtrage commun ───────────────────────────────────────────
def get_sig_col(df):
    return "signal_carte" if "signal_carte" in df.columns else ("signal" if "signal" in df.columns else None)

def filtrer(df):
    d = df.copy()
    sc = get_sig_col(d)
    if sc:
        d = d[d[sc].isin(signaux_choix)]
    if "score_final" in d.columns:
        d = d[d["score_final"] >= score_min]
    if "chaleur" in d.columns and chaleurs_choix:
        d = d[d["chaleur"].astype(str).isin(chaleurs_choix)]
    return d

df = filtrer(df_raw)
sig_col = get_sig_col(df)
total = len(df)

def score_color(s):
    if s >= 80: return "#c4440a"
    if s >= 60: return "#d4850a"
    if s >= 40: return "#1a4a8a"
    return "#b8b5ae"

def chaleur_tag(ch):
    ch = str(ch)
    if "très" in ch: return "tag-chaud", "🔴 " + ch
    if "chaud" in ch: return "tag-tiede", "🟠 " + ch
    if "tiède" in ch: return "tag-tiede", "🟡 " + ch
    return "tag-froid", "⚪ " + ch

# ════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
with tab_dash:
    # KPIs
    st.markdown("<div class='sh'>KPIs — " + DEPT_LABELS.get(st.session_state.run_dept, "") + f" / {st.session_state.run_annee}</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.metric("Prospects filtrés", f"{total:,}")
    with c2:
        avg = df["score_final"].mean() if "score_final" in df.columns and total else 0
        st.metric("Score moyen", f"{avg:.0f}/100")
    with c3:
        hot = (df["score_final"] >= 80).sum() if "score_final" in df.columns else 0
        st.metric("Top 20% (≥80)", f"{hot:,}")
    with c4:
        multi = (df["nb_signaux"] >= 2).sum() if "nb_signaux" in df.columns else 0
        st.metric("Multi-signal", f"{multi:,}")
    with c5:
        zones = df["nom_commune"].nunique() if "nom_commune" in df.columns else (df["commune"].nunique() if "commune" in df.columns else 0)
        st.metric("Communes", f"{zones:,}")
    with c6:
        avg_pm2 = df["prix_m2"].median() if "prix_m2" in df.columns and df["prix_m2"].notna().any() else 0
        st.metric("Prix/m² médian", f"{avg_pm2:,.0f} €")

    st.markdown("---")

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        # Distribution des scores
        st.markdown("<div class='sh'>Distribution des scores finaux</div>", unsafe_allow_html=True)
        if "score_final" in df.columns and total > 0:
            bins = [0, 20, 40, 60, 80, 100]
            labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
            colors = ["#e0dbd2", "#c8c2b8", "#f5a05a", "#c4440a80", "#c4440a"]
            counts = pd.cut(df["score_final"], bins=bins, labels=labels, right=True).value_counts().reindex(labels).fillna(0)
            total_dist = counts.sum()
            pcts = (counts / total_dist * 100).round(1)

            bar_html = '<div class="distrib-bar">' + "".join(
                f'<div class="db-seg" style="flex:{p};background:{c}" title="{l} : {int(n)} ({p}%)">{p:.0f}%</div>'
                for l, n, p, c in zip(labels, counts, pcts, colors) if p > 0
            ) + '</div>'
            st.markdown(bar_html, unsafe_allow_html=True)

            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
            for col_m, (lbl, cnt, clr) in zip([dc1,dc2,dc3,dc4,dc5], zip(labels, counts, colors)):
                with col_m:
                    st.markdown(f"""
<div style='text-align:center;padding:.5rem;background:var(--paper-2);border-radius:8px;border:1px solid var(--border)'>
  <div style='font-family:"Fraunces",serif;font-size:1.2rem;font-weight:700;color:{clr}'>{int(cnt):,}</div>
  <div style='font-size:.6rem;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.06em'>{lbl}</div>
</div>""", unsafe_allow_html=True)

    with col_b:
        # Répartition par signal
        st.markdown("<div class='sh'>Répartition par signal</div>", unsafe_allow_html=True)
        if sig_col and total > 0:
            sig_counts = df[sig_col].value_counts()
            SIG_COLORS_HEX = {"heritage":"#8a4a1a","divorce":"#c4440a","upgrade":"#1a6b4a","retraite":"#1a4a8a","primo":"#7a4aa0"}
            for sig, cnt in sig_counts.items():
                pct = cnt / total * 100
                color = SIG_COLORS_HEX.get(sig, "#888")
                label = SIGNAL_LABELS.get(sig, sig)
                st.markdown(f"""
<div style='display:flex;align-items:center;gap:.75rem;margin-bottom:.55rem'>
  <div style='width:9px;height:9px;border-radius:50%;background:{color};flex-shrink:0'></div>
  <div style='font-size:.75rem;color:var(--ink-light);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{label}</div>
  <div style='font-family:"Fraunces",serif;font-size:.9rem;font-weight:600;color:var(--ink)'>{cnt:,}</div>
  <div style='font-size:.65rem;color:var(--ink-faint);min-width:32px;text-align:right'>{pct:.0f}%</div>
</div>
<div style='height:3px;background:var(--paper-3);border-radius:2px;margin-bottom:.5rem'>
  <div style='height:100%;width:{pct:.1f}%;background:{color};border-radius:2px'></div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Top 5 prospects
    st.markdown("<div class='sh'>Top 5 prospects — score final le plus élevé</div>", unsafe_allow_html=True)
    if total > 0 and "score_final" in df.columns:
        top5 = df.head(5)
        for _, row in top5.iterrows():
            sf   = float(row.get("score_final", 0))
            sc   = score_color(sf)
            addr = str(row.get("adresse_complete", "—"))
            ch   = str(row.get("chaleur", ""))
            sig  = str(row.get("signal_label", row.get("signal_carte", "")))
            prix = float(row.get("valeur_fonciere", 0) or 0)
            surf = float(row.get("surface_reelle_bati", 0) or 0)
            pm2  = float(row.get("prix_m2", 0) or 0)
            dec  = float(row.get("decote_vs_median", 0) or 0)
            nb   = int(row.get("nb_signaux", 1))
            date = str(row.get("date_mutation", ""))[:10]
            comm = str(row.get("nom_commune", row.get("commune", "")))
            cp   = str(row.get("code_postal", ""))
            tag_cls, tag_lbl = chaleur_tag(ch)
            multi_tag = f'<span class="pr-tag tag-multi">⚡ {nb} signaux</span>' if nb >= 2 else ""
            st.markdown(f"""
<div class="prospect-row">
  <div class="pr-score">
    <div class="val" style="color:{sc}">{sf:.0f}</div>
    <div class="lbl">percentile</div>
  </div>
  <div class="pr-body">
    <div class="pr-addr">{addr}</div>
    <div class="pr-meta">
      <span>📍 {comm} {cp}</span>
      <span>📅 {date}</span>
      <span>🏠 {surf:.0f} m²</span>
    </div>
    <div class="pr-tags">
      <span class="pr-tag tag-signal">{sig}</span>
      <span class="pr-tag {tag_cls}">{tag_lbl}</span>
      {multi_tag}
    </div>
  </div>
  <div class="pr-right">
    <div class="pr-prix">{prix:,.0f} €</div>
    <div class="pr-surf">{pm2:,.0f} €/m²</div>
    {'<div class="pr-decote">−' + f'{abs(dec):.0f}% médiane</div>' if dec != 0 else ''}
  </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — PROSPECTS
# ════════════════════════════════════════════════════════════
with tab_prospects:
    st.markdown(f"<div class='sh'>{total:,} prospects · Score ≥ {score_min} · {', '.join(SIGNAL_LABELS.get(s,s) for s in signaux_choix)}</div>", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([1,1,2])
    with col_f1:
        vue = st.radio("Vue", ["Tableau", "Cartes"], horizontal=True, label_visibility="collapsed")
    with col_f2:
        tri = st.selectbox("Trier par", ["Score final ↓", "Prix ↓", "Surface ↓", "Date ↓", "Décote ↓"], label_visibility="collapsed")
    with col_f3:
        search = st.text_input("🔍 Filtrer par commune / adresse", placeholder="ex: Paris, Bordeaux…", label_visibility="collapsed")

    # Tri
    tri_map = {
        "Score final ↓": ("score_final", False),
        "Prix ↓": ("valeur_fonciere", False),
        "Surface ↓": ("surface_reelle_bati", False),
        "Date ↓": ("date_mutation", False),
        "Décote ↓": ("decote_vs_median", False),
    }
    sort_col, sort_asc = tri_map.get(tri, ("score_final", False))
    df_view = df.copy()
    if search.strip():
        mask = (
            df_view.get("nom_commune", pd.Series(dtype=str)).astype(str).str.upper().str.contains(search.upper(), na=False) |
            df_view.get("adresse_complete", pd.Series(dtype=str)).astype(str).str.upper().str.contains(search.upper(), na=False)
        )
        df_view = df_view[mask]
    if sort_col in df_view.columns:
        df_view = df_view.sort_values(sort_col, ascending=sort_asc)

    st.markdown(f"<div style='font-size:.7rem;color:var(--ink-faint);margin-bottom:.85rem'>{len(df_view):,} résultats</div>", unsafe_allow_html=True)

    if vue == "Cartes":
        for _, row in df_view.head(100).iterrows():
            sf   = float(row.get("score_final", 0))
            sc   = score_color(sf)
            addr = str(row.get("adresse_complete", "—"))
            ch   = str(row.get("chaleur", ""))
            sig  = str(row.get("signal_label", row.get("signal_carte", "")))
            prix = float(row.get("valeur_fonciere", 0) or 0)
            surf = float(row.get("surface_reelle_bati", 0) or 0)
            pm2  = float(row.get("prix_m2", 0) or 0)
            dec  = float(row.get("decote_vs_median", 0) or 0)
            nb   = int(row.get("nb_signaux", 1))
            date = str(row.get("date_mutation", ""))[:10]
            comm = str(row.get("nom_commune", row.get("commune", "")))
            cp   = str(row.get("code_postal", ""))
            seg  = str(row.get("segment_cible", ""))
            tag_cls, tag_lbl = chaleur_tag(ch)
            multi_tag = f'<span class="pr-tag tag-multi">⚡ {nb} signaux</span>' if nb >= 2 else ""
            st.markdown(f"""
<div class="prospect-row">
  <div class="pr-score">
    <div class="val" style="color:{sc}">{sf:.0f}</div>
    <div class="lbl">score</div>
  </div>
  <div class="pr-body">
    <div class="pr-addr">{addr}</div>
    <div class="pr-meta">
      <span>📍 {comm} {cp}</span><span>📅 {date}</span>
      <span>🏠 {surf:.0f} m²</span>
    </div>
    <div class="pr-tags">
      <span class="pr-tag tag-signal">{sig}</span>
      <span class="pr-tag {tag_cls}">{tag_lbl}</span>
      {multi_tag}
    </div>
    {f'<div style="font-size:.68rem;color:var(--ink-faint);margin-top:.35rem;font-family:Lora,serif">💬 {seg}</div>' if seg and seg != "—" else ""}
  </div>
  <div class="pr-right">
    <div class="pr-prix">{prix:,.0f} €</div>
    <div class="pr-surf">{pm2:,.0f} €/m²</div>
    {'<div class="pr-decote">−' + f'{abs(dec):.0f}%</div>' if dec != 0 else ''}
  </div>
</div>""", unsafe_allow_html=True)
        if len(df_view) > 100:
            st.caption(f"Affichage limité à 100 cartes. Téléchargez le CSV pour voir les {len(df_view):,} prospects.")
    else:
        # Vue tableau
        cols_show = [c for c in [
            "rang", "adresse_complete", "code_postal", "nom_commune",
            "signal_label", "score_final", "score_brut", "chaleur",
            "valeur_fonciere", "prix_m2", "surface_reelle_bati",
            "nombre_pieces_principales", "decote_vs_median", "anciennete_mois",
            "nb_signaux", "date_mutation", "nature_mutation", "type_local",
        ] if c in df_view.columns]
        rename = {
            "rang":"#","adresse_complete":"Adresse","code_postal":"CP","nom_commune":"Commune",
            "signal_label":"Signal","score_final":"Score","score_brut":"Brut","chaleur":"Chaleur",
            "valeur_fonciere":"Prix (€)","prix_m2":"€/m²","surface_reelle_bati":"Surface (m²)",
            "nombre_pieces_principales":"Pièces","decote_vs_median":"Décote %","anciennete_mois":"Ancienneté (mois)",
            "nb_signaux":"# Signaux","date_mutation":"Date","nature_mutation":"Nature","type_local":"Type",
        }
        st.dataframe(
            df_view[cols_show].rename(columns=rename).head(500),
            use_container_width=True, hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "Prix (€)": st.column_config.NumberColumn("Prix (€)", format="%.0f €"),
                "€/m²": st.column_config.NumberColumn("€/m²", format="%.0f €"),
                "Décote %": st.column_config.NumberColumn("Décote %", format="%.1f%%"),
            }
        )
        if len(df_view) > 500:
            st.caption(f"Affichage limité à 500 lignes sur {len(df_view):,}.")

    st.markdown("---")
    csv = df_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇ Exporter CSV complet", data=csv,
        file_name=f"prospects_{st.session_state.run_dept}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv")


# ════════════════════════════════════════════════════════════
# TAB 3 — ZONES
# ════════════════════════════════════════════════════════════
with tab_zones:
    zone_col = next((c for c in ["nom_commune","commune","code_postal"] if c in df.columns), None)
    if not zone_col or not sig_col or total == 0:
        st.info("Données insuffisantes pour le ranking par zone.")
    else:
        st.markdown("<div class='sh'>Ranking des zones par concentration de prospects</div>", unsafe_allow_html=True)

        agg_dict = {
            "Prospects":     (sig_col, "count"),
            "Score moy.":    ("score_final", "mean"),
            "Top 20%":       ("score_final", lambda x: (x >= 80).sum()),
            "Multi-signal":  ("nb_signaux",  lambda x: (x >= 2).sum()) if "nb_signaux" in df.columns else ("score_final", "count"),
        }
        if "prix_m2" in df.columns:
            agg_dict["€/m² médian"] = ("prix_m2", "median")
        if "valeur_fonciere" in df.columns:
            agg_dict["Prix moy. (€)"] = ("valeur_fonciere", "mean")
        if "decote_vs_median" in df.columns:
            agg_dict["Décote moy. %"] = ("decote_vs_median", "mean")

        ranking = (
            df.groupby(zone_col).agg(**agg_dict)
            .sort_values("Prospects", ascending=False)
            .reset_index()
        )
        ranking["Score moy."]   = ranking["Score moy."].round(1)
        if "€/m² médian"   in ranking.columns: ranking["€/m² médian"]   = ranking["€/m² médian"].round(0)
        if "Prix moy. (€)" in ranking.columns: ranking["Prix moy. (€)"] = ranking["Prix moy. (€)"].round(0)
        if "Décote moy. %" in ranking.columns: ranking["Décote moy. %"] = ranking["Décote moy. %"].round(1)

        max_prospects = ranking["Prospects"].max()

        # Cartes zones
        ca, cb = st.columns([1.4, 1])
        with ca:
            for _, zrow in ranking.head(15).iterrows():
                nm  = str(zrow[zone_col])
                cnt = int(zrow["Prospects"])
                smoy = float(zrow["Score moy."])
                top = int(zrow.get("Top 20%", 0))
                multi = int(zrow.get("Multi-signal", 0))
                pm2  = float(zrow.get("€/m² médian", 0) or 0)
                dec  = float(zrow.get("Décote moy. %", 0) or 0)
                pct  = cnt / max_prospects * 100
                sc   = score_color(smoy)
                st.markdown(f"""
<div class="zone-card">
  <div class="zc-head">
    <div>
      <div class="zc-name">{nm}</div>
    </div>
    <div style="font-family:'Fraunces',serif;font-size:1.3rem;font-weight:700;color:{sc}">{smoy:.0f}</div>
  </div>
  <div class="zone-bar"><div class="zone-bar-fill" style="width:{pct:.1f}%"></div></div>
  <div class="zone-stats">
    <span><b>{cnt}</b> prospects</span>
    <span><b>{top}</b> top 20%</span>
    <span><b>{multi}</b> multi-signal</span>
    {f'<span><b>{pm2:,.0f}</b> €/m²</span>' if pm2 else ''}
    {f'<span><b>−{dec:.0f}%</b> décote</span>' if dec > 0 else ''}
  </div>
</div>""", unsafe_allow_html=True)

        with cb:
            st.markdown("<div class='sh'>Tableau complet</div>", unsafe_allow_html=True)
            st.dataframe(ranking, use_container_width=True, hide_index=True,
                column_config={"Score moy.": st.column_config.ProgressColumn("Score moy.", min_value=0, max_value=100, format="%.1f")})


# ════════════════════════════════════════════════════════════
# TAB 4 — SEGMENTS MARKETING
# ════════════════════════════════════════════════════════════
with tab_segments:
    st.markdown("<div class='sh'>Segments de prospection — répartition & messages</div>", unsafe_allow_html=True)
    SIG_DOTS = {"heritage":"#8a4a1a","divorce":"#c4440a","upgrade":"#1a6b4a","retraite":"#1a4a8a","primo":"#7a4aa0"}

    if sig_col and total > 0:
        for sig_key in signaux_choix:
            subset = df[df[sig_col] == sig_key]
            if subset.empty:
                continue
            cnt  = len(subset)
            smoy = subset["score_final"].mean() if "score_final" in subset.columns else 0
            top  = (subset["score_final"] >= 80).sum() if "score_final" in subset.columns else 0
            pm2  = subset["prix_m2"].median() if "prix_m2" in subset.columns and subset["prix_m2"].notna().any() else 0
            dot  = SIG_DOTS.get(sig_key, "#888")
            lbl  = SIGNAL_LABELS.get(sig_key, sig_key)
            msg  = SIGNAL_SEGMENTS.get(sig_key, "")
            dec  = subset["decote_vs_median"].mean() if "decote_vs_median" in subset.columns and subset["decote_vs_median"].notna().any() else 0
            multi = (subset["nb_signaux"] >= 2).sum() if "nb_signaux" in subset.columns else 0

            st.markdown(f"""
<div class="seg-card">
  <div class="sc-head">
    <div class="seg-dot" style="background:{dot}"></div>
    <div class="sc-name">{lbl}</div>
    <div class="sc-count">{cnt:,} prospects · {cnt/total*100:.0f}%</div>
  </div>
  <div class="sc-msg">💬 {msg}</div>
  <div class="sc-stats">
    <span>Score moy. <b>{smoy:.0f}</b></span>
    <span>Top 20% <b>{top}</b></span>
    <span>Multi-signal <b>{multi}</b></span>
    {f'<span>€/m² médian <b>{pm2:,.0f}</b></span>' if pm2 else ''}
    {f'<span>Décote moy. <b>−{dec:.0f}%</b></span>' if dec > 0 else ''}
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='sh'>Matrice signal × chaleur</div>", unsafe_allow_html=True)
        if "chaleur" in df.columns:
            chaleurs_order = ["très chaud", "chaud", "tiède", "froid"]
            matrix = pd.crosstab(df[sig_col].map(SIGNAL_LABELS), df["chaleur"].astype(str))
            for ch in chaleurs_order:
                if ch not in matrix.columns:
                    matrix[ch] = 0
            matrix = matrix[[c for c in chaleurs_order if c in matrix.columns]]
            st.dataframe(matrix, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — CARTE
# ════════════════════════════════════════════════════════════
with tab_carte:
    has_coords = "latitude" in df.columns and "longitude" in df.columns
    df_map = df[df["latitude"].notna() & df["longitude"].notna() & (df["latitude"] != 0)] if has_coords else pd.DataFrame()

    if not has_coords or df_map.empty:
        st.info("⚠ Coordonnées GPS absentes dans ce fichier DVF.")
    else:
        st.markdown(f"<div class='sh'>{len(df_map):,} prospects géolocalisés — taille du cercle = score · couleur = signal</div>", unsafe_allow_html=True)

        SCOL_JS  = json.dumps(SIGNAL_COLORS)
        SLBL_JS  = json.dumps(SIGNAL_LABELS)
        pts = []
        for _, row in df_map.iterrows():
            lat  = float(row.get("latitude",  0) or 0)
            lng  = float(row.get("longitude", 0) or 0)
            if lat == 0 and lng == 0: continue
            pts.append({
                "lat":  lat, "lng": lng,
                "sig":  str(row.get(sig_col, "") or ""),
                "sf":   float(row.get("score_final", 50) or 50),
                "sb":   float(row.get("score_brut",  50) or 50),
                "addr": str(row.get("adresse_complete", "") or ""),
                "cp":   str(row.get("code_postal", "") or ""),
                "comm": str(row.get("nom_commune", row.get("commune","")) or ""),
                "prix": float(row.get("valeur_fonciere",0) or 0),
                "surf": float(row.get("surface_reelle_bati",0) or 0),
                "pm2":  float(row.get("prix_m2",0) or 0),
                "dec":  float(row.get("decote_vs_median",0) or 0),
                "date": str(row.get("date_mutation",""))[:10],
                "ch":   str(row.get("chaleur","") or ""),
                "nb":   int(row.get("nb_signaux",1) or 1),
            })

        clat = df_map["latitude"].median()
        clng = df_map["longitude"].median()
        data_js = json.dumps(pts)

        map_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:monospace;background:#f5f2ed}}
#map{{width:100%;height:580px}}
.lp{{font-family:monospace;font-size:.76rem;line-height:1.6;min-width:240px}}
.lp .badge{{display:inline-block;padding:.1rem .4rem;border-radius:3px;color:#fff;font-size:.62rem;font-weight:600;margin-bottom:.4rem}}
.lp .score{{font-size:1.5rem;font-weight:700;line-height:1;margin-bottom:.15rem}}
.lp .sub{{font-size:.68rem;color:#6b6860;margin-bottom:.4rem}}
.lp .addr{{font-size:.75rem;font-weight:600;margin-bottom:.35rem;color:#1a1814}}
.lp table{{width:100%;border-collapse:collapse;font-size:.68rem}}
.lp td{{padding:.12rem 0;border-bottom:1px solid rgba(0,0,0,.06)}}
.lp td:first-child{{color:#6b6860}}
.lp td:last-child{{font-weight:500;text-align:right}}
.lp .multi{{display:inline-block;background:rgba(245,160,90,.15);border:1px solid rgba(245,160,90,.3);color:#b07030;font-size:.6rem;padding:.1rem .35rem;border-radius:3px;margin-top:.35rem}}
</style>
</head><body><div id="map"></div>
<script>
const DATA={data_js};
const COLORS={SCOL_JS};
const LABELS={SLBL_JS};
const map=L.map('map').setView([{clat},{clng}],11);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'© OpenStreetMap',maxZoom:19}}).addTo(map);
const layer=L.layerGroup().addTo(map);
DATA.forEach(pt=>{{
  const color=COLORS[pt.sig]||'#888';
  const r=4+(pt.sf/100)*10;
  const sc=pt.sf>=80?'#c4440a':pt.sf>=60?'#d4850a':pt.sf>=40?'#1a4a8a':'#b8b5ae';
  const prix=pt.prix?pt.prix.toLocaleString('fr-FR')+' €':'—';
  const surf=pt.surf?pt.surf+' m²':'—';
  const pm2=pt.pm2?pt.pm2.toLocaleString('fr-FR')+' €/m²':'—';
  const dec=pt.dec?'−'+Math.abs(pt.dec).toFixed(0)+'% médiane':'—';
  const multi=pt.nb>=2?`<span class="multi">⚡ ${{pt.nb}} signaux</span>`:'';
  const pop=`<div class="lp">
    <div class="badge" style="background:${{color}}">${{LABELS[pt.sig]||pt.sig}}</div>
    <div class="score" style="color:${{sc}}">${{pt.sf.toFixed(0)}}<span style="font-size:.7rem;color:#6b6860">/100</span></div>
    <div class="sub">Score brut ${{pt.sb.toFixed(0)}} · ${{pt.ch}}</div>
    <div class="addr">${{pt.addr||'—'}}</div>
    <table>
      <tr><td>Commune</td><td>${{pt.comm}} ${{pt.cp}}</td></tr>
      <tr><td>Prix</td><td>${{prix}}</td></tr>
      <tr><td>Surface</td><td>${{surf}}</td></tr>
      <tr><td>Prix/m²</td><td>${{pm2}}</td></tr>
      <tr><td>Décote</td><td>${{dec}}</td></tr>
      <tr><td>Date</td><td>${{pt.date}}</td></tr>
    </table>
    ${{multi}}
  </div>`;
  L.circleMarker([pt.lat,pt.lng],{{radius:r,color,fillColor:color,fillOpacity:.65,weight:pt.nb>=2?2:1.2,opacity:.9}})
    .bindPopup(pop)
    .bindTooltip(`${{LABELS[pt.sig]||pt.sig}} · ${{pt.sf.toFixed(0)}}`,{{direction:'top',offset:[0,-4]}})
    .addTo(layer);
}});
if(DATA.length>0){{
  const lats=DATA.map(d=>d.lat),lngs=DATA.map(d=>d.lng);
  map.fitBounds([[Math.min(...lats),Math.min(...lngs)],[Math.max(...lats),Math.max(...lngs)]],{{padding:[30,30]}});
}}
</script></body></html>"""
        st.components.v1.html(map_html, height=590, scrolling=False)


# ════════════════════════════════════════════════════════════
# TAB 6 — DOCUMENTATION
# ════════════════════════════════════════════════════════════
with tab_doc:
    st.markdown("""
<div style='background:var(--ink);color:var(--paper);border-radius:var(--radius);padding:2.5rem 2rem;margin-bottom:1.5rem;position:relative;overflow:hidden'>
  <div style='position:absolute;inset:0;pointer-events:none;background-image:repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(245,240,230,.04) 39px,rgba(245,240,230,.04) 40px),repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(245,240,230,.04) 39px,rgba(245,240,230,.04) 40px)'></div>
  <div style='font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;background:rgba(245,160,90,.15);color:#f5a05a;border:1px solid rgba(245,160,90,.25);padding:.25rem .7rem;border-radius:20px;margin-bottom:1rem;display:inline-block;font-family:"DM Mono",monospace;position:relative'>Documentation technique v3.0</div>
  <h1 style='font-family:"Fraunces",serif!important;font-size:2rem!important;font-weight:700!important;color:var(--paper)!important;margin-bottom:.5rem;position:relative'>Pipeline DVF <em style="color:#f5a05a">×</em> BODACC</h1>
  <div style='font-family:"DM Mono",monospace;font-size:.74rem;color:rgba(245,242,237,.4);position:relative;line-height:1.7'>Score percentile · Malus qualité · Multi-signal · 5 signaux de vie · Export BI</div>
  <div style='display:flex;gap:2rem;margin-top:1.5rem;position:relative;flex-wrap:wrap'>
    <div><div style='font-family:"Fraunces",serif;font-size:1.7rem;font-weight:700;color:#f5a05a'>5</div><div style='font-size:.6rem;color:rgba(245,242,237,.35);text-transform:uppercase;letter-spacing:.08em'>Signaux</div></div>
    <div><div style='font-family:"Fraunces",serif;font-size:1.7rem;font-weight:700;color:#f5a05a'>100%</div><div style='font-size:.6rem;color:rgba(245,242,237,.35);text-transform:uppercase;letter-spacing:.08em'>Données publiques</div></div>
    <div><div style='font-family:"Fraunces",serif;font-size:1.7rem;font-weight:700;color:#f5a05a'>Percentile</div><div style='font-size:.6rem;color:rgba(245,242,237,.35);text-transform:uppercase;letter-spacing:.08em'>Score relatif dept</div></div>
    <div><div style='font-family:"Fraunces",serif;font-size:1.7rem;font-weight:700;color:#f5a05a'>v3</div><div style='font-size:.6rem;color:rgba(245,242,237,.35);text-transform:uppercase;letter-spacing:.08em'>Pipeline</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style='background:var(--paper);border:1px solid var(--border);border-radius:var(--radius);padding:1.75rem 2rem;margin-bottom:1.25rem'>
  <div style='font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-faint);margin-bottom:.35rem'>01 — Pourquoi un score percentile ?</div>
  <h2 style='font-family:"Fraunces",serif!important;font-size:1.3rem!important;font-weight:600!important;color:var(--ink)!important;margin-bottom:.75rem'>Le problème du score absolu</h2>
  <p style='font-family:"Lora",serif;font-size:.9rem;line-height:1.8;color:var(--ink-light)'>En v1 et v2, le score était absolu : une vente dans les 90 jours après une succession = 80 points, partout. Résultat : à Paris (département 75), des milliers de biens atteignaient 80+, rendant le ranking inexploitable.</p>
  <p style='font-family:"Lora",serif;font-size:.9rem;line-height:1.8;color:var(--ink-light)'>En v3, le score final est un <strong style="color:var(--ink)">percentile dans le département</strong>. Un score de 80 signifie "ce bien est dans le top 20% du département analysé, parmi tous les biens ayant déclenché un signal". C'est comparable entre villes et entre sessions.</p>
  <div style='background:var(--paper-2);border-radius:8px;padding:.85rem 1rem;margin-top:.75rem;font-size:.78rem;color:var(--ink-light);border:1px solid var(--border)'>
    <b style='color:var(--ink)'>Exemple :</b> Paris 75 génère 8 000 signaux divorce. Seuls les 20% les plus intenses (délai le plus court, adjudication, prix sous marché) atteignent score ≥ 80. À Périgueux, sur 40 signaux, les mêmes 20% atteignent aussi ≥ 80 — le score est comparable.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style='background:var(--paper);border:1px solid var(--border);border-radius:var(--radius);padding:1.75rem 2rem;margin-bottom:1.25rem'>
  <div style='font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-faint);margin-bottom:.35rem'>02 — Architecture du score v3</div>
  <h2 style='font-family:"Fraunces",serif!important;font-size:1.3rem!important;font-weight:600!important;color:var(--ink)!important;margin-bottom:.75rem'>4 étapes</h2>
  <div style='display:flex;align-items:center;gap:0;flex-wrap:wrap;row-gap:.5rem;margin:1rem 0'>
    <div style='flex:1;min-width:120px;background:var(--paper-2);border:1px solid var(--border);border-radius:7px;padding:.7rem .85rem;text-align:center'><div style='font-size:1rem;margin-bottom:.25rem'>⚡</div><span style='font-family:"DM Mono",monospace;font-size:.66rem;font-weight:500;color:var(--ink);display:block'>Score brut</span><div style='font-size:.6rem;color:var(--ink-faint)'>0–100 signal</div></div>
    <div style='font-size:.85rem;color:var(--ink-faint);padding:0 .35rem;flex-shrink:0'>→</div>
    <div style='flex:1;min-width:120px;background:var(--paper-2);border:1px solid var(--border);border-radius:7px;padding:.7rem .85rem;text-align:center'><div style='font-size:1rem;margin-bottom:.25rem'>⚠️</div><span style='font-family:"DM Mono",monospace;font-size:.66rem;font-weight:500;color:var(--ink);display:block'>Malus qualité</span><div style='font-size:.6rem;color:var(--ink-faint)'>−35 max</div></div>
    <div style='font-size:.85rem;color:var(--ink-faint);padding:0 .35rem;flex-shrink:0'>→</div>
    <div style='flex:1;min-width:120px;background:var(--paper-2);border:1px solid var(--border);border-radius:7px;padding:.7rem .85rem;text-align:center'><div style='font-size:1rem;margin-bottom:.25rem'>📊</div><span style='font-family:"DM Mono",monospace;font-size:.66rem;font-weight:500;color:var(--ink);display:block'>Percentile</span><div style='font-size:.6rem;color:var(--ink-faint)'>rank dept</div></div>
    <div style='font-size:.85rem;color:var(--ink-faint);padding:0 .35rem;flex-shrink:0'>→</div>
    <div style='flex:1;min-width:120px;background:var(--paper-2);border:1px solid var(--border);border-radius:7px;padding:.7rem .85rem;text-align:center'><div style='font-size:1rem;margin-bottom:.25rem'>⚡</div><span style='font-family:"DM Mono",monospace;font-size:.66rem;font-weight:500;color:var(--ink);display:block'>Multi-signal</span><div style='font-size:.6rem;color:var(--ink-faint)'>+5 / +12</div></div>
  </div>
  <table class="sc-table" style='margin-top:1rem'>
    <thead><tr><th>Étape</th><th>Description</th><th>Valeur</th></tr></thead>
    <tbody>
      <tr><td>Score brut</td><td>Intensité de l'événement de vie (délai, décote, adjudication…)</td><td>0–100</td></tr>
      <tr><td>Malus qualité</td><td>Pénalité pour données aberrantes appliquée avant normalisation</td><td>−20 à 0</td></tr>
      <tr><td>Normalisation percentile</td><td>Rank percentile dans le pool de signaux du département</td><td>0–100</td></tr>
      <tr><td>Bonus multi-signal</td><td>+5 si 2 signaux convergents · +12 si 3+ signaux</td><td>+5 / +12</td></tr>
    </tbody>
  </table>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style='background:var(--paper);border:1px solid var(--border);border-radius:var(--radius);padding:1.75rem 2rem;margin-bottom:1.25rem'>
  <div style='font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-faint);margin-bottom:.35rem'>03 — Malus qualité</div>
  <h2 style='font-family:"Fraunces",serif!important;font-size:1.3rem!important;font-weight:600!important;color:var(--ink)!important;margin-bottom:.75rem'>Filtres anti-bruit</h2>
  <p style='font-family:"Lora",serif;font-size:.9rem;line-height:1.8;color:var(--ink-light)'>Les malus s'appliquent sur le score brut avant la normalisation percentile. Un bien avec un malus fort aura un score brut dégradé, ce qui le placera plus bas dans le percentile.</p>
  <table class="sc-table">
    <thead><tr><th>Condition</th><th>Raison</th><th>Malus</th></tr></thead>
    <tbody>
      <tr><td>Type non résidentiel confirmé</td><td>Local commercial, dépendance, terrain — hors cible</td><td>−20</td></tr>
      <tr><td>Prix/m² &lt; 500 ou &gt; 35 000 €</td><td>Donnée corrompue ou bien atypique (château, ruine)</td><td>−15</td></tr>
      <tr><td>Surface &lt; 9 m² ou &gt; 600 m²</td><td>Cave, parking, ou très grand domaine hors segment</td><td>−10</td></tr>
      <tr><td>Prix total &lt; 15 000 €</td><td>Probable parking ou quote-part indivise</td><td>−10</td></tr>
    </tbody>
  </table>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style='background:rgba(196,68,10,.05);border:1px solid rgba(196,68,10,.2);border-radius:var(--radius);padding:1rem 1.25rem;margin-bottom:1.25rem;font-family:"Lora",serif;font-size:.88rem;line-height:1.7;color:var(--ink-light)'>
<strong style='color:var(--ink)'>Usage RGPD recommandé :</strong> utiliser les codes postaux et les segments pour créer des <strong>audiences publicitaires géolocalisées</strong> (Meta, Google). La prospection directe nominative nécessite un enrichissement via prestataire habilité. Les données DVF ne contiennent pas l'identité des propriétaires.
</div>""", unsafe_allow_html=True)

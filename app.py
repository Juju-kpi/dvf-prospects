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
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,500;0,700;1,300&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

:root {
  --ink: #1a1814;
  --ink-light: #6b6860;
  --ink-faint: #b8b5ae;
  --paper: #f5f2ed;
  --paper-2: #ede9e2;
  --paper-3: #e0dbd2;
  --accent: #f5a05a;
  --accent-red: #c4440a;
  --accent-green: #1a6b4a;
  --accent-blue: #1a4a8a;
  --accent-purple: #7a4aa0;
  --border: rgba(26,24,20,.12);
  --radius: 10px;
  --sidebar-bg: #1a1814;
}

html, body, [class*="css"] {
  font-family: 'DM Mono', monospace !important;
  color: var(--ink);
}
h1, h2, h3 {
  font-family: 'Fraunces', serif !important;
  letter-spacing: -.02em;
}

/* ─── App background ─── */
.stApp { background: var(--paper) !important; }
.stApp > header { background: var(--ink) !important; }

/* ─── Main content ─── */
.main .block-container {
  padding-top: 0 !important;
  padding-bottom: 3rem !important;
  max-width: 1400px !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] { background: var(--sidebar-bg) !important; border-right: none !important; }
[data-testid="stSidebar"] > div:first-child { background: var(--sidebar-bg) !important; padding: 1.25rem 1rem !important; }
[data-testid="stSidebar"] * { color: #f5f2ed !important; }

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
  color: rgba(245,242,237,.5) !important;
  font-size: .65rem !important;
  text-transform: uppercase !important;
  letter-spacing: .1em !important;
  font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.15) !important;
  color: #f5f2ed !important;
  border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSelectbox svg { fill: rgba(245,242,237,.5) !important; }
[data-testid="stSidebar"] .stMultiSelect > div > div {
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.15) !important;
  border-radius: 6px !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(245,242,237,.1) !important; margin: .75rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
  background: var(--accent) !important;
  color: var(--ink) !important;
  border: none !important;
  font-family: 'DM Mono', monospace !important;
  font-weight: 500 !important;
  font-size: .75rem !important;
  letter-spacing: .04em !important;
  width: 100% !important;
  border-radius: 6px !important;
  padding: .55rem 1rem !important;
  transition: opacity .15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover { opacity: .85 !important; background: var(--accent) !important; }

/* ─── Tabs ─── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  color: var(--ink-light) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: .72rem !important;
  letter-spacing: .05em !important;
  text-transform: uppercase !important;
  padding: .65rem 1.25rem !important;
  border-bottom: 2px solid transparent !important;
  transition: all .15s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: var(--ink) !important; }
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 2px solid var(--accent-red) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ─── Metrics ─── */
[data-testid="metric-container"] {
  background: var(--paper-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: .85rem 1rem !important;
}
[data-testid="stMetricLabel"] {
  font-size: .63rem !important;
  text-transform: uppercase !important;
  letter-spacing: .08em !important;
  color: var(--ink-light) !important;
  font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Fraunces', serif !important;
  font-size: 1.6rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  line-height: 1.1 !important;
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  overflow: hidden !important;
}
.stDataFrame td, .stDataFrame th {
  font-size: .74rem !important;
  font-family: 'DM Mono', monospace !important;
}

/* ─── Spinner ─── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ─── Alerts ─── */
.stSuccess { background: rgba(26,107,74,.08) !important; border: 1px solid rgba(26,107,74,.2) !important; border-radius: var(--radius) !important; color: var(--ink) !important; }
.stError { background: rgba(196,68,10,.08) !important; border: 1px solid rgba(196,68,10,.2) !important; border-radius: var(--radius) !important; }
.stInfo { background: var(--paper-2) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; color: var(--ink) !important; }

/* ─── Download button ─── */
[data-testid="stDownloadButton"] > button {
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: none !important;
  font-family: 'DM Mono', monospace !important;
  font-size: .72rem !important;
  border-radius: 6px !important;
  padding: .45rem 1rem !important;
  letter-spacing: .03em !important;
}
[data-testid="stDownloadButton"] > button:hover { opacity: .85 !important; background: var(--ink) !important; }

/* ─── Custom components ─── */
.page-header {
  background: var(--ink);
  color: var(--paper);
  margin: 0 -3rem 1.75rem -3rem;
  padding: 1.4rem 3rem;
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.page-header .logo { font-family: 'Fraunces', serif; font-size: 1.4rem; font-weight: 700; letter-spacing: -.03em; color: var(--paper); }
.page-header .logo span { color: #f5a05a; }
.page-header .tagline { font-size: .63rem; color: rgba(245,242,237,.38); letter-spacing: .1em; text-transform: uppercase; margin-top: .2rem; }
.page-header .status-pill { font-size: .63rem; background: rgba(245,160,90,.15); color: #f5a05a; border: 1px solid rgba(245,160,90,.3); padding: .3rem .75rem; border-radius: 20px; letter-spacing: .06em; white-space: nowrap; }

.section-head { font-size: .62rem; text-transform: uppercase; letter-spacing: .1em; color: var(--ink-faint); padding-bottom: .5rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; font-family: 'DM Mono', monospace; }

.info-box {
  background: var(--paper-2); border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: var(--radius); padding: 1rem 1.25rem; margin-bottom: 1.25rem;
  font-size: .78rem; line-height: 1.8; color: var(--ink-light);
}
.info-box b { color: var(--ink); }
.info-box .step-num {
  display: inline-block; width: 18px; height: 18px; background: var(--ink); color: var(--paper);
  border-radius: 50%; text-align: center; line-height: 18px; font-size: .65rem; font-weight: 700; margin-right: .35rem;
}

.score-table { width: 100%; border-collapse: collapse; font-size: .76rem; font-family: 'DM Mono', monospace; }
.score-table th { background: var(--ink); color: var(--paper); padding: .6rem .85rem; text-align: left; font-size: .63rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 500; }
.score-table tr:nth-child(even) td { background: var(--paper-2); }
.score-table td { padding: .6rem .85rem; border-bottom: 1px solid var(--border); color: var(--ink-light); vertical-align: top; line-height: 1.5; }
.score-table td:first-child { color: var(--ink); font-weight: 500; }
.score-table td:nth-child(3) { color: var(--accent-red); font-weight: 600; }

/* ─── Documentation ─── */
.doc-hero {
  background: var(--ink); color: var(--paper); border-radius: var(--radius);
  padding: 2.5rem 2rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.doc-hero::before {
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background-image:
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(245,240,230,.04) 39px, rgba(245,240,230,.04) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(245,240,230,.04) 39px, rgba(245,240,230,.04) 40px);
}
.doc-hero h1 { font-family: 'Fraunces', serif !important; font-size: 2rem !important; font-weight: 700 !important; line-height: 1.15 !important; letter-spacing: -.03em !important; color: var(--paper) !important; margin-bottom: .5rem; position: relative; }
.doc-hero h1 em { color: #f5a05a; font-style: italic; }
.doc-hero .badge-pill { display: inline-block; font-size: .62rem; letter-spacing: .1em; text-transform: uppercase; background: rgba(245,160,90,.15); color: #f5a05a; border: 1px solid rgba(245,160,90,.25); padding: .25rem .7rem; border-radius: 20px; margin-bottom: 1rem; position: relative; font-family: 'DM Mono', monospace; }
.doc-hero .sub { font-family: 'DM Mono', monospace; font-size: .75rem; color: rgba(245,242,237,.45); letter-spacing: .04em; position: relative; line-height: 1.7; max-width: 480px; }
.doc-hero-stats { display: flex; gap: 2.5rem; margin-top: 1.5rem; position: relative; flex-wrap: wrap; }
.doc-hero-stat .h-val { font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 700; color: #f5a05a; line-height: 1; }
.doc-hero-stat .h-lbl { font-family: 'DM Mono', monospace; font-size: .62rem; color: rgba(245,242,237,.4); letter-spacing: .08em; text-transform: uppercase; margin-top: .25rem; }

.doc-section { background: var(--paper); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.75rem 2rem; margin-bottom: 1.25rem; }
.doc-section h2 { font-family: 'Fraunces', serif !important; font-size: 1.3rem !important; font-weight: 600 !important; color: var(--ink) !important; margin-bottom: .35rem !important; letter-spacing: -.02em !important; }
.doc-section .sec-num { font-family: 'DM Mono', monospace; font-size: .6rem; text-transform: uppercase; letter-spacing: .1em; color: var(--ink-faint); margin-bottom: .35rem; }
.doc-section p { font-family: 'Lora', serif; font-size: .9rem; line-height: 1.8; color: var(--ink-light); margin-bottom: .75rem; }
.doc-section h3 { font-family: 'Fraunces', serif !important; font-size: 1.05rem !important; font-weight: 600 !important; color: var(--ink) !important; margin: 1.25rem 0 .5rem !important; }
.doc-section ul, .doc-section ol { padding-left: 1.3rem; margin-bottom: .75rem; }
.doc-section li { font-family: 'Lora', serif; font-size: .88rem; line-height: 1.7; color: var(--ink-light); margin-bottom: .3rem; }
.doc-section li strong { color: var(--ink); }

.pipe-flow { display: flex; align-items: center; gap: 0; margin: 1.25rem 0; flex-wrap: wrap; row-gap: .5rem; }
.pipe-step { flex: 1; min-width: 110px; background: var(--paper-2); border: 1px solid var(--border); border-radius: 7px; padding: .7rem .85rem; text-align: center; }
.pipe-step .ps-icon { font-size: 1.1rem; margin-bottom: .3rem; }
.pipe-step .ps-name { font-family: 'DM Mono', monospace; font-size: .68rem; font-weight: 500; color: var(--ink); display: block; }
.pipe-step .ps-desc { font-size: .62rem; color: var(--ink-faint); margin-top: .15rem; }
.pipe-arrow { font-size: .9rem; color: var(--ink-faint); padding: 0 .35rem; flex-shrink: 0; }

.doc-code { font-family: 'DM Mono', monospace; font-size: .78rem; background: var(--ink); color: #e8e4dc; padding: 1.25rem 1.5rem; border-radius: 8px; overflow-x: auto; line-height: 1.75; margin: 1rem 0; }
.doc-code .c { color: #6b8a6b; } .doc-code .k { color: #f5a070; } .doc-code .s { color: #a0c8a0; } .doc-code .f { color: #a0b8f0; }
.inline-code { font-family: 'DM Mono', monospace; font-size: .8rem; background: var(--paper-2); border: 1px solid var(--border); padding: .1rem .35rem; border-radius: 4px; color: var(--accent-red); }

.signal-row { display: flex; align-items: flex-start; gap: .85rem; padding: .85rem 0; border-bottom: 1px solid var(--border); }
.signal-row:last-child { border: none; }
.sig-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: .35rem; }
.sig-body .sig-name { font-family: 'DM Mono', monospace; font-size: .78rem; font-weight: 500; color: var(--ink); margin-bottom: .2rem; }
.sig-body .sig-desc { font-family: 'Lora', serif; font-size: .85rem; color: var(--ink-light); line-height: 1.65; }
.sig-body .sig-score { display: inline-block; font-family: 'DM Mono', monospace; font-size: .65rem; font-weight: 600; color: var(--accent-red); background: rgba(196,68,10,.08); border: 1px solid rgba(196,68,10,.15); padding: .15rem .45rem; border-radius: 4px; margin-top: .3rem; }

.steps-doc { display: flex; flex-direction: column; gap: .65rem; }
.step-doc { display: flex; align-items: flex-start; gap: .85rem; background: var(--paper-2); border: 1px solid var(--border); border-radius: 8px; padding: .85rem 1rem; }
.step-doc-num { width: 24px; height: 24px; border-radius: 50%; background: var(--ink); color: var(--paper); display: flex; align-items: center; justify-content: center; font-family: 'DM Mono', monospace; font-size: .65rem; font-weight: 700; flex-shrink: 0; }
.step-doc-content { font-family: 'Lora', serif; font-size: .88rem; line-height: 1.7; color: var(--ink-light); }
.step-doc-content strong { color: var(--ink); }

.limit-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
.limit-card { border-radius: 8px; padding: 1rem 1.1rem; font-family: 'Lora', serif; font-size: .85rem; line-height: 1.7; color: var(--ink-light); }
.limit-card.warn { background: rgba(196,68,10,.06); border: 1px solid rgba(196,68,10,.2); }
.limit-card.ok { background: rgba(26,107,74,.06); border: 1px solid rgba(26,107,74,.2); }
.limit-card .lc-head { font-family: 'DM Mono', monospace; font-size: .72rem; font-weight: 600; color: var(--ink); margin-bottom: .65rem; text-transform: uppercase; letter-spacing: .05em; }
.limit-card ul { padding-left: 1.1rem; } .limit-card li { margin-bottom: .35rem; } .limit-card li strong { color: var(--ink); }

.rgpd-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .75rem; margin: 1rem 0; }
.rgpd-item { background: var(--paper-2); border: 1px solid var(--border); border-radius: 8px; padding: .85rem 1rem; }
.ri-status { display: inline-block; font-family: 'DM Mono', monospace; font-size: .62rem; font-weight: 600; padding: .2rem .5rem; border-radius: 4px; text-transform: uppercase; letter-spacing: .05em; margin-bottom: .4rem; }
.ri-ok { background: rgba(26,107,74,.1); color: #1a6b4a; }
.ri-warn { background: rgba(196,68,10,.1); color: #c4440a; }
.ri-no { background: rgba(122,74,160,.1); color: #7a4aa0; }
.ri-label { font-family: 'DM Mono', monospace; font-size: .73rem; font-weight: 500; color: var(--ink); margin-bottom: .35rem; }
.rgpd-item p { font-family: 'Lora', serif; font-size: .8rem; line-height: 1.6; color: var(--ink-light); margin: 0 !important; }

.callout { border-radius: 8px; padding: 1rem 1.15rem; font-family: 'Lora', serif; font-size: .88rem; line-height: 1.7; color: var(--ink-light); margin: 1rem 0; }
.callout.warn { background: rgba(196,68,10,.06); border-left: 3px solid #c4440a; }
.callout.success { background: rgba(26,107,74,.06); border-left: 3px solid #1a6b4a; }
.callout strong { color: var(--ink); }

.schema-table { width: 100%; border-collapse: collapse; font-size: .76rem; font-family: 'DM Mono', monospace; margin: 1rem 0; }
.schema-table th { background: var(--ink); color: var(--paper); padding: .55rem .85rem; text-align: left; font-size: .62rem; letter-spacing: .08em; text-transform: uppercase; }
.schema-table td { padding: .55rem .85rem; border-bottom: 1px solid var(--border); color: var(--ink-light); vertical-align: top; }
.schema-table tr:nth-child(even) td { background: var(--paper-2); }
.schema-table td:first-child { color: var(--accent-red); }
.schema-table td:nth-child(2) { color: var(--accent-blue); }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.25rem 0 !important; }
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
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style='margin-bottom:1.5rem'>
  <div style='font-family:"Fraunces",serif;font-size:1.2rem;font-weight:700;letter-spacing:-.02em;color:#f5f2ed'>
    DVF <span style='color:#f5a05a'>×</span> BODACC
  </div>
  <div style='font-size:.6rem;color:rgba(245,242,237,.35);text-transform:uppercase;letter-spacing:.1em;margin-top:.2rem'>
    Prospects immobiliers
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='section-head' style='color:rgba(245,242,237,.3);border-bottom-color:rgba(245,242,237,.1)'>Paramètres</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='section-head' style='color:rgba(245,242,237,.3);border-bottom-color:rgba(245,242,237,.1)'>Filtres</div>", unsafe_allow_html=True)

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

    st.markdown("""
<div style='margin-top:1.5rem;font-size:.6rem;color:rgba(245,242,237,.28);line-height:1.9;font-family:"DM Mono",monospace'>
  Sources<br>
  DVF · data.gouv.fr<br>
  BODACC · OpenDataSoft<br><br>
  ⚠ Vérification RGPD requise<br>
  avant toute utilisation.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# EN-TÊTE PAGE
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
  <div>
    <div class="logo">DVF <span>×</span> BODACC</div>
    <div class="tagline">Identification de prospects immobiliers par signaux de vie</div>
  </div>
  <div class="status-pill">Pipeline v1.0</div>
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
# TABS PRINCIPALES
# ════════════════════════════════════════════════════════════
tab_analyse, tab_doc = st.tabs(["📊 Analyse & Carte", "📖 Documentation"])


# ════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENTATION
# ════════════════════════════════════════════════════════════
with tab_doc:

    st.markdown("""
<div class="doc-hero">
  <div class="badge-pill">Documentation technique v1.0</div>
  <h1>Pipeline DVF <em>×</em> BODACC</h1>
  <div class="sub">Identification automatique de prospects immobiliers par croisement de données publiques françaises. Signaux de vie · Scoring comportemental · Export CRM</div>
  <div class="doc-hero-stats">
    <div class="doc-hero-stat"><div class="h-val">5</div><div class="h-lbl">Signaux détectés</div></div>
    <div class="doc-hero-stat"><div class="h-val">100%</div><div class="h-lbl">Données publiques</div></div>
    <div class="doc-hero-stat"><div class="h-val">15</div><div class="h-lbl">Départements</div></div>
    <div class="doc-hero-stat"><div class="h-val">0–100</div><div class="h-lbl">Échelle de scoring</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Section 1 — Sources
    st.markdown("""
<div class="doc-section">
  <div class="sec-num">01 — Sources de données</div>
  <h2>DVF et BODACC</h2>
  <p>Le pipeline croise deux bases de données publiques françaises complémentaires pour générer des signaux de propension à vendre ou acheter un bien immobilier.</p>
  <div class="pipe-flow">
    <div class="pipe-step"><div class="ps-icon">🏛</div><span class="ps-name">DVF</span><div class="ps-desc">data.gouv.fr<br>Transactions foncières</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-icon">⚖️</div><span class="ps-name">BODACC</span><div class="ps-desc">OpenDataSoft<br>Annonces légales</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-icon">🔗</div><span class="ps-name">Croisement</span><div class="ps-desc">Jointure par<br>adresse / CP</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-icon">📊</div><span class="ps-name">Scoring</span><div class="ps-desc">5 signaux<br>0–100 pts</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><div class="ps-icon">📋</div><span class="ps-name">Export CSV</span><div class="ps-desc">Prospects triés<br>par score</div></div>
  </div>
  <h3>DVF — Demande de Valeurs Foncières</h3>
  <p>Publiée par la DGFiP, elle recense toutes les transactions immobilières enregistrées aux services de la publicité foncière. Elle couvre les ventes de biens bâtis et non bâtis, les adjudications et les échanges. Mise à jour trimestriellement avec un décalage de 3 à 6 mois. Disponible depuis 2014.</p>
  <h3>BODACC — Bulletin Officiel des Annonces Civiles et Commerciales</h3>
  <p>Publié par la DILA, il contient les annonces légales obligatoires : ventes de fonds de commerce, modifications d'entreprises, procédures collectives. <strong>Note :</strong> le BODACC ne contient pas directement les successions de particuliers — le signal "succession" est une corrélation temporelle entre une annonce et une transaction DVF rapprochée.</p>
</div>
""", unsafe_allow_html=True)

    # Section 2 — Signaux
    st.markdown("""
<div class="doc-section">
  <div class="sec-num">02 — Signaux détectés</div>
  <h2>Les 5 signaux de propension</h2>
  <p>Chaque prospect est associé à un signal principal. Les signaux représentent des événements de vie corrélés statistiquement à une mobilité résidentielle.</p>

  <div class="signal-row">
    <div class="sig-dot" style="background:#8a4a1a"></div>
    <div class="sig-body">
      <div class="sig-name">Succession BODACC</div>
      <div class="sig-desc">Vente d'un bien dans les 18 mois suivant une annonce BODACC liée à une succession dans le même secteur. Le délai court est un fort indicateur de contrainte de liquidation.</div>
      <div class="sig-score">50–100 pts · +30 si &lt;90j · +20 si &lt;180j · +20 adjudication</div>
    </div>
  </div>
  <div class="signal-row">
    <div class="sig-dot" style="background:#c4440a"></div>
    <div class="sig-body">
      <div class="sig-name">Divorce / séparation</div>
      <div class="sig-desc">Bien de type T3/T4 revendu dans les 3 ans suivant l'achat initial. Cette revente rapide d'un bien familial est corrélée à une séparation ou un changement de situation conjugale.</div>
      <div class="sig-score">60 pts (1–3 ans) · 80 pts (&lt;1 an)</div>
    </div>
  </div>
  <div class="signal-row">
    <div class="sig-dot" style="background:#1a6b4a"></div>
    <div class="sig-body">
      <div class="sig-name">Upgrade famille</div>
      <div class="sig-desc">Achat d'un T1/T2 dans les 4 dernières années, suggérant une primo-accession ou un célibataire devenu parent. Le propriétaire est désormais candidat à un bien plus grand.</div>
      <div class="sig-score">55 pts (score fixe)</div>
    </div>
  </div>
  <div class="signal-row">
    <div class="sig-dot" style="background:#1a4a8a"></div>
    <div class="sig-body">
      <div class="sig-name">Retraite / downsizing</div>
      <div class="sig-desc">Grand bien (T5+) vendu à plus de 10% sous la médiane de la commune. La décote suggère une motivation forte et une volonté de vendre rapidement.</div>
      <div class="sig-score">65 pts (décote 10–20%) · 80 pts (décote &gt;20%)</div>
    </div>
  </div>
  <div class="signal-row">
    <div class="sig-dot" style="background:#7a4aa0"></div>
    <div class="sig-body">
      <div class="sig-name">Primo-acheteur potentiel</div>
      <div class="sig-desc">T1/T2 vendu à moins de 70% du prix médian du code postal. Le bas prix suggère un bien d'entrée de gamme dont le propriétaire sera candidat à un achat supérieur.</div>
      <div class="sig-score">50 pts (score fixe)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Section 3 — Schéma CSV
    st.markdown("""
<div class="doc-section">
  <div class="sec-num">03 — Schéma de sortie</div>
  <h2>Structure du CSV généré</h2>
  <p>Le pipeline produit un fichier CSV structuré, trié par <span class="inline-code">score_signal</span> décroissant.</p>
  <table class="schema-table">
    <thead><tr><th>Colonne</th><th>Type</th><th>Description</th></tr></thead>
    <tbody>
      <tr><td>signal</td><td>str</td><td>Code du signal (heritage, divorce, upgrade, retraite, primo)</td></tr>
      <tr><td>signal_label</td><td>str</td><td>Libellé lisible du signal</td></tr>
      <tr><td>score_signal</td><td>float</td><td>Score de propension 0–100. Prospects chauds = score ≥ 70.</td></tr>
      <tr><td>chaleur</td><td>str</td><td>Catégorie qualitative : Très chaud / Chaud / Tiède</td></tr>
      <tr><td>adresse_complete</td><td>str</td><td>Adresse postale complète du bien (source DVF)</td></tr>
      <tr><td>code_postal</td><td>str</td><td>Code postal (5 chiffres)</td></tr>
      <tr><td>nom_commune</td><td>str</td><td>Commune de la transaction</td></tr>
      <tr><td>valeur_fonciere</td><td>float</td><td>Prix de vente en euros</td></tr>
      <tr><td>surface_reelle_bati</td><td>float</td><td>Surface habitable en m²</td></tr>
      <tr><td>nombre_pieces_principales</td><td>int</td><td>Nombre de pièces principales</td></tr>
      <tr><td>date_mutation</td><td>date</td><td>Date de la transaction (YYYY-MM-DD)</td></tr>
      <tr><td>nature_mutation</td><td>str</td><td>Type de transaction (Vente, Adjudication…)</td></tr>
      <tr><td>longitude / latitude</td><td>float</td><td>Coordonnées GPS pour la cartographie</td></tr>
    </tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    # Section 4 — Limites
    st.markdown("""
<div class="doc-section">
  <div class="sec-num">04 — Limites &amp; biais</div>
  <h2>Ce que le pipeline ne fait pas</h2>
  <p>Comprendre les limites est essentiel pour ne pas sur-interpréter les résultats.</p>
  <div class="limit-grid">
    <div class="limit-card warn">
      <div class="lc-head">⚠ Limites importantes</div>
      <ul>
        <li><strong>Pas de noms ni contacts</strong> — le DVF ne contient pas l'identité des vendeurs. Le pipeline identifie des adresses et des patterns, pas des personnes.</li>
        <li><strong>Succession = proxy indirect</strong> — le signal "héritage" est une corrélation temporelle, pas une cause avérée.</li>
        <li><strong>Données passées ≠ intentions futures</strong> — une transaction en 2024 est déjà réalisée. Le signal indique un profil actif, pas un bien à vendre aujourd'hui.</li>
        <li><strong>Faux positifs sur upgrade</strong> — un T2 acheté il y a 3 ans peut avoir été vendu sans lien avec une naissance.</li>
        <li><strong>Pas de données temps réel</strong> — le DVF est mis à jour trimestriellement avec un décalage de 3–6 mois.</li>
      </ul>
    </div>
    <div class="limit-card ok">
      <div class="lc-head">✓ Ce que le pipeline fait bien</div>
      <ul>
        <li><strong>Segmentation géographique précise</strong> — identification des zones à forte propension par code postal.</li>
        <li><strong>Scoring relatif fiable</strong> — les scores permettent de prioriser des zones et des types de biens.</li>
        <li><strong>Volume exploitable</strong> — des milliers de signaux par département pour alimenter des campagnes ciblées.</li>
        <li><strong>Reproductible et automatisable</strong> — peut tourner mensuellement pour suivre l'évolution du marché.</li>
        <li><strong>100% données publiques</strong> — aucun risque légal lié à l'acquisition des données sources.</li>
        <li><strong>Enrichissable</strong> — base idéale pour croiser avec d'autres sources (INSEE, notaires, CRM).</li>
      </ul>
    </div>
  </div>
  <div class="callout warn"><strong>Usage recommandé :</strong> utiliser le CSV comme base de segmentation pour des <em>audiences publicitaires</em> (Meta, Google) par zone géographique, et non comme liste de prospection directe nominative. Pour la prospection directe, un enrichissement via un prestataire habilité (RGPD) est nécessaire.</div>
</div>
""", unsafe_allow_html=True)

    # Section 5 — RGPD
    st.markdown("""
<div class="doc-section">
  <div class="sec-num">05 — Conformité RGPD</div>
  <h2>Cadre légal et bonnes pratiques</h2>
  <p>Le traitement de données à des fins commerciales est encadré par le RGPD et la loi Informatique et Libertés.</p>
  <div class="rgpd-grid">
    <div class="rgpd-item"><span class="ri-status ri-ok">✓ Autorisé</span><div class="ri-label">Ciblage géographique</div><p>Utiliser les codes postaux et communes pour cibler des campagnes Meta/Google sur des zones à forte propension.</p></div>
    <div class="rgpd-item"><span class="ri-status ri-ok">✓ Autorisé</span><div class="ri-label">Segments agrégés</div><p>Créer des audiences similaires (Lookalike) à partir de visiteurs de votre site sans traitement nominal.</p></div>
    <div class="rgpd-item"><span class="ri-status ri-ok">✓ Autorisé</span><div class="ri-label">Analyse statistique</div><p>Utiliser les données agrégées pour comprendre le marché et adapter une stratégie commerciale.</p></div>
    <div class="rgpd-item"><span class="ri-status ri-warn">⚠ Zone grise</span><div class="ri-label">Prospection postale</div><p>Envoyer un courrier à une adresse identifiée sans consentement préalable. Possible sous conditions avec base légale documentée.</p></div>
    <div class="rgpd-item"><span class="ri-status ri-warn">⚠ Zone grise</span><div class="ri-label">Enrichissement d'adresses</div><p>Croiser les adresses DVF avec un annuaire. Possible uniquement via un prestataire habilité avec consentement géré.</p></div>
    <div class="rgpd-item"><span class="ri-status ri-no">✗ Interdit</span><div class="ri-label">Fichier nominatif direct</div><p>Constituer une base Nom + Adresse + Événement de vie sans base légale explicite. Sanction CNIL possible.</p></div>
  </div>
  <div class="callout success"><strong>Recommandation :</strong> La voie la plus sûre est d'utiliser les insights géographiques du pipeline pour créer des <strong>landing pages SEO thématiques</strong> et des <strong>campagnes Meta/Google ciblées géographiquement</strong>. Conformité RGPD garantie.</div>
</div>
""", unsafe_allow_html=True)

    # Section 6 — Guide
    st.markdown("""
<div class="doc-section">
  <div class="sec-num">06 — Guide d'utilisation</div>
  <h2>Prise en main pas à pas</h2>
  <div class="steps-doc">
    <div class="step-doc"><div class="step-doc-num">1</div><div class="step-doc-content"><strong>Installer les dépendances</strong> — Lancer <span class="inline-code">pip install -r requirements.txt</span>. Python 3.8+ requis.</div></div>
    <div class="step-doc"><div class="step-doc-num">2</div><div class="step-doc-content"><strong>Configurer le département cible</strong> — Sélectionner le département et l'année dans la sidebar, ou modifier <span class="inline-code">DEPT</span> dans <span class="inline-code">pipeline.py</span>.</div></div>
    <div class="step-doc"><div class="step-doc-num">3</div><div class="step-doc-content"><strong>Lancer l'analyse</strong> — Cliquer sur <strong>→ Lancer l'analyse</strong>. Le pipeline télécharge DVF (~50–200 MB) et les données BODACC puis effectue le croisement.</div></div>
    <div class="step-doc"><div class="step-doc-num">4</div><div class="step-doc-content"><strong>Analyser les résultats</strong> — Explorez la carte interactive et le tableau des prospects. Filtrez par signal et score minimum.</div></div>
    <div class="step-doc"><div class="step-doc-num">5</div><div class="step-doc-content"><strong>Exporter</strong> — Téléchargez le CSV via <strong>⬇ Télécharger le CSV complet</strong>. Filtrez par <span class="inline-code">score_signal ≥ 70</span> pour les prospects chauds.</div></div>
    <div class="step-doc"><div class="step-doc-num">6</div><div class="step-doc-content"><strong>Activer</strong> — Utilisez les codes postaux pour créer des audiences publicitaires géolocalisées sur Meta ou Google Ads. Segmentez par signal pour adapter le message.</div></div>
  </div>
  <h3>Exemple d'intégration CRM</h3>
  <div class="doc-code"><span class="c"># Import dans HubSpot via API</span>
<span class="k">import</span> pandas <span class="k">as</span> pd

df = pd.<span class="f">read_csv</span>(<span class="s">'data/prospects_75_20241201.csv'</span>)
hot = df[df[<span class="s">'score_signal'</span>] >= 70].copy()
hot[<span class="s">'hubspot_tag'</span>] = hot[<span class="s">'signal'</span>].<span class="f">map</span>({
    <span class="s">'divorce'</span>:  <span class="s">'PROSPECT_DIVORCE'</span>,
    <span class="s">'retraite'</span>: <span class="s">'PROSPECT_RETRAITE'</span>,
    <span class="s">'heritage'</span>: <span class="s">'PROSPECT_HERITAGE'</span>,
    <span class="s">'upgrade'</span>:  <span class="s">'PROSPECT_UPGRADE'</span>,
    <span class="s">'primo'</span>:    <span class="s">'PROSPECT_PRIMO'</span>,
})
hot.<span class="f">to_csv</span>(<span class="s">'hubspot_import.csv'</span>, index=<span class="k">False</span>)</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE
# ════════════════════════════════════════════════════════════
with tab_analyse:
    df_raw = st.session_state.prospects

    if df_raw is None:
        st.markdown("""
<div class="info-box">
  <b>Comment ça marche ?</b><br><br>
  <span class="step-num">1</span> Choisissez un <b>département</b> et une <b>année</b> dans la sidebar.<br>
  <span class="step-num">2</span> Cliquez <b>→ Lancer l'analyse</b> — le pipeline télécharge DVF + BODACC, croise les données et score chaque prospect.<br>
  <span class="step-num">3</span> La carte et le tableau se remplissent automatiquement. Filtrez par signal et par score.<br>
  <span class="step-num">4</span> Téléchargez le CSV pour un usage externe (CRM, Meta Ads, Google Ads…).
</div>
""", unsafe_allow_html=True)

        st.markdown("<div class='section-head'>Grille de scoring</div>", unsafe_allow_html=True)
        rows = [
            ("Succession BODACC", "Vente dans les 18 mois après une succession BODACC", "50–100", "+30 si &lt;90j / +20 si &lt;180j / +10 si &lt;365j / +20 adjudication"),
            ("Divorce / séparation", "Bien T3/T4 revendu &lt; 3 ans après l'achat précédent", "60 ou 80", "+80 si délai &lt;1 an | +60 si 1–3 ans"),
            ("Upgrade famille", "Achat T1/T2 dans les 4 dernières années", "55", "Score fixe"),
            ("Retraite / downsizing", "T5+ vendu &gt; 10% sous la médiane commune", "65 ou 80", "+80 si décote &gt;20% | +65 si 10–20%"),
            ("Primo-acheteur", "T1/T2 vendu à &lt; 70% du prix médian du CP", "50", "Score fixe"),
        ]
        st.markdown(
            '<table class="score-table"><thead><tr><th>Signal</th><th>Critère</th><th>Score</th><th>Bonus / Détail</th></tr></thead><tbody>' +
            "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>" for r in rows) +
            "</tbody></table>", unsafe_allow_html=True
        )
        st.stop()

    # ── Filtrage ──────────────────────────────────────────────────
    df = df_raw.copy()
    if "signal_carte" in df.columns:
        sig_col = "signal_carte"
    elif "signal" in df.columns:
        sig_col = "signal"
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

    # ── Métriques ─────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Vue d'ensemble</div>", unsafe_allow_html=True)
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

    # ── Carte Leaflet ─────────────────────────────────────────────
    def build_leaflet_html(df: pd.DataFrame, sig_col: str) -> str:
        SIGNAL_COLORS_JS = json.dumps(SIGNAL_COLORS)
        SIGNAL_LABELS_JS = json.dumps(SIGNAL_LABELS)
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
</style>
</head><body>
<div id="map"></div>
<script>
const DATA   = {data_json};
const COLORS = {SIGNAL_COLORS_JS};
const LABELS = {SIGNAL_LABELS_JS};
const map = L.map('map').setView([{center_lat},{center_lng}], {zoom});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© OpenStreetMap contributors', maxZoom:18
}}).addTo(map);
if (DATA.length > 0) {{
  const layer = L.layerGroup().addTo(map);
  DATA.forEach(pt => {{
    const color  = COLORS[pt.signal] || '#888';
    const radius = 5 + (pt.score - 50) / 8;
    const prix   = pt.prix ? pt.prix.toLocaleString('fr-FR') + ' €' : '—';
    const surf   = pt.surface ? pt.surface + ' m²' : '—';
    const piec   = pt.pieces  ? pt.pieces + 'p' : '—';
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
    </div>`;
    L.circleMarker([pt.lat, pt.lng], {{
      radius, color, fillColor:color, fillOpacity:.65, weight:1.2, opacity:.9
    }}).bindPopup(popup).bindTooltip(
      `${{LABELS[pt.signal]||pt.signal}} · Score ${{pt.score}}`,
      {{direction:'top', offset:[0,-4]}}
    ).addTo(layer);
  }});
  const lats = DATA.map(d=>d.lat), lngs = DATA.map(d=>d.lng);
  map.fitBounds([[Math.min(...lats),Math.min(...lngs)],[Math.max(...lats),Math.max(...lngs)]], {{padding:[30,30]}});
}} else {{
  document.getElementById('map').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;background:#ede9e2;color:#6b6860;font-size:.85rem">⚠ Coordonnées GPS absentes</div>';
}}
</script>
</body></html>"""

    st.markdown("<div class='section-head'>Carte des prospects</div>", unsafe_allow_html=True)
    has_coords = "latitude" in df.columns and "longitude" in df.columns
    df_map = df[df["latitude"].notna() & df["longitude"].notna() & (df["latitude"] != 0)] if has_coords else pd.DataFrame()

    if not has_coords or df_map.empty:
        st.info("⚠ Coordonnées GPS absentes — les données DVF ne contiennent pas toujours latitude/longitude. Vérifiez que votre département est couvert par le fichier geo-dvf.")
    else:
        map_html = build_leaflet_html(df_map, sig_col)
        st.components.v1.html(map_html, height=530, scrolling=False)

    st.markdown("---")

    # ── Tableau + Ranking ──────────────────────────────────────────
    t1, t2 = st.tabs(["📋 Tableau prospects", "🏆 Ranking par zone"])

    with t1:
        st.markdown(f"<div style='font-size:.75rem;color:var(--ink-light);margin-bottom:.75rem'><b style='color:var(--ink)'>{total:,} prospects</b> — Score ≥ {score_min}, signaux : {', '.join(SIGNAL_LABELS.get(s,s) for s in signaux_choix)}</div>", unsafe_allow_html=True)

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

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇ Télécharger le CSV complet",
            data=csv_bytes,
            file_name=f"prospects_{st.session_state.run_dept}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with t2:
        st.markdown("<div class='section-head'>Top zones par nombre de prospects</div>", unsafe_allow_html=True)
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

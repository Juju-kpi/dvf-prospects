# -*- coding: utf-8 -*-
"""
DVF x BODACC — Outil BI de prospection vendeurs v4.2
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

from pipeline import (
    run_pipeline, SIGNAL_LABELS, SIGNAL_COLORS,
    SIGNAL_SEGMENTS, SIGNAL_CTA, SIGNAL_NORMALIZE
)

st.set_page_config(
    page_title="DVF x BODACC — Prospection vendeurs",
    page_icon="🏠", layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,500;0,700;1,300&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
:root{
  --ink:#1a1814;--inkl:#6b6860;--inkf:#b8b5ae;
  --paper:#f5f2ed;--p2:#ede9e2;--p3:#e0dbd2;
  --acc:#f5a05a;--red:#c4440a;--grn:#1a6b4a;--blu:#1a4a8a;--pur:#7a4aa0;
  --brd:rgba(26,24,20,.12);--r:10px;--sb:#1a1814;
}
html,body,[class*="css"]{font-family:'DM Mono',monospace!important;color:var(--ink);}
h1,h2,h3{font-family:'Fraunces',serif!important;letter-spacing:-.02em;}
.stApp{background:var(--paper)!important;}.stApp>header{background:var(--ink)!important;}
.main .block-container{padding-top:0!important;padding-bottom:3rem!important;max-width:1500px!important;}
[data-testid="stSidebar"]{background:var(--sb)!important;border-right:none!important;}
[data-testid="stSidebar"]>div:first-child{background:var(--sb)!important;padding:1.25rem 1rem!important;}
[data-testid="stSidebar"] *{color:#f5f2ed!important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stCheckbox label{
  color:rgba(245,242,237,.45)!important;font-size:.62rem!important;
  text-transform:uppercase!important;letter-spacing:.09em!important;}
[data-testid="stSidebar"] .stSelectbox>div>div,
[data-testid="stSidebar"] .stMultiSelect>div>div{
  background:rgba(255,255,255,.08)!important;border:1px solid rgba(255,255,255,.15)!important;border-radius:6px!important;}
[data-testid="stSidebar"] .stSelectbox svg{fill:rgba(245,242,237,.45)!important;}
[data-testid="stSidebar"] hr{border-color:rgba(245,242,237,.1)!important;margin:.6rem 0!important;}
[data-testid="stSidebar"] .stButton>button{
  background:var(--acc)!important;color:var(--ink)!important;border:none!important;
  font-family:'DM Mono',monospace!important;font-weight:500!important;font-size:.75rem!important;
  width:100%!important;border-radius:6px!important;padding:.55rem 1rem!important;}
[data-testid="stSidebar"] .stButton>button:hover{opacity:.85!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--brd)!important;gap:0!important;}
[data-testid="stTabs"] [data-baseweb="tab"]{
  background:transparent!important;border:none!important;color:var(--inkl)!important;
  font-family:'DM Mono',monospace!important;font-size:.67rem!important;letter-spacing:.05em!important;
  text-transform:uppercase!important;padding:.6rem 1rem!important;border-bottom:2px solid transparent!important;}
[data-testid="stTabs"] [data-baseweb="tab"]:hover{color:var(--ink)!important;}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--ink)!important;border-bottom:2px solid var(--red)!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none!important;}
[data-testid="stTabs"] [data-baseweb="tab-panel"]{padding-top:1.25rem!important;}
[data-testid="metric-container"]{background:var(--p2)!important;border:1px solid var(--brd)!important;border-radius:var(--r)!important;padding:.8rem 1rem!important;}
[data-testid="stMetricLabel"]{font-size:.58rem!important;text-transform:uppercase!important;letter-spacing:.08em!important;color:var(--inkl)!important;}
[data-testid="stMetricValue"]{font-family:'Fraunces',serif!important;font-size:1.45rem!important;font-weight:700!important;color:var(--ink)!important;line-height:1.1!important;}
[data-testid="stDataFrame"]{border-radius:var(--r)!important;border:1px solid var(--brd)!important;overflow:hidden!important;}
.stDataFrame td,.stDataFrame th{font-size:.71rem!important;font-family:'DM Mono',monospace!important;}
.stSuccess{background:rgba(26,107,74,.08)!important;border:1px solid rgba(26,107,74,.2)!important;border-radius:var(--r)!important;}
.stError{background:rgba(196,68,10,.08)!important;border:1px solid rgba(196,68,10,.2)!important;border-radius:var(--r)!important;}
.stInfo{background:var(--p2)!important;border:1px solid var(--brd)!important;border-radius:var(--r)!important;}
.stSpinner>div{border-top-color:var(--acc)!important;}
[data-testid="stDownloadButton"]>button{
  background:var(--ink)!important;color:var(--paper)!important;border:none!important;
  font-family:'DM Mono',monospace!important;font-size:.7rem!important;border-radius:6px!important;padding:.42rem .9rem!important;}
[data-testid="stDownloadButton"]>button:hover{opacity:.85!important;}
.ph{background:var(--ink);margin:0 -3rem 0 -3rem;padding:1.35rem 3rem;display:flex;align-items:center;justify-content:space-between;gap:1rem;}
.ph .logo{font-family:'Fraunces',serif;font-size:1.35rem;font-weight:700;letter-spacing:-.03em;color:#f5f2ed;}
.ph .logo em{color:#f5a05a;font-style:normal;}
.ph .sub{font-size:.6rem;color:rgba(245,242,237,.3);letter-spacing:.1em;text-transform:uppercase;margin-top:.15rem;}
.ph .vtag{font-size:.6rem;background:rgba(245,160,90,.15);color:#f5a05a;border:1px solid rgba(245,160,90,.3);padding:.28rem .7rem;border-radius:20px;white-space:nowrap;}
.sh{font-size:.59rem;text-transform:uppercase;letter-spacing:.1em;color:var(--inkf);padding-bottom:.4rem;border-bottom:1px solid var(--brd);margin-bottom:.9rem;}
.shd{font-size:.59rem;text-transform:uppercase;letter-spacing:.1em;color:rgba(245,242,237,.27);padding-bottom:.4rem;border-bottom:1px solid rgba(245,242,237,.1);margin-bottom:.8rem;}
.pc{display:flex;align-items:flex-start;gap:.9rem;padding:.9rem 1rem;border:1px solid var(--brd);border-radius:var(--r);margin-bottom:.5rem;background:var(--paper);transition:box-shadow .15s;}
.pc:hover{box-shadow:0 2px 14px rgba(26,24,20,.08);}
.pc.hot{border-left:3px solid var(--red);}
.pc.cls{border-left:3px solid var(--grn);}
.pc.mul{border-left:3px solid var(--acc);}
.psc{text-align:center;min-width:52px;}
.psc .v{font-family:'Fraunces',serif;font-size:1.45rem;font-weight:700;line-height:1;}
.psc .l{font-size:.52rem;text-transform:uppercase;letter-spacing:.06em;color:var(--inkf);}
.pbd{flex:1;min-width:0;}
.pad{font-weight:500;font-size:.8rem;color:var(--ink);margin-bottom:.18rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pmt{font-size:.67rem;color:var(--inkl);line-height:1.75;}
.pmt span{margin-right:.6rem;}
.ptg{display:flex;flex-wrap:wrap;gap:.28rem;margin-top:.35rem;}
.tg{font-size:.58rem;padding:.1rem .38rem;border-radius:4px;font-weight:500;}
.ts{background:var(--p2);border:1px solid var(--brd);color:var(--inkl);}
.th{background:rgba(196,68,10,.09);border:1px solid rgba(196,68,10,.18);color:var(--red);}
.tw{background:rgba(245,160,90,.1);border:1px solid rgba(245,160,90,.22);color:#b07030;}
.tc{background:rgba(26,74,138,.08);border:1px solid rgba(26,74,138,.15);color:var(--blu);}
.tf{background:var(--p3);border:1px solid var(--brd);color:var(--inkf);}
.tclus{background:rgba(26,107,74,.09);border:1px solid rgba(26,107,74,.2);color:var(--grn);}
.tmul{background:rgba(245,160,90,.12);border:1px solid rgba(245,160,90,.25);color:#c47820;}
.tins{background:rgba(122,74,160,.08);border:1px solid rgba(122,74,160,.18);color:var(--pur);}
.tp1{background:rgba(196,68,10,.09);border:1px solid rgba(196,68,10,.2);color:var(--red);}
.tp2{background:rgba(245,160,90,.1);border:1px solid rgba(245,160,90,.2);color:#b07030;}
.tp3{background:rgba(26,74,138,.08);border:1px solid rgba(26,74,138,.15);color:var(--blu);}
.prt{text-align:right;min-width:90px;}
.ppx{font-family:'Fraunces',serif;font-size:.95rem;font-weight:600;color:var(--ink);}
.psf{font-size:.65rem;color:var(--inkf);margin-top:.1rem;}
.pdc{font-size:.63rem;color:var(--red);font-weight:600;}
.pmg{font-family:'Lora',serif;font-size:.72rem;color:var(--inkl);margin-top:.4rem;line-height:1.6;font-style:italic;}
.pcta{font-family:'DM Mono',monospace;font-size:.66rem;color:var(--blu);margin-top:.25rem;background:rgba(26,74,138,.05);border:1px solid rgba(26,74,138,.12);border-radius:5px;padding:.3rem .5rem;line-height:1.5;}
.zc{background:var(--paper);border:1px solid var(--brd);border-radius:var(--r);padding:.9rem 1.15rem;margin-bottom:.45rem;}
.zch{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.45rem;}
.znm{font-weight:600;font-size:.82rem;color:var(--ink);}
.zsc{font-family:'Fraunces',serif;font-size:1.2rem;font-weight:700;}
.zb{height:4px;background:var(--p3);border-radius:2px;margin:.35rem 0;}
.zbf{height:100%;border-radius:2px;}
.zst{display:flex;gap:1.2rem;font-size:.65rem;color:var(--inkl);flex-wrap:wrap;}.zst b{color:var(--ink);}
.sg{background:var(--paper);border:1px solid var(--brd);border-radius:var(--r);padding:1rem 1.2rem;margin-bottom:.6rem;}
.sgh{display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;}
.sgd{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.sgn{font-weight:600;font-size:.8rem;color:var(--ink);}
.sgc{font-size:.65rem;color:var(--inkf);margin-left:auto;}
.sgm{font-family:'Lora',serif;font-size:.82rem;color:var(--inkl);line-height:1.65;margin-bottom:.4rem;}
.sgcta{background:var(--p2);border-radius:6px;padding:.5rem .7rem;font-size:.7rem;color:var(--inkl);margin-bottom:.4rem;line-height:1.6;}
.sgcta b{color:var(--ink);}
.sgs{display:flex;gap:1.3rem;font-size:.65rem;color:var(--inkl);flex-wrap:wrap;}.sgs b{color:var(--ink);}
.dist-bar{display:flex;height:20px;border-radius:6px;overflow:hidden;margin:.65rem 0;}
.db-s{display:flex;align-items:center;justify-content:center;font-size:.58rem;font-weight:600;color:#fff;}
.ic{background:var(--p2);border:1px solid var(--brd);border-radius:8px;padding:.85rem 1rem;}
.icv{font-family:'Fraunces',serif;font-size:1.35rem;font-weight:700;color:var(--ink);line-height:1;}
.icl{font-size:.59rem;text-transform:uppercase;letter-spacing:.07em;color:var(--inkf);margin-top:.18rem;}
.ics{font-size:.66rem;color:var(--inkl);margin-top:.2rem;font-family:'Lora',serif;line-height:1.5;}
.tc-card{background:var(--paper);border:1px solid var(--brd);border-radius:var(--r);padding:1.1rem 1.25rem;margin-bottom:.65rem;}
.tc-card h4{font-family:'Fraunces',serif!important;font-size:1rem!important;font-weight:600!important;color:var(--ink)!important;margin-bottom:.5rem!important;}
.mc{display:flex;align-items:flex-end;gap:2px;height:55px;padding:.15rem 0;}
.mcb{min-width:5px;border-radius:2px 2px 0 0;opacity:.72;flex:1;cursor:default;}
.hbar{display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;}
.hbar-nm{font-family:'DM Mono',monospace;font-size:.71rem;min-width:55px;color:var(--ink);}
.hbar-bg{flex:1;height:5px;background:var(--p3);border-radius:3px;}
.hbar-fill{height:100%;border-radius:3px;}
.hbar-val{font-size:.67rem;color:var(--inkl);min-width:65px;text-align:right;}
.lhero{background:var(--ink);border-radius:var(--r);padding:2.75rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.lhero::before{content:'';position:absolute;inset:0;pointer-events:none;background-image:repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(245,240,230,.04) 39px,rgba(245,240,230,.04) 40px),repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(245,240,230,.04) 39px,rgba(245,240,230,.04) 40px);}
.lpill{display:inline-block;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;background:rgba(245,160,90,.15);color:#f5a05a;border:1px solid rgba(245,160,90,.25);padding:.22rem .65rem;border-radius:20px;margin-bottom:.9rem;position:relative;font-family:'DM Mono',monospace;}
.lh1{font-family:'Fraunces',serif!important;font-size:2.1rem!important;font-weight:700!important;color:#f5f2ed!important;line-height:1.1!important;letter-spacing:-.03em!important;margin-bottom:.55rem;position:relative;}
.lh1 em{color:#f5a05a;font-style:italic;}
.lsub{font-family:'Lora',serif;font-size:.88rem;color:rgba(245,242,237,.48);line-height:1.8;max-width:560px;position:relative;}
.lhst{display:flex;gap:2.5rem;margin-top:1.75rem;position:relative;flex-wrap:wrap;}
.lhst .v{font-family:'Fraunces',serif;font-size:1.75rem;font-weight:700;color:#f5a05a;line-height:1;}
.lhst .l{font-size:.59rem;color:rgba(245,242,237,.32);text-transform:uppercase;letter-spacing:.08em;margin-top:.2rem;}
.how{display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem;margin:1.5rem 0;}
.hwc{background:var(--paper);border:1px solid var(--brd);border-radius:var(--r);padding:1.1rem 1.15rem;}
.hwn{font-family:'Fraunces',serif;font-size:1.55rem;font-weight:700;color:var(--p3);line-height:1;margin-bottom:.35rem;}
.hwt{font-weight:600;font-size:.78rem;color:var(--ink);margin-bottom:.28rem;}
.hwb{font-family:'Lora',serif;font-size:.78rem;color:var(--inkl);line-height:1.65;}
.se-row{display:flex;align-items:flex-start;gap:.85rem;padding:.88rem 0;border-bottom:1px solid var(--brd);}
.se-row:last-child{border:none;}
.se-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:.28rem;}
.se-nm{font-family:'DM Mono',monospace;font-size:.78rem;font-weight:500;color:var(--ink);margin-bottom:.17rem;}
.se-desc{font-family:'Lora',serif;font-size:.83rem;color:var(--inkl);line-height:1.65;}
.se-when{font-size:.64rem;color:var(--red);font-weight:500;margin-top:.22rem;font-family:'DM Mono',monospace;}
.score-arch{background:var(--p2);border-radius:var(--r);padding:1.25rem 1.5rem;margin:1rem 0;border:1px solid var(--brd);}
.sa-steps{display:flex;flex-direction:column;gap:.45rem;margin-top:.7rem;}
.sa-step{display:flex;align-items:flex-start;gap:.7rem;}
.sa-num{width:21px;height:21px;border-radius:50%;background:var(--ink);color:var(--paper);display:flex;align-items:center;justify-content:center;font-size:.59rem;font-weight:700;flex-shrink:0;}
.sa-txt{font-size:.77rem;color:var(--inkl);line-height:1.6;}.sa-txt b{color:var(--ink);}
.ug{display:grid;grid-template-columns:1fr 1fr;gap:.85rem;margin:1rem 0;}
.uc{border-radius:8px;padding:1rem 1.1rem;}
.uc-ok{background:rgba(26,107,74,.06);border:1px solid rgba(26,107,74,.2);}
.uc-warn{background:rgba(245,160,90,.06);border:1px solid rgba(245,160,90,.2);}
.uc-no{background:rgba(196,68,10,.06);border:1px solid rgba(196,68,10,.18);}
.uc-h{font-family:'DM Mono',monospace;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem;color:var(--ink);}
.uc ul{padding-left:1.1rem;margin:0;}
.uc li{font-family:'Lora',serif;font-size:.8rem;color:var(--inkl);line-height:1.7;margin-bottom:.18rem;}
hr{border:none!important;border-top:1px solid var(--brd)!important;margin:1rem 0!important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {"prospects": None, "tendances": {}, "run_dept": None, "run_annee": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

DEPT_LABELS = {
    "75":"75 — Paris","69":"69 — Rhone","13":"13 — Bouches-du-Rhone",
    "33":"33 — Gironde","31":"31 — Haute-Garonne","06":"06 — Alpes-Maritimes",
    "59":"59 — Nord","67":"67 — Bas-Rhin","44":"44 — Loire-Atlantique",
    "34":"34 — Herault","76":"76 — Seine-Maritime","38":"38 — Isere",
    "92":"92 — Hauts-de-Seine","93":"93 — Seine-Saint-Denis","94":"94 — Val-de-Marne",
}
SIG_DOTS = {"heritage":"#8a4a1a","divorce":"#c4440a","upgrade":"#1a6b4a","retraite":"#1a4a8a","primo":"#7a4aa0"}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='margin-bottom:1.3rem'>"
        "<div style='font-family:Fraunces,serif;font-size:1.1rem;font-weight:700;letter-spacing:-.02em;color:#f5f2ed'>"
        "DVF <span style='color:#f5a05a'>x</span> BODACC</div>"
        "<div style='font-size:.56rem;color:rgba(245,242,237,.27);text-transform:uppercase;letter-spacing:.1em;margin-top:.1rem'>"
        "Prospection vendeurs v4.2</div></div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='shd'>Parametres pipeline</div>", unsafe_allow_html=True)
    dept    = st.selectbox("Departement", list(DEPT_LABELS.keys()), format_func=lambda x: DEPT_LABELS[x])
    annee   = st.selectbox("Annee DVF", [2024, 2023, 2022, 2021])
    fenetre = st.select_slider("Fenetre succession (mois)", [9, 12, 18, 24], value=18,
                               help="9=forts signaux · 18=standard · 24=filet large")
    rayon   = st.select_slider("Rayon cluster (km)", [0.25, 0.5, 1.0, 2.0], value=0.5)

    st.markdown("---")
    st.markdown("<div class='shd'>Filtres resultats</div>", unsafe_allow_html=True)
    score_min = st.slider("Score minimum", 0, 90, 60, 5,
                          help="Percentile dept : 60=top 40% · 80=top 20% · 90=top 10%")
    sigs = st.multiselect("Signaux", list(SIGNAL_LABELS.keys()), default=list(SIGNAL_LABELS.keys()),
                          format_func=lambda x: SIGNAL_LABELS[x])
    chals = st.multiselect("Chaleur CRM", ["tres chaud","chaud","tiede","froid"],
                           default=["tres chaud","chaud","tiede"])
    prios = st.multiselect("Priorite", ["P1 — Contact immediat","P2 — Dans les 30j","P3 — Nurturing","P4 — A surveiller"],
                           default=["P1 — Contact immediat","P2 — Dans les 30j"])
    cls_only = st.checkbox("Hotspots seulement")

    st.markdown("---")
    run_btn = st.button("Lancer l'analyse", type="primary")

    st.markdown(
        "<div style='margin-top:1.3rem;font-size:.55rem;color:rgba(245,242,237,.2);line-height:2;font-family:DM Mono,monospace'>"
        "DVF · data.gouv.fr<br>BODACC · OpenDataSoft<br>INSEE · API communes<br><br>"
        "Score = percentile dept<br>80 = top 20%<br><br>Usage geo / segment<br>RGPD : pas de nominatif</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='ph'>"
    "<div><div class='logo'>DVF <em>x</em> BODACC — Prospection vendeurs</div>"
    "<div class='sub'>Detecter les proprietaires susceptibles de vendre · Score percentile · Liquidite · INSEE · Clusters</div></div>"
    "<div class='vtag'>v4.2 · Scoring contextuel</div>"
    "</div>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Pipeline — {DEPT_LABELS[dept]} / {annee} · fenetre {fenetre} mois..."):
        try:
            prospects, tendances = run_pipeline(
                dept=dept, annee=annee,
                fenetre_succession_mois=fenetre,
                enrichir_cadastre=False,
                rayon_cluster_km=rayon,
            )
            st.session_state.prospects = prospects
            st.session_state.tendances = tendances
            st.session_state.run_dept  = dept
            st.session_state.run_annee = annee
            n   = len(prospects)
            hot = int((prospects["score_final"] >= 80).sum()) if "score_final" in prospects.columns else 0
            cls = int(prospects.get("cluster_chaud", pd.Series(False)).sum())
            st.success(f"{n:,} prospects identifies — {hot:,} top 20% — {cls:,} dans un hotspot")
        except Exception as e:
            st.error(f"Erreur pipeline : {e}")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
(tab_home, tab_dash, tab_pros, tab_zones,
 tab_segs, tab_mkt, tab_carte) = st.tabs([
    "Guide", "Dashboard", "Prospects",
    "Zones chaudes", "Segments & CTA",
    "Marche", "Carte",
])

df_raw    = st.session_state.prospects
tendances = st.session_state.tendances

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_sc(d):
    return "signal_carte" if "signal_carte" in d.columns else ("signal" if "signal" in d.columns else None)

def filtrer(d):
    d = d.copy()
    sc = get_sc(d)
    if sc: d = d[d[sc].isin(sigs)]
    if "score_final" in d.columns: d = d[d["score_final"] >= score_min]
    if "chaleur" in d.columns and chals: d = d[d["chaleur"].astype(str).isin(chals)]
    if "priorite" in d.columns and prios: d = d[d["priorite"].isin(prios)]
    if cls_only and "cluster_chaud" in d.columns: d = d[d["cluster_chaud"] == True]
    return d

def scol(s):
    s = float(s)
    if s >= 80: return "#c4440a"
    if s >= 60: return "#d4850a"
    if s >= 40: return "#1a4a8a"
    return "#b8b5ae"

def chtag(ch):
    ch = str(ch)
    if "tres" in ch: return "th","tres chaud"
    if "chaud" == ch: return "tw","chaud"
    if "chaud" in ch: return "tw","chaud"
    if "tiede" in ch: return "tc","tiede"
    return "tf","froid"

def ptag(p):
    p = str(p)
    if "P1" in p: return "tp1", p
    if "P2" in p: return "tp2", p
    if "P3" in p: return "tp3", p
    return "ts", p

def render(row, show_cta=False):
    sf  = float(row.get("score_final", 0) or 0)
    sco = float(row.get("score_confiance", sf) or sf)
    adr = str(row.get("adresse_complete","—") or "—")
    ch  = str(row.get("chaleur","") or "")
    sig = str(row.get("signal_label","") or "")
    sk  = str(row.get("signal_carte", row.get("signal","")) or "")
    pri = str(row.get("priorite","") or "")
    px  = float(row.get("valeur_fonciere",0) or 0)
    sf2 = float(row.get("surface_reelle_bati",0) or 0)
    pm2 = float(row.get("prix_m2",0) or 0)
    dec = float(row.get("decote_vs_median",0) or 0)
    nb  = int(row.get("nb_signaux",1) or 1)
    dt  = str(row.get("date_mutation",""))[:10]
    cm  = str(row.get("nom_commune", row.get("commune","")) or "")
    cp  = str(row.get("code_postal","") or "")
    liq = float(row.get("liquidite_cp",0) or 0)
    cld = bool(row.get("cluster_chaud", False))
    cdn = int(row.get("cluster_densite",0) or 0)
    tp  = float(row.get("insee_taux_proprio",0) or 0)
    anc = float(row.get("anciennete_mois",0) or 0)
    seg = str(row.get("segment_cible","") or "")
    cta = str(row.get("cta","") or "")

    tc, tl = chtag(ch)
    pc, pl = ptag(pri)
    cls_s  = "pc hot" if sf >= 80 else ("pc cls" if cld else ("pc mul" if nb >= 2 else "pc"))
    mul_tag = f'<span class="tg tmul">+{nb} signaux</span>' if nb >= 2 else ""
    clt_tag = f'<span class="tg tclus">hotspot {cdn}</span>' if cld else ""
    liq_tag = f'<span class="tg tc">liq.{liq:.0f}</span>' if liq >= 60 else ""
    ins_tag = f'<span class="tg tins">proprio {tp:.0f}%</span>' if tp > 0 else ""
    anc_txt = f"<span>{anc:.0f} mois</span>" if anc > 0 else ""
    dec_html = f'<div class="pdc">-{abs(dec):.0f}% med.</div>' if dec > 0 else ""
    seg_html = f'<div class="pmg">{seg}</div>' if seg else ""
    cta_html = f'<div class="pcta">Action : {cta}</div>' if show_cta and cta else ""
    conf_html = f'<span class="tg ts">confiance {sco:.0f}</span>'

    return (
        f'<div class="{cls_s}">'
        f'<div class="psc"><div class="v" style="color:{scol(sf)}">{sf:.0f}</div><div class="l">score</div></div>'
        f'<div class="pbd">'
        f'<div class="pad">{adr}</div>'
        f'<div class="pmt"><span>📍 {cm} {cp}</span><span>📅 {dt}</span><span>🏠 {sf2:.0f} m2</span>{anc_txt}</div>'
        f'<div class="ptg"><span class="tg ts">{sig}</span><span class="tg {tc}">{tl}</span>'
        f'<span class="tg {pc}">{pl}</span>{mul_tag}{clt_tag}{liq_tag}{ins_tag}{conf_html}</div>'
        f'{seg_html}{cta_html}'
        f'</div>'
        f'<div class="prt"><div class="ppx">{px:,.0f} E</div><div class="psf">{pm2:,.0f} E/m2</div>{dec_html}</div>'
        f'</div>'
    )

def mini_bar_chart(values, labels, color="#f5a05a", height_px=55):
    mx = max(values) if values else 1
    parts = []
    for v, l in zip(values, labels):
        h = max(int(v / mx * 100), 4)
        is_peak = v >= mx * 0.9
        is_low  = v <= mx * 0.3
        cls = "peak" if is_peak else ("low" if is_low else ""
        )
        bg = "#c4440a" if is_peak else ("#b8b5ae" if is_low else color)
        parts.append(f'<div class="mcb {cls}" style="height:{h}%;background:{bg}" title="{l}: {v:,.0f}"></div>')
    return f'<div class="mc" style="height:{height_px}px">{"".join(parts)}</div>'

def hbar(label, value, max_val, color="#f5a05a", val_fmt="{:.0f}"):
    pct = value / max_val * 100 if max_val > 0 else 0
    val_str = val_fmt.format(value)
    return (
        f'<div class="hbar">'
        f'<div class="hbar-nm">{label}</div>'
        f'<div class="hbar-bg"><div class="hbar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
        f'<div class="hbar-val">{val_str}</div>'
        f'</div>'
    )

def section_card(title, content_html):
    return (
        f'<div class="tc-card">'
        f'<h4>{title}</h4>'
        f'{content_html}'
        f'</div>'
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — GUIDE
# ════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown(
        '<div class="lhero">'
        '<div class="lpill">Outil de prospection vendeurs</div>'
        '<h1 class="lh1">Identifiez les proprietaires<br><em>susceptibles de vendre</em></h1>'
        '<div class="lsub">Ce pipeline croise DVF, BODACC, INSEE et les donnees de marche pour detecter '
        '5 signaux de vie qui precedent statistiquement une decision de vente immobiliere. '
        'Chaque prospect recoit un score percentile dans son departement : 80 = top 20%.</div>'
        '<div class="lhst">'
        '<div><div class="v">5</div><div class="l">Signaux de vie</div></div>'
        '<div><div class="v">4</div><div class="l">Sources donnees</div></div>'
        '<div><div class="v">7</div><div class="l">Etapes scoring</div></div>'
        '<div><div class="v">100%</div><div class="l">Open data</div></div>'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<div class='sh'>Comment ca marche en 4 etapes</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="how">'
        '<div class="hwc"><div class="hwn">01</div><div class="hwt">Choisir le secteur</div>'
        '<div class="hwb">Selectionnez departement et annee DVF. Le pipeline telecharge automatiquement '
        'les transactions (50-200 MB), BODACC et les donnees INSEE de contexte.</div></div>'
        '<div class="hwc"><div class="hwn">02</div><div class="hwt">Detection des signaux</div>'
        '<div class="hwb">5 evenements de vie sont detectes : succession, divorce, famille qui grandit, '
        'retraite/downsizing, primo-acheteur. Chaque bien peut cumule plusieurs signaux.</div></div>'
        '<div class="hwc"><div class="hwn">03</div><div class="hwt">Scoring contextuel</div>'
        '<div class="hwb">7 etapes : brut signal + malus qualite + bonus INSEE + cluster geo + '
        'percentile dept + liquidite marche + multi-signal. Score 80 = top 20% du dept.</div></div>'
        '<div class="hwc"><div class="hwn">04</div><div class="hwt">Activer la prospection</div>'
        '<div class="hwb">Export CSV pour CRM. Codes postaux pour Meta/Google Ads. '
        'Les segments donnent le message adapte et le canal recommande pour chaque signal.</div></div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("<div class='sh'>Les 5 signaux de vie — pourquoi ils predisent une vente</div>", unsafe_allow_html=True)
    sigs_info = [
        ("heritage", "#8a4a1a",
         "Succession / heritage",
         "Annonce BODACC suivie d'une vente dans les 18 mois dans le meme secteur. "
         "Les heritiers vendent rapidement pour partager le capital ou solder la succession. "
         "Signal renforce si adjudication (vente forcee) ou vente dans les 90 jours.",
         "Signal le plus fiable — taux de conversion en vente reelle le plus eleve"),
        ("divorce",  "#c4440a",
         "Divorce / separation",
         "T3/T4 revendu 30j a 3 ans apres l'achat precedent sur la meme adresse. "
         "Revente rapide d'un bien familial = fort indice de separation. "
         "Signal plus fort si delai < 1 an (rupture recente, besoin de liquidite urgent).",
         "Signal tres actionnable — proprietaires en situation de contrainte financiere"),
        ("upgrade",  "#1a6b4a",
         "Upgrade famille",
         "Proprietaire d'un T1/T2 residentiel achete il y a 2 a 4 ans, surface > 15 m2. "
         "La fenetre 2-4 ans est celle ou le besoin d'espace devient pressant (naissance, couple). "
         "Ce proprietaire est aujourd'hui acheteur potentiel d'un bien plus grand — et donc vendeur.",
         "Signal d'intention — la personne ne sait pas encore qu'elle va vendre"),
        ("retraite", "#1a4a8a",
         "Retraite / downsizing",
         "T5+ (>= 80 m2) vendu avec decote > 10% vs mediane des T5+ du meme code postal. "
         "La decote signale une motivation forte : liquidation rapide, entree en maison de retraite, "
         "ou demenagement vers une zone moins chere. Vendeur motive, souvent negocie.",
         "Signal de pression — vendeur motive, negocie souvent sous le marche"),
        ("primo",    "#7a4aa0",
         "Primo-acheteur potentiel",
         "T1/T2 vendu a < 70% de la mediane des T1/T2 du CP, surface >= 15 m2, prix >= 15 000 E. "
         "Ce bien d'entree de gamme a ete achete par un primo-accedant. "
         "Dans 3 a 7 ans, cette personne deviendra candidate a un bien superieur.",
         "Signal de cycle long — audience a prechauffer en amont"),
    ]
    for key, color, name, desc, when in sigs_info:
        st.markdown(
            f'<div class="se-row">'
            f'<div class="se-dot" style="background:{color}"></div>'
            f'<div><div class="se-nm">{name}</div>'
            f'<div class="se-desc">{desc}</div>'
            f'<div class="se-when">{when}</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("<div class='sh'>Architecture du score — 7 etapes</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="score-arch">'
        '<div style="font-size:.78rem;color:var(--inkl);font-family:Lora,serif;margin-bottom:.5rem">'
        'Le score final est un <b style="color:var(--ink)">percentile dans le departement</b>. '
        'Score 80 = top 20% de tous les signaux detectes dans ce departement. '
        'Comparable entre Paris (50 000 signaux) et une ville moyenne (500 signaux).</div>'
        '<div class="sa-steps">'
        '<div class="sa-step"><div class="sa-num">1</div><div class="sa-txt"><b>Score brut signal</b> — intensite de l\'evenement : delai vente, decote, adjudication. Echelle 0-100 absolue.</div></div>'
        '<div class="sa-step"><div class="sa-num">2</div><div class="sa-txt"><b>Malus qualite</b> — -20 non residentiel · -15 prix/m2 aberrant · -10 surface hors normes · -10 prix &lt; 15 000 E. Filtre les donnees corrompues.</div></div>'
        '<div class="sa-step"><div class="sa-num">3</div><div class="sa-txt"><b>Bonus INSEE</b> — +0 a +5 selon contexte socio-demo. Divorce dans zone 90% locataires = 0. Retraite dans zone senior = +5.</div></div>'
        '<div class="sa-step"><div class="sa-num">4</div><div class="sa-txt"><b>Cluster geographique</b> — +4 a +8 si >= 3 prospects dans le rayon parametre. Un hotspot = zone de transition demographique reelle.</div></div>'
        '<div class="sa-step"><div class="sa-num">5</div><div class="sa-txt"><b>Normalisation percentile</b> — rank(pct=True) sur le score enrichi. Transforme le score absolu en position relative dans le dept.</div></div>'
        '<div class="sa-step"><div class="sa-num">6</div><div class="sa-txt"><b>Bonus liquidite marche</b> — +0 a +10 selon l\'activite du CP (volume transactions + delai rotation). Marche liquide = signal plus fiable.</div></div>'
        '<div class="sa-step"><div class="sa-num">7</div><div class="sa-txt"><b>Bonus multi-signal</b> — +5 si 2 signaux convergents sur la meme adresse · +12 si 3+ signaux. Rares et tres fiables.</div></div>'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("<div class='sh'>Usage legal recommande</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="ug">'
        '<div class="uc uc-ok"><div class="uc-h">Autorise</div><ul>'
        '<li>Campagnes geociblees Meta/Google Ads sur les codes postaux identifies</li>'
        '<li>Audiences LAL (Lookalike) a partir de vos clients actuels</li>'
        '<li>Analyse statistique du marche et strategie commerciale</li>'
        '<li>Contenu SEO cible sur les communes a fort potentiel</li>'
        '<li>Segmentation des zones prioritaires pour votre equipe commerciale</li>'
        '</ul></div>'
        '<div class="uc uc-warn"><div class="uc-h">Zone grise — documenter la base legale</div><ul>'
        '<li>Flyer postal a une adresse DVF (possible via base "interet legitime" RGPD)</li>'
        '<li>Enrichissement adresse via annuaire (prestataire habilite requis)</li>'
        '<li>Partage du CSV avec des tiers sans accord de traitement</li>'
        '</ul></div>'
        '</div>'
        '<div class="uc uc-no" style="margin-top:0"><div class="uc-h">Interdit — sanctions CNIL</div><ul>'
        '<li>Fichier nominatif Nom + Adresse + Evenement de vie sans consentement ou base legale</li>'
        '<li>Croisement DVF/annuaire de facon autonome pour retrouver des noms</li>'
        '<li>Demarchage telephonique direct sans opt-in sur des prospects identifies par DVF</li>'
        '</ul></div>',
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════════════════
# GUARD
# ════════════════════════════════════════════════════════════════════════════
if df_raw is None:
    for t in [tab_dash, tab_pros, tab_zones, tab_segs, tab_mkt, tab_carte]:
        with t:
            st.info("Lancez une analyse depuis la sidebar.")
    st.stop()

df      = filtrer(df_raw)
sig_col = get_sc(df)
total   = len(df)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_dash:
    dept_lbl = DEPT_LABELS.get(st.session_state.run_dept or "", "")
    st.markdown(f"<div class='sh'>{dept_lbl} / {st.session_state.run_annee} — {total:,} prospects filtres</div>",
                unsafe_allow_html=True)

    hot   = int((df["score_final"] >= 80).sum()) if "score_final" in df.columns else 0
    warm  = int(((df["score_final"] >= 60) & (df["score_final"] < 80)).sum()) if "score_final" in df.columns else 0
    multi = int((df.get("nb_signaux", pd.Series(1)) >= 2).sum())
    cls_n = int(df.get("cluster_chaud", pd.Series(False)).sum())
    p1    = int((df.get("priorite", pd.Series("")) == "P1 — Contact immediat").sum())
    avg   = float(df["score_final"].mean()) if "score_final" in df.columns and total else 0
    zones = int(df["nom_commune"].nunique()) if "nom_commune" in df.columns else 0
    lavg  = float(df["liquidite_cp"].mean()) if "liquidite_cp" in df.columns and df["liquidite_cp"].notna().any() else 0

    c = st.columns(8)
    with c[0]: st.metric("Prospects", f"{total:,}")
    with c[1]: st.metric("Score moyen", f"{avg:.0f}")
    with c[2]: st.metric("P1 immediat", f"{p1:,}")
    with c[3]: st.metric("Top 20% (>=80)", f"{hot:,}")
    with c[4]: st.metric("Chauds 60-79", f"{warm:,}")
    with c[5]: st.metric("Hotspots", f"{cls_n:,}")
    with c[6]: st.metric("Multi-signal", f"{multi:,}")
    with c[7]: st.metric("Liquidite moy.", f"{lavg:.0f}")

    st.markdown("---")
    ca, cb = st.columns([1.3, 1])

    with ca:
        st.markdown("<div class='sh'>Distribution des scores percentile</div>", unsafe_allow_html=True)
        if "score_final" in df.columns and total > 0:
            bins  = [0, 20, 40, 60, 80, 100]
            lbls  = ["0-20","20-40","40-60","60-80","80-100"]
            clrs  = ["#e0dbd2","#c8c2b8","#f5a05a","#c4440a80","#c4440a"]
            cts   = pd.cut(df["score_final"], bins=bins, labels=lbls, right=True).value_counts().reindex(lbls).fillna(0)
            pcts  = (cts / cts.sum() * 100).round(1)
            bar_html = "".join(
                f'<div class="db-s" style="flex:{p};background:{clr}" title="{l}:{int(n)}">{p:.0f}%</div>'
                for l, n, p, clr in zip(lbls, cts, pcts, clrs) if p > 0
            )
            st.markdown(f'<div class="dist-bar">{bar_html}</div>', unsafe_allow_html=True)
            cs2 = st.columns(5)
            for col_m, (lbl, cnt, clr) in zip(cs2, zip(lbls, cts, clrs)):
                with col_m:
                    st.markdown(
                        f'<div style="text-align:center;padding:.45rem;background:var(--p2);border-radius:8px;border:1px solid var(--brd)">'
                        f'<div style="font-family:Fraunces,serif;font-size:1.1rem;font-weight:700;color:{clr}">{int(cnt):,}</div>'
                        f'<div style="font-size:.58rem;color:var(--inkf);text-transform:uppercase;letter-spacing:.05em">{lbl}</div>'
                        f'</div>', unsafe_allow_html=True
                    )
        st.markdown("---")
        st.markdown("<div class='sh'>Distribution priorite prospection</div>", unsafe_allow_html=True)
        if "priorite" in df.columns and total > 0:
            pri_counts = df["priorite"].value_counts()
            pri_colors = {"P1 — Contact immediat":"#c4440a","P2 — Dans les 30j":"#f5a05a",
                          "P3 — Nurturing":"#1a4a8a","P4 — A surveiller":"#b8b5ae"}
            for pr, cnt in pri_counts.items():
                pct = cnt / total * 100
                col = pri_colors.get(str(pr), "#888")
                st.markdown(
                    f'<div class="hbar">'
                    f'<div class="hbar-nm" style="min-width:160px;font-size:.68rem">{pr}</div>'
                    f'<div class="hbar-bg"><div class="hbar-fill" style="width:{pct:.1f}%;background:{col}"></div></div>'
                    f'<div class="hbar-val">{cnt:,} ({pct:.0f}%)</div>'
                    f'</div>', unsafe_allow_html=True
                )

    with cb:
        st.markdown("<div class='sh'>Par signal detecte</div>", unsafe_allow_html=True)
        if sig_col and total > 0:
            for sig, cnt in df[sig_col].value_counts().items():
                pct = cnt / total * 100
                col = SIG_DOTS.get(sig, "#888")
                lbl = SIGNAL_LABELS.get(sig, sig)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.4rem">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{col};flex-shrink:0"></div>'
                    f'<div style="font-size:.72rem;color:var(--inkl);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{lbl}</div>'
                    f'<div style="font-family:Fraunces,serif;font-size:.86rem;font-weight:600;color:var(--ink)">{cnt:,}</div>'
                    f'<div style="font-size:.63rem;color:var(--inkf);min-width:28px;text-align:right">{pct:.0f}%</div>'
                    f'</div>'
                    f'<div style="height:3px;background:var(--p3);border-radius:2px;margin-bottom:.4rem">'
                    f'<div style="height:100%;width:{pct:.1f}%;background:{col};border-radius:2px"></div>'
                    f'</div>', unsafe_allow_html=True
                )
        st.markdown("---")
        st.markdown("<div class='sh'>Score confiance moyen par signal</div>", unsafe_allow_html=True)
        if sig_col and "score_confiance" in df.columns and total > 0:
            sc_by_sig = df.groupby(sig_col)["score_confiance"].mean().sort_values(ascending=False)
            mx_c = float(sc_by_sig.max()) if not sc_by_sig.empty else 1
            for sig, val in sc_by_sig.items():
                col = SIG_DOTS.get(sig, "#888")
                lbl = SIGNAL_LABELS.get(sig, sig)
                st.markdown(hbar(lbl[:22], val, mx_c, col, "{:.1f}"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='sh'>Top 5 prospects — priorite absolue</div>", unsafe_allow_html=True)
    if total > 0 and "score_final" in df.columns:
        for _, row in df.head(5).iterrows():
            st.markdown(render(row), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROSPECTS
# ════════════════════════════════════════════════════════════════════════════
with tab_pros:
    st.markdown(f"<div class='sh'>{total:,} prospects · Score >= {score_min}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: vue = st.radio("Vue", ["Cartes", "Tableau"], horizontal=True, label_visibility="collapsed")
    with c2:
        tri_opt = ["Score", "Confiance", "Priorite", "Prix", "Surface", "Liquidite", "Cluster", "Anciennete"]
        tri = st.selectbox("Tri", tri_opt, label_visibility="collapsed")
    with c3: srch = st.text_input("Commune / adresse", placeholder="Filtrer...", label_visibility="collapsed")

    tri_map = {
        "Score":      ("score_final",         False),
        "Confiance":  ("score_confiance",     False),
        "Priorite":   ("priorite",            True),
        "Prix":       ("valeur_fonciere",     False),
        "Surface":    ("surface_reelle_bati", False),
        "Liquidite":  ("liquidite_cp",        False),
        "Cluster":    ("cluster_densite",     False),
        "Anciennete": ("anciennete_mois",     True),
    }
    sc2, sa = tri_map.get(tri, ("score_final", False))
    dv = df.copy()
    if srch.strip():
        mask = (
            dv.get("nom_commune", pd.Series(dtype=str)).astype(str).str.upper().str.contains(srch.upper(), na=False) |
            dv.get("adresse_complete", pd.Series(dtype=str)).astype(str).str.upper().str.contains(srch.upper(), na=False)
        )
        dv = dv[mask]
    if sc2 in dv.columns:
        dv = dv.sort_values(sc2, ascending=sa)

    st.markdown(f"<div style='font-size:.68rem;color:var(--inkf);margin-bottom:.75rem'>{len(dv):,} resultats</div>",
                unsafe_allow_html=True)

    if vue == "Cartes":
        for _, row in dv.head(100).iterrows():
            st.markdown(render(row, show_cta=True), unsafe_allow_html=True)
        if len(dv) > 100:
            st.caption(f"100/{len(dv):,} affiches. Telechargez le CSV.")
    else:
        show_c = [c for c in [
            "rang","adresse_complete","code_postal","nom_commune","signal_label",
            "score_final","score_brut","score_confiance","chaleur","priorite",
            "valeur_fonciere","prix_m2","surface_reelle_bati","nombre_pieces_principales",
            "decote_vs_median","anciennete_mois","nb_signaux","cluster_densite",
            "liquidite_cp","bonus_liquidite","bonus_insee",
            "insee_taux_proprio","insee_age_median","insee_revenu_median",
            "date_mutation","nature_mutation","type_local",
        ] if c in dv.columns]
        ren = {
            "rang":"#","adresse_complete":"Adresse","code_postal":"CP","nom_commune":"Commune",
            "signal_label":"Signal","score_final":"Score","score_brut":"Brut",
            "score_confiance":"Confiance","chaleur":"Chaleur","priorite":"Priorite",
            "valeur_fonciere":"Prix(E)","prix_m2":"E/m2","surface_reelle_bati":"Surface(m2)",
            "nombre_pieces_principales":"Pieces","decote_vs_median":"Decote%",
            "anciennete_mois":"Ancien.(mois)","nb_signaux":"#Sig","cluster_densite":"Cluster",
            "liquidite_cp":"Liquidite","bonus_liquidite":"BonusLiq","bonus_insee":"BonusINSEE",
            "insee_taux_proprio":"Proprio%","insee_age_median":"AgeMed",
            "insee_revenu_median":"RevenuMed(E)","date_mutation":"Date",
            "nature_mutation":"Nature","type_local":"Type",
        }
        st.dataframe(
            dv[show_c].rename(columns=ren).head(500),
            use_container_width=True, hide_index=True,
            column_config={
                "Score":    st.column_config.ProgressColumn("Score",    min_value=0, max_value=100, format="%d"),
                "Confiance":st.column_config.ProgressColumn("Confiance",min_value=0, max_value=100, format="%d"),
                "Liquidite":st.column_config.ProgressColumn("Liquidite",min_value=0, max_value=100, format="%.0f"),
                "Prix(E)":  st.column_config.NumberColumn("Prix(E)",   format="%.0f E"),
                "E/m2":     st.column_config.NumberColumn("E/m2",      format="%.0f E"),
            }
        )
        if len(dv) > 500:
            st.caption(f"500/{len(dv):,} affiches.")

    st.markdown("---")
    csv_b = dv.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("Exporter CSV", data=csv_b,
        file_name=f"prospects_{st.session_state.run_dept}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ZONES CHAUDES
# ════════════════════════════════════════════════════════════════════════════
with tab_zones:
    zcol = next((c for c in ["nom_commune","commune","code_postal"] if c in df.columns), None)
    if not zcol or not sig_col or total == 0:
        st.info("Donnees insuffisantes.")
    else:
        st.markdown("<div class='sh'>Ranking zones par potentiel vendeurs</div>", unsafe_allow_html=True)
        agg = {
            "Prospects":  (sig_col, "count"),
            "Score moy.": ("score_final", "mean"),
            "Top 20%":    ("score_final", lambda x: (x >= 80).sum()),
            "P1":         ("priorite", lambda x: (x == "P1 — Contact immediat").sum()) if "priorite" in df.columns else ("score_final", lambda x: (x >= 80).sum()),
        }
        if "cluster_chaud"   in df.columns: agg["Hotspots"]     = ("cluster_chaud",   "sum")
        if "nb_signaux"      in df.columns: agg["Multi-sig"]    = ("nb_signaux",      lambda x: (x >= 2).sum())
        if "liquidite_cp"    in df.columns: agg["Liquidite"]    = ("liquidite_cp",    "mean")
        if "valeur_fonciere" in df.columns: agg["Prix moy.(E)"] = ("valeur_fonciere", "mean")
        if "decote_vs_median"in df.columns: agg["Decote moy.%"] = ("decote_vs_median","mean")
        if "isbn_taux_proprio" in df.columns or "insee_taux_proprio" in df.columns:
            agg["Proprio%"] = ("insee_taux_proprio","mean")

        ranking = df.groupby(zcol).agg(**agg).sort_values("Prospects", ascending=False).reset_index()
        for col_r in ["Score moy.","Liquidite","Decote moy.%","Proprio%"]:
            if col_r in ranking.columns: ranking[col_r] = ranking[col_r].round(1)
        if "Prix moy.(E)" in ranking.columns: ranking["Prix moy.(E)"] = ranking["Prix moy.(E)"].round(0)
        if "Hotspots" in ranking.columns: ranking["Hotspots"] = ranking["Hotspots"].astype(int)
        if "P1" in ranking.columns: ranking["P1"] = ranking["P1"].astype(int)

        mx_p = ranking["Prospects"].max()
        ca2, cb2 = st.columns([1.4, 1])
        with ca2:
            for _, zr in ranking.head(15).iterrows():
                nm  = str(zr[zcol]); cnt = int(zr["Prospects"]); smoy = float(zr["Score moy."])
                top = int(zr.get("Top 20%",0)); p1n = int(zr.get("P1",0))
                hs  = int(zr.get("Hotspots",0)); ms = int(zr.get("Multi-sig",0))
                liq = float(zr.get("Liquidite",0) or 0); dec = float(zr.get("Decote moy.%",0) or 0)
                pct = cnt / mx_p * 100; col = scol(smoy)
                st.markdown(
                    f'<div class="zc"><div class="zch">'
                    f'<div><div class="znm">{nm}</div></div>'
                    f'<div class="zsc" style="color:{col}">{smoy:.0f}</div></div>'
                    f'<div class="zb"><div class="zbf" style="width:{pct:.1f}%;background:{col}"></div></div>'
                    f'<div class="zst">'
                    f'<span><b>{cnt}</b> prospects</span>'
                    f'<span><b>{p1n}</b> P1</span>'
                    f'<span><b>{top}</b> top20%</span>'
                    f'<span><b>{hs}</b> hotspots</span>'
                    f'<span><b>{ms}</b> multi</span>'
                    f'{f"<span>liq.<b>{liq:.0f}</b></span>" if liq else ""}'
                    f'{f"<span>-<b>{dec:.0f}%</b></span>" if dec > 0 else ""}'
                    f'</div></div>', unsafe_allow_html=True
                )
        with cb2:
            st.dataframe(ranking, use_container_width=True, hide_index=True,
                column_config={"Score moy.": st.column_config.ProgressColumn("Score moy.", min_value=0, max_value=100, format="%.1f")})

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SEGMENTS & CTA
# ════════════════════════════════════════════════════════════════════════════
with tab_segs:
    st.markdown("<div class='sh'>Segments vendeurs — message et canal recommandes</div>", unsafe_allow_html=True)
    if sig_col and total > 0:
        for sk in sigs:
            sub = df[df[sig_col] == sk]
            if sub.empty: continue
            cnt  = len(sub); smoy = float(sub["score_final"].mean()) if "score_final" in sub.columns else 0
            top  = int((sub["score_final"] >= 80).sum()) if "score_final" in sub.columns else 0
            p1n  = int((sub.get("priorite", pd.Series("")) == "P1 — Contact immediat").sum())
            ms   = int((sub.get("nb_signaux", pd.Series(1)) >= 2).sum())
            hs   = int(sub.get("cluster_chaud", pd.Series(False)).sum())
            pm2  = float(sub["prix_m2"].median()) if "prix_m2" in sub.columns and sub["prix_m2"].notna().any() else 0
            liq  = float(sub["liquidite_cp"].mean()) if "liquidite_cp" in sub.columns and sub["liquidite_cp"].notna().any() else 0
            tp   = float(sub["insee_taux_proprio"].mean()) if "insee_taux_proprio" in sub.columns and sub["insee_taux_proprio"].notna().any() else 0
            am   = float(sub["insee_age_median"].mean()) if "insee_age_median" in sub.columns and sub["insee_age_median"].notna().any() else 0
            rm   = float(sub["insee_revenu_median"].mean()) if "insee_revenu_median" in sub.columns and sub["insee_revenu_median"].notna().any() else 0
            dec  = float(sub["decote_vs_median"].mean()) if "decote_vs_median" in sub.columns and sub["decote_vs_median"].notna().any() else 0
            col  = SIG_DOTS.get(sk, "#888"); lbl = SIGNAL_LABELS.get(sk, sk)
            msg  = SIGNAL_SEGMENTS.get(sk, ""); cta = SIGNAL_CTA.get(sk, "")
            st.markdown(
                f'<div class="sg">'
                f'<div class="sgh"><div class="sgd" style="background:{col}"></div>'
                f'<div class="sgn">{lbl}</div><div class="sgc">{cnt:,} · {cnt/total*100:.0f}%</div></div>'
                f'<div class="sgm">{msg}</div>'
                f'<div class="sgcta"><b>Canal &amp; message recommandes :</b> {cta}</div>'
                f'<div class="sgs">'
                f'<span>Score moy. <b>{smoy:.0f}</b></span>'
                f'<span>P1 <b>{p1n}</b></span>'
                f'<span>Top 20% <b>{top}</b></span>'
                f'<span>Hotspots <b>{hs}</b></span>'
                f'<span>Multi-sig <b>{ms}</b></span>'
                f'{f"<span>Liq. <b>{liq:.0f}</b></span>" if liq else ""}'
                f'</div>'
                f'<div style="margin-top:.5rem;display:flex;gap:1.2rem;font-size:.62rem;color:var(--inkl)">'
                f'{f"<span>Proprio <b>{tp:.0f}%</b></span>" if tp else ""}'
                f'{f"<span>Age med. <b>{am:.0f} ans</b></span>" if am else ""}'
                f'{f"<span>Revenu med. <b>{rm:,.0f} E</b></span>" if rm else ""}'
                f'{f"<span>E/m2 med. <b>{pm2:,.0f}</b></span>" if pm2 else ""}'
                f'{f"<span>Decote moy. <b>-{dec:.0f}%</b></span>" if dec > 0 else ""}'
                f'</div></div>', unsafe_allow_html=True
            )
        st.markdown("---")
        st.markdown("<div class='sh'>Matrice signal x chaleur</div>", unsafe_allow_html=True)
        if "chaleur" in df.columns:
            ch_order = ["tres chaud","chaud","tiede","froid"]
            mx_tab   = pd.crosstab(df[sig_col].map(SIGNAL_LABELS), df["chaleur"].astype(str))
            for ch in ch_order:
                if ch not in mx_tab.columns: mx_tab[ch] = 0
            st.dataframe(mx_tab[[c for c in ch_order if c in mx_tab.columns]], use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='sh'>Matrice signal x priorite</div>", unsafe_allow_html=True)
        if "priorite" in df.columns:
            p_order = ["P1 — Contact immediat","P2 — Dans les 30j","P3 — Nurturing","P4 — A surveiller"]
            mx_pri  = pd.crosstab(df[sig_col].map(SIGNAL_LABELS), df["priorite"].astype(str))
            for p in p_order:
                if p not in mx_pri.columns: mx_pri[p] = 0
            st.dataframe(mx_pri[[c for c in p_order if c in mx_pri.columns]], use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — MARCHE & TENDANCES
# ════════════════════════════════════════════════════════════════════════════
with tab_mkt:
    st.markdown("<div class='sh'>Donnees de marche — DVF brut (tous biens residentiels)</div>",
                unsafe_allow_html=True)
    if not tendances:
        st.info("Lancez une analyse pour voir les tendances du marche.")
    else:
        ca3, cb3 = st.columns(2)
        with ca3:
            # Volume mensuel
            if "volume_par_mois" in tendances and not tendances["volume_par_mois"].empty:
                tv = tendances["volume_par_mois"]
                vals = tv["nb_transactions"].tolist()
                lbls_v = tv["mois"].tolist()
                mx_v = max(vals) if vals else 1
                peak_m = lbls_v[vals.index(mx_v)] if vals else ""
                low_m  = lbls_v[vals.index(min(vals))] if vals else ""
                chart_html = mini_bar_chart(vals, lbls_v, "#f5a05a")
                content = (
                    f'{chart_html}'
                    f'<div style="font-size:.66rem;color:var(--inkl);margin-top:.3rem">'
                    f'Pic : <b style="color:var(--ink)">{peak_m}</b> ({mx_v:,}) · '
                    f'Creux : <b style="color:var(--ink)">{low_m}</b> ({min(vals):,})'
                    f'</div>'
                )
                st.markdown(section_card("Volume mensuel de transactions", content), unsafe_allow_html=True)
                with st.expander("Voir le tableau"):
                    st.dataframe(tv.rename(columns={"mois":"Mois","nb_transactions":"Transactions"}),
                                 use_container_width=True, hide_index=True)

            # Volume par type de bien
            if "volume_par_type" in tendances and not tendances["volume_par_type"].empty:
                vt = tendances["volume_par_type"]
                mx_t = int(vt["nb"].max())
                rows = "".join(
                    hbar(str(r["type_local"])[:18], int(r["nb"]), mx_t, "#1a4a8a", "{:,.0f}")
                    for _, r in vt.iterrows()
                )
                st.markdown(section_card("Volume par type de bien", rows), unsafe_allow_html=True)

        with cb3:
            # Prix median mensuel
            if "prix_median_par_mois" in tendances and not tendances["prix_median_par_mois"].empty:
                tp2 = tendances["prix_median_par_mois"]
                vals2 = tp2["prix_median"].tolist()
                lbls2 = tp2["mois"].tolist()
                chart2 = mini_bar_chart(vals2, lbls2, "#1a4a8a")
                if len(vals2) > 1:
                    delta = vals2[-1] - vals2[0]
                    sign  = "+" if delta > 0 else ""
                    base  = vals2[0]
                    pct_d = delta / base * 100 if base > 0 else 0
                    clr2  = "var(--grn)" if delta > 0 else "var(--red)"
                    evo   = f'<div style="font-size:.66rem;color:var(--inkl);margin-top:.3rem">Evolution : <b style="color:{clr2}">{sign}{delta:,.0f} E ({pct_d:.1f}%)</b></div>'
                else:
                    evo = ""
                content2 = chart2 + evo
                st.markdown(section_card("Evolution du prix median", content2), unsafe_allow_html=True)
                with st.expander("Voir le tableau"):
                    st.dataframe(tp2.rename(columns={"mois":"Mois","prix_median":"Prix median (E)"}),
                                 use_container_width=True, hide_index=True)

            # Evolution trimestrielle
            if "prix_par_trimestre" in tendances and not tendances["prix_par_trimestre"].empty:
                qt = tendances["prix_par_trimestre"]
                vals_q = qt["prix_median"].tolist()
                lbls_q = qt["trimestre"].tolist()
                chart_q = mini_bar_chart(vals_q, lbls_q, "#7a4aa0")
                st.markdown(section_card("Prix median par trimestre", chart_q), unsafe_allow_html=True)

        st.markdown("---")
        ca4, cb4 = st.columns(2)
        with ca4:
            # Top CP volume
            if "top_cp_volume" in tendances and not tendances["top_cp_volume"].empty:
                tv2 = tendances["top_cp_volume"]
                mx4 = int(tv2["volume"].max())
                rows4 = "".join(
                    hbar(str(r["code_postal"]), int(r["volume"]), mx4, "#f5a05a", "{:,.0f}")
                    for _, r in tv2.iterrows()
                )
                st.markdown(section_card("Top 15 CP par volume de ventes", rows4), unsafe_allow_html=True)
            # Prix/m2 par nombre de pieces
            if "pm2_par_pieces" in tendances and not tendances["pm2_par_pieces"].empty:
                pp = tendances["pm2_par_pieces"]
                mx_pp = float(pp["pm2_med"].max())
                rows_pp = "".join(
                    hbar(f"T{int(r['nombre_pieces_principales'])}", float(r["pm2_med"]), mx_pp, "#1a6b4a", "{:,.0f} E/m2")
                    for _, r in pp.iterrows()
                )
                st.markdown(section_card("Prix/m2 median par type (T1-T6)", rows_pp), unsafe_allow_html=True)

        with cb4:
            # Top CP prix/m2
            if "prix_m2_par_cp" in tendances and not tendances["prix_m2_par_cp"].empty:
                tp3 = tendances["prix_m2_par_cp"].head(15)
                mx5 = float(tp3["prix_m2_median"].max())
                rows5 = "".join(
                    hbar(str(r["code_postal"]), float(r["prix_m2_median"]), mx5, "#1a4a8a", "{:,.0f} E/m2")
                    for _, r in tp3.iterrows()
                )
                st.markdown(section_card("Prix/m2 median par CP (top 15)", rows5), unsafe_allow_html=True)

        # INSEE
        if "insee_taux_proprio" in df.columns and df["insee_taux_proprio"].notna().any():
            st.markdown("---")
            st.markdown("<div class='sh'>Contexte INSEE — moyenne sur les prospects filtres</div>",
                        unsafe_allow_html=True)
            tp_a = float(df["insee_taux_proprio"].mean())
            am_a = float(df["insee_age_median"].mean()) if "insee_age_median" in df.columns else 0
            rm_a = float(df["insee_revenu_median"].mean()) if "insee_revenu_median" in df.columns else 0
            ci = st.columns(3)
            interps = [
                (f"{tp_a:.1f}%", "Taux proprietaires",
                 "Zone majoritairement proprietaire — signaux vendeurs credibles" if tp_a > 60 else "Zone locataire majoritaire — signaux a nuancer"),
                (f"{am_a:.1f} ans", "Age median",
                 "Zone senior — signal retraite tres renforce" if am_a > 55 else ("Zone mixte" if am_a > 42 else "Zone jeune — signal upgrade/primo renforce")),
                (f"{rm_a:,.0f} E", "Revenu median/UC",
                 "Pouvoir d'achat eleve — upgrade probable" if rm_a > 28000 else "Primo-accedants potentiels — signal primo fiable"),
            ]
            for col_i, (val, lbl, sub) in zip(ci, interps):
                with col_i:
                    st.markdown(
                        f'<div class="ic"><div class="icv">{val}</div>'
                        f'<div class="icl">{lbl}</div>'
                        f'<div class="ics">{sub}</div></div>',
                        unsafe_allow_html=True
                    )

        # Liquidite prospects
        if "liquidite_cp" in df.columns and total > 0:
            st.markdown("---")
            st.markdown("<div class='sh'>Liquidite marche — repartition des prospects filtres</div>",
                        unsafe_allow_html=True)
            liq_bins  = [0, 25, 50, 75, 100]
            liq_lbls  = ["Faible (<25)","Moderee (25-50)","Bonne (50-75)","Tres bonne (>75)"]
            liq_clrs  = ["#b8b5ae","#f5a05a","#1a4a8a","#1a6b4a"]
            liq_dist  = pd.cut(df["liquidite_cp"], bins=liq_bins, labels=liq_lbls, right=True).value_counts().reindex(liq_lbls).fillna(0)
            tot_liq   = liq_dist.sum()
            bar_html  = "".join(
                f'<div class="db-s" style="flex:{n/tot_liq*100:.1f};background:{clr}" title="{l}:{int(n)}">{n/tot_liq*100:.0f}%</div>'
                for (l, n), clr in zip(liq_dist.items(), liq_clrs) if n > 0
            )
            st.markdown(f'<div class="dist-bar">{bar_html}</div>', unsafe_allow_html=True)
            lc = st.columns(4)
            for col_m, (lbl, cnt), clr in zip(lc, liq_dist.items(), liq_clrs):
                with col_m:
                    st.markdown(
                        f'<div style="text-align:center;padding:.4rem;background:var(--p2);border-radius:8px;border:1px solid var(--brd)">'
                        f'<div style="font-family:Fraunces,serif;font-size:1.05rem;font-weight:700;color:{clr}">{int(cnt):,}</div>'
                        f'<div style="font-size:.58rem;color:var(--inkf);text-transform:uppercase;letter-spacing:.05em">{lbl}</div>'
                        f'</div>', unsafe_allow_html=True
                    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — CARTE
# ════════════════════════════════════════════════════════════════════════════
with tab_carte:
    has_c = "latitude" in df.columns and "longitude" in df.columns
    df_map = df[df["latitude"].notna() & df["longitude"].notna() & (df["latitude"] != 0)] if has_c else pd.DataFrame()
    if not has_c or df_map.empty:
        st.info("Coordonnees GPS absentes dans ce fichier DVF.")
    else:
        st.markdown(
            f"<div class='sh'>{len(df_map):,} prospects geocodifies · taille=score · couleur=signal · contour epais=hotspot</div>",
            unsafe_allow_html=True
        )
        SC_JS = json.dumps(SIGNAL_COLORS)
        SL_JS = json.dumps(SIGNAL_LABELS)
        pts = []
        for _, row in df_map.iterrows():
            lat = float(row.get("latitude", 0) or 0)
            lng = float(row.get("longitude", 0) or 0)
            if lat == 0 and lng == 0: continue
            pts.append({
                "lat": lat, "lng": lng,
                "sig": str(row.get(sig_col, "") or ""),
                "sf":  float(row.get("score_final", 50) or 50),
                "sb":  float(row.get("score_brut",  50) or 50),
                "sco": float(row.get("score_confiance", 50) or 50),
                "adr": str(row.get("adresse_complete","") or ""),
                "cp":  str(row.get("code_postal","") or ""),
                "cm":  str(row.get("nom_commune", row.get("commune","")) or ""),
                "px":  float(row.get("valeur_fonciere",0) or 0),
                "sf2": float(row.get("surface_reelle_bati",0) or 0),
                "pm2": float(row.get("prix_m2",0) or 0),
                "dec": float(row.get("decote_vs_median",0) or 0),
                "dt":  str(row.get("date_mutation",""))[:10],
                "ch":  str(row.get("chaleur","") or ""),
                "pri": str(row.get("priorite","") or ""),
                "nb":  int(row.get("nb_signaux",1) or 1),
                "cls": bool(row.get("cluster_chaud",False)),
                "cdn": int(row.get("cluster_densite",0) or 0),
                "liq": float(row.get("liquidite_cp",0) or 0),
                "tp":  float(row.get("insee_taux_proprio",0) or 0),
                "am":  float(row.get("insee_age_median",0) or 0),
                "anc": float(row.get("anciennete_mois",0) or 0),
            })
        clat = float(df_map["latitude"].median())
        clng = float(df_map["longitude"].median())
        djs  = json.dumps(pts)

        map_html = (
            "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
            "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css'>"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js'></script>"
            "<style>"
            "*{box-sizing:border-box;margin:0;padding:0}"
            "body{font-family:monospace;background:#f5f2ed}"
            "#map{width:100%;height:590px}"
            ".lp{font-family:monospace;font-size:.74rem;line-height:1.6;min-width:260px}"
            ".lp .b{display:inline-block;padding:.1rem .4rem;border-radius:3px;color:#fff;font-size:.6rem;font-weight:600;margin-bottom:.3rem}"
            ".lp .s{font-size:1.4rem;font-weight:700;line-height:1;margin-bottom:.12rem}"
            ".lp .su{font-size:.62rem;color:#6b6860;margin-bottom:.3rem}"
            ".lp .a{font-size:.72rem;font-weight:600;margin-bottom:.28rem;color:#1a1814}"
            ".lp table{width:100%;border-collapse:collapse;font-size:.66rem}"
            ".lp td{padding:.1rem 0;border-bottom:1px solid rgba(0,0,0,.06)}"
            ".lp td:first-child{color:#6b6860}.lp td:last-child{font-weight:500;text-align:right}"
            ".cls-t{display:inline-block;background:rgba(26,107,74,.12);border:1px solid rgba(26,107,74,.25);color:#1a6b4a;font-size:.58rem;padding:.08rem .32rem;border-radius:3px;margin-top:.28rem}"
            ".mul-t{display:inline-block;background:rgba(245,160,90,.12);border:1px solid rgba(245,160,90,.25);color:#b07030;font-size:.58rem;padding:.08rem .32rem;border-radius:3px;margin-top:.28rem;margin-left:.2rem}"
            ".pri-t{display:inline-block;background:rgba(196,68,10,.09);border:1px solid rgba(196,68,10,.18);color:#c4440a;font-size:.58rem;padding:.08rem .32rem;border-radius:3px;margin-top:.28rem;margin-left:.2rem}"
            "</style></head><body><div id='map'></div>"
            "<script>"
            f"const D={djs},C={SC_JS},L={SL_JS};"
            f"const map=L.map('map').setView([{clat},{clng}],11);"
            "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'(c) OpenStreetMap',maxZoom:19}).addTo(map);"
            "const ly=L.layerGroup().addTo(map);"
            "D.forEach(pt=>{"
            "const col=C[pt.sig]||'#888';"
            "const r=4+(pt.sf/100)*10;"
            "const sc=pt.sf>=80?'#c4440a':pt.sf>=60?'#d4850a':pt.sf>=40?'#1a4a8a':'#b8b5ae';"
            "const prix=pt.px?pt.px.toLocaleString('fr-FR')+' E':'--';"
            "const surf=pt.sf2?pt.sf2+' m2':'--';"
            "const pm2=pt.pm2?pt.pm2.toLocaleString('fr-FR')+' E/m2':'--';"
            "const dec=pt.dec?'-'+Math.abs(pt.dec).toFixed(0)+'% med.':'--';"
            "const cls=pt.cls?`<span class='cls-t'>hotspot ${pt.cdn}</span>`:'';"
            "const mul=pt.nb>=2?`<span class='mul-t'>+${pt.nb} signaux</span>`:'';"
            "const pri=pt.pri?`<span class='pri-t'>${pt.pri.substring(0,2)}</span>`:'';"
            "const pop=`<div class='lp'>"
            "<div class='b' style='background:${col}'>${L[pt.sig]||pt.sig}</div>"
            "<div class='s' style='color:${sc}'>${pt.sf.toFixed(0)}<span style='font-size:.65rem;color:#6b6860'>/100</span></div>"
            "<div class='su'>brut ${pt.sb.toFixed(0)} &middot; confiance ${pt.sco.toFixed(0)} &middot; ${pt.ch} &middot; liq.${pt.liq.toFixed(0)}</div>"
            "<div class='a'>${pt.adr||'--'}</div>"
            "<table>"
            "<tr><td>Commune</td><td>${pt.cm} ${pt.cp}</td></tr>"
            "<tr><td>Prix</td><td>${prix}</td></tr>"
            "<tr><td>Surface</td><td>${surf}</td></tr>"
            "<tr><td>Prix/m2</td><td>${pm2}</td></tr>"
            "<tr><td>Decote</td><td>${dec}</td></tr>"
            "<tr><td>Date</td><td>${pt.dt}</td></tr>"
            "<tr><td>Anciennete</td><td>${pt.anc.toFixed(0)} mois</td></tr>"
            "${pt.tp?`<tr><td>Proprio%</td><td>${pt.tp.toFixed(0)}%</td></tr>`:''}"
            "${pt.am?`<tr><td>Age med.</td><td>${pt.am.toFixed(0)} ans</td></tr>`:''}"
            "</table>"
            "${cls}${mul}${pri}"
            "</div>`;"
            "L.circleMarker([pt.lat,pt.lng],{"
            "radius:r,color:col,fillColor:col,fillOpacity:.65,"
            "weight:pt.cls?3:pt.nb>=2?2:1.2,opacity:.9"
            "}).bindPopup(pop)"
            ".bindTooltip(`${L[pt.sig]||pt.sig} ${pt.sf.toFixed(0)}${pt.cls?' hotspot':''}${pt.nb>=2?' +multi':''}`,{direction:'top',offset:[0,-4]})"
            ".addTo(ly);"
            "});"
            "if(D.length>0){"
            "const la=D.map(d=>d.lat),lo=D.map(d=>d.lng);"
            "map.fitBounds([[Math.min(...la),Math.min(...lo)],[Math.max(...la),Math.max(...lo)]],{padding:[30,30]});}"
            "</script></body></html>"
        )
        st.components.v1.html(map_html, height=600, scrolling=False)

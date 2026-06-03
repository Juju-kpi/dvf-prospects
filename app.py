"""
DVF × BODACC — App v4.0
Liquidité · INSEE · Clusters · Cadastre · Tendances marché
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

from pipeline import run_pipeline, SIGNAL_LABELS, SIGNAL_COLORS, SIGNAL_SEGMENTS

st.set_page_config(page_title="DVF BODACC — Prospection v4", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,500;0,700;1,300&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
:root{--ink:#1a1814;--ink-light:#6b6860;--ink-faint:#b8b5ae;--paper:#f5f2ed;--paper-2:#ede9e2;--paper-3:#e0dbd2;--accent:#f5a05a;--red:#c4440a;--green:#1a6b4a;--blue:#1a4a8a;--purple:#7a4aa0;--border:rgba(26,24,20,.12);--r:10px;--sb:#1a1814}
html,body,[class*="css"]{font-family:'DM Mono',monospace!important;color:var(--ink)}
h1,h2,h3{font-family:'Fraunces',serif!important;letter-spacing:-.02em}
.stApp{background:var(--paper)!important}.stApp>header{background:var(--ink)!important}
.main .block-container{padding-top:0!important;padding-bottom:3rem!important;max-width:1500px!important}
[data-testid="stSidebar"]{background:var(--sb)!important;border-right:none!important}
[data-testid="stSidebar"]>div:first-child{background:var(--sb)!important;padding:1.25rem 1rem!important}
[data-testid="stSidebar"] *{color:#f5f2ed!important}
[data-testid="stSidebar"] .stSelectbox label,[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label{color:rgba(245,242,237,.45)!important;font-size:.62rem!important;text-transform:uppercase!important;letter-spacing:.09em!important}
[data-testid="stSidebar"] .stSelectbox>div>div,[data-testid="stSidebar"] .stMultiSelect>div>div{background:rgba(255,255,255,.08)!important;border:1px solid rgba(255,255,255,.15)!important;border-radius:6px!important}
[data-testid="stSidebar"] .stSelectbox svg{fill:rgba(245,242,237,.45)!important}
[data-testid="stSidebar"] hr{border-color:rgba(245,242,237,.1)!important;margin:.6rem 0!important}
[data-testid="stSidebar"] .stButton>button{background:var(--accent)!important;color:var(--ink)!important;border:none!important;font-family:'DM Mono',monospace!important;font-weight:500!important;font-size:.75rem!important;width:100%!important;border-radius:6px!important;padding:.55rem 1rem!important}
[data-testid="stSidebar"] .stButton>button:hover{opacity:.85!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;gap:0!important}
[data-testid="stTabs"] [data-baseweb="tab"]{background:transparent!important;border:none!important;color:var(--ink-light)!important;font-family:'DM Mono',monospace!important;font-size:.68rem!important;letter-spacing:.05em!important;text-transform:uppercase!important;padding:.6rem 1rem!important;border-bottom:2px solid transparent!important}
[data-testid="stTabs"] [data-baseweb="tab"]:hover{color:var(--ink)!important}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--ink)!important;border-bottom:2px solid var(--red)!important}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{display:none!important}
[data-testid="stTabs"] [data-baseweb="tab-panel"]{padding-top:1.25rem!important}
[data-testid="metric-container"]{background:var(--paper-2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;padding:.8rem 1rem!important}
[data-testid="stMetricLabel"]{font-size:.58rem!important;text-transform:uppercase!important;letter-spacing:.08em!important;color:var(--ink-light)!important}
[data-testid="stMetricValue"]{font-family:'Fraunces',serif!important;font-size:1.45rem!important;font-weight:700!important;color:var(--ink)!important;line-height:1.1!important}
[data-testid="stDataFrame"]{border-radius:var(--r)!important;border:1px solid var(--border)!important;overflow:hidden!important}
.stDataFrame td,.stDataFrame th{font-size:.71rem!important;font-family:'DM Mono',monospace!important}
.stSuccess{background:rgba(26,107,74,.08)!important;border:1px solid rgba(26,107,74,.2)!important;border-radius:var(--r)!important}
.stError{background:rgba(196,68,10,.08)!important;border:1px solid rgba(196,68,10,.2)!important;border-radius:var(--r)!important}
.stInfo{background:var(--paper-2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important}
.stSpinner>div{border-top-color:var(--accent)!important}
[data-testid="stDownloadButton"]>button{background:var(--ink)!important;color:var(--paper)!important;border:none!important;font-family:'DM Mono',monospace!important;font-size:.7rem!important;border-radius:6px!important;padding:.42rem .9rem!important}
[data-testid="stDownloadButton"]>button:hover{opacity:.85!important}
.ph{background:var(--ink);color:var(--paper);margin:0 -3rem 1.75rem -3rem;padding:1.35rem 3rem;display:flex;align-items:center;justify-content:space-between;gap:1rem}
.ph .logo{font-family:'Fraunces',serif;font-size:1.35rem;font-weight:700;letter-spacing:-.03em;color:var(--paper)}
.ph .logo span{color:#f5a05a}
.ph .tagline{font-size:.6rem;color:rgba(245,242,237,.32);letter-spacing:.1em;text-transform:uppercase;margin-top:.15rem}
.ph .badge{font-size:.6rem;background:rgba(245,160,90,.15);color:#f5a05a;border:1px solid rgba(245,160,90,.3);padding:.28rem .7rem;border-radius:20px;letter-spacing:.06em;white-space:nowrap}
.sh{font-size:.59rem;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-faint);padding-bottom:.4rem;border-bottom:1px solid var(--border);margin-bottom:.9rem}
.sh-d{font-size:.59rem;text-transform:uppercase;letter-spacing:.1em;color:rgba(245,242,237,.28);padding-bottom:.4rem;border-bottom:1px solid rgba(245,242,237,.1);margin-bottom:.8rem}
.pr{display:flex;align-items:flex-start;gap:.9rem;padding:.9rem 1rem;border:1px solid var(--border);border-radius:var(--r);margin-bottom:.5rem;background:var(--paper);transition:box-shadow .15s}
.pr:hover{box-shadow:0 2px 12px rgba(26,24,20,.07)}
.pr-sc{text-align:center;min-width:50px}.pr-sc .v{font-family:'Fraunces',serif;font-size:1.45rem;font-weight:700;line-height:1}.pr-sc .l{font-size:.52rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-faint)}
.pr-bd{flex:1;min-width:0}.pr-addr{font-weight:500;font-size:.8rem;color:var(--ink);margin-bottom:.18rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pr-meta{font-size:.68rem;color:var(--ink-light);line-height:1.7}.pr-meta span{margin-right:.65rem}
.pr-tags{display:flex;flex-wrap:wrap;gap:.28rem;margin-top:.35rem}
.tag{font-size:.58rem;padding:.1rem .38rem;border-radius:4px;font-weight:500}
.t-sig{background:var(--paper-2);border:1px solid var(--border);color:var(--ink-light)}
.t-hot{background:rgba(196,68,10,.09);border:1px solid rgba(196,68,10,.18);color:var(--red)}
.t-warm{background:rgba(245,160,90,.1);border:1px solid rgba(245,160,90,.22);color:#b07030}
.t-cool{background:rgba(26,74,138,.08);border:1px solid rgba(26,74,138,.15);color:var(--blue)}
.t-cold{background:var(--paper-3);border:1px solid var(--border);color:var(--ink-faint)}
.t-cluster{background:rgba(26,107,74,.09);border:1px solid rgba(26,107,74,.2);color:var(--green)}
.t-multi{background:rgba(245,160,90,.12);border:1px solid rgba(245,160,90,.25);color:#c47820}
.pr-rt{text-align:right;min-width:88px}.pr-prix{font-family:'Fraunces',serif;font-size:.95rem;font-weight:600;color:var(--ink)}.pr-surf{font-size:.65rem;color:var(--ink-faint);margin-top:.12rem}.pr-dec{font-size:.63rem;color:var(--red);font-weight:600}
.zc{background:var(--paper);border:1px solid var(--border);border-radius:var(--r);padding:.9rem 1.15rem;margin-bottom:.45rem}
.zc-h{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.5rem}
.zc-nm{font-weight:600;font-size:.82rem;color:var(--ink)}.zc-sc{font-family:'Fraunces',serif;font-size:1.25rem;font-weight:700}
.zbar{height:4px;background:var(--paper-3);border-radius:2px;margin:.4rem 0}
.zbar-f{height:100%;border-radius:2px}
.zst{display:flex;gap:1.25rem;font-size:.66rem;color:var(--ink-light)}.zst b{color:var(--ink)}
.seg-c{background:var(--paper);border:1px solid var(--border);border-radius:var(--r);padding:1rem 1.2rem;margin-bottom:.6rem}
.seg-h{display:flex;align-items:center;gap:.6rem;margin-bottom:.45rem}
.sd{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.seg-nm{font-weight:600;font-size:.8rem;color:var(--ink)}.seg-cnt{font-size:.66rem;color:var(--ink-faint);margin-left:auto}
.seg-msg{font-family:'Lora',serif;font-size:.8rem;color:var(--ink-light);line-height:1.65;margin-bottom:.45rem}
.seg-st{display:flex;gap:1.4rem;font-size:.66rem;color:var(--ink-light)}.seg-st b{color:var(--ink)}
.insee-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin:.75rem 0}
.insee-card{background:var(--paper-2);border:1px solid var(--border);border-radius:8px;padding:.85rem 1rem}
.ic-val{font-family:'Fraunces',serif;font-size:1.4rem;font-weight:700;color:var(--ink);line-height:1}
.ic-lbl{font-size:.6rem;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-faint);margin-top:.2rem}
.ic-sub{font-size:.67rem;color:var(--ink-light);margin-top:.2rem;font-family:'Lora',serif}
.liq-bar{display:flex;height:18px;border-radius:5px;overflow:hidden;margin:.5rem 0}
.liq-seg{display:flex;align-items:center;justify-content:center;font-size:.58rem;font-weight:600;color:#fff}
.trend-card{background:var(--paper);border:1px solid var(--border);border-radius:var(--r);padding:1.1rem 1.25rem;margin-bottom:.65rem}
.trend-card h4{font-family:'Fraunces',serif!important;font-size:1.05rem!important;font-weight:600!important;color:var(--ink)!important;margin-bottom:.6rem!important}
.mini-chart{display:flex;align-items:flex-end;gap:3px;height:60px;padding:.25rem 0}
.mc-bar{min-width:8px;border-radius:2px 2px 0 0;background:var(--accent);opacity:.7;flex:1;transition:opacity .15s}
.mc-bar:hover{opacity:1}
.mc-bar.hot{background:var(--red)}
.info-box{background:var(--paper-2);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:var(--r);padding:.9rem 1.15rem;margin-bottom:1.1rem;font-size:.76rem;line-height:1.8;color:var(--ink-light)}
.info-box b{color:var(--ink)}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1rem 0!important}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────
for k,v in {"prospects":None,"tendances":{},"run_dept":None,"run_annee":None}.items():
    if k not in st.session_state: st.session_state[k]=v

DEPT_LABELS={
    "75":"75 — Paris","69":"69 — Rhône","13":"13 — Bouches-du-Rhône",
    "33":"33 — Gironde","31":"31 — Haute-Garonne","06":"06 — Alpes-Maritimes",
    "59":"59 — Nord","67":"67 — Bas-Rhin","44":"44 — Loire-Atlantique",
    "34":"34 — Hérault","76":"76 — Seine-Maritime","38":"38 — Isère",
    "92":"92 — Hauts-de-Seine","93":"93 — Seine-Saint-Denis","94":"94 — Val-de-Marne",
}

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='margin-bottom:1.3rem'>
  <div style='font-family:"Fraunces",serif;font-size:1.12rem;font-weight:700;letter-spacing:-.02em;color:#f5f2ed'>DVF <span style='color:#f5a05a'>×</span> BODACC</div>
  <div style='font-size:.56rem;color:rgba(245,242,237,.28);text-transform:uppercase;letter-spacing:.1em;margin-top:.12rem'>Prospection immobilière v4</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='sh-d'>Paramètres</div>", unsafe_allow_html=True)
    dept  = st.selectbox("Département", list(DEPT_LABELS.keys()), format_func=lambda x: DEPT_LABELS[x])
    annee = st.selectbox("Année DVF", [2024,2023,2022,2021])
    fenetre = st.select_slider("Fenêtre succession", [9,12,18,24], value=18,
        help="Mois après BODACC pour chercher une vente liée\n9=fort · 18=std · 24=large")
    enrichir_cadastre = st.checkbox("Enrichissement cadastre (lent)", value=False,
        help="Appels API Cadastre pour récupérer l'année de construction. Ajoute ~2 min.")
    rayon_cluster = st.select_slider("Rayon cluster (km)", [0.25,0.5,1.0,2.0], value=0.5,
        help="Rayon de détection des hotspots géographiques")

    st.markdown("---")
    st.markdown("<div class='sh-d'>Filtres résultats</div>", unsafe_allow_html=True)
    score_min     = st.slider("Score minimum", 0, 90, 60, 5,
                               help="Score percentile · 60=top 40% · 80=top 20%")
    signaux_choix = st.multiselect("Signaux", list(SIGNAL_LABELS.keys()),
                                   default=list(SIGNAL_LABELS.keys()),
                                   format_func=lambda x: SIGNAL_LABELS[x])
    chaleurs_choix= st.multiselect("Chaleur CRM", ["très chaud","chaud","tiède","froid"],
                                   default=["très chaud","chaud","tiède"])
    clusters_only = st.checkbox("Hotspots seulement", value=False,
                                 help="Affiche uniquement les prospects dans un cluster géographique")

    st.markdown("---")
    run_clicked = st.button("→ Lancer l'analyse", type="primary")

    st.markdown("""<div style='margin-top:1.4rem;font-size:.56rem;color:rgba(245,242,237,.22);line-height:2;font-family:"DM Mono",monospace'>
Sources · DVF data.gouv.fr<br>BODACC OpenDataSoft<br>INSEE communes<br>API BAN / Cadastre<br><br>
Score = percentile dept.<br>80 = top 20% · 95 = top 5%<br><br>⚠ RGPD — usage geo/segment</div>""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""<div class="ph">
  <div><div class="logo">DVF <span>×</span> BODACC</div>
  <div class="tagline">Prospection immobilière · Score percentile · Liquidité · INSEE · Clusters</div></div>
  <div class="badge">v4.0 · Scoring contextuel</div>
</div>""", unsafe_allow_html=True)

# ── Lancement ────────────────────────────────────────────────
if run_clicked:
    with st.spinner(f"Pipeline — {DEPT_LABELS[dept]} / {annee} · fenêtre {fenetre} mois…"):
        try:
            prospects, tendances = run_pipeline(
                dept=dept, annee=annee,
                fenetre_succession_mois=fenetre,
                enrichir_cadastre=enrichir_cadastre,
                rayon_cluster_km=rayon_cluster,
            )
            st.session_state.prospects  = prospects
            st.session_state.tendances  = tendances
            st.session_state.run_dept   = dept
            st.session_state.run_annee  = annee
            hot = (prospects["score_final"]>=80).sum() if "score_final" in prospects.columns else 0
            cls = (prospects.get("cluster_chaud", pd.Series(False))).sum()
            st.success(f"✓ {len(prospects):,} prospects — {hot:,} top 20% — {cls:,} dans un hotspot")
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()

# ── Tabs ─────────────────────────────────────────────────────
tabs = st.tabs(["📊 Dashboard","🎯 Prospects","🗺 Zones","🏷 Segments","📈 Marché","📍 Carte","📖 Doc"])
tab_dash,tab_pros,tab_zones,tab_seg,tab_market,tab_carte,tab_doc = tabs

df_raw  = st.session_state.prospects
tendances = st.session_state.tendances

# ── Vide ─────────────────────────────────────────────────────
if df_raw is None:
    with tab_dash:
        st.markdown("""<div class="info-box"><b>DVF × BODACC v4 — Outil de prospection contextuel</b><br><br>
<b>Nouveau en v4 :</b><br>
• <b>Score liquidité</b> — le signal est pondéré par l'activité réelle du marché local (CP)<br>
• <b>Données INSEE</b> — taux propriétaires, âge médian, revenu médian enrichissent le contexte<br>
• <b>Clusters géographiques</b> — détection des hotspots (zones à forte concentration de signaux)<br>
• <b>Cadastre</b> — enrichissement optionnel avec l'année de construction<br>
• <b>Tendances marché</b> — volume mensuel, saisonnalité, évolution prix/m²<br><br>
Sélectionnez un département et cliquez <b>→ Lancer l'analyse</b>.</div>""", unsafe_allow_html=True)
    st.stop()

# ── Filtrage ─────────────────────────────────────────────────
def get_sig_col(d):
    return "signal_carte" if "signal_carte" in d.columns else ("signal" if "signal" in d.columns else None)

def filtrer(d):
    d = d.copy()
    sc = get_sig_col(d)
    if sc: d = d[d[sc].isin(signaux_choix)]
    if "score_final" in d.columns: d = d[d["score_final"] >= score_min]
    if "chaleur" in d.columns and chaleurs_choix: d = d[d["chaleur"].astype(str).isin(chaleurs_choix)]
    if clusters_only and "cluster_chaud" in d.columns: d = d[d["cluster_chaud"]==True]
    return d

df      = filtrer(df_raw)
sig_col = get_sig_col(df)
total   = len(df)

def sc(s): return "#c4440a" if s>=80 else "#d4850a" if s>=60 else "#1a4a8a" if s>=40 else "#b8b5ae"
def ch_tag(ch):
    ch=str(ch)
    if "très" in ch: return "t-hot","🔴 "+ch
    if "chaud" in ch: return "t-warm","🟠 "+ch
    if "tiède" in ch: return "t-cool","🟡 "+ch
    return "t-cold","⚪ "+ch

def render_prospect(row, show_seg=False):
    sf   = float(row.get("score_final",0))
    addr = str(row.get("adresse_complete","—"))
    ch   = str(row.get("chaleur",""))
    sig  = str(row.get("signal_label",row.get("signal_carte","")))
    prix = float(row.get("valeur_fonciere",0) or 0)
    surf = float(row.get("surface_reelle_bati",0) or 0)
    pm2  = float(row.get("prix_m2",0) or 0)
    dec  = float(row.get("decote_vs_median",0) or 0)
    nb   = int(row.get("nb_signaux",1) or 1)
    date = str(row.get("date_mutation",""))[:10]
    comm = str(row.get("nom_commune",row.get("commune","")))
    cp   = str(row.get("code_postal",""))
    liq  = float(row.get("liquidite_cp",0) or 0)
    cls  = bool(row.get("cluster_chaud",False))
    cdn  = int(row.get("cluster_densite",0) or 0)
    seg  = str(row.get("segment_cible",""))
    tc,tl = ch_tag(ch)
    multi_tag = f'<span class="tag t-multi">⚡ {nb} signaux</span>' if nb>=2 else ""
    cluster_tag = f'<span class="tag t-cluster">🔥 cluster {cdn}</span>' if cls else ""
    liq_badge = f'<span class="tag t-cool">💧 liq.{liq:.0f}</span>' if liq>0 else ""
    seg_div = f'<div style="font-size:.66rem;color:var(--ink-faint);margin-top:.3rem;font-family:Lora,serif">💬 {seg}</div>' if show_seg and seg and seg!="—" else ""
    return f"""<div class="pr">
  <div class="pr-sc"><div class="v" style="color:{sc(sf)}">{sf:.0f}</div><div class="l">score</div></div>
  <div class="pr-bd">
    <div class="pr-addr">{addr}</div>
    <div class="pr-meta"><span>📍 {comm} {cp}</span><span>📅 {date}</span><span>🏠 {surf:.0f} m²</span></div>
    <div class="pr-tags"><span class="tag t-sig">{sig}</span><span class="tag {tc}">{tl}</span>{multi_tag}{cluster_tag}{liq_badge}</div>
    {seg_div}
  </div>
  <div class="pr-rt"><div class="pr-prix">{prix:,.0f} €</div><div class="pr-surf">{pm2:,.0f} €/m²</div>{'<div class="pr-dec">−'+f'{abs(dec):.0f}% médiane</div>' if dec>0 else ''}</div>
</div>"""

# ════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown(f"<div class='sh'>{DEPT_LABELS.get(st.session_state.run_dept,'')} / {st.session_state.run_annee}</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    hot   = (df["score_final"]>=80).sum() if "score_final" in df.columns else 0
    multi = (df["nb_signaux"]>=2).sum() if "nb_signaux" in df.columns else 0
    cls   = df.get("cluster_chaud",pd.Series(False)).sum()
    avg   = df["score_final"].mean() if "score_final" in df.columns and total else 0
    zones = df["nom_commune"].nunique() if "nom_commune" in df.columns else 0
    avg_liq = df["liquidite_cp"].mean() if "liquidite_cp" in df.columns and df["liquidite_cp"].notna().any() else 0
    with c1: st.metric("Prospects",f"{total:,}")
    with c2: st.metric("Score moyen",f"{avg:.0f}/100")
    with c3: st.metric("Top 20% (≥80)",f"{hot:,}")
    with c4: st.metric("Hotspots",f"{int(cls):,}")
    with c5: st.metric("Multi-signal",f"{multi:,}")
    with c6: st.metric("Communes",f"{zones:,}")
    with c7: st.metric("Liquidité moy.",f"{avg_liq:.0f}/100")

    st.markdown("---")
    col_a,col_b = st.columns([1.3,1])

    with col_a:
        st.markdown("<div class='sh'>Distribution des scores</div>", unsafe_allow_html=True)
        if "score_final" in df.columns and total>0:
            bins=[0,20,40,60,80,100]; lbls=["0–20","20–40","40–60","60–80","80–100"]
            clrs=["#e0dbd2","#c8c2b8","#f5a05a","#c4440a80","#c4440a"]
            counts=pd.cut(df["score_final"],bins=bins,labels=lbls,right=True).value_counts().reindex(lbls).fillna(0)
            pcts=(counts/counts.sum()*100).round(1)
            st.markdown('<div class="liq-bar">'+"".join(
                f'<div class="liq-seg" style="flex:{p};background:{c}" title="{l}:{int(n)}">{p:.0f}%</div>'
                for l,n,p,c in zip(lbls,counts,pcts,clrs) if p>0
            )+'</div>', unsafe_allow_html=True)
            cs=st.columns(5)
            for col_m,(lbl,cnt,clr) in zip(cs,zip(lbls,counts,clrs)):
                with col_m:
                    st.markdown(f'<div style="text-align:center;padding:.45rem;background:var(--paper-2);border-radius:8px;border:1px solid var(--border)"><div style="font-family:\'Fraunces\',serif;font-size:1.1rem;font-weight:700;color:{clr}">{int(cnt):,}</div><div style="font-size:.58rem;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.05em">{lbl}</div></div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='sh'>Par signal</div>", unsafe_allow_html=True)
        SH={"heritage":"#8a4a1a","divorce":"#c4440a","upgrade":"#1a6b4a","retraite":"#1a4a8a","primo":"#7a4aa0"}
        if sig_col and total>0:
            for sig,cnt in df[sig_col].value_counts().items():
                pct=cnt/total*100; color=SH.get(sig,"#888"); lbl=SIGNAL_LABELS.get(sig,sig)
                st.markdown(f'<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.45rem"><div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div><div style="font-size:.73rem;color:var(--ink-light);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{lbl}</div><div style="font-family:\'Fraunces\',serif;font-size:.88rem;font-weight:600;color:var(--ink)">{cnt:,}</div><div style="font-size:.63rem;color:var(--ink-faint);min-width:30px;text-align:right">{pct:.0f}%</div></div><div style="height:3px;background:var(--paper-3);border-radius:2px;margin-bottom:.45rem"><div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:2px"></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='sh'>Top 5 prospects</div>", unsafe_allow_html=True)
    if total>0:
        for _,row in df.head(5).iterrows():
            st.markdown(render_prospect(row, show_seg=True), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — PROSPECTS
# ════════════════════════════════════════════════════════════
with tab_pros:
    c1,c2,c3 = st.columns([1,1,2])
    with c1: vue=st.radio("Vue",["Cartes","Tableau"],horizontal=True,label_visibility="collapsed")
    with c2: tri=st.selectbox("Tri",["Score ↓","Prix ↓","Surface ↓","Liquidité ↓","Cluster ↓"],label_visibility="collapsed")
    with c3: srch=st.text_input("🔍 Commune / adresse","",label_visibility="collapsed",placeholder="Filtrer…")

    tri_map={"Score ↓":("score_final",False),"Prix ↓":("valeur_fonciere",False),"Surface ↓":("surface_reelle_bati",False),"Liquidité ↓":("liquidite_cp",False),"Cluster ↓":("cluster_densite",False)}
    sc2,sa=tri_map.get(tri,("score_final",False))
    dv=df.copy()
    if srch.strip():
        mask=(dv.get("nom_commune",pd.Series(dtype=str)).astype(str).str.upper().str.contains(srch.upper(),na=False)|
              dv.get("adresse_complete",pd.Series(dtype=str)).astype(str).str.upper().str.contains(srch.upper(),na=False))
        dv=dv[mask]
    if sc2 in dv.columns: dv=dv.sort_values(sc2,ascending=sa)

    st.markdown(f"<div style='font-size:.68rem;color:var(--ink-faint);margin-bottom:.75rem'>{len(dv):,} résultats</div>",unsafe_allow_html=True)

    if vue=="Cartes":
        for _,row in dv.head(100).iterrows():
            st.markdown(render_prospect(row,show_seg=True),unsafe_allow_html=True)
        if len(dv)>100: st.caption(f"100/{len(dv):,} affichés. Téléchargez le CSV.")
    else:
        cols_show=[c for c in ["rang","adresse_complete","code_postal","nom_commune","signal_label",
            "score_final","score_brut","chaleur","valeur_fonciere","prix_m2","surface_reelle_bati",
            "nombre_pieces_principales","decote_vs_median","anciennete_mois","nb_signaux",
            "cluster_densite","liquidite_cp","insee_taux_proprio","insee_age_median",
            "insee_revenu_median","date_mutation","nature_mutation"] if c in dv.columns]
        rename={"rang":"#","adresse_complete":"Adresse","code_postal":"CP","nom_commune":"Commune",
            "signal_label":"Signal","score_final":"Score","score_brut":"Brut","chaleur":"Chaleur",
            "valeur_fonciere":"Prix (€)","prix_m2":"€/m²","surface_reelle_bati":"Surface (m²)",
            "nombre_pieces_principales":"Pièces","decote_vs_median":"Décote%","anciennete_mois":"Ancienneté(mois)",
            "nb_signaux":"#Sig","cluster_densite":"Cluster","liquidite_cp":"Liquidité",
            "insee_taux_proprio":"Proprio%","insee_age_median":"ÂgeMéd","insee_revenu_median":"RevenuMéd(€)",
            "date_mutation":"Date","nature_mutation":"Nature"}
        st.dataframe(dv[cols_show].rename(columns=rename).head(500),use_container_width=True,hide_index=True,
            column_config={"Score":st.column_config.ProgressColumn("Score",min_value=0,max_value=100,format="%d"),
                           "Liquidité":st.column_config.ProgressColumn("Liquidité",min_value=0,max_value=100,format="%.0f"),
                           "Prix (€)":st.column_config.NumberColumn("Prix (€)",format="%.0f €"),
                           "€/m²":st.column_config.NumberColumn("€/m²",format="%.0f €")})
    st.markdown("---")
    csv=dv.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("⬇ Exporter CSV",data=csv,file_name=f"prospects_{st.session_state.run_dept}_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")

# ════════════════════════════════════════════════════════════
# TAB 3 — ZONES
# ════════════════════════════════════════════════════════════
with tab_zones:
    zcol=next((c for c in ["nom_commune","commune","code_postal"] if c in df.columns),None)
    if not zcol or not sig_col or total==0:
        st.info("Données insuffisantes.")
    else:
        st.markdown("<div class='sh'>Ranking zones</div>",unsafe_allow_html=True)
        agg={"Prospects":(sig_col,"count"),"Score moy.":("score_final","mean"),"Top 20%":("score_final",lambda x:(x>=80).sum())}
        if "cluster_chaud" in df.columns: agg["Hotspots"]=("cluster_chaud","sum")
        if "liquidite_cp"  in df.columns: agg["Liquidité moy."]=("liquidite_cp","mean")
        if "valeur_fonciere" in df.columns: agg["Prix moy.(€)"]=("valeur_fonciere","mean")
        if "decote_vs_median" in df.columns: agg["Décote moy.%"]=("decote_vs_median","mean")
        if "insee_taux_proprio" in df.columns: agg["Proprio%"]=("insee_taux_proprio","mean")
        ranking=(df.groupby(zcol).agg(**agg).sort_values("Prospects",ascending=False).reset_index())
        ranking["Score moy."]=ranking["Score moy."].round(1)
        for c in ["Liquidité moy.","Prix moy.(€)","Décote moy.%","Proprio%"]:
            if c in ranking.columns: ranking[c]=ranking[c].round(1)
        mx=ranking["Prospects"].max()
        ca,cb=st.columns([1.4,1])
        with ca:
            for _,zr in ranking.head(15).iterrows():
                nm=str(zr[zcol]); cnt=int(zr["Prospects"]); smoy=float(zr["Score moy."])
                top=int(zr.get("Top 20%",0)); liq=float(zr.get("Liquidité moy.",0) or 0)
                hs=int(zr.get("Hotspots",0)); pct=cnt/mx*100
                color=sc(smoy)
                st.markdown(f'<div class="zc"><div class="zc-h"><div><div class="zc-nm">{nm}</div></div><div class="zc-sc" style="color:{color}">{smoy:.0f}</div></div><div class="zbar"><div class="zbar-f" style="width:{pct:.1f}%;background:{color}"></div></div><div class="zst"><span><b>{cnt}</b> prospects</span><span><b>{top}</b> top 20%</span><span><b>{hs}</b> hotspots</span>{f"<span>liq.<b>{liq:.0f}</b></span>" if liq else ""}</div></div>', unsafe_allow_html=True)
        with cb:
            st.dataframe(ranking,use_container_width=True,hide_index=True,
                column_config={"Score moy.":st.column_config.ProgressColumn("Score moy.",min_value=0,max_value=100,format="%.1f")})

# ════════════════════════════════════════════════════════════
# TAB 4 — SEGMENTS
# ════════════════════════════════════════════════════════════
with tab_seg:
    SH={"heritage":"#8a4a1a","divorce":"#c4440a","upgrade":"#1a6b4a","retraite":"#1a4a8a","primo":"#7a4aa0"}
    st.markdown("<div class='sh'>Segments marketing</div>",unsafe_allow_html=True)
    if sig_col and total>0:
        for sk in signaux_choix:
            sub=df[df[sig_col]==sk]
            if sub.empty: continue
            cnt=len(sub); smoy=sub["score_final"].mean() if "score_final" in sub.columns else 0
            top=(sub["score_final"]>=80).sum() if "score_final" in sub.columns else 0
            multi=(sub["nb_signaux"]>=2).sum() if "nb_signaux" in sub.columns else 0
            hs=sub.get("cluster_chaud",pd.Series(False)).sum()
            pm2=sub["prix_m2"].median() if "prix_m2" in sub.columns and sub["prix_m2"].notna().any() else 0
            liq=sub["liquidite_cp"].mean() if "liquidite_cp" in sub.columns and sub["liquidite_cp"].notna().any() else 0
            tp=sub["insee_taux_proprio"].mean() if "insee_taux_proprio" in sub.columns and sub["insee_taux_proprio"].notna().any() else 0
            am=sub["insee_age_median"].mean() if "insee_age_median" in sub.columns and sub["insee_age_median"].notna().any() else 0
            rm=sub["insee_revenu_median"].mean() if "insee_revenu_median" in sub.columns and sub["insee_revenu_median"].notna().any() else 0
            st.markdown(f'''<div class="seg-c">
  <div class="seg-h"><div class="sd" style="background:{SH.get(sk,"#888")}"></div><div class="seg-nm">{SIGNAL_LABELS.get(sk,sk)}</div><div class="seg-cnt">{cnt:,} · {cnt/total*100:.0f}%</div></div>
  <div class="seg-msg">💬 {SIGNAL_SEGMENTS.get(sk,"—")}</div>
  <div class="seg-st"><span>Score <b>{smoy:.0f}</b></span><span>Top20% <b>{top}</b></span><span>Hotspots <b>{int(hs)}</b></span><span>Multi <b>{multi}</b></span>{f"<span>Liq.<b>{liq:.0f}</b></span>" if liq else ""}</div>
  <div style="margin-top:.55rem;display:flex;gap:1.2rem;font-size:.64rem;color:var(--ink-light)">
    {f"<span>Proprio <b>{tp:.0f}%</b></span>" if tp else ""}
    {f"<span>ÂgeMéd <b>{am:.0f} ans</b></span>" if am else ""}
    {f"<span>RevenuMéd <b>{rm:,.0f} €</b></span>" if rm else ""}
    {f"<span>€/m² méd <b>{pm2:,.0f}</b></span>" if pm2 else ""}
  </div>
</div>''', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='sh'>Matrice signal × chaleur</div>",unsafe_allow_html=True)
        if "chaleur" in df.columns:
            matrix=pd.crosstab(df[sig_col].map(SIGNAL_LABELS),df["chaleur"].astype(str))
            for ch in ["très chaud","chaud","tiède","froid"]:
                if ch not in matrix.columns: matrix[ch]=0
            st.dataframe(matrix[[c for c in ["très chaud","chaud","tiède","froid"] if c in matrix.columns]],use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — MARCHÉ & TENDANCES
# ════════════════════════════════════════════════════════════
with tab_market:
    st.markdown("<div class='sh'>Tendances du marché — données DVF brutes (tous biens résidentiels)</div>",unsafe_allow_html=True)

    if not tendances:
        st.info("Lancez une analyse pour voir les tendances du marché.")
    else:
        # Volume mensuel
        if "volume_par_mois" in tendances:
            t_vol=tendances["volume_par_mois"]
            st.markdown("<div class='trend-card'><h4>Volume de transactions par mois</h4>", unsafe_allow_html=True)
            if not t_vol.empty:
                mx=t_vol["nb_transactions"].max()
                bars="".join(
                    f'<div class="mc-bar {"hot" if row["nb_transactions"]>=mx*0.8 else ""}" style="height:{row["nb_transactions"]/mx*100:.0f}%;min-height:4px" title="{row["mois"]} : {row["nb_transactions"]:,} transactions"></div>'
                    for _,row in t_vol.iterrows()
                )
                st.markdown(f'<div class="mini-chart">{bars}</div>',unsafe_allow_html=True)
                peak=t_vol.loc[t_vol["nb_transactions"].idxmax(),"mois"]
                low=t_vol.loc[t_vol["nb_transactions"].idxmin(),"mois"]
                st.markdown(f'<div style="font-size:.68rem;color:var(--ink-light);margin-top:.3rem">Pic : <b style="color:var(--ink)">{peak}</b> ({t_vol["nb_transactions"].max():,} trans.) · Creux : <b style="color:var(--ink)">{low}</b> ({t_vol["nb_transactions"].min():,} trans.)</div>',unsafe_allow_html=True)
                c1,c2=st.columns(2)
                with c1: st.dataframe(t_vol.rename(columns={"mois":"Mois","nb_transactions":"Transactions"}),use_container_width=True,hide_index=True)
            st.markdown("</div>",unsafe_allow_html=True)

        # Prix médian mensuel
        if "prix_median_par_mois" in tendances:
            t_prix=tendances["prix_median_par_mois"]
            st.markdown("<div class='trend-card'><h4>Évolution du prix médian</h4>",unsafe_allow_html=True)
            if not t_prix.empty:
                mx2=t_prix["prix_median"].max()
                bar2_parts=[]
                for _,row in t_prix.iterrows():
                    h=row["prix_median"]/mx2*100
                    m=str(row["mois"]); p=f'{row["prix_median"]:,.0f}'
                    bar2_parts.append(f'<div class="mc-bar" style="height:{h:.0f}%;min-height:4px;background:var(--blue)" title="{m} : {p} €"></div>')
                bars2="".join(bar2_parts)
                st.markdown(f'<div class="mini-chart">{bars2}</div>',unsafe_allow_html=True)
                delta=t_prix["prix_median"].iloc[-1]-t_prix["prix_median"].iloc[0] if len(t_prix)>1 else 0
                sign="↑" if delta>0 else "↓" if delta<0 else "→"
                clr="var(--green)" if delta>0 else "var(--red)"
                base=t_prix["prix_median"].iloc[0]
                pct_delta=delta/base*100 if base else 0
                st.markdown(f'<div style="font-size:.68rem;color:var(--ink-light);margin-top:.3rem">Évolution : <b style="color:{clr}">{sign} {abs(delta):,.0f} €</b> ({pct_delta:.1f}%)</div>',unsafe_allow_html=True)
                st.dataframe(t_prix.rename(columns={"mois":"Mois","prix_median":"Prix médian (€)"}),use_container_width=True,hide_index=True)
            st.markdown("</div>",unsafe_allow_html=True)

        # Liquidité par CP
        ca,cb=st.columns(2)
        with ca:
            if "top_cp_volume" in tendances:
                st.markdown("<div class='trend-card'><h4>Top 10 CP par volume</h4>",unsafe_allow_html=True)
                tv=tendances["top_cp_volume"]
                mx3=tv["volume"].max()
                for _,r in tv.iterrows():
                    pct=r["volume"]/mx3*100
                    st.markdown(f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem"><div style="font-family:\'DM Mono\',monospace;font-size:.72rem;min-width:55px;color:var(--ink)">{r["code_postal"]}</div><div style="flex:1;height:6px;background:var(--paper-3);border-radius:3px"><div style="height:100%;width:{pct:.0f}%;background:var(--accent);border-radius:3px"></div></div><div style="font-size:.68rem;color:var(--ink-light);min-width:35px;text-align:right">{r["volume"]:,}</div></div>',unsafe_allow_html=True)
                st.markdown("</div>",unsafe_allow_html=True)

        with cb:
            if "prix_m2_par_cp" in tendances:
                st.markdown("<div class='trend-card'><h4>Prix/m² médian par CP</h4>",unsafe_allow_html=True)
                tp2=tendances["prix_m2_par_cp"].head(10)
                mx4=tp2["prix_m2_median"].max() if not tp2.empty else 1
                for _,r in tp2.iterrows():
                    pct=r["prix_m2_median"]/mx4*100
                    st.markdown(f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem"><div style="font-family:\'DM Mono\',monospace;font-size:.72rem;min-width:55px;color:var(--ink)">{r["code_postal"]}</div><div style="flex:1;height:6px;background:var(--paper-3);border-radius:3px"><div style="height:100%;width:{pct:.0f}%;background:var(--blue);border-radius:3px"></div></div><div style="font-size:.68rem;color:var(--ink-light);min-width:60px;text-align:right">{r["prix_m2_median"]:,.0f} €</div></div>',unsafe_allow_html=True)
                st.markdown("</div>",unsafe_allow_html=True)

        # Liquidité prospects filtrés
        if "liquidite_cp" in df.columns and total>0:
            st.markdown("---")
            st.markdown("<div class='sh'>Liquidité marché — prospects filtrés</div>",unsafe_allow_html=True)
            liq_dist=pd.cut(df["liquidite_cp"],bins=[0,25,50,75,100],labels=["Faible","Modérée","Bonne","Très bonne"],right=True).value_counts().reindex(["Faible","Modérée","Bonne","Très bonne"]).fillna(0)
            liq_clrs=["#b8b5ae","#f5a05a","#1a4a8a","#1a6b4a"]
            tot_liq=liq_dist.sum()
            st.markdown('<div class="liq-bar">'+"".join(
                f'<div class="liq-seg" style="flex:{n/tot_liq*100:.1f};background:{c}" title="{l}:{int(n)}">{n/tot_liq*100:.0f}%</div>'
                for (l,n),c in zip(liq_dist.items(),liq_clrs) if n>0
            )+'</div>',unsafe_allow_html=True)
            lc=st.columns(4)
            for col_m,(lbl,cnt),clr in zip(lc,liq_dist.items(),liq_clrs):
                with col_m:
                    st.markdown(f'<div style="text-align:center;padding:.4rem;background:var(--paper-2);border-radius:8px;border:1px solid var(--border)"><div style="font-family:\'Fraunces\',serif;font-size:1.05rem;font-weight:700;color:{clr}">{int(cnt):,}</div><div style="font-size:.58rem;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.05em">{lbl}</div></div>',unsafe_allow_html=True)

        # INSEE agrégé
        if "insee_taux_proprio" in df.columns and df["insee_taux_proprio"].notna().any():
            st.markdown("---")
            st.markdown("<div class='sh'>Contexte socio-démographique INSEE — moyenne sur les prospects filtrés</div>",unsafe_allow_html=True)
            tp_avg=df["insee_taux_proprio"].mean(); am_avg=df["insee_age_median"].mean() if "insee_age_median" in df.columns else 0; rm_avg=df["insee_revenu_median"].mean() if "insee_revenu_median" in df.columns else 0
            st.markdown(f'<div class="insee-grid"><div class="insee-card"><div class="ic-val">{tp_avg:.1f}%</div><div class="ic-lbl">Taux propriétaires</div><div class="ic-sub">{"Zone majoritairement propriétaire" if tp_avg>60 else "Zone locataire majoritaire"}</div></div><div class="insee-card"><div class="ic-val">{am_avg:.1f} ans</div><div class="ic-lbl">Âge médian</div><div class="ic-sub">{"Zone senior — signal retraite renforcé" if am_avg>50 else "Zone jeune — signal upgrade/primo renforcé"}</div></div><div class="insee-card"><div class="ic-val">{rm_avg:,.0f} €</div><div class="ic-lbl">Revenu médian/UC</div><div class="ic-sub">{"Pouvoir d\'achat élevé" if rm_avg>28000 else "Pouvoir d\'achat modéré — primo-accédants"}</div></div></div>',unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 6 — CARTE
# ════════════════════════════════════════════════════════════
with tab_carte:
    has_c="latitude" in df.columns and "longitude" in df.columns
    df_map=df[df["latitude"].notna()&df["longitude"].notna()&(df["latitude"]!=0)] if has_c else pd.DataFrame()
    if not has_c or df_map.empty:
        st.info("⚠ Coordonnées GPS absentes dans ce fichier DVF.")
    else:
        st.markdown(f"<div class='sh'>{len(df_map):,} prospects géolocalisés · taille = score · couleur = signal · contour épais = hotspot cluster</div>",unsafe_allow_html=True)
        SCJS=json.dumps(SIGNAL_COLORS); SLJS=json.dumps(SIGNAL_LABELS)
        pts=[]
        for _,row in df_map.iterrows():
            lat=float(row.get("latitude",0) or 0); lng=float(row.get("longitude",0) or 0)
            if lat==0 and lng==0: continue
            pts.append({"lat":lat,"lng":lng,"sig":str(row.get(sig_col,"") or ""),
                "sf":float(row.get("score_final",50) or 50),"sb":float(row.get("score_brut",50) or 50),
                "addr":str(row.get("adresse_complete","") or ""),"cp":str(row.get("code_postal","") or ""),
                "comm":str(row.get("nom_commune",row.get("commune","")) or ""),
                "prix":float(row.get("valeur_fonciere",0) or 0),"surf":float(row.get("surface_reelle_bati",0) or 0),
                "pm2":float(row.get("prix_m2",0) or 0),"dec":float(row.get("decote_vs_median",0) or 0),
                "date":str(row.get("date_mutation",""))[:10],"ch":str(row.get("chaleur","") or ""),
                "nb":int(row.get("nb_signaux",1) or 1),"cls":bool(row.get("cluster_chaud",False)),
                "cdn":int(row.get("cluster_densite",0) or 0),"liq":float(row.get("liquidite_cp",0) or 0),
                "tp":float(row.get("insee_taux_proprio",0) or 0),"am":float(row.get("insee_age_median",0) or 0)})
        clat=df_map["latitude"].median(); clng=df_map["longitude"].median()
        djs=json.dumps(pts)
        html=f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:monospace;background:#f5f2ed}}#map{{width:100%;height:590px}}
.lp{{font-family:monospace;font-size:.74rem;line-height:1.6;min-width:250px}}
.lp .b{{display:inline-block;padding:.1rem .4rem;border-radius:3px;color:#fff;font-size:.6rem;font-weight:600;margin-bottom:.35rem}}
.lp .s{{font-size:1.45rem;font-weight:700;line-height:1;margin-bottom:.15rem}}
.lp .su{{font-size:.65rem;color:#6b6860;margin-bottom:.35rem}}
.lp .a{{font-size:.73rem;font-weight:600;margin-bottom:.3rem;color:#1a1814}}
.lp table{{width:100%;border-collapse:collapse;font-size:.67rem}}
.lp td{{padding:.1rem 0;border-bottom:1px solid rgba(0,0,0,.06)}}
.lp td:first-child{{color:#6b6860}}
.lp td:last-child{{font-weight:500;text-align:right}}
.lp .cluster{{display:inline-block;background:rgba(26,107,74,.12);border:1px solid rgba(26,107,74,.25);color:#1a6b4a;font-size:.58rem;padding:.1rem .35rem;border-radius:3px;margin-top:.3rem}}
.lp .multi{{display:inline-block;background:rgba(245,160,90,.12);border:1px solid rgba(245,160,90,.25);color:#b07030;font-size:.58rem;padding:.1rem .35rem;border-radius:3px;margin-top:.3rem;margin-left:.25rem}}
</style></head><body><div id="map"></div>
<script>
const D={djs},C={SCJS},L={SLJS};
const map=L.map('map').setView([{clat},{clng}],11);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'© OpenStreetMap',maxZoom:19}}).addTo(map);
const ly=L.layerGroup().addTo(map);
D.forEach(pt=>{{
  const col=C[pt.sig]||'#888';
  const r=4+(pt.sf/100)*10;
  const sc=pt.sf>=80?'#c4440a':pt.sf>=60?'#d4850a':pt.sf>=40?'#1a4a8a':'#b8b5ae';
  const prix=pt.prix?pt.prix.toLocaleString('fr-FR')+' €':'—';
  const surf=pt.surf?pt.surf+' m²':'—';
  const pm2=pt.pm2?pt.pm2.toLocaleString('fr-FR')+' €/m²':'—';
  const dec=pt.dec?'−'+Math.abs(pt.dec).toFixed(0)+'%':'—';
  const cls_tag=pt.cls?`<span class="cluster">🔥 hotspot ${{pt.cdn}} prospects</span>`:'';
  const multi_tag=pt.nb>=2?`<span class="multi">⚡ ${{pt.nb}} signaux</span>`:'';
  const pop=`<div class="lp"><div class="b" style="background:${{col}}">${{L[pt.sig]||pt.sig}}</div>
    <div class="s" style="color:${{sc}}">${{pt.sf.toFixed(0)}}<span style="font-size:.65rem;color:#6b6860">/100</span></div>
    <div class="su">Brut ${{pt.sb.toFixed(0)}} · ${{pt.ch}} · liq.${{pt.liq.toFixed(0)}}</div>
    <div class="a">${{pt.addr||'—'}}</div>
    <table><tr><td>Commune</td><td>${{pt.comm}} ${{pt.cp}}</td></tr>
      <tr><td>Prix</td><td>${{prix}}</td></tr><tr><td>Surface</td><td>${{surf}}</td></tr>
      <tr><td>Prix/m²</td><td>${{pm2}}</td></tr><tr><td>Décote</td><td>${{dec}}</td></tr>
      <tr><td>Date</td><td>${{pt.date}}</td></tr>
      ${{pt.tp?`<tr><td>Proprio</td><td>${{pt.tp.toFixed(0)}}%</td></tr>`:''}}
      ${{pt.am?`<tr><td>Âge méd.</td><td>${{pt.am.toFixed(0)}} ans</td></tr>`:''}}
    </table>
    ${{cls_tag}}${{multi_tag}}
  </div>`;
  L.circleMarker([pt.lat,pt.lng],{{
    radius:r,color:col,fillColor:col,fillOpacity:.65,
    weight:pt.cls?3:pt.nb>=2?2:1.2,opacity:.9
  }}).bindPopup(pop).bindTooltip(`${{L[pt.sig]||pt.sig}} · ${{pt.sf.toFixed(0)}}${{pt.cls?" 🔥":""}}`,{{direction:'top',offset:[0,-4]}}).addTo(ly);
}});
if(D.length>0){{const la=D.map(d=>d.lat),lo=D.map(d=>d.lng);map.fitBounds([[Math.min(...la),Math.min(...lo)],[Math.max(...la),Math.max(...lo)]],{{padding:[30,30]}});}}
</script></body></html>"""
        st.components.v1.html(html,height=600,scrolling=False)

# ════════════════════════════════════════════════════════════
# TAB 7 — DOC
# ════════════════════════════════════════════════════════════
with tab_doc:
    st.markdown("""<div style='background:var(--ink);border-radius:var(--r);padding:2.25rem 2rem;margin-bottom:1.5rem;position:relative;overflow:hidden'>
<div style='position:absolute;inset:0;pointer-events:none;background-image:repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(245,240,230,.04) 39px,rgba(245,240,230,.04) 40px),repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(245,240,230,.04) 39px,rgba(245,240,230,.04) 40px)'></div>
<div style='font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;background:rgba(245,160,90,.15);color:#f5a05a;border:1px solid rgba(245,160,90,.25);padding:.22rem .65rem;border-radius:20px;margin-bottom:.9rem;display:inline-block;font-family:"DM Mono",monospace;position:relative'>Documentation v4.0</div>
<h1 style='font-family:"Fraunces",serif!important;font-size:1.85rem!important;font-weight:700!important;color:var(--paper)!important;margin-bottom:.45rem;position:relative'>Pipeline DVF <em style="color:#f5a05a">×</em> BODACC</h1>
<div style='font-family:"DM Mono",monospace;font-size:.72rem;color:rgba(245,242,237,.38);position:relative;line-height:1.75'>Score percentile · Liquidité · INSEE · Clusters · Cadastre · Tendances</div>
<div style='display:flex;gap:1.75rem;margin-top:1.35rem;position:relative;flex-wrap:wrap'>
  <div><div style='font-family:"Fraunces",serif;font-size:1.55rem;font-weight:700;color:#f5a05a'>5</div><div style='font-size:.58rem;color:rgba(245,242,237,.32);text-transform:uppercase;letter-spacing:.07em'>Signaux</div></div>
  <div><div style='font-family:"Fraunces",serif;font-size:1.55rem;font-weight:700;color:#f5a05a'>7</div><div style='font-size:.58rem;color:rgba(245,242,237,.32);text-transform:uppercase;letter-spacing:.07em'>Étapes scoring</div></div>
  <div><div style='font-family:"Fraunces",serif;font-size:1.55rem;font-weight:700;color:#f5a05a'>4</div><div style='font-size:.58rem;color:rgba(245,242,237,.32);text-transform:uppercase;letter-spacing:.07em'>Sources données</div></div>
  <div><div style='font-family:"Fraunces",serif;font-size:1.55rem;font-weight:700;color:#f5a05a'>Percentile</div><div style='font-size:.58rem;color:rgba(245,242,237,.32);text-transform:uppercase;letter-spacing:.07em'>Score relatif dept</div></div>
</div></div>""", unsafe_allow_html=True)

    for title,body in [
        ("Sources de données v4","""<b>DVF</b> (data.gouv.fr) : transactions immobilières depuis 2014, mis à jour trimestriellement.<br>
<b>BODACC</b> (OpenDataSoft) : annonces légales commerciales — proxy succession.<br>
<b>INSEE</b> (API communes) : taux propriétaires, âge médian, revenu médian par commune.<br>
<b>API BAN + Cadastre</b> (optionnel) : géocodage et année de construction par adresse."""),
        ("Architecture du score final v4","""<b>1. Score brut</b> — intensité de l'événement (délai, décote, adjudication) → 0–100<br>
<b>2. Malus qualité</b> — bien non résidentiel (−20), prix/m² aberrant (−15), surface hors norme (−10), prix trop bas (−10)<br>
<b>3. Bonus INSEE</b> — contexte socio-démo favorable au signal → +0 à +5<br>
<b>4. Bonus Cadastre</b> — bien ancien + signal compatible → +0 à +10<br>
<b>5. Bonus Cluster</b> — hotspot géographique → +4 à +8<br>
<b>6. Normalisation percentile</b> — rank dans le pool dept → 0–100<br>
<b>7. Bonus liquidité + multi-signal</b> → score final 0–100"""),
        ("Clusters géographiques","""Algorithme de clustering spatial basé sur la distance haversine. Pour chaque prospect géolocalisé, on compte les voisins dans un rayon paramétrable (défaut 0.5 km). Si ≥ 3 prospects dans ce rayon → hotspot confirmé.<br><br>
Un hotspot = zone de concentration anormale de signaux, souvent liée à une copropriété en difficulté, un quartier en transition démographique, ou un programme de rénovation.<br><br>
Sur la carte : <b>contour épais</b> = prospect dans un hotspot. Le tooltip affiche 🔥 et le nombre de voisins."""),
        ("Liquidité marché","""Calculée par code postal sur les données DVF de l'année analysée.<br><br>
<b>Volume</b> : nombre de transactions résidentielles → percentile dans le département.<br>
<b>Rotation</b> : délai médian entre 2 ventes consécutives sur la même adresse → inversé en percentile (délai court = marché liquide).<br><br>
Score final = 60% volume + 40% rotation. Un marché liquide (score >70) signifie que les prix sont fiables et les transactions représentatives → signal plus crédible. Un marché illiquide (<30) = données peu représentatives, signal potentiellement bruité."""),
        ("Données INSEE", (
            "Récupérées via l'API Open Data pour chaque commune du département.<br><br>"
            "<b>Taux propriétaires</b> : signal divorce/heritage peu pertinent si zone à 90% locataires.<br>"
            "<b>Age médian</b> : signal retraite amplifié si zone senior (>50 ans).<br>"
            "<b>Revenu médian</b> : signal upgrade amplifié si revenus suffisants pour un 2e achat.<br><br>"
            "Ces données s'appliquent en bonus contextuel (+0 à +5 pts) sur le score brut, avant la normalisation percentile."
        )),
        ("RGPD & Usage légal", (
            "<b>OK Autorise</b> : ciblage géographique par CP pour Meta/Google Ads, audiences LAL, analyse statistique.<br>"
            "<b>ATTENTION Zone grise</b> : prospection postale à une adresse sans consentement"
            " (base légale intérêt légitime à documenter).<br>"
            "<b>INTERDIT</b> : fichier nominatif Nom+Adresse+Evenement de vie sans consentement CNIL.<br><br>"
            "Le DVF ne contient pas l'identité des propriétaires depuis 2021 (arrêté CNIL)."
            " Pour enrichir en contacts nominatifs, passer par un prestataire habilité"
            " (Kompass, InfoLegale) qui gère les bases légales RGPD."
        )),
    ]:
        st.markdown(f'<div style="background:var(--paper);border:1px solid var(--border);border-radius:var(--r);padding:1.5rem 1.75rem;margin-bottom:1rem"><h2 style="font-family:\'Fraunces\',serif!important;font-size:1.15rem!important;font-weight:600!important;color:var(--ink)!important;margin-bottom:.65rem">{title}</h2><div style="font-family:\'Lora\',serif;font-size:.87rem;line-height:1.85;color:var(--ink-light)">{body}</div></div>',unsafe_allow_html=True)

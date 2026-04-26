"""
UNMAPPED — Main Application
=============================
Open-source infrastructure layer for informal skills and economic opportunity in LMICs.
Streamlit UI with three integrated modules:
  Module 1: Skills Signal Engine    → ISCO-08 Skills Passport
  Module 2: AI Readiness & Displacement Lens → Resilience Score
  Module 3: Opportunity Dashboard   → Youth View + Policy View
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import google.generativeai as genai 

import os
from dotenv import load_dotenv  

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY :
    genai.configure(api_key=GOOGLE_API_KEY)
else :
    st.error("Google API key not found. Please check your .env file and ensure GOOGLE_API_KEY is set.")

from data_loader import (
    get_available_countries,
    get_country_context,
    get_all_taxonomy_skills,
    compute_skill_population_divergence,
    compute_returns_to_education,
    get_sector_growth_data,
    load_labor_context,
    load_automation_risk,
)
from signal_engine import run_skills_signal_engine, format_passport_for_display
from risk_analyzer import run_risk_analysis

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UNMAPPED",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Core palette */
  :root {
    --unmapped-dark:    #0f172a;
    --unmapped-slate:   #1e293b;
    --unmapped-accent:  #6366f1;
    --unmapped-gold:    #f59e0b;
    --unmapped-green:   #22c55e;
    --unmapped-red:     #ef4444;
    --unmapped-text:    #e2e8f0;
    --unmapped-muted:   #94a3b8;
  }

  /* Hide default Streamlit chrome for cleaner feel */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Branded header */
  .unmapped-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #0f172a 100%);
    border-bottom: 2px solid #6366f1;
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }
  .unmapped-header h1 { color: #e2e8f0; margin: 0; font-size: 2rem; letter-spacing: -0.5px; }
  .unmapped-header p  { color: #94a3b8; margin: 0.3rem 0 0; font-size: 0.92rem; }
  .unmapped-accent { color: #6366f1; }
  .unmapped-gold-text { color: #f59e0b; }

  /* Metric cards */
  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
  }
  .metric-card h3 { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;
                    letter-spacing: 0.06em; margin: 0 0 0.3rem; }
  .metric-card .value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; margin: 0; }
  .metric-card .sub   { color: #64748b;  font-size: 0.78rem; margin-top: 0.2rem; }

  /* Passport card */
  .passport-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #6366f1;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
  }
  .passport-headline { color: #a5b4fc; font-size: 1.1rem; font-weight: 600;
                       font-style: italic; margin-bottom: 0.8rem; }
  .isco-badge {
    display: inline-block;
    background: #312e81;
    color: #c7d2fe;
    border: 1px solid #6366f1;
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.8rem;
  }
  .competency-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.2rem 0.2rem 0;
    border: 1px solid #0284c7;
  }
  .durable-pill {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.2rem 0.2rem 0;
    border: 1px solid #16a34a;
  }
  .risk-pill {
    display: inline-block;
    background: #450a0a;
    color: #fca5a5;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.2rem 0.2rem 0;
    border: 1px solid #dc2626;
  }

  /* Section headers */
  .section-header {
    color: #94a3b8;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
  }

  /* Opportunity card */
  .opp-card {
    background: #0f2337;
    border: 1px solid #1e4976;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
  }
  .opp-card h4 { color: #bfdbfe; margin: 0 0 0.3rem; font-size: 0.92rem; }
  .opp-card p  { color: #64748b;  margin: 0;          font-size: 0.78rem; }

  /* Resilience score */
  .resilience-score-box {
    text-align: center;
    background: #1e293b;
    border-radius: 14px;
    padding: 1.4rem;
    border: 1px solid #334155;
  }
  .resilience-number { font-size: 3.5rem; font-weight: 800; line-height: 1; }
  .resilience-label  { font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem; }

  /* Alert boxes */
  .alert-critical { background:#2d0505; border:1px solid #ef4444;
                    border-left:4px solid #ef4444; border-radius:8px;
                    padding:0.8rem 1rem; color:#fca5a5; font-size:0.85rem; }
  .alert-success  { background:#052d14; border:1px solid #22c55e;
                    border-left:4px solid #22c55e; border-radius:8px;
                    padding:0.8rem 1rem; color:#86efac; font-size:0.85rem; }
  .alert-info     { background:#0c1a2e; border:1px solid #3b82f6;
                    border-left:4px solid #3b82f6; border-radius:8px;
                    padding:0.8rem 1rem; color:#93c5fd; font-size:0.85rem; }
  .alert-warning  { background:#2d1d05; border:1px solid #f59e0b;
                    border-left:4px solid #f59e0b; border-radius:8px;
                    padding:0.8rem 1rem; color:#fcd34d; font-size:0.85rem; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 0.82rem; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] { color: #64748b; }
  .stTabs [aria-selected="true"] { color: #6366f1; border-bottom-color: #6366f1; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────────
if "passport"         not in st.session_state: st.session_state.passport         = None
if "passport_display" not in st.session_state: st.session_state.passport_display = None
if "risk_analysis"    not in st.session_state: st.session_state.risk_analysis    = None
if "active_country"   not in st.session_state: st.session_state.active_country   = "Algeria"
if "analysis_done"    not in st.session_state: st.session_state.analysis_done    = False


# ── SIDEBAR: Context Configurator ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem; text-align:center;">
      <span style="font-size:1.6rem;">🗺️</span>
      <h2 style="color:#e2e8f0; margin:0.2rem 0 0; font-size:1.1rem; letter-spacing:-0.3px;">
        UNMAPPED
      </h2>
      <p style="color:#475569; font-size:0.7rem; margin:0;">
        Open Infrastructure for Informal Talent
      </p>
    </div>
    <hr style="border-color:#1e293b; margin: 0.8rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">⚙️ Context Configurator</p>', unsafe_allow_html=True)

    countries = get_available_countries()
    selected_country = st.selectbox(
        "Country Context",
        options=countries,
        index=countries.index(st.session_state.active_country)
        if st.session_state.active_country in countries else 0,
        help="Switch country to update all signals, wage floors, and risk timelines instantly.",
    )

    if selected_country != st.session_state.active_country:
        st.session_state.active_country = selected_country
        # Reset analysis if country changes
        st.session_state.analysis_done    = False
        st.session_state.passport         = None
        st.session_state.passport_display = None
        st.session_state.risk_analysis    = None
        st.rerun()

    country = st.session_state.active_country
    ctx     = get_country_context(country)

    if ctx:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:0.8rem;">
          <h3>📍 {ctx.get('country','—')}</h3>
          <p class="value" style="font-size:1rem;">{ctx.get('region','—')}</p>
          <p class="sub">{ctx.get('notes','')[:80]}…</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Informal Rate", f"{ctx.get('informal_employment_rate_pct','?')}%")
            st.metric("GDP Growth",    f"{ctx.get('gdp_growth_pct','?')}%")
        with col2:
            st.metric("Internet",      f"{ctx.get('internet_penetration_pct','?')}%")
            st.metric("Youth Unemp.",  f"{ctx.get('youth_unemployment_pct','?')}%")

    st.markdown('<hr style="border-color:#1e293b; margin:1rem 0;">', unsafe_allow_html=True)

    # ── Econometric Signals Panel ──────────────────────────────────────────────
    st.markdown('<p class="section-header">📊 Econometric Signals</p>', unsafe_allow_html=True)

    spd = compute_skill_population_divergence(country)
    if spd.get("value") is not None:
        st.markdown(f"""
        <div class="metric-card">
          <h3>Skill-Population Divergence</h3>
          <p class="value" style="color:{spd['color']};">{spd['value']}</p>
          <p class="sub">{spd['label']} — {spd['interpretation'][:100]}…</p>
        </div>
        """, unsafe_allow_html=True)

    rte = compute_returns_to_education(country)
    if rte.get("value") is not None:
        st.markdown(f"""
        <div class="metric-card">
          <h3>Returns to Education (Mincerian)</h3>
          <p class="value" style="color:#6366f1;">{rte['value']}%</p>
          <p class="sub">Per year of recognized credential<br>
          +${rte['annual_gain_per_year_edu']}/yr wage potential</p>
        </div>
        """, unsafe_allow_html=True)

    # GDP Per Capita Signal
    if ctx:
        st.markdown(f"""
        <div class="metric-card">
          <h3>GDP per Capita</h3>
          <p class="value" style="color:#a5b4fc;">${ctx.get('gdp_per_capita_usd',0):,.0f}</p>
          <p class="sub">Growing at {ctx.get('gdp_growth_pct','?')}%/yr (World Bank)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e293b; margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#1e3a5f; font-size:0.65rem; text-align:center;">'
        'Data: WDI · ILOSTAT · Frey-Osborne · ILO<br>'
        'UNMAPPED v0.1 · Open Source · Protocol, Not Product</p>',
        unsafe_allow_html=True
    )


# ── HEADER ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="unmapped-header">
  <h1>🗺️ <span class="unmapped-accent">UN</span>MAPPED
    <span style="font-size:0.7rem; color:#475569; font-weight:400;
                 background:#1e293b; padding:0.2rem 0.6rem; border-radius:4px;
                 margin-left:0.5rem; vertical-align:middle;">
      v0.1-prototype · {country} context active
    </span>
  </h1>
  <p>Closing the gap between informal skills and economic opportunity in LMICs · 
  <span style="color:#6366f1;">Open Infrastructure Layer</span></p>
</div>
""", unsafe_allow_html=True)


# ── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🪪 Module 1 · Skills Passport",
    "🔬 Module 2 · Resilience Lens",
    "📈 Module 3 · Opportunity Dashboard",
    "🔍 Data Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — SKILLS SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🪪 Skills Signal Engine")
    st.markdown(
        "Describe your informal work experience in your own words. "
        "UNMAPPED will map it to an internationally recognized **ISCO-08** profile "
        "and generate your **Human-Readable Skills Passport**."
    )

    col_input, col_examples = st.columns([3, 1])
    with col_input:
        user_input = st.text_area(
            "Your experience (in your own words)",
            placeholder=(
                "e.g. \"I've been repairing phones and selling accessories at my stall "
                "since I was 17. I fix screens, replace batteries, and manage stock for "
                "about 20 phones a day. I handle all the money and deal with suppliers.\""
            ),
            height=120,
            help="Be specific — years of experience, daily tasks, and volume handled all help.",
        )

    with col_examples:
        st.markdown("**💡 Try these:**")
        examples = [
            "Run a phone repair stall for 4 years",
            "Driving motorcycle taxi in the city daily",
            "Sewing and tailoring clothes from home",
            "Selling food at the market every morning",
            "Community health volunteer for 3 years",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state["prefill"] = ex

    # Handle prefill from example buttons
    if "prefill" in st.session_state and not user_input:
        user_input = st.session_state.pop("prefill")
        st.rerun()

    analyze_btn = st.button(
        "🚀 Generate Skills Passport",
        type="primary",
        disabled=not user_input.strip(),
        use_container_width=True,
    )

    if analyze_btn and user_input.strip():
        with st.spinner("🔍 UNMAPPED is mapping skills via GEMINI 1.5…"):
            passport = run_skills_signal_engine(user_input, country)

        if "error" in passport:
            st.error(f"Signal engine error: {passport.get('error')}")
            st.info("check if your GOOGLE_API_KEY is valid and has Gemini 1.5 Flahs enabled.")

        else:
            st.session_state.passport = passport
            st.session_state.passport_display = format_passport_for_display(passport)
            st.session_state.analysis_done = True

            # Auto-run risk analysis
            with st.spinner("🔬 Computing AI resilience scores…"):
                isco_code = passport.get("isco_code", "5221")
                st.session_state.risk_analysis = run_risk_analysis(
                    isco_code, country, st.session_state.passport_display
                )

    # ── Passport Display ───────────────────────────────────────────────────────
    if st.session_state.passport_display:
        pd_data = st.session_state.passport_display
        ra      = st.session_state.risk_analysis or {}
        res     = ra.get("resilience", {})

        st.markdown("---")
        st.markdown("### 📄 Your Skills Passport")

        # Header row
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(f"""
            <div class="passport-card">
              <div class="isco-badge">ISCO-08 · {pd_data['isco_code']} · {pd_data['isco_major_group']}</div>
              <p class="passport-headline">"{pd_data['headline']}"</p>
              <p style="color:#94a3b8; font-size:0.85rem; margin:0;">{pd_data['experience_summary']}</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            readiness = pd_data.get("formal_sector_readiness", "medium")
            r_color   = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(readiness, "#94a3b8")
            st.markdown(f"""
            <div class="metric-card">
              <h3>Formal Readiness</h3>
              <p class="value" style="color:{r_color}; font-size:1.3rem;">{readiness.upper()}</p>
              <p class="sub">{pd_data.get('years_equivalent','')}</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            form_ease = pd_data.get("formalization_ease", "medium")
            f_color   = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(form_ease, "#94a3b8")
            st.markdown(f"""
            <div class="metric-card">
              <h3>Formalization</h3>
              <p class="value" style="color:{f_color}; font-size:1.3rem;">{form_ease.upper()}</p>
              <p class="sub">Ease of certification</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            conf      = pd_data.get("confidence", "medium")
            c_color   = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(conf, "#94a3b8")
            st.markdown(f"""
            <div class="metric-card">
              <h3>ISCO Match</h3>
              <p class="value" style="color:{c_color}; font-size:1.3rem;">{conf.upper()}</p>
              <p class="sub">Taxonomy confidence</p>
            </div>
            """, unsafe_allow_html=True)

        # Skills breakdown
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<p class="section-header">🧠 Core Competencies</p>', unsafe_allow_html=True)
            pills_html = "".join(
                f'<span class="competency-pill">{c}</span>'
                for c in pd_data.get("core_competencies", [])
            )
            st.markdown(pills_html, unsafe_allow_html=True)

            st.markdown('<p class="section-header" style="margin-top:1rem;">🏅 Hidden Qualifications</p>', unsafe_allow_html=True)
            for q in pd_data.get("hidden_qualifications", []):
                st.markdown(f"✦ {q}")

        with col_r:
            st.markdown('<p class="section-header">🔄 Transferable Skills</p>', unsafe_allow_html=True)
            pills_html = "".join(
                f'<span class="durable-pill">{s}</span>'
                for s in pd_data.get("transferable_skills", [])
            )
            st.markdown(pills_html, unsafe_allow_html=True)

            st.markdown('<p class="section-header" style="margin-top:1rem;">🎯 Adjacent Roles</p>', unsafe_allow_html=True)
            for role in pd_data.get("durable_adjacent_roles", []):
                st.markdown(f"→ {role}")

        # Certification pathway
        if pd_data.get("certification_pathway"):
            st.markdown(f"""
            <div class="alert-info" style="margin-top:1rem;">
              📜 <strong>Recommended Certification Pathway:</strong> {pd_data['certification_pathway']}
            </div>
            """, unsafe_allow_html=True)

        # Next step message
        if pd_data.get("next_step_message"):
            st.markdown(f"""
            <div class="alert-success" style="margin-top:0.8rem;">
              🚀 <strong>Your next step:</strong> {pd_data['next_step_message']}
            </div>
            """, unsafe_allow_html=True)

        # JSON export
        with st.expander("🔧 Raw Passport JSON (for developers / integrations)"):
            st.json(st.session_state.passport)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — AI READINESS & DISPLACEMENT LENS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔬 AI Readiness & Displacement Lens")

    if not st.session_state.analysis_done:
        st.markdown("""
        <div class="alert-info">
          ℹ️ Complete the Skills Passport in <strong>Module 1</strong> first to unlock the Resilience Lens.
        </div>
        """, unsafe_allow_html=True)
    else:
        ra   = st.session_state.risk_analysis or {}
        res  = ra.get("resilience", {})
        tr   = ra.get("task_risk", {})
        disp = ra.get("displacement", {})
        opps = ra.get("opportunities", [])
        narr = ra.get("narrative", "")
        pd_d = st.session_state.passport_display

        score = res.get("score", 0) or 0
        color = res.get("color", "#94a3b8")
        tier  = res.get("tier", "—")

        # ── Resilience Score Gauge ─────────────────────────────────────────────
        col_gauge, col_narr = st.columns([1, 2])
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Resilience Score", "font": {"color": "#94a3b8", "size": 14}},
                number={"font": {"color": color, "size": 48}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569",
                             "tickfont": {"color": "#475569"}},
                    "bar":  {"color": color, "thickness": 0.3},
                    "bgcolor": "#1e293b",
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0,  38], "color": "#450a0a"},
                        {"range": [38, 55], "color": "#431407"},
                        {"range": [55, 72], "color": "#1c1917"},
                        {"range": [72, 100], "color": "#052e16"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 3},
                        "thickness": 0.8,
                        "value": score,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter, sans-serif"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"""
            <div style="text-align:center; margin-top:-1rem;">
              <span style="font-size:1.1rem; font-weight:700; color:{color};">{tier}</span><br>
              <span style="font-size:0.75rem; color:#64748b;">{res.get('label','')}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_narr:
            st.markdown('<p class="section-header">🤖 AI Resilience Brief</p>', unsafe_allow_html=True)
            if narr:
                st.markdown(f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:10px;
                            padding:1.2rem; color:#cbd5e1; font-size:0.88rem; line-height:1.7;">
                  {narr.replace(chr(10), "<br>")}
                </div>
                """, unsafe_allow_html=True)

            # Displacement timeline
            if disp:
                st.markdown(f"""
                <div class="alert-warning" style="margin-top:0.8rem;">
                  ⏱️ <strong>Local Disruption Timeline:</strong>
                  {disp.get('adjusted_timeline','—')} in {country}
                  (Global baseline: {disp.get('base_timeline','—')} · 
                  Lag factor: {disp.get('lag_factor','?')}x due to {disp.get('internet_penetration','?')}% internet penetration)
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Resilience Component Radar ─────────────────────────────────────────
        col_radar, col_tasks = st.columns(2)
        with col_radar:
            st.markdown('<p class="section-header">🕸️ Resilience Component Breakdown</p>', unsafe_allow_html=True)
            components = res.get("components", {})
            if components:
                cats   = list(components.keys())
                values = list(components.values())
                values.append(values[0])  # close polygon
                cats.append(cats[0])

                fig_radar = go.Figure(go.Scatterpolar(
                    r=values, theta=cats,
                    fill="toself",
                    fillcolor=f"rgba(99,102,241,0.15)",
                    line=dict(color="#6366f1", width=2),
                    name="Resilience Profile",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="#0f172a",
                        radialaxis=dict(visible=True, range=[0, 100],
                                        tickfont=dict(color="#475569", size=9),
                                        gridcolor="#1e293b"),
                        angularaxis=dict(tickfont=dict(color="#94a3b8", size=10),
                                         gridcolor="#1e293b"),
                    ),
                    showlegend=False,
                    height=320,
                    margin=dict(l=40, r=40, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        with col_tasks:
            st.markdown('<p class="section-header">⚡ Task-Level Risk Analysis</p>', unsafe_allow_html=True)

            at_risk = tr.get("at_risk", [])
            durable = tr.get("durable", [])

            if at_risk:
                st.markdown("**⚠️ Tasks at Risk of Automation:**")
                for task in at_risk:
                    bar_color = "#ef4444" if task["risk_pct"] > 70 else "#f97316"
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                      <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                        <span style="color:#fca5a5; font-size:0.8rem;">{task['task']}</span>
                        <span style="color:{bar_color}; font-size:0.8rem; font-weight:600;">{task['risk_pct']}%</span>
                      </div>
                      <div style="background:#1e293b; border-radius:3px; height:5px;">
                        <div style="background:{bar_color}; width:{task['risk_pct']}%; height:5px; border-radius:3px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            if durable:
                st.markdown("**🛡️ Durable Tasks (AI-resistant):**")
                for task in durable:
                    st.markdown(f"""
                    <span class="durable-pill">{task['icon']} {task['task']}</span>
                    """, unsafe_allow_html=True)

        # ── Adjacent Opportunities ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">🎯 Adjacent Durable Opportunities</p>', unsafe_allow_html=True)

        if opps:
            cols = st.columns(min(len(opps), 3))
            for i, opp in enumerate(opps):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="opp-card">
                      <h4>{opp['icon']} {opp['role']}</h4>
                      <p>{opp['category']} · Est. ${opp['wage_estimate_usd']:,.0f}/mo</p>
                      <p style="color:#475569; margin-top:0.3rem;">{opp['growth_alignment'][:60]}…</p>
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — OPPORTUNITY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Opportunity Dashboard")

    view_mode = st.radio(
        "Dashboard View",
        ["🎒 Youth View — Grounded Matching", "🏛️ Policy View — Aggregate Signals"],
        horizontal=True,
    )

    ctx = get_country_context(country)
    spd = compute_skill_population_divergence(country)
    rte = compute_returns_to_education(country)
    sgd = get_sector_growth_data(country)

    if "Youth View" in view_mode:
        st.markdown(f"#### 🎒 Grounded Matching · {country}")
        st.markdown(
            f"Real wage floors and reachable growth sectors for workers in **{country}** "
            f"based on WDI and ILOSTAT data."
        )

        # Wage comparison chart
        if ctx:
            wage_floor   = ctx.get("wage_floor_usd_month", 100)
            informal_w   = ctx.get("median_informal_wage_usd_month", 60)
            digital_w    = wage_floor * 1.4
            green_w      = wage_floor * 1.5
            skilled_w    = wage_floor * 1.7

            wage_data = pd.DataFrame({
                "Pathway":     ["Current Informal", "Wage Floor (Formal)", "Digital Economy",
                                "Green Economy", "Skilled Technician"],
                "Monthly USD": [informal_w, wage_floor, digital_w, green_w, skilled_w],
                "Color":       ["#64748b", "#6366f1", "#3b82f6", "#22c55e", "#f59e0b"],
            })

            fig_wage = px.bar(
                wage_data,
                x="Monthly USD",
                y="Pathway",
                orientation="h",
                color="Pathway",
                color_discrete_sequence=wage_data["Color"].tolist(),
                text="Monthly USD",
            )
            fig_wage.update_traces(
                texttemplate="$%{x:,.0f}",
                textposition="outside",
                marker_line_width=0,
            )
            fig_wage.update_layout(
                height=300,
                margin=dict(l=20, r=60, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#1e293b", color="#475569", title="Monthly Income (USD)"),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8", title=""),
                showlegend=False,
                font=dict(family="Inter, sans-serif", color="#94a3b8"),
            )
            st.plotly_chart(fig_wage, use_container_width=True)

        # Growth sectors
        st.markdown('<p class="section-header">🚀 Top Growth Sectors in {}</p>'.format(country), unsafe_allow_html=True)
        if sgd.get("top_sectors"):
            scols = st.columns(3)
            sector_icons = ["🌐", "⚙️", "💡"]
            for i, sector in enumerate(sgd["top_sectors"]):
                with scols[i]:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                      <p style="font-size:1.8rem; margin:0;">{sector_icons[i]}</p>
                      <h3 style="margin:0.3rem 0;">{sector}</h3>
                      <p class="sub">Top growth sector #{i+1}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Key signals for youth
        st.markdown("---")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Wage Floor",       f"${ctx.get('wage_floor_usd_month','?')}/mo",
                      f"+{rte.get('wage_gap',0):.0f} above informal")
        with col_s2:
            st.metric("Returns to Edu",   f"{rte.get('value','?')}%/yr",
                      "Per recognized credential")
        with col_s3:
            st.metric("Digital Jobs Share", f"{ctx.get('digital_jobs_share_pct','?')}%",
                      "Of formal jobs — growing")
        with col_s4:
            st.metric("TVET Enrollment", f"{ctx.get('tvet_enrollment_pct','?')}%",
                      "Of eligible youth")

    else:
        # ── Policy View ────────────────────────────────────────────────────────
        st.markdown("#### 🏛️ Policy View · Cross-Country Aggregate Signals")

        all_ctx = load_labor_context()

        # SPD across all countries
        spd_data = []
        for _, row in all_ctx.iterrows():
            c = row["country"]
            s = compute_skill_population_divergence(c)
            if s.get("value") is not None:
                spd_data.append({
                    "Country": c,
                    "Skill-Population Divergence": s["value"],
                    "Informal Rate %": row.get("informal_employment_rate_pct", 0),
                    "GDP Growth %": row.get("gdp_growth_pct", 0),
                    "Internet %": row.get("internet_penetration_pct", 0),
                    "Youth Unemployment %": row.get("youth_unemployment_pct", 0),
                    "Returns to Education %": row.get("returns_to_education_pct", 0),
                })

        spd_df = pd.DataFrame(spd_data).sort_values("Skill-Population Divergence", ascending=False)

        col_chart, col_scatter = st.columns(2)
        with col_chart:
            st.markdown('<p class="section-header">Skill-Population Divergence by Country</p>', unsafe_allow_html=True)
            fig_spd = px.bar(
                spd_df,
                x="Country",
                y="Skill-Population Divergence",
                color="Skill-Population Divergence",
                color_continuous_scale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                text_auto=".1f",
            )
            fig_spd.update_traces(marker_line_width=0)
            fig_spd.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=60),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#1e293b", color="#475569", tickangle=45),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
                coloraxis_showscale=False,
                font=dict(family="Inter, sans-serif", color="#94a3b8"),
            )
            st.plotly_chart(fig_spd, use_container_width=True)

        with col_scatter:
            st.markdown('<p class="section-header">Returns to Education vs. Informal Employment Rate</p>', unsafe_allow_html=True)
            fig_scatter = px.scatter(
                spd_df,
                x="Informal Rate %",
                y="Returns to Education %",
                size="Youth Unemployment %",
                color="GDP Growth %",
                color_continuous_scale=[[0,"#3b82f6"],[1,"#22c55e"]],
                hover_name="Country",
                text="Country",
            )
            fig_scatter.update_traces(textposition="top center", textfont_size=9)
            fig_scatter.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#1e293b", color="#475569",
                           title="Informal Employment Rate (%)"),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8",
                           title="Returns to Education (%)"),
                font=dict(family="Inter, sans-serif", color="#94a3b8"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Population projection table
        st.markdown('<p class="section-header">📊 Cross-Country Econometric Signal Table</p>', unsafe_allow_html=True)
        display_cols = [
            "Country", "Skill-Population Divergence",
            "Informal Rate %", "Returns to Education %",
            "Youth Unemployment %", "Internet %", "GDP Growth %",
        ]
        styled_df = spd_df[display_cols].copy()
        styled_df["Skill-Population Divergence"] = styled_df["Skill-Population Divergence"].map("{:.1f}".format)
        styled_df["Informal Rate %"]             = styled_df["Informal Rate %"].map("{:.1f}%".format)
        styled_df["Returns to Education %"]      = styled_df["Returns to Education %"].map("{:.1f}%".format)
        styled_df["Youth Unemployment %"]        = styled_df["Youth Unemployment %"].map("{:.1f}%".format)
        styled_df["Internet %"]                  = styled_df["Internet %"].map("{:.1f}%".format)
        styled_df["GDP Growth %"]                = styled_df["GDP Growth %"].map("{:.1f}%".format)

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Internet vs AI adoption policy signal
        st.markdown('<p class="section-header">📡 Internet Penetration vs. AI Adoption — Disruption Risk Matrix</p>', unsafe_allow_html=True)
        ar_df = all_ctx[["country", "internet_penetration_pct", "ai_adoption_index",
                          "informal_employment_rate_pct", "population_millions"]].dropna()
        ar_df.columns = ["Country", "Internet %", "AI Adoption", "Informal Rate %", "Population"]

        fig_matrix = px.scatter(
            ar_df,
            x="Internet %",
            y="AI Adoption",
            size="Population",
            color="Informal Rate %",
            color_continuous_scale=[[0,"#22c55e"],[0.5,"#f59e0b"],[1,"#ef4444"]],
            hover_name="Country",
            text="Country",
            labels={"AI Adoption": "AI Adoption Index (0–1)"},
        )
        fig_matrix.update_traces(textposition="top center", textfont_size=9)
        fig_matrix.add_hline(y=0.4, line_dash="dot", line_color="#6366f1",
                             annotation_text="AI adoption threshold",
                             annotation_font_color="#6366f1")
        fig_matrix.add_vline(x=50, line_dash="dot", line_color="#f59e0b",
                             annotation_text="50% internet",
                             annotation_font_color="#f59e0b")
        fig_matrix.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1e293b", color="#475569"),
            yaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
        )
        st.plotly_chart(fig_matrix, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔍 Data Explorer")
    st.markdown(
        "Inspect the three core data layers powering UNMAPPED. "
        "All CSV files are open and replaceable — this is a protocol, not a product."
    )

    data_view = st.selectbox("Select dataset", [
        "📚 Taxonomy (ISCO-08 Mapping)",
        "🌍 Labor Market Context",
        "🤖 Automation Risk Index",
    ])

    if "Taxonomy" in data_view:
        from data_loader import load_taxonomy
        df = load_taxonomy()
        st.markdown(f"**{len(df)} informal skill mappings** · ISCO-08 compliant")
        search = st.text_input("Search taxonomy", placeholder="e.g. phone, tailor, driver")
        if search:
            mask = df.apply(lambda row: search.lower() in row.to_string().lower(), axis=1)
            df   = df[mask]
        st.dataframe(df, use_container_width=True, hide_index=True)

    elif "Labor Market" in data_view:
        df = load_labor_context()
        st.markdown(f"**{len(df)} country contexts** · WDI / ILOSTAT inspired")
        cols_show = [
            "country", "region", "wage_floor_usd_month", "informal_employment_rate_pct",
            "internet_penetration_pct", "gdp_growth_pct", "returns_to_education_pct",
            "youth_unemployment_pct", "ai_adoption_index",
        ]
        st.dataframe(df[cols_show], use_container_width=True, hide_index=True)

    else:
        df = load_automation_risk()
        st.markdown(f"**{len(df)} ISCO codes profiled** · Frey-Osborne / ILO inspired")
        cols_show = [
            "isco_code", "isco_title", "automation_probability",
            "frey_osborne_category", "ilo_risk_tier",
            "task_routine_score", "lmic_displacement_timeline_years",
            "green_economy_relevance",
        ]
        st.dataframe(df[cols_show], use_container_width=True, hide_index=True)

        # Automation risk histogram
        st.markdown('<p class="section-header" style="margin-top:1.5rem;">Automation Probability Distribution (all ISCO codes)</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="automation_probability", nbins=10,
            color="frey_osborne_category",
            color_discrete_map={
                "Low Risk":       "#22c55e",
                "Medium Risk":    "#f59e0b",
                "High Risk":      "#f97316",
                "Very High Risk": "#ef4444",
            },
        )
        fig_hist.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1e293b", color="#475569", title="Automation Probability"),
            yaxis=dict(gridcolor="#1e293b", color="#94a3b8", title="ISCO Code Count"),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            font=dict(family="Inter, sans-serif", color="#94a3b8"),
            bargap=0.1,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

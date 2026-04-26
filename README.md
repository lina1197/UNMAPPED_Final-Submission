# 🗺️ UNMAPPED
**Open-Source Infrastructure Layer for Informal Skills & Economic Opportunity in LMICs**

> *"A protocol, not a product. Country-specific data are inputs, not hardcoded logic."*

---

## What is UNMAPPED?

UNMAPPED closes the gap between informal economic activity and formal opportunity pathways by building a modular, open infrastructure that:

- Translates informal work experience into internationally recognized **ISCO-08** profiles
- Computes **AI displacement risk** calibrated to local infrastructure (not just global indices)
- Surfaces **econometric signals** (Returns to Education, Skill-Population Divergence, GDP growth) directly in the UI
- Supports a **Dual Interface**: Youth View (grounded matching) and Policy View (aggregate signals)

---

## Architecture

```
unmapped/
├── app.py                  ← Streamlit UI (Context Configurator, 3 Modules, Data Explorer)
├── signal_engine.py        ← Module 1: Claude API → ISCO-08 Skills Passport
├── risk_analyzer.py        ← Module 2: Resilience Score + Displacement Timeline
├── data_loader.py          ← Data layer: loaders, econometric signal functions
├── requirements.txt
└── data/
    ├── taxonomy.csv              ← 30 informal skills → ISCO-08 mappings
    ├── labor_market_context.csv  ← 13 LMIC country contexts (WDI/ILOSTAT-inspired)
    └── automation_risk.csv       ← 27 ISCO codes, Frey-Osborne/ILO risk profiles
```

---

## Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Protocol, Not Product** | All country data are CSV inputs. Swap `labor_market_context.csv` for any country without touching code. |
| **Standardized Taxonomy** | Maps all informal skills to ISCO-08 (or ESCO-compatible) codes via Claude API + fuzzy pre-filter |
| **Constraint-First UI** | Streamlit renders on low-bandwidth mobile. No heavy JS. Sidebar = instant country switch. |

---

## Modules

### Module 1 — Skills Signal Engine
- **Input**: Free-text informal experience description (any language model can parse)
- **AI Logic**: Claude API maps to ISCO-08, extracts core competencies, hidden qualifications
- **Output**: Human-Readable **Skills Passport** with certification pathways

### Module 2 — AI Readiness & Displacement Lens
- **Resilience Score**: Composite score (0–100) weighting automation resistance, social intelligence, digital augmentability
- **Local Calibration**: Frey-Osborne timelines adjusted by `internet_penetration` and `ai_adoption_index`
- **Task Decomposition**: Shows *which specific tasks* are at risk vs. durable

### Module 3 — Opportunity Dashboard
- **Youth View**: Real wage floors, growth sectors, Returns to Education per country
- **Policy View**: Cross-country Skill-Population Divergence, GDP vs. informal rate scatter, AI adoption matrix

---

## Econometric Signals

| Signal | Formula | Source Inspiration |
|--------|---------|---------------------|
| **Skill-Population Divergence (SPD)** | `(sector_growth × 10) - formal_employment_rate` | World Bank WDI |
| **Returns to Education** | Mincerian rate from context CSV | ILOSTAT, Psacharopoulos & Patrinos |
| **Local AI Disruption Timeline** | `base_timeline × (1 + connectivity_lag_factor)` | Frey-Osborne + ILO |
| **Resilience Score** | Weighted composite of 5 ISCO-level indices | ILO Skills for Jobs DB |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run the app
streamlit run app.py
```

---

## Adding a New Country

1. Add a row to `data/labor_market_context.csv` with the country's WDI/ILOSTAT data
2. The Context Configurator sidebar will automatically include it
3. All signals, wage floors, and risk timelines update instantly

No code changes required. **This is the protocol in action.**

---

## Data Sources (Inspiration)

- [World Bank WDI](https://databank.worldbank.org/source/world-development-indicators)
- [ILOSTAT](https://ilostat.ilo.org/)
- [ILO ISCO-08 Taxonomy](https://www.ilo.org/public/english/bureau/stat/isco/)
- [Frey & Osborne (2013) — The Future of Employment](https://www.oxfordmartin.ox.ac.uk/downloads/academic/The_Future_of_Employment.pdf)
- [ILO — World Employment and Social Outlook](https://www.ilo.org/global/research/global-reports/weso/lang--en/index.htm)

---

## License

MIT — fork it, deploy it, extend it. This is open infrastructure.

---

*UNMAPPED v0.1 · Senior AI & Econometric Data Engineering Prototype*

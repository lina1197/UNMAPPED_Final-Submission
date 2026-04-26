"""
UNMAPPED — Data Loader Module
==============================
Loads, validates, and preprocesses the three core CSV data layers.
All paths are relative so the system is portable.
"""

import os
import pandas as pd
import numpy as np
from functools import lru_cache

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _resolve_data_path(filename: str) -> str:
    """Resolve a data file path with a fallback to the repository root."""
    candidate = os.path.join(DATA_DIR, filename)
    if os.path.exists(candidate):
        return candidate

    candidate = os.path.join(BASE_DIR, filename)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Data file not found: {filename}. Checked {DATA_DIR} and {BASE_DIR}."
    )


TAXONOMY_PATH         = _resolve_data_path("taxonomy.csv")
LABOR_CONTEXT_PATH    = _resolve_data_path("labor_market_context.csv")
AUTOMATION_RISK_PATH  = _resolve_data_path("automation_risk.csv")


# ── Loaders ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_taxonomy() -> pd.DataFrame:
    """Load and normalize the informal-skills-to-ISCO-08 taxonomy."""
    df = pd.read_csv(TAXONOMY_PATH, dtype=str)
    df.columns = df.columns.str.strip()
    df["isco_code"] = df["isco_code"].str.strip()
    return df


@lru_cache(maxsize=1)
def load_labor_context() -> pd.DataFrame:
    """Load country-level econometric context (WDI/ILOSTAT-inspired)."""
    df = pd.read_csv(LABOR_CONTEXT_PATH)
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "wage_floor_usd_month", "median_informal_wage_usd_month",
        "sector_growth_pct", "informal_employment_rate_pct",
        "internet_penetration_pct", "mobile_penetration_pct",
        "youth_unemployment_pct", "population_millions",
        "gdp_per_capita_usd", "gdp_growth_pct",
        "returns_to_education_pct", "gini_coefficient",
        "urban_population_pct", "electricity_access_pct",
        "ai_adoption_index", "digital_jobs_share_pct",
        "microfinance_access_pct", "tvet_enrollment_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@lru_cache(maxsize=1)
def load_automation_risk() -> pd.DataFrame:
    """Load ISCO-level automation risk data (Frey-Osborne / ILO-inspired)."""
    df = pd.read_csv(AUTOMATION_RISK_PATH)
    df.columns = df.columns.str.strip()
    df["isco_code"] = df["isco_code"].astype(str).str.strip()

    numeric_cols = [
        "task_routine_score", "automation_probability",
        "manual_dexterity_index", "social_perception_index",
        "creative_intelligence_index", "physical_demand_index",
        "digital_augmentation_potential",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Context Helpers ────────────────────────────────────────────────────────────

def get_country_context(country: str) -> dict:
    """Return a single country's econometric context as a dict."""
    df = load_labor_context()
    row = df[df["country"].str.lower() == country.lower()]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_available_countries() -> list:
    """Return sorted list of countries in the context dataset."""
    return sorted(load_labor_context()["country"].tolist())


def get_isco_risk(isco_code: str) -> dict:
    """Fetch automation risk record for a given ISCO code."""
    df = load_automation_risk()
    row = df[df["isco_code"] == str(isco_code)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def match_skill_to_isco(user_input: str) -> list[dict]:
    """
    Fuzzy keyword match against taxonomy for quick pre-filtering
    before the Claude API call. Returns top candidate records.
    """
    df = load_taxonomy()
    user_lower = user_input.lower()
    keywords = [w for w in user_lower.split() if len(w) > 3]

    scores = []
    for _, row in df.iterrows():
        haystack = " ".join([
            str(row.get("informal_skill", "")),
            str(row.get("informal_skill_alias", "")),
            str(row.get("isco_title", "")),
            str(row.get("skill_cluster", "")),
            str(row.get("transferable_skills", "")),
        ]).lower()
        score = sum(kw in haystack for kw in keywords)
        if score > 0:
            scores.append((score, row.to_dict()))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [record for _, record in scores[:3]]


# ── Econometric Signal Computation ────────────────────────────────────────────

def compute_skill_population_divergence(country: str) -> dict:
    """
    SPD = gap between high-growth-sector demand and available skilled labor.
    Proxy: (sector_growth_pct * 10) - (1 - informal_employment_rate/100) * 100
    Higher = greater urgency for reskilling investment.
    """
    ctx = get_country_context(country)
    if not ctx:
        return {"value": None, "interpretation": "No data"}

    informal_rate = ctx.get("informal_employment_rate_pct", 70)
    sector_growth = ctx.get("sector_growth_pct", 4)
    formal_rate   = 100 - informal_rate

    spd = (sector_growth * 10) - formal_rate
    spd = round(spd, 2)

    if spd > 15:
        label = "Critical"
        color = "#ef4444"
        interpretation = (
            f"Severe skill-to-opportunity gap. {round(informal_rate)}% of workers "
            f"operate informally while formal sectors grow at {sector_growth}%/yr. "
            "Reskilling investment has highest ROI here."
        )
    elif spd > 5:
        label = "Elevated"
        color = "#f97316"
        interpretation = (
            f"Significant divergence between sector growth and formal skill supply. "
            f"Targeted TVET and certification programs urgently needed."
        )
    else:
        label = "Moderate"
        color = "#eab308"
        interpretation = (
            f"Moderate skill gap. Formalization programs can absorb informal workers "
            "with focused effort."
        )

    return {
        "value": spd,
        "label": label,
        "color": color,
        "interpretation": interpretation,
        "informal_rate": informal_rate,
        "sector_growth": sector_growth,
    }


def compute_returns_to_education(country: str) -> dict:
    """
    Returns to Education (Mincerian) signal from context data.
    Contextualizes what a 1-year credential gain is worth in wage terms.
    """
    ctx = get_country_context(country)
    if not ctx:
        return {"value": None}

    rte = ctx.get("returns_to_education_pct", 8.5)
    wage_floor = ctx.get("wage_floor_usd_month", 100)
    informal_wage = ctx.get("median_informal_wage_usd_month", 60)
    wage_gap = wage_floor - informal_wage
    potential_gain = round(informal_wage * (rte / 100) * 12, 1)  # annual

    return {
        "value": rte,
        "wage_floor": wage_floor,
        "informal_wage": informal_wage,
        "wage_gap": wage_gap,
        "annual_gain_per_year_edu": potential_gain,
        "interpretation": (
            f"Each additional year of recognized education/training "
            f"yields ~{rte}% wage increase in {country}. "
            f"Closing the formal-informal wage gap (${wage_gap}/mo) "
            f"could add ~${potential_gain}/yr to a worker's income."
        ),
    }


def compute_ai_displacement_timeline(
    isco_code: str,
    country: str
) -> dict:
    """
    Adjust Frey-Osborne displacement timeline by local internet penetration.
    Low connectivity = slower AI disruption (infrastructure lag effect).
    """
    risk = get_isco_risk(isco_code)
    ctx  = get_country_context(country)

    if not risk or not ctx:
        return {}

    base_low, base_high = _parse_timeline(
        risk.get("lmic_displacement_timeline_years", "10-15")
    )
    internet  = ctx.get("internet_penetration_pct", 50)
    ai_index  = ctx.get("ai_adoption_index", 0.3)

    # Infrastructure lag multiplier: lower penetration → slower disruption
    lag_factor = 1 + (1 - internet / 100) * 0.5 + (1 - ai_index) * 0.3
    adj_low  = round(base_low  * lag_factor, 1)
    adj_high = round(base_high * lag_factor, 1)

    return {
        "base_timeline": f"{base_low}–{base_high} years",
        "adjusted_timeline": f"{adj_low}–{adj_high} years",
        "lag_factor": round(lag_factor, 2),
        "internet_penetration": internet,
        "ai_adoption_index": ai_index,
        "note": (
            f"With {internet}% internet penetration and AI adoption index of {ai_index}, "
            f"market disruption in {country} is estimated {adj_low}–{adj_high} years "
            f"(vs global baseline {base_low}–{base_high} years)."
        ),
    }


def _parse_timeline(timeline_str: str) -> tuple[float, float]:
    """Parse '8-12' or '15+' into (low, high) floats."""
    s = str(timeline_str).replace(" years", "").strip()
    if "+" in s:
        val = float(s.replace("+", ""))
        return val, val + 5
    if "-" in s:
        parts = s.split("-")
        return float(parts[0]), float(parts[1])
    val = float(s)
    return val, val


def get_all_taxonomy_skills() -> list[str]:
    """Return list of all informal skills for autocomplete."""
    df = load_taxonomy()
    return sorted(df["informal_skill"].tolist())


def get_sector_growth_data(country: str) -> dict:
    """Return top growth sectors and GDP signal for dashboard."""
    ctx = get_country_context(country)
    if not ctx:
        return {}
    return {
        "top_sectors": [
            ctx.get("top_growth_sector_1", "—"),
            ctx.get("top_growth_sector_2", "—"),
            ctx.get("top_growth_sector_3", "—"),
        ],
        "gdp_growth": ctx.get("gdp_growth_pct", 0),
        "gdp_per_capita": ctx.get("gdp_per_capita_usd", 0),
        "isco_demand_shift": ctx.get("isco_demand_shift", "—"),
        "digital_jobs_share": ctx.get("digital_jobs_share_pct", 0),
    }

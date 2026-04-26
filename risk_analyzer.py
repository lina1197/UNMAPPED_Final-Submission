"""
UNMAPPED — Risk Analyzer (Module 2)
=====================================
Computes the AI Readiness & Displacement Lens:
- Resilience Score (composite)
- Local disruption timeline (calibrated to infrastructure)
- Skill durability breakdown
- Adjacent opportunity mapping
"""
import os
import json
import google.generativeai as genai 
from dotenv import load_dotenv
from data_loader import (
    get_isco_risk,
    get_country_context,
    compute_ai_displacement_timeline,
    load_taxonomy,
)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME  = "gemini-flash-latest"


# ── Resilience Score Computation ───────────────────────────────────────────────

def compute_resilience_score(
    isco_code: str,
    country: str,
    passport_data: dict | None = None,
) -> dict:
    """
    Composite Resilience Score (0–100) using:
      - Automation probability (inverted)       → 35% weight
      - Social perception index                 → 20% weight
      - Digital augmentation potential          → 20% weight
      - Manual dexterity (for physical trades)  → 10% weight
      - Creative intelligence index             → 15% weight
    
    Adjusted by country's internet penetration and AI adoption index.
    """
    risk = get_isco_risk(isco_code)
    ctx  = get_country_context(country)

    if not risk:
        return {"score": None, "error": "No risk data for this ISCO code"}

    # Raw score components
    auto_prob     = risk.get("automation_probability", 0.5)
    social        = risk.get("social_perception_index", 0.5)
    digital_aug   = risk.get("digital_augmentation_potential", 0.5)
    dexterity     = risk.get("manual_dexterity_index", 0.5)
    creativity    = risk.get("creative_intelligence_index", 0.5)

    # Weighted composite (before country adjustment)
    raw_score = (
        (1 - auto_prob)  * 0.35 +
        social           * 0.20 +
        digital_aug      * 0.20 +
        dexterity        * 0.10 +
        creativity       * 0.15
    ) * 100

    # Country adjustment: low internet = slower disruption = higher resilience today
    internet     = ctx.get("internet_penetration_pct", 50) if ctx else 50
    ai_index     = ctx.get("ai_adoption_index", 0.3) if ctx else 0.3
    country_boost = (1 - internet / 100) * 8 + (1 - ai_index) * 5
    final_score   = min(100, raw_score + country_boost)

    # Tier classification
    tier, color, label = _classify_resilience(final_score)

    # Component breakdown for radar chart
    components = {
        "Automation Resistance":   round((1 - auto_prob) * 100, 1),
        "Social Intelligence":     round(social * 100, 1),
        "Digital Augmentability":  round(digital_aug * 100, 1),
        "Physical Craft Value":    round(dexterity * 100, 1),
        "Creative Complexity":     round(creativity * 100, 1),
    }

    return {
        "score": round(final_score, 1),
        "raw_score": round(raw_score, 1),
        "country_boost": round(country_boost, 1),
        "tier": tier,
        "color": color,
        "label": label,
        "components": components,
        "isco_code": isco_code,
        "country": country,
    }


def _classify_resilience(score: float) -> tuple[str, str, str]:
    if score >= 72:
        return "Resilient", "#22c55e", "Your skills are highly resilient to AI disruption"
    elif score >= 55:
        return "Moderate", "#f59e0b", "Moderate resilience — targeted upskilling will protect your income"
    elif score >= 38:
        return "Vulnerable", "#f97316", "Vulnerability detected — act now to build durable skills"
    else:
        return "At Risk", "#ef4444", "High risk of displacement — urgent reskilling pathway needed"


# ── Task-Level Risk Decomposition ─────────────────────────────────────────────

def decompose_task_risk(isco_code: str) -> dict:
    """
    Break down which specific tasks are at risk vs. which are durable.
    Returns structured task cards for UI rendering.
    """
    risk = get_isco_risk(isco_code)
    if not risk:
        return {}

    at_risk_raw = risk.get("at_risk_tasks", "")
    durable_raw = risk.get("durable_tasks", "")

    at_risk_tasks = [t.strip() for t in at_risk_raw.split(",") if t.strip()]
    durable_tasks = [t.strip() for t in durable_raw.split(",") if t.strip()]

    auto_prob = risk.get("automation_probability", 0.5)
    routine   = risk.get("task_routine_score", 0.5)

    # Assign individual risk levels to tasks heuristically
    task_risk_cards = []
    n = len(at_risk_tasks)
    for i, task in enumerate(at_risk_tasks):
        individual_risk = auto_prob + (routine * 0.1) - (i * 0.05)
        individual_risk = max(0.3, min(0.99, individual_risk))
        task_risk_cards.append({
            "task": task,
            "risk_pct": round(individual_risk * 100),
            "status": "at_risk",
            "icon": "⚠️",
        })

    durable_cards = []
    for task in durable_tasks:
        durable_cards.append({
            "task": task,
            "risk_pct": round((1 - auto_prob) * 100),
            "status": "durable",
            "icon": "🛡️",
        })

    return {
        "at_risk": task_risk_cards,
        "durable": durable_cards,
        "overall_automation_prob": auto_prob,
        "frey_osborne_category": risk.get("frey_osborne_category"),
        "ilo_risk_tier": risk.get("ilo_risk_tier"),
    }


# ── Adjacent Opportunity Mapping ───────────────────────────────────────────────

def get_adjacent_opportunities(
    isco_code: str,
    country: str,
    passport_durable_skills: list[str] | None = None,
) -> list[dict]:
    """
    Map durable skills to adjacent formal-sector roles reachable within
    the country's wage floor and growth sectors.
    """
    risk    = get_isco_risk(isco_code)
    ctx     = get_country_context(country)
    taxonomy = load_taxonomy()

    if not risk or not ctx:
        return []

    resilience_pathway = risk.get("resilience_pathway", "")
    green_relevance    = risk.get("green_economy_relevance", "Low")
    care_relevance     = risk.get("care_economy_relevance", "Low")
    wage_floor         = ctx.get("wage_floor_usd_month", 100)
    top_sectors        = [
        ctx.get("top_growth_sector_1", ""),
        ctx.get("top_growth_sector_2", ""),
        ctx.get("top_growth_sector_3", ""),
    ]

    # Pull adjacent skills from taxonomy
    tax_row = taxonomy[taxonomy["isco_code"] == str(isco_code)]
    durable_adjacent = []
    if not tax_row.empty:
        durable_adjacent = [
            s.strip() for s in
            tax_row.iloc[0].get("durable_adjacent_skills", "").split(",")
            if s.strip()
        ]

    opportunities = []

    # From taxonomy durable adjacents
    for adj in durable_adjacent[:3]:
        opportunities.append({
            "role": adj,
            "source": "Taxonomy adjacency",
            "wage_estimate_usd": round(wage_floor * 1.2, 0),
            "growth_alignment": _check_sector_alignment(adj, top_sectors),
            "category": "Adjacent Formal Role",
            "urgency": "medium",
            "icon": "🎯",
        })

    # Green economy opportunity
    if green_relevance in ["High", "Medium"]:
        opportunities.append({
            "role": "Green Economy / Climate Tech Specialist",
            "source": "ILO Green Jobs",
            "wage_estimate_usd": round(wage_floor * 1.5, 0),
            "growth_alignment": "High — climate adaptation is a top global priority",
            "category": "Green Economy",
            "urgency": "high",
            "icon": "🌱",
        })

    # Care economy opportunity
    if care_relevance in ["High", "Medium"]:
        opportunities.append({
            "role": "Care Economy / Community Services Role",
            "source": "ILO Care Work Report",
            "wage_estimate_usd": round(wage_floor * 1.1, 0),
            "growth_alignment": "Growing — aging populations and health system gaps",
            "category": "Care Economy",
            "urgency": "medium",
            "icon": "🤝",
        })

    # Digital economy opportunity
    digital_aug = risk.get("digital_augmentation_potential", 0.5)
    if digital_aug > 0.6:
        opportunities.append({
            "role": "Digital Platform Worker / Remote Gig Economy",
            "source": "World Bank Digital Economy",
            "wage_estimate_usd": round(wage_floor * 1.3, 0),
            "growth_alignment": "High — internet penetration enabling remote work access",
            "category": "Digital Economy",
            "urgency": "high",
            "icon": "💻",
        })

    return opportunities[:5]


def _check_sector_alignment(role: str, top_sectors: list[str]) -> str:
    role_lower = role.lower()
    for sector in top_sectors:
        if sector and any(
            kw in role_lower for kw in sector.lower().split()
        ):
            return f"High — aligns with top growth sector: {sector}"
    return "Moderate — transferable across multiple sectors"


# ── AI-Powered Resilience Narrative ───────────────────────────────────────────

RESILIENCE_SYSTEM = """You are UNMAPPED's Risk Analyst. Given a worker's Skills Passport 
and their Resilience Score, write a short, plain-language resilience brief (4-6 sentences).

Rules:
- Be honest but empowering; avoid fear-mongering
- Explain which tasks are safe and which are at risk in plain terms
- Suggest 1 specific durable skill to invest in immediately
- Mention the local timeline (not global) for disruption
- Never use jargon without explaining it

Respond in plain text only, no JSON, no markdown headers."""

def generate_resilience_narrative(
    passport_display: dict,
    resilience_data: dict,
    displacement_data: dict,
    country: str,
) -> str:
    """Use Gemini to generate a plain-language resilience brief."""
    
    prompt = f"""
Worker Profile:
- Role: {passport_display.get('isco_title', '—')}
- Country: {country}
- Resilience Score: {resilience_data.get('score', '—')}/100 ({resilience_data.get('tier', '—')})
- At-risk tasks: {', '.join(passport_display.get('at_risk_tasks', []))}
- Durable tasks: {', '.join(passport_display.get('durable_tasks', []))}
- Adjusted disruption timeline: {displacement_data.get('adjusted_timeline', '—')}
- Resilience pathway: {passport_display.get('resilience_pathway', '—')}
- Internet penetration in {country}: {displacement_data.get('internet_penetration', '—')}%

Write the resilience brief now.
"""

    # 2. Initialize the model with System Instructions
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=RESILIENCE_SYSTEM
    )

    # 3. Generate response
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate resilience brief: {str(e)}"


# ── Full Module 2 Pipeline ─────────────────────────────────────────────────────

def run_risk_analysis(
    isco_code: str,
    country: str,
    passport_display: dict,
) -> dict:
    """
    Full Module 2 pipeline returning all risk analysis signals.
    """
    resilience   = compute_resilience_score(isco_code, country)
    task_risk    = decompose_task_risk(isco_code)
    displacement = compute_ai_displacement_timeline(isco_code, country)
    opportunities = get_adjacent_opportunities(isco_code, country)

    # Note: We pass the data to Gemini here
    narrative = generate_resilience_narrative(
        passport_display, resilience, displacement, country
    )

    return {
        "resilience": resilience,
        "task_risk": task_risk,
        "displacement": displacement,
        "opportunities": opportunities,
        "narrative": narrative,
    }
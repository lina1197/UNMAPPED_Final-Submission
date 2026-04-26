"""
UNMAPPED — Signal Engine (Module 1)
=====================================
Uses google generative AIto map informal experience descriptions to ISCO-08 profiles
and generate a Human-Readable Skills Passport in plain language.
"""

import json
import os
import google.generativeai as genai 
from dotenv import load_dotenv 
from data_loader import (
    load_taxonomy,
    match_skill_to_isco,
    get_isco_risk,
    get_country_context,
)

# Configuration

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini 1.5 Flash is highly capable and fits well within free-tier limits
MODEL_NAME = "gemini-flash-latest"

# ── System Prompt ──────────────────────────────────────────────────────────────

SIGNAL_ENGINE_SYSTEM = """You are UNMAPPED's Skills Signal Engine — an econometric AI assistant 
that translates informal work experience into internationally recognized occupational profiles.

Your audience is a first-generation worker in a Low- or Middle-Income Country (LMIC), 
someone like "Amara" who runs a phone stall in Lagos or "Ravi" who repairs motorcycles in Chennai.

Your task:
1. Parse the user's described experience into concrete tasks and capabilities.
2. Map these to the best-fitting ISCO-08 occupational code from the reference taxonomy.
3. Generate a Human-Readable Skills Passport in plain, dignified, empowering language.

STRICT OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:
{
  "isco_code": "4-digit code as string",
  "isco_title": "Official ISCO-08 title",
  "isco_major_group": "Group number and label",
  "confidence": "high | medium | low",
  "skills_passport": {
    "headline": "One powerful sentence describing who this person is professionally",
    "experience_summary": "2-3 sentences translating their informal experience into formal terms",
    "core_competencies": ["competency 1", "competency 2", "competency 3", "competency 4", "competency 5"],
    "hidden_qualifications": ["qualification 1", "qualification 2", "qualification 3"],
    "years_equivalent": "e.g. '3-5 years relevant experience'",
    "formal_sector_readiness": "high | medium | low",
    "readiness_explanation": "One sentence explaining readiness level"
  },
  "transferable_skills": ["skill 1", "skill 2", "skill 3"],
  "durable_adjacent_roles": ["role 1", "role 2", "role 3"],
  "certification_pathway": "Primary recommended certification pathway",
  "formalization_ease": "high | medium | low",
  "next_step_message": "One actionable, encouraging sentence for the worker"
}

Tone rules:
- Never use condescending language ("just a vendor", "unskilled", "low-level")
- Translate informal work into its true economic contribution
- Be specific: "managed cash flow of ~$200/day across 3 product lines" not "handled money"
- Use the taxonomy reference if provided for precision"""


# ── Core API Call ──────────────────────────────────────────────────────────────

def generate_skills_passport(
    user_input: str,
    country: str = "Nigeria",
    taxonomy_hints: list[dict] | None = None,
) -> dict:
    
    # 1. Prepare Context
    taxonomy_ref = ""
    if taxonomy_hints:
        taxonomy_ref = "\n\nREFERENCE TAXONOMY CANDIDATES:\n" + json.dumps(
            taxonomy_hints, indent=2
        )

    country_ctx = get_country_context(country)
    context_note = ""
    if country_ctx:
        context_note = (
            f"\n\nCOUNTRY CONTEXT: {country} | "
            f"Informal employment rate: {country_ctx.get('informal_employment_rate_pct', '?')}% | "
            f"Top growth sectors: {country_ctx.get('top_growth_sector_1', '?')}, "
            f"{country_ctx.get('top_growth_sector_2', '?')} | "
            f"Wage floor: ${country_ctx.get('wage_floor_usd_month', '?')}/month"
        )

    user_message = (
        f"Worker's informal experience:\n\"{user_input}\"\n"
        f"Country: {country}"
        f"{context_note}"
        f"{taxonomy_ref}"
        "\n\nGenerate the Skills Passport JSON."
    )

    # 2. Initialize Model with System Instruction
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SIGNAL_ENGINE_SYSTEM
    )

    # 3. Generate Content
    # We use response_mime_type to force Gemini to return valid JSON
    response = model.generate_content(
        user_message,
        generation_config={"response_mime_type": "application/json"}
    )

    try:
        # Gemini usually returns clean JSON string when mime_type is set
        return json.loads(response.text)
    except json.JSONDecodeError:
        # Fallback for parsing if something went wrong
        raw = response.text.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        try:
            return json.loads(raw)
        except:
            return {"error": "Failed to parse Skills Passport JSON", "raw": response.text}


# ── Enrichment: Merge with Local Data Layers ──────────────────────────────────

def enrich_passport_with_risk(passport: dict) -> dict:
    """Merge automation risk data into the Skills Passport."""
    isco_code = passport.get("isco_code", "")
    risk = get_isco_risk(isco_code)
    if risk:
        passport["automation_risk"] = {
            "probability": risk.get("automation_probability"),
            "routine_score": risk.get("task_routine_score"),
            "frey_osborne_category": risk.get("frey_osborne_category"),
            "ilo_risk_tier": risk.get("ilo_risk_tier"),
            "timeline_years": risk.get("lmic_displacement_timeline_years"),
            "at_risk_tasks": risk.get("at_risk_tasks", "").split(", "),
            "durable_tasks": risk.get("durable_tasks", "").split(", "),
            "resilience_pathway": risk.get("resilience_pathway"),
            "digital_augmentation_potential": risk.get("digital_augmentation_potential"),
            "green_economy_relevance": risk.get("green_economy_relevance"),
            "care_economy_relevance": risk.get("care_economy_relevance"),
        }
    return passport


def run_skills_signal_engine(
    user_input: str,
    country: str = "Nigeria"
) -> dict:
    """
    Full pipeline: fuzzy match → Claude API → ISCO enrichment → risk merge.
    Returns a complete Skills Passport with all signals attached.
    """
    # Step 1: Pre-filter taxonomy for context
    hints = match_skill_to_isco(user_input)

    # Step 2: Claude API mapping
    passport = generate_skills_passport(user_input, country, hints)

    if "error" in passport:
        return passport

    # Step 3: Merge automation risk
    passport = enrich_passport_with_risk(passport)

    # Step 4: Tag with pipeline metadata
    passport["_meta"] = {
        "input": user_input,
        "country": country,
        "taxonomy_hints_count": len(hints),
        "model": MODEL_NAME,
    }

    return passport


# ── Passport Formatter for UI ──────────────────────────────────────────────────

def format_passport_for_display(passport: dict) -> dict:
    """
    Flatten passport into display-ready sections for Streamlit rendering.
    """
    sp = passport.get("skills_passport", {})
    ar = passport.get("automation_risk", {})

    return {
        "isco_code": passport.get("isco_code", "—"),
        "isco_title": passport.get("isco_title", "—"),
        "isco_major_group": passport.get("isco_major_group", "—"),
        "confidence": passport.get("confidence", "—"),
        "headline": sp.get("headline", ""),
        "experience_summary": sp.get("experience_summary", ""),
        "core_competencies": sp.get("core_competencies", []),
        "hidden_qualifications": sp.get("hidden_qualifications", []),
        "years_equivalent": sp.get("years_equivalent", ""),
        "formal_sector_readiness": sp.get("formal_sector_readiness", "—"),
        "readiness_explanation": sp.get("readiness_explanation", ""),
        "transferable_skills": passport.get("transferable_skills", []),
        "durable_adjacent_roles": passport.get("durable_adjacent_roles", []),
        "certification_pathway": passport.get("certification_pathway", ""),
        "formalization_ease": passport.get("formalization_ease", "—"),
        "next_step_message": passport.get("next_step_message", ""),
        "automation_probability": ar.get("probability"),
        "risk_category": ar.get("frey_osborne_category", "—"),
        "ilo_risk_tier": ar.get("ilo_risk_tier", "—"),
        "at_risk_tasks": ar.get("at_risk_tasks", []),
        "durable_tasks": ar.get("durable_tasks", []),
        "resilience_pathway": ar.get("resilience_pathway", ""),
        "green_relevance": ar.get("green_economy_relevance", "—"),
        "care_relevance": ar.get("care_economy_relevance", "—"),
    }
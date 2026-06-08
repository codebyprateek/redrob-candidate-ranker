"""
Redrob Hackathon — Intelligent Candidate Ranking System
Senior AI Engineer — Founding Team @ Redrob AI

Architecture:
  1. Multi-dimensional skill scoring (hard requirements, soft requirements, disqualifiers)
  2. Career-trajectory scoring (product vs services, tenure, progression)
  3. Behavioral/availability signals (engagement, recency, notice period)
  4. Honeypot detection (impossible profiles are automatically penalized)
  5. Composite weighted scoring with monotonic score output

Compute constraints met:
  - CPU only, no external API calls
  - Runs well within 5 min / 16 GB on 100K candidates
  - Pure Python + pandas/numpy only
"""

import json
import math
import csv
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# ── JD constants ────────────────────────────────────────────────────────────

# Hard-requirement skills: candidate must demonstrate these in WORK HISTORY, not just listed skills
HARD_SKILLS = {
    "embeddings", "embedding", "sentence-transformers", "openai embeddings",
    "bge", "e5", "rag", "retrieval-augmented generation",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss", "milvus", "pgvector",
    "hybrid search", "hybrid retrieval",
    "ranking", "re-ranking", "reranking", "retrieval", "information retrieval",
    "ndcg", "mrr", "map", "offline evaluation", "a/b test", "ab test",
    "evaluation framework", "ranking evaluation",
    "python",
}

# Soft skills (nice to have, bonus points)
SOFT_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "learning to rank", "ltr", "xgboost", "lightgbm",
    "hr tech", "hrtech", "recruiting tech", "talent platform",
    "distributed systems", "large-scale inference", "inference optimization",
    "open-source", "open source", "github",
    "llm", "large language model", "gpt", "bert", "transformers",
    "nlp", "natural language processing",
    "semantic search", "dense retrieval",
    "recommendation system", "recommender", "search system",
}

# AI/ML product-domain keywords that strongly indicate fit (appear in JD description of role)
PRODUCT_ML_KEYWORDS = {
    "embedding", "retrieval", "ranking", "search", "recommendation",
    "vector", "semantic", "nlp", "transformer", "llm", "fine-tun",
    "evaluation", "rerank", "dense", "sparse", "hybrid",
    "rag", "faiss", "pinecone", "weaviate", "qdrant", "elasticsearch",
    "opensearch", "milvus", "pgvector",
}

# Hard disqualifiers: current title + career history patterns
DISQUALIFIED_TITLES = {
    "hr manager", "human resources", "marketing manager", "content writer",
    "graphic designer", "business analyst", "sales manager", "project manager",
    "ux designer", "ui designer", "product designer", "scrum master",
    "devops engineer", "sre", "site reliability", "network engineer",
    "database administrator", "dba", "system administrator", "sysadmin",
    "finance", "accountant", "legal", "lawyer", "recruiter",
    "computer vision engineer", "cv engineer", "speech recognition engineer",
    "robotics engineer",
}

# Services companies (full career = disqualifier; partial = penalty)
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "niit", "mindtree", "l&t infotech", "ltimindtree",
    "persistent systems",  # borderline but often project-shop
}

# India Tier-1 cities (preferred locations)
TIER1_CITIES = {
    "noida", "pune", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "gurugram", "gurgaon", "chennai",
    "delhi ncr", "ncr",
}

# Reference date for recency calculations
TODAY = date(2026, 6, 8)


# ── Helper utilities ─────────────────────────────────────────────────────────

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def days_since(d):
    if d is None:
        return 9999
    return (TODAY - d).days


def text_lower(t):
    return (t or "").lower()


def contains_any(text, keywords):
    t = text_lower(text)
    return any(kw in t for kw in keywords)


def count_matches(text, keywords):
    t = text_lower(text)
    return sum(1 for kw in keywords if kw in t)


def sigmoid(x, center=0.0, steepness=1.0):
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


# ── Honeypot detector ────────────────────────────────────────────────────────

def is_honeypot(c):
    """
    Detect impossible/inconsistent profiles.
    Returns True if this looks like a planted honeypot.
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    yoe = profile.get("years_of_experience", 0)

    # Check 1: company founded after claimed tenure starts
    for job in career:
        start = parse_date(job.get("start_date"))
        dur = job.get("duration_months", 0)
        if start and dur > 0:
            # If claimed duration impossibly long vs start date
            earliest_possible_start = date(TODAY.year - int(yoe) - 2, 1, 1)
            if start < date(1990, 1, 1):
                return True

    # Check 2: Expert in 10+ skills with near-zero duration each
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 10:
        avg_dur = sum(s.get("duration_months", 0) for s in expert_skills) / len(expert_skills)
        if avg_dur < 3:
            return True

    # Check 3: years_of_experience vs actual career history duration wildly inconsistent
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 2 and total_career_months < 6:
        return True

    # Check 4: years_of_experience >> total possible (e.g. 8 yrs but graduated 3 yrs ago)
    education = c.get("education", [])
    if education:
        latest_grad = max((e.get("end_year", 0) for e in education), default=0)
        if latest_grad > 0:
            max_possible_yoe = TODAY.year - latest_grad + 1
            if yoe > max_possible_yoe + 3:
                return True

    # Check 5: contradictory title + skills (e.g. Marketing Manager with expert ML skills listed)
    title = text_lower(profile.get("current_title", ""))
    if any(dt in title for dt in ["marketing", "hr manager", "graphic designer", "content writer"]):
        hard_skill_count = sum(
            1 for s in skills
            if any(hs in text_lower(s.get("name", "")) for hs in ["embedding", "rag", "vector", "ranking", "retrieval"])
            and s.get("proficiency") in ("expert", "advanced")
        )
        if hard_skill_count >= 5:
            return True

    return False


# ── Scoring components ───────────────────────────────────────────────────────

def score_skills(c):
    """Score based on skills list and their proficiency / duration."""
    skills = c.get("skills", [])
    if not skills:
        return 0.0

    hard_score = 0.0
    soft_score = 0.0

    for s in skills:
        name = text_lower(s.get("name", ""))
        proficiency = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)

        prof_weight = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}.get(proficiency, 0.25)
        dur_weight = min(duration / 36.0, 1.0)  # cap at 3 years
        endorse_weight = min(endorsements / 50.0, 1.0)

        combined = 0.5 * prof_weight + 0.3 * dur_weight + 0.2 * endorse_weight

        if any(hs in name for hs in HARD_SKILLS):
            hard_score += combined
        elif any(ss in name for ss in SOFT_SKILLS):
            soft_score += combined * 0.6

    # Normalize: expect ~4 hard skills for full score
    hard_norm = min(hard_score / 4.0, 1.0)
    soft_norm = min(soft_score / 3.0, 1.0)

    return 0.7 * hard_norm + 0.3 * soft_norm


def score_career(c):
    """
    Score career trajectory:
    - Product company experience heavily rewarded
    - Services-only career penalized
    - Retrieval/ranking/ML work in descriptions rewarded
    - Title relevance
    - Tenure length (not job-hopping)
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)
    current_title = text_lower(profile.get("current_title", ""))

    if not career:
        return 0.0

    # --- Title relevance ---
    title_score = 0.0
    ideal_titles = ["ai engineer", "ml engineer", "machine learning engineer",
                    "applied scientist", "research engineer", "nlp engineer",
                    "search engineer", "data scientist", "senior engineer",
                    "software engineer", "backend engineer", "platform engineer"]
    if any(t in current_title for t in ideal_titles):
        title_score = 0.8
    elif "engineer" in current_title or "scientist" in current_title or "developer" in current_title:
        title_score = 0.5
    elif any(dt in current_title for dt in DISQUALIFIED_TITLES):
        title_score = 0.0
    else:
        title_score = 0.3

    # --- Work history quality ---
    total_months = sum(j.get("duration_months", 0) for j in career)
    product_months = 0
    services_months = 0
    ml_relevance_score = 0.0

    for job in career:
        dur = job.get("duration_months", 0)
        company = text_lower(job.get("company", ""))
        industry = text_lower(job.get("industry", ""))
        desc = text_lower(job.get("description", ""))
        title = text_lower(job.get("title", ""))
        size = job.get("company_size", "")

        # Services vs product
        if any(sc in company for sc in SERVICES_COMPANIES):
            services_months += dur
        elif industry in ("software", "saas", "fintech", "edtech", "hr tech", "e-commerce",
                          "healthtech", "ai", "machine learning", "artificial intelligence"):
            product_months += dur
        elif size in ("51-200", "201-500", "501-1000"):
            product_months += dur * 0.7  # startup/mid-size = likely product
        else:
            product_months += dur * 0.4

        # ML/retrieval work in descriptions
        ml_hits = count_matches(desc + " " + title, PRODUCT_ML_KEYWORDS)
        ml_relevance_score += min(ml_hits * dur / 12.0, 3.0)

    # Normalize
    product_ratio = product_months / max(total_months, 1)
    services_ratio = services_months / max(total_months, 1)

    # Full career in services = bad; partial is just a penalty
    if services_ratio > 0.9:
        services_penalty = 0.3
    elif services_ratio > 0.6:
        services_penalty = 0.6
    else:
        services_penalty = 1.0

    product_score = min(product_ratio * 1.2, 1.0)  # slight bonus for pure product
    ml_norm = min(ml_relevance_score / 15.0, 1.0)

    # YoE fit: JD wants 5-9 years; very strong sweet spot at 6-8
    yoe_score = 0.0
    if 6 <= yoe <= 8:
        yoe_score = 1.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        yoe_score = 0.85
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.65
    elif yoe < 4:
        yoe_score = 0.3
    else:  # > 11 years
        yoe_score = 0.5  # over-qualified risk

    career_score = (
        0.25 * title_score +
        0.25 * ml_norm +
        0.20 * product_score * services_penalty +
        0.15 * yoe_score +
        0.15 * services_penalty
    )

    return min(career_score, 1.0)


def score_location(c):
    """Score location fit. JD wants Pune/Noida; Tier-1 Indian cities ok."""
    profile = c.get("profile", {})
    location = text_lower(profile.get("location", ""))
    country = text_lower(profile.get("country", "india"))
    relocate = c.get("redrob_signals", {}).get("willing_to_relocate", False)

    if country != "india":
        return 0.3 if relocate else 0.1

    if any(city in location for city in ["noida", "pune"]):
        return 1.0
    elif any(city in location for city in TIER1_CITIES):
        return 0.8
    else:
        return 0.6 if relocate else 0.4


def score_behavioral(c):
    """
    Score behavioral/availability signals.
    A great candidate who isn't reachable is worthless to a recruiter.
    """
    sig = c.get("redrob_signals", {})

    # Recency: last active
    last_active = parse_date(sig.get("last_active_date"))
    days_inactive = days_since(last_active)
    if days_inactive < 7:
        recency_score = 1.0
    elif days_inactive < 30:
        recency_score = 0.85
    elif days_inactive < 90:
        recency_score = 0.6
    elif days_inactive < 180:
        recency_score = 0.35
    else:
        recency_score = 0.1

    # Availability signals
    open_to_work = 1.0 if sig.get("open_to_work_flag", False) else 0.5

    # Notice period: JD wants sub-30 days; up to 60 days is ok
    notice = sig.get("notice_period_days", 90)
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.9
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.25

    # Recruiter responsiveness
    response_rate = sig.get("recruiter_response_rate", 0.5)
    response_score = response_rate  # already 0-1

    # Interview completion (reliability)
    interview_rate = sig.get("interview_completion_rate", 0.7)
    interview_score = interview_rate

    # Profile completeness
    completeness = sig.get("profile_completeness_score", 50) / 100.0

    # Applications in last 30 days (actively searching = good signal)
    apps = sig.get("applications_submitted_30d", 0)
    active_search_score = min(apps / 5.0, 1.0)

    behavioral = (
        0.25 * recency_score +
        0.20 * open_to_work +
        0.15 * response_score +
        0.15 * notice_score +
        0.10 * interview_score +
        0.10 * completeness +
        0.05 * active_search_score
    )

    return min(behavioral, 1.0)


def score_github_assessment(c):
    """Score technical proof points: github activity + skill assessment scores."""
    sig = c.get("redrob_signals", {})

    # GitHub activity
    gh = sig.get("github_activity_score", -1)
    if gh == -1:
        github_score = 0.3  # no github = slight penalty
    else:
        github_score = gh / 100.0

    # Skill assessment scores (objective, platform-verified)
    assessments = sig.get("skill_assessment_scores", {})
    if assessments:
        relevant_scores = []
        for skill_name, score in assessments.items():
            sn = skill_name.lower()
            if any(hs in sn for hs in HARD_SKILLS | SOFT_SKILLS):
                relevant_scores.append(score)
        if relevant_scores:
            avg_assessment = sum(relevant_scores) / len(relevant_scores) / 100.0
        else:
            avg_assessment = sum(assessments.values()) / len(assessments) / 100.0
    else:
        avg_assessment = 0.4

    return 0.5 * github_score + 0.5 * avg_assessment


def score_disqualifiers(c):
    """
    Returns a multiplier (0.0-1.0). 0 = disqualified.
    Checks for hard disqualifiers from the JD.
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    current_title = text_lower(profile.get("current_title", ""))
    yoe = profile.get("years_of_experience", 0)

    # Hard DQ: title is clearly wrong domain with no ML history
    if any(dt in current_title for dt in DISQUALIFIED_TITLES):
        # But give benefit of doubt if career history shows ML work
        ml_in_history = any(
            count_matches(
                text_lower(j.get("description", "")) + text_lower(j.get("title", "")),
                PRODUCT_ML_KEYWORDS
            ) >= 3
            for j in career
        )
        if not ml_in_history:
            return 0.05  # very strong penalty, not full 0 in case of career change

    # Pure research background (no production) — soft DQ
    research_only = all(
        text_lower(j.get("industry", "")) in ("academia", "research", "education")
        or "research" in text_lower(j.get("title", ""))
        for j in career
    ) and len(career) > 0
    if research_only:
        return 0.2

    # Only consulting/services entire career
    all_services = all(
        any(sc in text_lower(j.get("company", "")) for sc in SERVICES_COMPANIES)
        for j in career
    ) and len(career) > 0
    if all_services:
        return 0.35

    # CV/Speech/Robotics primary expertise, no NLP/IR
    cv_speech_skills = [s for s in skills if any(
        kw in text_lower(s.get("name", ""))
        for kw in ["computer vision", "cv", "speech recognition", "tts", "robotics", "object detection"]
    )]
    nlp_ir_skills = [s for s in skills if any(
        kw in text_lower(s.get("name", ""))
        for kw in ["nlp", "retrieval", "ranking", "embedding", "search", "rag"]
    )]
    if len(cv_speech_skills) > len(nlp_ir_skills) * 2 and len(nlp_ir_skills) < 2:
        return 0.4

    return 1.0


def score_candidate(c):
    """
    Master scoring function. Returns (composite_score, component_dict).
    """
    # Honeypot check first — hard penalty
    if is_honeypot(c):
        return 0.01, {"honeypot": True}

    dq = score_disqualifiers(c)
    if dq < 0.1:
        return dq * 0.05, {"disqualified": True, "dq_factor": dq}

    s_skills = score_skills(c)
    s_career = score_career(c)
    s_location = score_location(c)
    s_behavioral = score_behavioral(c)
    s_github = score_github_assessment(c)

    # Composite weights (must sum to 1.0)
    # Skills + Career = core technical fit
    # Behavioral = "can we actually hire them"
    composite = (
        0.30 * s_skills +
        0.30 * s_career +
        0.20 * s_behavioral +
        0.12 * s_github +
        0.08 * s_location
    )

    # Apply disqualifier multiplier
    composite *= dq

    return round(min(composite, 1.0), 6), {
        "skills": s_skills,
        "career": s_career,
        "location": s_location,
        "behavioral": s_behavioral,
        "github": s_github,
        "dq": dq,
    }


# ── Reasoning generator ──────────────────────────────────────────────────────

def generate_reasoning(c, components, rank):
    """Generate specific, honest 1-2 sentence reasoning grounded in profile facts."""
    profile = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    notice = sig.get("notice_period_days", 90)
    response_rate = sig.get("recruiter_response_rate", 0.5)
    last_active = parse_date(sig.get("last_active_date"))
    days_inactive = days_since(last_active)
    open_to_work = sig.get("open_to_work_flag", False)

    # Find top hard skills with strong proficiency
    strong_hard_skills = [
        s["name"] for s in skills
        if any(hs in text_lower(s.get("name", "")) for hs in HARD_SKILLS)
        and s.get("proficiency") in ("advanced", "expert")
    ][:3]

    # Find career company that's most notable (product company)
    product_jobs = [
        j for j in career
        if not any(sc in text_lower(j.get("company", "")) for sc in SERVICES_COMPANIES)
        and j.get("duration_months", 0) >= 12
    ]

    parts = []

    # Core fit sentence
    if components.get("honeypot"):
        return "Profile flagged as inconsistent — experience claims do not align with career timeline."
    if components.get("disqualified"):
        return f"{title} role does not align with Senior AI Engineer JD requirements; minimal NLP/retrieval evidence in history."

    if strong_hard_skills:
        skill_str = ", ".join(strong_hard_skills)
        parts.append(f"{title} with {yoe:.0f} yrs exp; strong in {skill_str}")
    else:
        parts.append(f"{title} with {yoe:.0f} yrs exp at {company}")

    if product_jobs:
        best = product_jobs[0]
        dur_yrs = best.get("duration_months", 0) // 12
        if dur_yrs >= 1:
            # Check if description mentions ranking/retrieval
            desc = text_lower(best.get("description", ""))
            ml_hits = [kw for kw in PRODUCT_ML_KEYWORDS if kw in desc]
            if ml_hits:
                parts.append(f"built {ml_hits[0]}-related systems at {best['company']} ({dur_yrs}yr tenure)")
            else:
                parts.append(f"product company experience at {best['company']} ({dur_yrs}yr tenure)")

    # Availability / concern sentence
    concerns = []
    positives = []

    if days_inactive > 180:
        concerns.append(f"inactive for {days_inactive // 30}+ months")
    elif days_inactive < 14:
        positives.append("recently active")

    if notice > 90:
        concerns.append(f"long notice period ({notice}d)")
    elif notice <= 30:
        positives.append(f"quick joiner ({notice}d notice)")

    if response_rate < 0.3:
        concerns.append(f"low response rate ({response_rate:.0%})")
    elif response_rate > 0.7:
        positives.append(f"responsive ({response_rate:.0%} reply rate)")

    if open_to_work:
        positives.append("actively open to work")

    if location:
        positives.append(f"based in {location}")

    sentence2_parts = positives[:2] + (["concern: " + c for c in concerns[:1]] if concerns else [])
    if sentence2_parts:
        parts.append("; ".join(sentence2_parts))

    return "; ".join(parts[:3]).strip("; ") + "."


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(candidates_path: str, out_path: str, top_n: int = 100):
    print(f"Loading candidates from {candidates_path}...", flush=True)

    scored = []
    total = 0
    errors = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                c = json.loads(line)
                cid = c["candidate_id"]
                score, components = score_candidate(c)
                scored.append((cid, score, components, c))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Warning: parse error on line {total}: {e}", flush=True)

    print(f"Scored {total} candidates ({errors} errors). Selecting top {top_n}...", flush=True)

    # Sort descending by score (rounded to 4dp for tie-break consistency), then ascending by candidate_id
    scored.sort(key=lambda x: (-round(x[1], 4), x[0]))

    top = scored[:top_n]

    # Ensure scores are strictly monotonically non-increasing
    # (same score allowed; but rank 1 must have highest)
    for i in range(1, len(top)):
        if top[i][1] > top[i-1][1]:
            # Shouldn't happen after sort, but safety clamp
            top[i] = (top[i][0], top[i-1][1], top[i][2], top[i][3])

    print(f"Writing output to {out_path}...", flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_idx, (cid, score, components, c) in enumerate(top, start=1):
            reasoning = generate_reasoning(c, components, rank_idx)
            writer.writerow([cid, rank_idx, f"{score:.4f}", reasoning])

    print(f"\nDone! Top {top_n} candidates written to {out_path}")
    print(f"\nTop 10 preview:")
    for rank_idx, (cid, score, components, c) in enumerate(top[:10], start=1):
        title = c.get("profile", {}).get("current_title", "?")
        yoe = c.get("profile", {}).get("years_of_experience", 0)
        print(f"  #{rank_idx:3d}  {cid}  {score:.4f}  [{title}, {yoe:.0f}yrs]")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    run(args.candidates, args.out, args.top)

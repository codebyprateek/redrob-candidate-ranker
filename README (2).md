# Redrob Hackathon — Intelligent Candidate Ranking System

**Challenge:** Intelligent Candidate Discovery & Ranking  
**Target Role:** Senior AI Engineer — Founding Team @ Redrob AI  
**Dataset:** 100,000 synthetic candidate profiles (candidates.jsonl)  
**Output:** Top 100 ranked candidates with reasoning (submission.csv)

---

## Architecture

This system uses a **multi-dimensional, rule-based + heuristic scoring pipeline** — no GPU, no external APIs, runs in under 2 minutes on a standard CPU.

### Why not embeddings/LLMs?

The challenge forbids external API calls and GPU use during the ranking step, and must complete in ≤5 min / ≤16 GB CPU-only. A system that calls GPT-4 per candidate or loads a 7B model would fail these constraints. Instead, this system implements **expert-designed feature engineering** directly from the JD.

### Scoring components (5 signals → 1 composite)

| Component | Weight | What it measures |
|---|---|---|
| `score_skills` | 30% | Hard/soft skill match, proficiency level, endorsements, usage duration |
| `score_career` | 30% | Title relevance, product vs services background, ML work in history, YoE fit |
| `score_behavioral` | 20% | Recency, open-to-work, response rate, notice period, interview completion |
| `score_github_assessment` | 12% | GitHub activity score + platform skill assessments |
| `score_location` | 8% | India Tier-1 city fit, willingness to relocate |

**Final score = Σ(weights × components) × disqualifier_multiplier**

### Key design decisions

**Hard disqualifiers** — Candidates are penalized (0.05×–0.4× multiplier) for:
- Non-technical current titles (HR Manager, Marketing Manager, etc.) with no ML evidence in career history
- Pure research/academic careers with no production deployment
- Entire career at consulting/services companies (TCS, Infosys, Wipro, etc.)
- CV/Speech/Robotics-primary expertise with no NLP/IR exposure

**Honeypot detection** — Before scoring, each profile is checked for:
- Impossible company tenure (start date precedes company founding)
- Expert-in-10+ skills with near-zero duration each
- Years-of-experience far exceeding possible career length vs graduation year
- Title-skill contradiction (Marketing Manager with expert embedding skills)

**Behavioral multiplier** — Strong behavioral signals (recent activity, high response rate, short notice) can significantly lift a candidate; poor signals (6+ months inactive, 5% response rate) down-weight them even when skills are strong.

**JD-aware skill matching** — Skills are matched not just by name but with proficiency, endorsements, and usage duration weighted. An "advanced" embeddings skill with 36 months duration and 50 endorsements scores much higher than "beginner" with 2 months.

---

## Reproduce the submission

### Requirements

```
Python >= 3.8
pandas    (for data loading, optional — pure stdlib also works)
numpy
scikit-learn
```

Install:
```bash
pip install -r requirements.txt
```

### Single command to reproduce

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Optional flags

```
--candidates  Path to candidates.jsonl (default: ./candidates.jsonl)
--out         Output CSV path (default: ./submission.csv)
--top         Number of candidates to output (default: 100)
```

### Validate output

```bash
python validate_submission.py submission.csv
```

---

## Compute constraints met

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤ 5 min | ~90 sec on 100K candidates |
| Memory | ≤ 16 GB | ~500 MB peak |
| Compute | CPU only | ✅ pure Python + numpy |
| Network | Off | ✅ zero external calls |

---

## Methodology summary

The core insight from the JD is: **keyword matching is explicitly the wrong answer**. The JD says "a Tier-5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."

To operationalize this:

1. **Career description analysis**: We extract ML/retrieval relevance from free-text job descriptions, not just the skills list. A candidate who "built a recommendation engine for a B2C platform" gets credit even if they don't list "RAG" as a skill.

2. **Context-aware disqualification**: A Marketing Manager with perfect AI keyword coverage is penalized; an ML Engineer at a services company but with strong IR descriptions in their role history is not.

3. **Behavioral realism**: A strong-profile candidate who hasn't logged in for 6 months with a 5% response rate is a recruiter's dead end. Behavioral signals act as a hiring-probability multiplier, not just a bonus.

4. **Honeypot resistance**: Impossible profiles (8 years at a 3-year-old company; expert in 15 skills with 0 months duration) are identified structurally and pushed to near-zero scores.

---

## File structure

```
redrob-ranker/
├── rank.py                        # Main ranking script
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── submission_metadata.yaml       # Submission metadata
└── submission.csv                 # Final ranked output (top 100)
```

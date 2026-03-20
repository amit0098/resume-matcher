# 💼 Job Description Skill Extractor + Resume Matcher

> **NLP-powered Resume Screening Tool** — Extract skills, compute match scores, identify skill gaps

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/amit0098/resume-matcher)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20Transformers-orange)](https://spacy.io)
[![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Problem Statement

Recruiters spend 6-8 seconds manually screening each resume — inefficient and inconsistent. Candidates get rejected without knowing why. ATS systems reject good candidates due to keyword mismatch.

**This system solves all three problems:**
- Automatically matches resumes to job descriptions with an objective score
- Tells candidates exactly which skills are missing
- Ranks multiple candidates for a single job role

---

## Live Demo

**Try it now → [huggingface.co/spaces/amit0098/resume-matcher](https://huggingface.co/spaces/amit0098/resume-matcher)**

Paste any Job Description + Upload your Resume (PDF/DOCX/TXT) → Get instant match score + gap analysis

---

## Pipeline Architecture

```
Input: Job Description + Resume (PDF / DOCX / TXT / Paste)
              |
    Stage 1: Text Preprocessing
    lowercase → clean → normalize whitespace
              |
    Stage 2: Skill Extraction
    regex + 133-skill taxonomy → JD skills dict + Resume skills dict
              |
    Stage 3: Three Matching Methods (parallel)
    ├── TF-IDF Cosine Similarity   (keyword level)
    ├── Semantic Similarity         (meaning level - BERT)
    └── Skill Overlap Score         (direct skill match)
              |
    Stage 4: Weighted Final Score
    Final = 30% TF-IDF + 40% Semantic + 30% Skill Overlap
              |
    Output: Match Score + Matched Skills + Missing Skills + Recommendations
```

---

## Features

| Feature | Description |
|---|---|
| Skill Extraction | Extracts 133+ skills across 8 categories from any text |
| TF-IDF Matching | Keyword-level cosine similarity with bigram support |
| Semantic Matching | Meaning-level similarity using BERT (all-MiniLM-L6-v2) |
| Skill Gap Analysis | Matched, Missing, Extra skills breakdown |
| Resume Upload | Supports PDF, DOCX, and TXT file uploads |
| Candidate Ranking | Rank multiple candidates for a single job |
| Live Deployment | Streamlit app on Hugging Face Spaces |

---

## Matching Score Formula

```python
Final Score = 0.30 x TF-IDF Score
            + 0.40 x Semantic Score    # highest weight - handles synonyms
            + 0.30 x Skill Overlap Score
```

| Score Range | Verdict | Action |
|---|---|---|
| 65% and above | STRONG MATCH | Apply with confidence |
| 45 to 64% | MODERATE MATCH | Apply and upskill |
| Below 45% | WEAK MATCH | Upskill first |

---

## Sample Results

**Amit Kumar vs 5 Job Roles:**

```
Candidate: Amit Kumar
------------------------------------------------------------------
MODERATE | Data Scientist - Flipkart    | Score: 43.8%  [TF-IDF:13.7  Sem:64.9  Skill:45.8]
MODERATE | NLP Engineer - Swiggy        | Score: 48.3%  [TF-IDF:17.3  Sem:55.4  Skill:70.0]
WEAK     | ML Engineer - Razorpay       | Score: 38.9%  [TF-IDF:11.8  Sem:64.7  Skill:31.6]
MODERATE | Data Analyst - Zepto         | Score: 45.3%  [TF-IDF:14.2  Sem:55.6  Skill:62.5]
MODERATE | GenAI Engineer - Meesho      | Score: 51.5%  [TF-IDF:14.4  Sem:64.3  Skill:71.4]
```

**Gap Analysis — Amit Kumar vs GenAI Engineer (Meesho):**

```
Final Score: 51.5% - MODERATE

Matched Skills (15): python, nlp, llm, rag, faiss, langchain,
                     hugging face, docker, git, transformers,
                     prompt engineering, mistral, communication...

Missing Skills (6):  chromadb, embeddings, gpt, llama, llamaindex, mlflow

Recommendation: Add missing skills to resume to improve score
```

**Key Insight:** TF-IDF scores are consistently low (13-17%) while Semantic scores are high (55-65%). This reveals resumes use abbreviations like NLP and LLM while JDs use full forms like Natural Language Processing and Large Language Models — a real ATS problem.

---

## Tech Stack

| Category | Libraries |
|---|---|
| NLP Processing | spaCy, regex |
| Keyword Matching | scikit-learn TF-IDF, cosine similarity |
| Semantic Similarity | sentence-transformers (all-MiniLM-L6-v2) |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly, wordcloud |
| File Parsing | pypdf (PDF), python-docx (DOCX) |
| Deployment | Streamlit, Hugging Face Spaces |

---

## Repository Structure

```
resume-matcher/
│
├── app.py                              # Streamlit web app
├── Resume_Matcher.ipynb     # Complete Colab notebook (32 cells)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/amit0098/resume-matcher.git
cd resume-matcher

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run the app
streamlit run app.py
```

---

## Run on Google Colab

1. Open `Resume_Matcher_Amit_Kumar.ipynb` in Google Colab
2. Runtime → Change runtime type → CPU (no GPU needed)
3. Run all cells — completes in 10-15 minutes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amit0098/resume-matcher/blob/main/Resume_Matcher_Amit_Kumar.ipynb)

---

## Skill Taxonomy

The system covers 133 skills across 8 categories:

| Category | Examples |
|---|---|
| Programming | Python, Java, SQL, R, Scala, JavaScript |
| ML and AI | Machine Learning, Deep Learning, NLP, Computer Vision |
| Frameworks | TensorFlow, PyTorch, scikit-learn, XGBoost, spaCy |
| GenAI and LLMs | LLM, RAG, Prompt Engineering, FAISS, LangChain |
| Cloud and DevOps | AWS, Azure, Docker, Kubernetes, MLflow |
| Databases | SQL, MongoDB, PostgreSQL, Snowflake, Redis |
| Visualization | Power BI, Tableau, Excel, Matplotlib, Plotly |
| Soft Skills | Communication, Leadership, Agile, Problem Solving |

---

## Industry Applications

- **HR Tech** — Automated resume screening for Naukri, LinkedIn, Greenhouse
- **Corporate HR** — Rank candidates for any open position instantly
- **Job Seekers** — Know your match % before applying, fix your resume
- **Career Counselors** — Identify skill gaps and suggest learning paths
- **Staffing Agencies** — Screen thousands of resumes in minutes

---

## Author

**Amit Kumar**
Data Scientist | Bangalore, India

- GitHub: [github.com/amit0098](https://github.com/amit0098)
- Live App: [huggingface.co/spaces/amit0098/resume-matcher](https://huggingface.co/spaces/amit0098/resume-matcher)

---

## License

This project is licensed under the MIT License.

---

*Capstone Project 2 — PGA 45 | NLP Track*

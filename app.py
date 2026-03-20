import streamlit as st
import re
import numpy as np
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import matplotlib.pyplot as plt

# ── Page Config ──
st.set_page_config(page_title="Resume Matcher", page_icon="💼", layout="wide")
st.title("💼 Job Description Skill Extractor + Resume Matcher")
st.caption("NLP-powered Resume Screening Tool | By Amit Kumar | github.com/amit0098")

# ── Skill Taxonomy ──
SKILL_TAXONOMY = {
    "Programming": ["python","java","javascript","r","scala","c++","sql","bash","typescript","go"],
    "ML & AI": ["machine learning","deep learning","nlp","natural language processing",
                "computer vision","feature engineering","classification","regression",
                "clustering","time series","statistics","a/b testing"],
    "Frameworks": ["tensorflow","pytorch","scikit-learn","xgboost","lightgbm","hugging face",
                   "transformers","spacy","langchain","fastapi","keras","nltk"],
    "GenAI & LLMs": ["llm","rag","prompt engineering","faiss","chromadb","langchain",
                     "llamaindex","fine-tuning","embeddings","openai","mistral","llama"],
    "Cloud & DevOps": ["aws","azure","gcp","docker","kubernetes","git","mlflow",
                       "airflow","ci/cd","s3","ec2","sagemaker"],
    "Databases": ["sql","mysql","postgresql","mongodb","redis","snowflake",
                  "redshift","elasticsearch","nosql","hadoop","spark"],
    "Visualization": ["power bi","tableau","excel","matplotlib","plotly",
                      "seaborn","looker","dashboard"],
    "Soft Skills": ["communication","teamwork","leadership","problem solving",
                    "agile","scrum","stakeholder management","presentation"]
}
ALL_SKILLS = {s.lower(): cat for cat, skills in SKILL_TAXONOMY.items() for s in skills}

# ── Load Models ──
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    nlp   = spacy.load("en_core_web_sm")
    return model, nlp

with st.spinner("Loading AI models... (first load takes ~1 min)"):
    sem_model, nlp = load_models()

# ── Resume Text Extraction ──
def extract_text_from_pdf(file_bytes):
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception:
        return None

def extract_text_from_docx(file_bytes):
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text.strip()
    except Exception:
        return None

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode("utf-8").strip()
    except Exception:
        return file_bytes.decode("latin-1").strip()

# ── Core NLP Functions ──
def extract_skills(text):
    text_l = text.lower()
    found  = {}
    for skill, cat in ALL_SKILLS.items():
        if re.search(r"\b" + re.escape(skill) + r"\b", text_l):
            found[skill] = cat
    return found

def match_score(jd_text, res_text):
    vec   = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    mat   = vec.fit_transform([jd_text.lower(), res_text.lower()])
    tfidf = round(float(cosine_similarity(mat[0:1], mat[1:2])[0][0]) * 100, 1)

    j_emb = sem_model.encode(jd_text[:800]).reshape(1,-1)
    r_emb = sem_model.encode(res_text[:800]).reshape(1,-1)
    sem   = round(float(cosine_similarity(j_emb, r_emb)[0][0]) * 100, 1)

    jd_s  = set(extract_skills(jd_text))
    re_s  = set(extract_skills(res_text))
    skill = round(len(jd_s & re_s) / max(len(jd_s), 1) * 100, 1)

    final    = round(0.3*tfidf + 0.4*sem + 0.3*skill, 1)
    matching = sorted(list(jd_s & re_s))
    missing  = sorted(list(jd_s - re_s))
    extra    = sorted(list(re_s - jd_s))
    return tfidf, sem, skill, final, matching, missing, extra

# ── UI ──
st.markdown("---")
col1, col2 = st.columns(2)

# LEFT — JD
with col1:
    st.subheader("📋 Job Description")
    jd_text = st.text_area(
        "Paste Job Description here:",
        height=320,
        placeholder="Paste the full job description here...\n\nExample:\nWe are looking for a Data Scientist with Python, machine learning, SQL, TensorFlow..."
    )

# RIGHT — Resume
with col2:
    st.subheader("📄 Your Resume")
    resume_mode = st.radio(
        "How do you want to add your resume?",
        ["📁 Upload File (PDF / DOCX / TXT)", "✏️ Paste Text"],
        horizontal=True
    )

    res_text = ""

    if resume_mode == "📁 Upload File (PDF / DOCX / TXT)":
        uploaded = st.file_uploader(
            "Upload your resume:",
            type=["pdf", "docx", "txt"],
            help="Supported: PDF, DOCX, TXT"
        )
        if uploaded is not None:
            file_bytes = uploaded.read()
            fname = uploaded.name.lower()
            with st.spinner(f"Extracting text from {uploaded.name}..."):
                if fname.endswith(".pdf"):
                    res_text = extract_text_from_pdf(file_bytes)
                elif fname.endswith(".docx"):
                    res_text = extract_text_from_docx(file_bytes)
                elif fname.endswith(".txt"):
                    res_text = extract_text_from_txt(file_bytes)

            if res_text:
                st.success(f"✅ Resume extracted! ({len(res_text.split())} words)")
                with st.expander("👁️ Preview extracted text"):
                    st.text(res_text[:1000] + ("..." if len(res_text) > 1000 else ""))
            else:
                st.error("❌ Could not extract text. Try a different file or use Paste Text instead.")
    else:
        res_text = st.text_area(
            "Paste your resume text:",
            height=320,
            placeholder="Paste your resume content here...\n\nInclude your skills, experience, education..."
        )

# ── Analyze ──
st.markdown("---")
if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
    if not jd_text.strip():
        st.warning("⚠️ Please paste a Job Description first.")
    elif not res_text or not res_text.strip():
        st.warning("⚠️ Please upload or paste your Resume.")
    else:
        with st.spinner("Analyzing your resume against the job description..."):
            tfidf_s, sem_s, skill_s, final_s, matched, missing, extra = match_score(jd_text, res_text)

        st.markdown("---")
        st.subheader("📊 Match Analysis Results")

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Final Score",    f"{final_s}%")
        c2.metric("🔤 Keyword Match",  f"{tfidf_s}%")
        c3.metric("🧠 Semantic Match", f"{sem_s}%")
        c4.metric("⚙️ Skill Match",    f"{skill_s}%")

        # Verdict
        if final_s >= 65:
            st.success("🟢 STRONG MATCH — Your profile is well-aligned. Apply with confidence!")
        elif final_s >= 45:
            st.warning("🟡 MODERATE MATCH — You match well but missing some skills. Upskill and apply!")
        else:
            st.error("🔴 WEAK MATCH — Significant gaps found. Focus on upskilling before applying.")

        # Chart
        fig, ax = plt.subplots(figsize=(10, 3))
        metrics = ["Keyword\n(TF-IDF)", "Semantic\nSimilarity", "Skill\nOverlap", "FINAL\nSCORE"]
        values  = [tfidf_s, sem_s, skill_s, final_s]
        colors  = ["#5BA3D9", "#2E75B6", "#9DC3E6", "#1F4E79"]
        bars = ax.bar(metrics, values, color=colors, width=0.5)
        ax.set_ylim(0, 115)
        ax.axhline(y=65, color="green",  linestyle="--", alpha=0.6, label="Strong (65%)")
        ax.axhline(y=45, color="orange", linestyle="--", alpha=0.6, label="Moderate (45%)")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2., v+1.5, f"{v}%",
                    ha="center", fontweight="bold", fontsize=11)
        ax.set_title("Match Score Breakdown", fontweight="bold", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Score (%)")
        st.pyplot(fig)

        # Skill columns
        st.markdown("---")
        st.subheader("🎯 Skill Analysis")
        ca, cb, cc = st.columns(3)

        with ca:
            st.markdown(f"### ✅ Matched ({len(matched)})")
            for s in matched:
                st.success(f"✓ {s}")
            if not matched:
                st.info("No matching skills found")

        with cb:
            st.markdown(f"### ❌ Missing ({len(missing)})")
            for s in missing:
                st.error(f"✗ {s}")
            if not missing:
                st.success("No missing skills!")
            else:
                st.caption("👆 Add these to your resume")

        with cc:
            st.markdown(f"### ➕ Extra in Resume ({len(extra)})")
            for s in extra[:10]:
                st.info(f"+ {s}")
            if not extra:
                st.info("No extra skills")
            else:
                st.caption("Skills beyond JD requirements")

        # Recommendation
        st.markdown("---")
        st.subheader("💡 Recommendation")
        if missing:
            st.info(f"**Top skills to add to your resume:** `{', '.join(missing[:5])}`")
        if final_s >= 65:
            st.success("Tailor your cover letter to highlight your matched skills and apply!")
        elif final_s >= 45:
            st.warning("You are close! Apply now and simultaneously upskill on the missing areas.")
        else:
            st.error("Build the missing skills first through courses or projects, then apply.")

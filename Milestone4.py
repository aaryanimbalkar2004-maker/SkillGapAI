# ==================================================
# Skill Gap Analyzer — Standalone Streamlit App
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fpdf import FPDF
import io

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ==================================================
# Utility Functions
# ==================================================

def extract_skills(text):
    """
    Extract skills from text using NLP (POS tagging + Named Entities)
    """
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in word_tokenize(text) if word.isalpha() and word not in stop_words]

    doc = nlp(" ".join(tokens))
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

    # Combine tokens + noun phrases
    keywords = set(tokens + noun_chunks)

    # Example predefined skill keywords (can expand)
    skill_keywords = {
        'python', 'java', 'c++', 'html', 'css', 'javascript',
        'sql', 'mysql', 'mongodb', 'aws', 'azure', 'docker',
        'kubernetes', 'git', 'machine learning', 'deep learning',
        'data analysis', 'nlp', 'react', 'node.js', 'flask',
        'django', 'communication', 'leadership', 'teamwork',
        'problem solving', 'linux', 'networking', 'cloud computing'
    }

    extracted = [kw for kw in keywords if kw in skill_keywords]
    return list(set(extracted))


def compare_skills(resume_skills, jd_skills):
    """
    Compare extracted skills and compute match percentage
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = list(resume_set & jd_set)
    missing = list(jd_set - resume_set)

    match_percent = round((len(matched) / len(jd_set)) * 100, 2) if jd_set else 0

    return matched, missing, match_percent


def export_pdf(resume_name, matched, missing, match_percent):
    """
    Export analysis as PDF report (returns binary data)
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Skill Gap Analysis Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, txt=f"Candidate: {resume_name}", ln=True)
    pdf.cell(200, 10, txt=f"Overall Match: {match_percent}%", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Matched Skills:", ln=True)
    pdf.set_font("Arial", "", 11)
    for s in matched:
        pdf.cell(200, 8, txt=f"- {s}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Missing Skills:", ln=True)
    pdf.set_font("Arial", "", 11)
    for s in missing:
        pdf.cell(200, 8, txt=f"- {s}", ln=True)

    # ✅ Get PDF data as bytes (not write to a file)
    pdf_bytes = pdf.output(dest='S').encode('latin1')

    return pdf_bytes


# ==================================================
# Streamlit UI
# ==================================================

st.set_page_config(page_title="Skill Gap Analyzer", layout="wide")
st.title("🧠 Skill Gap Analyzer Dashboard")

st.sidebar.header("📂 Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (txt/pdf/docx)", type=["txt", "pdf", "docx"])
jd_file = st.sidebar.file_uploader("Upload Job Description (txt/pdf/docx)", type=["txt", "pdf", "docx"])

def read_file(file):
    if file is None:
        return ""
    content = ""
    if file.name.endswith(".txt"):
        content = file.read().decode("utf-8", errors="ignore")
    elif file.name.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file)
            content = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception:
            content = ""
    elif file.name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(file)
            content = " ".join([p.text for p in doc.paragraphs])
        except Exception:
            content = ""
    return content


# ==================================================
# Analysis Section
# ==================================================

if st.sidebar.button("🔍 Analyze"):
    resume_text = read_file(resume_file)
    jd_text = read_file(jd_file)

    if not resume_text or not jd_text:
        st.error("Please upload both resume and job description files.")
    else:
        st.subheader("📄 Extracting Skills...")

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)

        matched, missing, match_percent = compare_skills(resume_skills, jd_skills)

        st.success("Analysis Complete ✅")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total JD Skills", len(jd_skills))
        with col2:
            st.metric("Matched Skills", len(matched))
        with col3:
            st.metric("Match Percentage", f"{match_percent}%")

        st.write("---")

        st.subheader("✅ Matched Skills")
        st.write(", ".join(matched) if matched else "No matches found.")

        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing) if missing else "No missing skills.")

        st.subheader("💡 Recommendations")
        if missing:
            st.info("Focus on learning these skills: " + ", ".join(missing))
        else:
            st.success("Excellent! All key skills matched.")

        # Export options
        st.write("---")
        st.subheader("📤 Export Report")

        pdf_data = export_pdf(resume_file.name if resume_file else "Candidate", matched, missing, match_percent)
        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="skill_gap_report.pdf",
            mime="application/pdf"
        )

        csv_data = pd.DataFrame({
            "Matched Skills": matched + [""] * (len(missing) - len(matched)) if len(missing) > len(matched) else matched,
            "Missing Skills": missing + [""] * (len(matched) - len(missing)) if len(matched) > len(missing) else missing
        }).to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="skill_gap_report.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Upload a resume and job description, then click **Analyze** to start.")

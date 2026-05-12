import streamlit as st
import PyPDF2
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Matcher", page_icon="📄")

st.title("AI Resume Keyword Matcher")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

analyze = st.button("Analyze Resume")


def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + " "

    return text


if analyze and uploaded_file and job_description:

    resume_text = extract_text_from_pdf(
        io.BytesIO(uploaded_file.read())
    )

    text_data = [resume_text, job_description]

    cv = CountVectorizer()

    matrix = cv.fit_transform(text_data)

    similarity_score = cosine_similarity(matrix)[0][1]

    match_percentage = round(similarity_score * 100, 2)

    st.subheader("Resume Match Score")

    st.success(f"{match_percentage}% Match")

    resume_words = set(resume_text.lower().split())

    jd_words = set(job_description.lower().split())

    missing_skills = jd_words - resume_words

    st.subheader("Missing Keywords")

    if missing_skills:
        st.write(list(missing_skills)[:20])
    else:
        st.write("No major keywords missing")
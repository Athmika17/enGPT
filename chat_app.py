import streamlit as st
from transformers import pipeline
import PyPDF2, glob

# Load notes
def load_notes():
    text = ""
    for pdf_file in glob.glob("*.pdf"):
        try:
            with open(pdf_file, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.warning(f"⚠️ Could not read {pdf_file}: {e}")  # Use Streamlit warning
    if text.strip() == "":
        # Fallback notes if no PDFs found
        text = (
            "Ohm's Law: V = IR.\n"
            "Thermodynamics (First Law): Energy cannot be created or destroyed.\n"
            "Stress in a beam: σ = My/I.\n"
            "Newton’s Second Law: F = ma.\n"
        )
    return text

engineering_notes = load_notes()
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Web UI
st.title("🎓 StudentGPT")
st.write("Ask me questions from your notes (like ChatGPT)")

user_q = st.text_input("You:")
if user_q:
    if engineering_notes.strip() == "":
        st.error("No notes available for answering questions.")
    else:
        result = qa_pipeline(question=user_q, context=engineering_notes)
        st.write("**StudentGPT:**", result['answer'])

from transformers import pipeline
import PyPDF2

# ------------------------
# Load Knowledge (from PDF or text)
# ------------------------
import glob
import PyPDF2

def load_notes():
    """Reads text from all PDFs in the project folder."""
    text = ""
    for pdf_file in glob.glob("*.pdf"):  # looks for ALL PDFs in folder
        try:
            with open(pdf_file, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
        except Exception as e:
            print(f"⚠️ Could not read {pdf_file}: {e}")
    if text.strip() == "":
        # Fallback if no PDFs are found
        return """
        Ohm's Law: V = IR.
        Thermodynamics (First Law): Energy cannot be created or destroyed.
        Stress in a beam: σ = My/I.
        Newton’s Second Law: F = ma.
        """
    return text

engineering_notes = load_notes()

# ------------------------
# Load AI Model
# ------------------------
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ------------------------
# Chatbot
# ------------------------
def student_gpt():
    print("🎓 StudentGPT: Ask me engineering questions! Type 'bye' to exit.")
    while True:
        question = input("You: ")
        if question.lower() == "bye":
            print("🎓 StudentGPT: Goodbye, and happy studying! 📚")
            break
        result = qa_pipeline(question=question, context=engineering_notes)
        print("🎓 StudentGPT:", result['answer'])

if __name__ == "__main__":
    student_gpt()

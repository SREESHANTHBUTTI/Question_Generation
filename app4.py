import os
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
import google.generativeai as genai
from fpdf import FPDF  # pip install fpdf

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCXgQiDcfmwkm30xMYDuG6nDvkQ1ib9rmE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-1.5-pro")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp/'  # Store uploaded files in a temp directory
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Ensure temp folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            text = ''.join([(page.extract_text() or '') for page in pdf.pages])
        return text.strip()
    elif ext == 'docx':
        doc = docx.Document(file_path)
        text = ' '.join([para.text for para in doc.paragraphs])
        return text.strip()
    elif ext == 'txt':
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.read().strip()
    return None

def question_mcqs_generator(input_text, num_questions):
    prompt = f"""
    Generate {num_questions} multiple-choice questions (MCQs) based on the following text:
    
    {input_text}
    
    **MCQ Format:**
    - Question
    - Four options labeled (A, B, C, D)
    - Indicate the correct answer.

    **Example Output:**
    ## MCQ
    Question: What is the capital of France?
    A) Berlin
    B) Madrid
    C) Paris
    D) Rome
    Correct Answer: C
    """
    try:
        response = model.generate_content(prompt).text.strip()
        return response if response else "Error: No response from AI."
    except Exception as e:
        return f"Error generating MCQs: {str(e)}"

def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("## MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)  # Add a line break

    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.output(pdf_path)
    return pdf_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_mcqs():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text_from_file(file_path)

        if text:
            num_questions = int(request.form['num_questions'])
            mcqs = question_mcqs_generator(text, num_questions)

            pdf_filename = "generated_mcqs.pdf"
            pdf_path = create_pdf(mcqs, pdf_filename)

            return render_template('results.html', mcqs=mcqs, pdf_filename=pdf_filename)
    return "Invalid file format"

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

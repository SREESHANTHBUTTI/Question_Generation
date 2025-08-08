import os
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fpdf import FPDF  # pip install fpdf
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Load the T5 model for question generation
MODEL_NAME = "t5-small"  # You can also use 't5-base' for better accuracy
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text
    elif ext == 'docx':
        doc = docx.Document(file_path)
        text = ' '.join([para.text for para in doc.paragraphs])
        return text
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

def generate_questions_t5(input_text, num_questions):
    """Generates multiple-choice questions using T5 model."""
    prompt = f"generate {num_questions} questions: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=256, num_return_sequences=num_questions)

    mcqs = []
    for i, generated in enumerate(output):
        question = tokenizer.decode(generated, skip_special_tokens=True)
        mcqs.append(f"## MCQ {i+1}\n{question}")

    return "\n\n".join(mcqs)

def save_mcqs_to_file(mcqs, filename):
    results_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(mcqs)
    return results_path

def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("## MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)  # Add a line break

    pdf_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
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
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from the uploaded file
        text = extract_text_from_file(file_path)

        if text:
            num_questions = int(request.form['num_questions'])
            mcqs = generate_questions_t5(text, num_questions)

            # Save the generated MCQs to a file
            txt_filename = f"generated_mcqs_{filename.rsplit('.', 1)[0]}.txt"
            pdf_filename = f"generated_mcqs_{filename.rsplit('.', 1)[0]}.pdf"
            save_mcqs_to_file(mcqs, txt_filename)
            create_pdf(mcqs, pdf_filename)

            # Display and allow downloading
            return render_template('results.html', mcqs=mcqs, txt_filename=txt_filename, pdf_filename=pdf_filename)
    return "Invalid file format"

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULTS_FOLDER']):
        os.makedirs(app.config['RESULTS_FOLDER'])
    app.run(debug=True)

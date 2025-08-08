import os
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
from fpdf import FPDF  # pip install fpdf
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

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

def question_mcqs_generator(input_text, num_questions):
    prompt = f"generate {num_questions} MCQs from: {input_text}"
    input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
    output_ids = model.generate(input_ids, max_length=512)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("\n\n"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)  # Add a line break

    pdf.output(filename)
    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_mcqs():
    text = request.form.get('text_input', '').strip()
    file = request.files.get('file')

    if not text and file and allowed_file(file.filename):
        filename = file.filename
        file_path = filename
        file.save(file_path)
        text = extract_text_from_file(file_path)
    
    if not text:
        return "No valid text input provided."

    num_questions = int(request.form['num_questions'])
    mcqs = question_mcqs_generator(text, num_questions)
    pdf_filename = "generated_mcqs.pdf"
    create_pdf(mcqs, pdf_filename)

    return render_template('results.html', mcqs=mcqs, pdf_filename=pdf_filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

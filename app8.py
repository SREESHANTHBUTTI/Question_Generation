import os
from flask import Flask, render_template, request, send_file
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fpdf import FPDF  # pip install fpdf

# Initialize Flask app
app = Flask(__name__)

# Load the T5 model for question generation
MODEL_NAME = "t5-small"  # You can also use 't5-base' for better accuracy
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_questions_t5(text, num_questions=5):
    inputs = tokenizer.encode("generate question: " + text, return_tensors="pt")
    
    questions = []
    for _ in range(num_questions):  # Call model multiple times instead of setting num_return_sequences
        output = model.generate(**inputs, max_length=256, num_return_sequences=1)
        question = tokenizer.decode(output[0], skip_special_tokens=True)
        questions.append(question)

    return "\n\n".join(questions)  # Join multiple questions with spacing

#def generate_questions_t5(input_text, num_questions):
#    """Generates multiple-choice questions using T5 model."""
#    prompt = f"generate {num_questions} questions: {input_text}"
#   inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

#    with torch.no_grad():
        output = model.generate(**inputs, max_length=256, num_return_sequences=num_questions)

#    mcqs = []
#   for i, generated in enumerate(output):
#        question = tokenizer.decode(generated, skip_special_tokens=True)
#       mcqs.append(f"## MCQ {i+1}\n{question}")

#    return "\n\n".join(mcqs)

def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("## MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)  # Add a line break

    pdf.output(filename)
    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@app.route('/generate', methods=['POST'])
def generate_mcqs():
    text = request.form.get('text_input', '').strip()  # Fix here
    file = request.files.get('file')

    if not text and file and allowed_file(file.filename):
        filename = file.filename
        file_path = filename
        file.save(file_path)
        text = extract_text_from_file(file_path)
    
    if not text:
        return "No valid text input provided."

    num_questions = int(request.form.get('num_questions', 5))  # Default to 5 questions
    mcqs =generate_questions_t5(text, num_questions)
    pdf_filename = "generated_mcqs.pdf"
    create_pdf(mcqs, pdf_filename)

    return render_template('results.html', mcqs=mcqs, pdf_filename=pdf_filename)

#def generate_mcqs():
#    text = request.form['text']
#    #text = request.form.get('text_input', '').strip()
#   num_questions = int(request.form['num_questions'])
#   mcqs = generate_questions_t5(text, num_questions)

#   pdf_filename = "generated_mcqs.pdf"
#    create_pdf(mcqs, pdf_filename)

#    return render_template('results.html', mcqs=mcqs, pdf_filename=pdf_filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__== "__main__":
    app.run(debug=True)
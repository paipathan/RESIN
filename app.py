from pathlib import Path
from flask import Flask, request, render_template, send_from_directory
import os
from pdfminer.high_level import extract_text
import subprocess
import json
import sys
import fitz  
import re

def normalize_text(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()

def fuzzy_match(a, b, threshold=0.1):
    a_words = normalize_text(a)
    b_words = normalize_text(b)
    if not a_words or not b_words:
        return False
    overlap = len(set(a_words) & set(b_words))
    ratio = overlap / max(len(a_words), len(b_words))
    return ratio >= threshold

def get_color(labels):
    score = 0
    if 'Metrics' in labels: score += 1
    if 'Actionable' in labels: score += 1
    if 'Domain-Specific' in labels: score += 1
    if score == 0:
        return (1, 0, 0)         # Red
    elif score == 1:
        return (1, 0.5, 0)       # Orange
    elif score == 2:
        return (1, 1, 0)         # Yellow
    else:
        return (0, 1, 0)         # Green

app = Flask(__name__)
app.config['TEMP_FOLDER'] = 'temp'

os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

def return_text(file_path):
    return extract_text(file_path)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    for child in Path(app.config['TEMP_FOLDER']).iterdir():
        os.remove(child)
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return 'No file part'

    file = request.files['pdf_file']

    if file.filename == '':
        return 'No selected file'

    if file and file.filename.lower().endswith('.pdf'):
        temp_path = os.path.join(app.config['TEMP_FOLDER'], file.filename)
        file.save(temp_path)

        resume_text = return_text(temp_path)

        python_executable = sys.executable
        result = subprocess.run(
            [python_executable, 'scan.py'],
            capture_output=True,
            text=True,
            cwd="C:/Users/Abhay Jani/Desktop/RESIN"
        )


        output = result.stdout.strip()
        
        image_dir = 'temp'  # Update path if different
        image_files = [f for f in os.listdir(image_dir) if f.startswith("page_") and f.endswith(".png")]
        num_pages = len(image_files)

        return render_template(
            'result.html',
            pdf_filename=file.filename,
            resume_text=resume_text,
            scanned_result = json.loads(output),
            num_pages = num_pages
        )

    return 'Invalid file type'

@app.route('/temp/<path:filename>')
def serve_temp_file(filename):
    return send_from_directory(app.config['TEMP_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

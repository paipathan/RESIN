from pdfminer.high_level import extract_text
import os
import spacy
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
import json
import fitz  
import re
from pdf2image import convert_from_path


result = []

nlps = [spacy.load('actionable_classifier'), 
        spacy.load('metric_classifier'), 
        spacy.load('domain_classifier')]

sentence_tokens = sent_tokenize(extract_text("temp/" + os.listdir('temp')[0]))

for i, sentence in enumerate(sentence_tokens):
    result.append([sentence])
    for nlp in nlps:    
        doc = nlp(sentence)
        prediction = max(doc.cats, key=doc.cats.get)
        result[i].append(prediction)


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

# Convert to tuple format for matching
result_map = [(entry[0], entry[1:]) for entry in result]

doc = fitz.open("temp/" + os.listdir('temp')[0])

for page in doc:
    blocks = page.get_text("blocks")
    for block in blocks:
        text = block[4]
        rect = fitz.Rect(block[0], block[1], block[2], block[3])
        for entry_text, labels in result_map:
            if fuzzy_match(text, entry_text):
                highlight_color = get_color(labels)
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=None, fill=highlight_color)
                annot.set_opacity(0.5)
                annot.set_border(width=0)
                annot.update()

                break



poppler_path = r"C:\Users\Abhay Jani\Desktop\poppler-24.08.0\Library\bin"


print(json.dumps(result))
doc.save("temp/output_highlighted.pdf")
images = convert_from_path("temp/output_highlighted.pdf", output_folder="temp", poppler_path=poppler_path)

for i, img in enumerate(images):
    img.save(f"temp/page_{i + 1}.png")
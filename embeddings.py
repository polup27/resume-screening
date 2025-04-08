import os
from docx import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        reader = PdfReader(filepath)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text
    elif filepath.endswith('.docx'):
        doc = Document(filepath)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        return ""

def calculate_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

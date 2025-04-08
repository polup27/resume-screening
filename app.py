from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from embeddings import extract_text_from_file, calculate_similarity

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    match_score = None
    if request.method == 'POST':
        job_desc = request.form['job_desc']
        resume_file = request.files['resume']
        filename = secure_filename(resume_file.filename)

        # Save resume to uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)

        resume_text = extract_text_from_file(filepath)
        match_score = calculate_similarity(job_desc, resume_text)
        match_score = round(match_score * 100, 2)

    return render_template('index.html', match_score=match_score)

if __name__ == '__main__':
    app.run(debug=True)

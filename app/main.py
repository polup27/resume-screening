import os, json
from app.parser import extract_text_from_pdf
from app.model import get_embeddings
from app.ranker import rank_resumes

resume_folder = 'data/resumes'
job_description_file = 'data/job_descriptions.txt'

# Extract resume texts
resume_texts = []
file_names = []
for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        path = os.path.join(resume_folder, file)
        text = extract_text_from_pdf(path)
        resume_texts.append(text)
        file_names.append(file)

# Get job description
with open(job_description_file, 'r') as f:
    job_description = f.read()

# Embeddings
resume_vecs = get_embeddings(resume_texts)
job_vec = get_embeddings([job_description])[0]

# Ranking
rankings = rank_resumes(resume_vecs, job_vec)

# Show results
for idx, score in rankings:
    print(f"{file_names[idx]} → Score: {score[0]:.4f}")

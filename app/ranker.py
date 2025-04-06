from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(resume_vectors, job_vector):
    scores = cosine_similarity(resume_vectors, [job_vector])
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked

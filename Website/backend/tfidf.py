import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_top_documents_tfidf(txt_files, query):
    # Step 1: Load documents and filenames
    documents = []
    filenames = []

    for file_name in txt_files:
        if file_name.endswith('.txt'):
            filenames.append(file_name)
            with open("./text_files/" +  file_name, 'r', encoding='utf-8') as file:
                documents.append(file.read())

    # Step 2: Define the query
    query = query

    # Step 3: Preprocess and compute TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    # Step 4: Compute similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Step 5: Retrieve top 5 documents
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]
    top_5_results = [(filenames[i], cosine_similarities[i]) for i in top_5_indices]

    ans = []
    for filename, _ in top_5_results:
        ans.append(filename)  
    return ans[:3]



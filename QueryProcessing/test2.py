# %%
"""
#### Code for retrieval using TF-IDF
"""

# %%
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Step 1: Load documents and filenames
folder_path = "description"  # Replace with your folder path
documents = []
filenames = []  # To store filenames

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        filenames.append(file_name)  # Save the filename
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            documents.append(file.read())

# Step 2: Define the query
query = "Brown Men's Shirt"

# Step 3: Preprocess and compute TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
tfidf_matrix = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

# Step 4: Compute similarity
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Step 5: Retrieve top 5 documents
top_5_indices = cosine_similarities.argsort()[-5:][::-1]  # Indices of top 5 similar documents
top_5_results = [(filenames[i], cosine_similarities[i]) for i in top_5_indices]  # Pair filename with score

# Print the results
print("Top 5 documents:")
for filename, score in top_5_results:
    print(f"Filename: {filename}, Similarity Score: {score}")


# %%
import os
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# %%
# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load documents and filenames
folder_path = "description"  # Replace with your folder path
documents = []
filenames = []  # To store filenames

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        filenames.append(file_name)  # Save the filename
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            documents.append(file.read())

# Step 2: Preprocess documents
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove punctuation and stopwords
    return tokens

tokenized_documents = [preprocess(doc) for doc in documents]

# Step 3: Initialize BM25
bm25 = BM25Okapi(tokenized_documents)

# Step 4: Define and preprocess the query
query = "Brown Men's Shirt"
tokenized_query = preprocess(query)

# Step 5: Compute BM25 scores
scores = bm25.get_scores(tokenized_query)

# Step 6: Retrieve top 5 documents
top_5_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
top_5_results = [(filenames[i], scores[i]) for i in top_5_indices]

# Print the results
print("Top 5 documents:")
for filename, score in top_5_results:
    print(f"Filename: {filename}, BM25 Score: {score}")

# %%
import os
from sentence_transformers import SentenceTransformer, util


# %%
# Step 1: Load documents and filenames
folder_path = "description"  # Replace with your folder path
documents = []
filenames = []  # To store filenames

for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):
        filenames.append(file_name)  # Save the filename
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            documents.append(file.read())

# Step 2: Load SentenceBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight, fast model for embedding

# Step 3: Compute embeddings for documents
document_embeddings = model.encode(documents, convert_to_tensor=True)

# Step 4: Define and compute embedding for the query
query = "Brown Men's Shirt"
query_embedding = model.encode(query, convert_to_tensor=True)

# Step 5: Compute cosine similarity
cosine_similarities = util.cos_sim(query_embedding, document_embeddings).flatten()

# Step 6: Retrieve top 5 documents
top_5_indices = cosine_similarities.argsort(descending=True)[:5]
top_5_results = [(filenames[i], cosine_similarities[i].item()) for i in top_5_indices]

# Print the results
print("Top 5 documents:")
for filename, score in top_5_results:
    print(f"Filename: {filename}, Similarity Score: {score}")
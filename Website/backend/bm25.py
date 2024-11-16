import os
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
# Initialize the stopwords once
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

def retrieve_top_documents(txt_files, query):

    if tokenized_documents is None or bm25 is None:
        documents = []
        filenames = []

        for file_name in txt_files:
            if file_name.endswith('.txt'):
                filenames.append(file_name)
                with open("./text_files/" +  file_name, 'r', encoding='utf-8') as file:
                    documents.append(file.read())

        tokenized_documents = [preprocess(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_documents)

    # Step 4: Define and preprocess the query
    tokenized_query = preprocess(query)

    # Step 5: Compute BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Step 6: Retrieve top 5 documents
    top_5_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    top_5_results = [(filenames[i], scores[i]) for i in top_5_indices]

    ans = []
    for filename, _ in top_5_results:
        ans.append(filename)  
    return ans[:3]

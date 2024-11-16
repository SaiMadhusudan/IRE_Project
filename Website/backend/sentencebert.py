import os
from sentence_transformers import SentenceTransformer, util

def retrieve_top_documents_setencebert(txt_files, query,model):
    # Step 1: Load documents and filenames
    documents = []
    filenames = []

    for file_name in txt_files:
        if file_name.endswith('.txt'):
            filenames.append(file_name)
            with open("./text_files/" +  file_name, 'r', encoding='utf-8') as file:
                documents.append(file.read())

    # Step 2: 
    model = model.to('cuda:0')  # Move the model to CUDA:0

    # Step 3: Compute embeddings for documents
    document_embeddings = model.encode(documents, convert_to_tensor=True, device='cuda:0')

    # Step 4: Define and compute embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True, device='cuda:0')

    # Step 5: Compute cosine similarity
    cosine_similarities = util.cos_sim(query_embedding.cpu(), document_embeddings.cpu()).numpy().flatten()

    # Step 6: Retrieve top 5 documents
    top_5_indices = cosine_similarities.argsort()[::-1][:5]
    top_5_results = [(filenames[i], cosine_similarities[i]) for i in top_5_indices]


    ans = []
    for filename, _ in top_5_results:
        ans.append(filename)  
    return ans[:3]



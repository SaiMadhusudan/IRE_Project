import os
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Set environment variable for API key
os.environ['ANTHROPIC_API_KEY'] = 'key'

# Initialize LLM and embedding models
llm = Anthropic(temperature=0.0, model='claude-3-haiku-20240307')
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Configure settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Load documents from the specified directory
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# Create an index from the loaded documents
index = VectorStoreIndex.from_documents(documents)

# Set up retriever for querying the index
retriever = index.as_retriever(verbose=True)

# Retrieve documents based on the query
response = retriever.retrieve("Can you suggest me some luxury cars from the given list?")

# Display the retrieved documents in a clean format
for idx, doc in enumerate(response, start=1):
    print(f"{idx}. {doc}")

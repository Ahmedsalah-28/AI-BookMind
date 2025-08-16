import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBED_MODEL

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def embed_and_normalize(texts):
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(texts)
    return normalize(np.array(vectors)), embeddings

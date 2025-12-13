import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
import logging
from config import PDF_PATH, CHROMA_DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Chunk the text
def chunk_text(text, max_len=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunks.append(text[start:end])
        start += (max_len - overlap)
    return chunks

# Create embeddings using Hugging Face SentenceTransformer
def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks, show_progress_bar=True)

# Store chunks and embeddings in ChromaDB
def store_in_chroma(chunks, embeddings):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    emb_list = [emb.tolist() for emb in embeddings]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=emb_list)

# Build DB pipeline
def build_database():
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    store_in_chroma(chunks, embeddings)
    logging.info(f"Stored {len(chunks)} chunks into ChromaDB successfully!")

if __name__ == "__main__":
    build_database()
import chromadb
from huggingface_hub import InferenceClient
from config import CHROMA_DB_PATH, COLLECTION_NAME, HF_API_KEY, TOP_K_CHUNKS

# Initialize ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Retrieve relevant chunks
def retrieve_chunks(query, n_results=TOP_K_CHUNKS):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]

# Initialize Hugging Face Inference client
hf_client = InferenceClient(api_key=HF_API_KEY)

# Generate answer using Hugging Face
def generate_answer_hf(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are an expert assistant specialized in Polycab Fans products. 
Use ONLY the context provided below to answer the user's question. 
Do NOT provide information that is not present in the context. 
If the answer cannot be found in the context, respond exactly with: 
'The information is not available in the provided catalogue.'

Answer clearly, concisely, and professionally. 
Preserve any units, numbers, and technical terminology as given in the context. 
Do not invent or assume anything beyond what is stated in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = hf_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message["content"]

# Polycab Fans FAQ Bot

This repository contains a **RAG-based FAQ system** for the Polycab Fans catalogue.  
Users can ask questions related to the catalogue, and the bot retrieves relevant information from the PDF and generates concise answers using LLM.

---

## Features

- Extracts text from the PDF catalogue of Polycab Fans.  
- Splits the text into **semantic chunks** for better retrieval.  
- Generates **vector embeddings** using Hugging Face `SentenceTransformer`.  
- Stores embeddings in **ChromaDB** for efficient retrieval.  
- Uses a **DeepSeek LLM via Hugging Face Inference API** to generate context-aware answers.  
- Streamlit interface for easy interaction with the bot.  

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/hsnandi/rag_streamlit
cd rag_streamlit
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Build the vector database:

```bash
python build_db.py
```
This will extract text from the PDF, create embeddings, and store them in ChromaDB.

5. Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Usage

1. Open the Streamlit interface in your browser.

2. Enter a question about Polycab Fans in the input box.

3. Click Get Answer.

4. The bot will retrieve relevant chunks from the PDF and generate a concise answer using the LLM.

---

## Implementation Notes

- Text Extraction: Uses pdfplumber to extract text from PDF pages.

- Chunking & Embeddings: Text is split into overlapping chunks to preserve context. Embeddings are generated using SentenceTransformer.

- Vector Database: ChromaDB is used for fast retrieval of relevant chunks.

- Answer Generation: A Hugging Face LLM (chat-based model) generates responses based strictly on the retrieved context.

### Optimization Note:

While this implementation works efficiently with open-source models and tools, using more advanced or specialized models for text extraction and LLM inference can further improve retrieval accuracy, answer quality, and performance.

---

## Project Highlights

- Fully modular and reusable code.

- Clear separation between data preparation, retrieval, and answer generation.

- Demonstrates prompt engineering skills to prevent hallucinations and ensure context-aware answers.

- Fast and interactive FAQ bot with Streamlit interface.

---

## Dependencies

- streamlit
- sentence-transformers
- chromadb
- huggingface_hub
- pdfplumber
- torch
- numpy

All dependencies are listed in requirements.txt.


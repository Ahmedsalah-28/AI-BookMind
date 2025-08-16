# 💬 AI/ML Books RAG Chat Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot for AI, Machine Learning, and LLM books.  
This project allows you to ask questions about content in your PDFs and get **accurate answers based on the documents**.

---

## 📚 Books Included

- `0406_Software_Engineering_for_Data_Scientists`  
- `Hands-On Large Language Models Language Understanding and Generation`  
- `Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)` by Aurelien Geron

👉 You can also **add any other books you want** to the `data/` folder, and the assistant will automatically process them.


---

## ⚡ Features

- Extracts text from PDFs and splits it into chunks for better retrieval.
- Embeds text using **sentence-transformers** for semantic search.
- Builds a **FAISS vector index** for fast similarity search.
- Integrates **RAG pipeline with memory** for conversational context.
- Answers questions **only based on provided content**, with source references.
- Streamlit web interface for easy interaction.

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ai-ml-books-rag-chat.git
cd ai-ml-books-rag-chat

Create a Conda environment:

conda create -n ai_books_rag python=3.11 -y
conda activate ai_books_rag


Install dependencies from requirements.txt:

pip install -r requirements.txt

Create a .env file based on .env.example and add your OpenRouter API key:

OPENROUTER_API_KEY=your_openrouter_api_key_here

🚀 Usage
1. Build the FAISS index from your PDFs:
python build_index.py

2. Run the Streamlit chatbot:
streamlit run app.py


Enter questions about AI, ML, and LLMs.

The bot answers based only on your PDF content.

Sources are displayed for reference.

🗂 File Structure
.
├─ data/                 # Folder containing PDFs
├─ processed_chunks/     # Auto-generated text chunks
├─ faiss_index/          # FAISS vector store
├─ config.py             # Configuration file
├─ pdf_utils.py          # PDF text extraction and cleaning
├─ chunk_utils.py        # Splitting text into chunks
├─ embedding_utils.py    # Embedding and normalization
├─ build_index.py        # Build FAISS index
├─ app.py                # Streamlit chat interface
├─ requirements.txt      # Project dependencies
├─ .env.example          # Environment variables example
└─ README.md             # Project documentation

"

📜 License

MIT License - free to use, modify, and share.



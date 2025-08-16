from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def split_into_chunks(text, metadata=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents([text], metadatas=[metadata or {}])

def save_chunks(chunks, book_title):
    book_dir = Path(OUTPUT_DIR) / book_title.replace(" ", "_")
    book_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        file_path = book_dir / f"chunk_{i}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)

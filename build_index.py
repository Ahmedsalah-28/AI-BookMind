from pathlib import Path
from config import PDF_FOLDER, INDEX_DIR
from pdf_utils import extract_text_from_pdf, clean_text
from chunk_utils import split_into_chunks, save_chunks
from embedding_utils import get_embeddings
from langchain_community.vectorstores import FAISS

def main():
    all_docs = []
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    print(f"📚 Found {len(pdf_files)} PDF files in '{PDF_FOLDER}'")

    for pdf_path in pdf_files:
        title = pdf_path.stem
        print(f"📖 Processing: {title}")
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw_text)
        docs = split_into_chunks(cleaned, metadata={"book_title": title})
        save_chunks(docs, title)
        all_docs.extend(docs)

    # Build FAISS index directly from documents
    vectorstore = FAISS.from_documents(
        all_docs,
        get_embeddings()
    )
    vectorstore.save_local(INDEX_DIR)

    print(f"✅ Processed {len(all_docs)} chunks from {len(pdf_files)} books.")
    print(f"💾 FAISS index saved in: {INDEX_DIR}")

if __name__ == "__main__":
    main()

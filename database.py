import os
import shutil
from typing import List
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema.document import Document

# File paths
CHROMA_PATH = "chroma"
PDF_FILES = [
    "data/gelir_vergisi.pdf",
    "data/katma_deger.pdf",
    "data/ozel_tuketim.pdf",
    "data/kurumlar.pdf",
    "data/motorlu_tasit.pdf",
]
TXT_DIR = "data/processed_txt"


def main():
    documents = load_documents(PDF_FILES)
    chunks = split_documents(documents)
    add_to_chroma(chunks)  

def get_embeddings_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings  

def load_documents(pdf_files: List[str]) -> List[Document]:
    """
    Converts multiple PDF files into text files and returns their content as a list of Documents.
    """
    os.makedirs(TXT_DIR, exist_ok=True)
    documents = []
    
    for pdf_file in pdf_files:
        # Generate the corresponding text file path
        base_name = os.path.basename(pdf_file).replace(".pdf", ".txt")
        txt_path = os.path.join(TXT_DIR, base_name)
        
        # Convert PDF to text
        with open(pdf_file, "rb") as file, open(txt_path, "w", encoding="utf-8") as txt_file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            txt_file.write(text)
        print(f"PDF converted to text file at: {txt_path}")
        
        # Create a single Document with the full text
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            content = txt_file.read()
        documents.append(Document(page_content=content, metadata={"source": pdf_file}))

    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits the content of the documents into chunks based on 'Madde' or 'MADDE' headings
    and adds metadata for each chunk.
    """
    madde_documents = []
    for document in documents:
        text = document.page_content
        lines = text.split("\n")
        
        madde_content = []
        for line in lines:
            line = line.strip()
            # Case-sensitive check for "Madde" or "MADDE"
            if line.startswith("Madde") or line.startswith("MADDE"): 
                if madde_content:  
                    madde_documents.append(Document(
                        page_content="\n".join(madde_content),
                        metadata=document.metadata
                    ))
                    madde_content = []  
                madde_content.append(line)  
            elif madde_content:
                madde_content.append(line) 
        
        # Add the last madde
        if madde_content:
            madde_documents.append(Document(
                page_content="\n".join(madde_content),
                metadata=document.metadata
            ))

    return madde_documents

def add_to_chroma(chunks: list[Document]):
    # Ensure the database directory exists.
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH, exist_ok=True)

    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings_function())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
   
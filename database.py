import os
import shutil
from typing import List
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema.document import Document

# File paths
CHROMA_PATH = "chroma"
CHROMA_PATHS = [
    "chroma/gelir_vergisi",
    "chroma/katma_deger",
    "chroma/ozel_tuketim",
    "chroma/kurumlar",
    "chroma/motorlu_tasit",
    ]    

# Directory containing the PDF files
PDF_DIR = "data"  # Specify the directory where your PDFs are stored
TXT_DIR = "data/processed_txt"  # Directory for storing the converted text files
    
def main(chroma_path: str):
    # Dynamically get PDF files from the directory
    pdf_files = get_pdf_files(PDF_DIR)
    if not pdf_files:
        print(f"No PDF files found in directory: {PDF_DIR}")
        return

    documents = load_documents(pdf_files)
    chunks = split_documents(documents)  # Ensure `split_documents` is defined elsewhere
    add_to_chroma(chunks, chroma_path)  # Ensure `add_to_chroma` is defined elsewhere

def get_embeddings_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings  

def get_pdf_files(directory: str) -> List[str]:
    """
    Scans the specified directory for PDF files and returns their paths.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".pdf")]

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

def add_to_chroma(chunks, chroma_path):
    # Ensure the database directory exists.
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path, exist_ok=True)

    # Load the existing database.
    embeddings = get_embeddings_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add.")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(PDF_DIR):
        shutil.rmtree(PDF_DIR)
        print(f"'{PDF_DIR}' path cleared.")
    else:
        print(f"'{PDF_DIR}' path does not exist.")

    # Reinitialize the database directory
    os.makedirs(PDF_DIR, exist_ok=True)

if __name__ == "__main__":
    main()

import re
import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

# File paths
CHROMA_PATH = "chroma"

# Chroma initialization moved outside the function to avoid multiple initializations
db = None

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def get_embeddings_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def extract_score(response_text):
    # Regular expression to find a number between 0 and 1
    match = re.search(r'\b0(?:\.\d+)?|1(?:\.0+)?\b', response_text)
    if match:
        return float(match.group())
    else:
        raise ValueError(f"Could not extract a valid score from response: {response_text}")

def init_chroma():
    global db
    if db is None:
        embedding_function = get_embeddings_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        print("Chroma initialized.")

# Integrate reranking in your query_rag function
def query_rag(query_text: str, prompt_template: str, model_name: str):
    # Prepare the DB.
    print("Initializing ChromaDB...")
    embedding_function = get_embeddings_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    print("Searching for similar documents...")
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("No results found in ChromaDB.")
        return "İlgili bağlam bulunamadı.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Use the provided prompt template
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Query the model with the selected model name
    print(f"Querying the model: {model_name}...")
    model = OllamaLLM(model=model_name)
    response_text = model.invoke(prompt)

    # Prepare source content for display
    sources = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown Source"),
        }
        for doc, _score in results
    ]    

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text, sources


if __name__ == "__main__":
    main()
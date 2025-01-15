import re
import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def get_embeddings_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings  

def query_rag(query_text: str, prompt_template: str, model_name: str, selected_path: str):
    # Prepare the DB.
    print("Initializing ChromaDB...")
    embedding_function = get_embeddings_function()
    print(f"Selected path: {selected_path}")
    db = Chroma(persist_directory=selected_path, embedding_function=embedding_function)

    # Search the DB.
    print("Searching for similar documents...")
    results = db.similarity_search_with_score(query_text, k=3)

    if not results:
        print("No results found in ChromaDB.")
        return "İlgili bağlam bulunamadı.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Use the provided prompt template
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(f"Prompt: {prompt}\n")

    # Query the model with the selected model name
    print(f"Querying the model: {model_name}...")
    model = OllamaLLM(model=model_name)
    response_text = model.invoke(prompt)

    # Prepare source content for display
    # Prepare source-content and score pairs
    sources = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown Source"),
            "score": score
        }
        for doc, score in results
    ]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)

    return response_text, sources

if __name__ == "__main__":
    main()
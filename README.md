
# CSE 496 - Graduation Project || Financial Documents Question Answering

This project provides a question-answering system based on financial documents in Turkish. The system processes the PDF documents you upload and generates accurate, context-aware answers to your questions related to tax laws.

## Features
- Category Selection: Choose the tax type related to your question.
- Document Management: View existing documents, upload new ones, and update the database.
- Database Clearing: Clear the entire database with a single click.
- Model Selection: Choose different Ollama models to answer your questions.
- Ask Questions and Get Answers: Write your question, see the relevant answer along with the sources.

## Requirements
To run the project, the following tools and libraries are required:
- Python 3.9 or higher
- Streamlit
- PyPDF2
- langchain
- langchain-chroma
- langchain-ollama

Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Setup and Execution
- Clone the Repository:
```bash
git clone https://github.com/eaysu/financial-question-answering.git
cd financial-question-answering
```
- Prepare Data Folders:
```bash
/data/income_tax.pdf
```
- Update the Database
- Run the Streamlit Application:
```bash
streamlit run app.py
```
- Interface:
    - Step 1: Select the tax category related to your question.
    - Step 2: Manage the documents in the data folder.
    - Step 3: Upload new PDF documents and update the database.
    - Step 4: Select the Ollama model to answer your questions.
    - Step 5: Enter your question and review the answers.


## Usage Workflow
- Launch the application.
- Select the category to define the context of your question.
- Upload your PDF documents and update the database.
- Enter your questions in the interface and get the answers.
- Review the sources if necessary.
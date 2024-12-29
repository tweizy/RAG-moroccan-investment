# Cohere RAG Streamlit Application

Welcome to the Cohere RAG Streamlit Application, a Retrieval-Augmented Generation (RAG) system designed to provide insightful answers about Moroccan investment opportunities. This application leverages Cohere's powerful language models, FAISS for efficient similarity search, and Streamlit for an intuitive web interface.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Application Locally](#running-the-application-locally)
- [Deployment](#deployment)
- [Additional Information](#additional-information)
- [Acknowledgements](#acknowledgements)

## Features
- PDF Text Extraction: Converts PDF documents into structured Markdown files.
- Text Chunking: Breaks down large texts into manageable, semantically meaningful chunks.
- Embeddings Generation: Utilizes SentenceTransformers to generate embeddings for each chunk.
- Efficient Similarity Search: Employs FAISS to enable fast retrieval of relevant chunks based on user queries.
- Interactive Web Interface: Streamlit-based UI with responsive design, pastel color themes, and interactive elements.
- Dynamic Retrieval Display: Shows retrieved chunks with expandable sections for detailed views.

## Prerequisites

Before setting up the project, ensure you have the following installed on your system:
- Python 3.8+
- pip (Python package manager)
- Virtual Environment Tools (venv or virtualenv)
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

Begin by cloning the repository to your local machine:

```bash
git clone https://github.com/tweizy/RAG-moroccan-investment
cd RAG-moroccan-investment
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies and avoid conflicts:

```bash
python3 -m venv venv
```

Activate the virtual environment:
- On macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

- On Windows (Command Prompt):

  ```bash
  venv\Scripts\activate.bat
  ```

- On Windows (PowerShell):

  ```bash
  venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install NLTK Data

The application relies on NLTK's punkt tokenizer. Install the necessary NLTK data:

```bash
python -m nltk.downloader punkt
```


### 5. Set Up Environment Variables

Create a `.env` file in the root directory of the project to store your Cohere API key securely:
1. Create the `.env` File:

   ```bash
   touch .env
   ```

2. Add Your Cohere API Key:
   Open `.env` in a text editor and add the following line:

   ```
   COHERE_API_KEY=your_actual_cohere_api_key
   ```

   Replace `your_actual_cohere_api_key` with your Cohere API key.

3. Ensure `.env` is Ignored by Git:
   Make sure `.env` is listed in `.gitignore` to prevent accidental commits of sensitive information.

## Data Preparation

To prepare the data for the RAG system, follow these steps in sequence:

### 1. Extract Text from PDFs

Use the `pdf_extraction.py` script to extract text from your PDF documents:

```bash
python scripts/pdf_extraction.py
```

Details:
- Input Directory: `./data/pdfs/` (Place all your original PDF files here.)
- Output Directory: `./data/extracted_texts/` (Markdown files will be saved here.)
- Logging: Extraction logs are saved in `extraction.log`.

### 2. Chunk the Extracted Text

Once text extraction is complete, use the `chunking.py` script to divide the extracted texts into manageable chunks:

```bash
python scripts/chunking.py
```

Details:
- Input Directory: `./data/extracted_texts/` (Contains Markdown files.)
- Output Directory: `./data/chunks/` (Chunked data will be saved as `chunked_data.json`.)

Parameters:
- Chunk Size: Approximately 400 words per chunk.
- Overlap Sentences: 1 sentence overlap between consecutive chunks to preserve context.

### 3. Generate Embeddings

Generate embeddings for each chunk using the `embeddings.py` script:

```bash
python scripts/embeddings.py
```

Details:
- Input File: `./data/chunks/chunked_data.json`
- Output File: `./data/chunks/embedded_chunks.json`
- Model: `all-MiniLM-L6-v2` 

Note: The script uses SentenceTransformers to generate embeddings and appends them to each chunk.

### 4. Build the Vector Database

Build the FAISS vector database for efficient similarity search using the `build_vdb.py` script:

```bash
python vectordb/build_vdb.py
```

Details:
- Input File: `./data/chunks/embedded_chunks.json`
- FAISS Index File: `./vectordb/faiss_index.index`
- ID Map File: `./vectordb/faiss_id_map.txt`
- Metadata Database: `./vectordb/metadata.db`

Steps:
1. Load Embedded Chunks: Reads `embedded_chunks.json`.
2. Initialize Metadata DB: Sets up `metadata.db` with chunk metadata.
3. Insert Metadata: Populates the SQLite database with chunk IDs, PDF names, and chunk texts.
4. Initialize FAISS Index: Creates a FAISS index with a dimension matching the embedding size (384 for `all-MiniLM-L6-v2`).
5. Add Vectors to FAISS: Inserts embedding vectors and maintains an ID map for retrieval.
6. Save FAISS Index and ID Map: Persists the FAISS index and corresponding ID map to disk.

Outcome: A fully built vector database ready for similarity searches during runtime.

## Running the Application Locally

After completing the data preparation steps, you can run the Streamlit application locally.

1. Ensure Environment Variables are Set

   Make sure your `.env` file contains the correct `COHERE_API_KEY`.

2. Launch the Streamlit App

   Run the following command from the root directory of the project:

   ```bash
   streamlit run app.py
   ```

   Expected Output:

   Streamlit will start the application and provide a local URL (typically `http://localhost:8501`). Open this URL in your web browser to access the app.

3. Using the Application
   - Enter a Query: Input your question related to Moroccan investment insights in the provided text box.
   - Generate Answer: Click the "Generate" button to initiate the retrieval and generation process.
   - View Results: The app will display the generated answer and list the retrieved chunks. Click on each chunk to view its full content.

## Acknowledgements
- Cohere: For providing powerful language models.
- FAISS: For efficient similarity search and clustering of dense vectors.
- Streamlit: For enabling rapid development of interactive web applications.
- SentenceTransformers: For state-of-the-art sentence embeddings.
- pdfplumber: For PDF text extraction.
- NLTK: For natural language processing tasks.

Happy Querying!
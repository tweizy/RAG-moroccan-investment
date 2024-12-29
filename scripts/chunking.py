import os
import re
import json
import uuid
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess_text(text):
    """
    Cleans the text by removing page headers and any unwanted patterns.

    :param text: The raw text extracted from a Markdown file.
    :return: Cleaned text.
    """
    # Remove page headers like '--- Page 121 ---'
    cleaned_text = re.sub(r'^--- Page \d+ ---$', '', text, flags=re.MULTILINE)
    
    # Remove lines that consist only of dashes or other separators
    cleaned_text = re.sub(r'^[-]+$', '', cleaned_text, flags=re.MULTILINE)
    
    # Remove excessive whitespace resulting from the removals
    cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text).strip()
    
    return cleaned_text


def _create_chunk(sentences, pdf_name):
    """
    Helper function to create a chunk dictionary from a list of sentences.

    :param sentences: List of sentences to include in the chunk.
    :param pdf_name: Name of the source PDF file.
    :return: Dictionary representing the chunk with metadata.
    """
    chunk_text = " ".join(sentences).strip()
    chunk_id = str(uuid.uuid4())
    return {
        "chunk_id": chunk_id,
        "pdf_name": pdf_name,
        "chunk_text": chunk_text
    }


def chunk_text(
    text,
    pdf_name,
    chunk_size=400,
    overlap_sentences=1,
    paragraph_separator="\n\n"
):
    """
    Breaks a given text into semantically meaningful chunks, respecting sentence boundaries.
    Each chunk is built by accumulating sentences until the chunk_size in words is reached or exceeded.
    A small overlap of 'overlap_sentences' ensures continuity.

    :param text: The full text extracted from a PDF.
    :param pdf_name: Name of the source PDF file (or .md file).
    :param chunk_size: The approximate word limit for each chunk.
    :param overlap_sentences: Number of sentences to overlap between consecutive chunks (helps preserve context).
    :param paragraph_separator: String that separates paragraphs in your content (usually a double newline).
    :return: A list of dictionaries, each representing a chunk with metadata.
    """
    # Split the text into paragraphs to maintain paragraph integrity
    paragraphs = text.split(paragraph_separator)

    chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue  # Skip empty paragraphs

        # Split paragraph into sentences
        sentences = sent_tokenize(paragraph)

        # Initialize buffer to accumulate sentences for the current chunk
        buffer = []
        buffer_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(word_tokenize(sentence))

            # Check if adding the current sentence exceeds the chunk size
            if buffer_word_count + sentence_word_count > chunk_size and buffer:
                # Finalize the current chunk
                chunk = _create_chunk(buffer, pdf_name)
                chunks.append(chunk)

                # Reset buffer with overlapping sentences
                if overlap_sentences > 0:
                    buffer = buffer[-overlap_sentences:]
                    buffer_word_count = sum(len(word_tokenize(s)) for s in buffer)
                else:
                    buffer = []
                    buffer_word_count = 0

            # Add the current sentence to the buffer
            buffer.append(sentence)
            buffer_word_count += sentence_word_count

        # After processing all sentences, check if any sentences remain in the buffer
        if buffer:
            chunk = _create_chunk(buffer, pdf_name)
            chunks.append(chunk)

    return chunks


def chunk_directory(
    input_dir="./extracted_texts",
    output_dir="./chunks",
    chunk_size=400,
    overlap_sentences=1
):
    """
    Reads all `.md` files in the input directory, preprocesses and chunks each file's content,
    and stores them in a single JSON file 'chunked_data.json' under output_dir.

    :param input_dir: Directory containing Markdown files extracted from PDFs.
    :param output_dir: Directory to store the resulting chunk data.
    :param chunk_size: Maximum approximate word count per chunk.
    :param overlap_sentences: Number of overlapping sentences between adjacent chunks.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory '{output_dir}'.")
    else:
        print(f"[INFO] Using existing output directory '{output_dir}'.")

    all_chunks = []
    total_files = 0
    skipped_chunks = 0

    # Iterate over all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".md"):
            total_files += 1
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Preprocess text to remove unwanted patterns
            cleaned_text = preprocess_text(text)

            # Skip files that are empty after preprocessing
            if not cleaned_text:
                print(f"[WARN] '{file_name}' is empty after preprocessing. Skipping.")
                continue

            # Derive the original PDF name from the Markdown file name
            pdf_name = file_name.replace(".md", ".pdf")
            print(f"[INFO] Chunking '{file_name}' (original PDF: {pdf_name}) ...")

            # Perform chunking on the cleaned text
            chunks = chunk_text(
                text=cleaned_text,
                pdf_name=pdf_name,
                chunk_size=chunk_size,
                overlap_sentences=overlap_sentences
            )

            # Filter out unwanted chunks (e.g., headers or very short chunks)
            filtered_chunks = []
            for chunk in chunks:
                # Skip chunks that match unwanted patterns (e.g., page headers)
                if re.match(r'^--- Page \d+ ---$', chunk["chunk_text"].strip()):
                    skipped_chunks += 1
                    continue  # Skip this chunk

                # Optionally, skip chunks that are too short to be meaningful
                if len(word_tokenize(chunk["chunk_text"])) < 20:
                    skipped_chunks += 1
                    continue  # Skip short, likely non-informative chunks

                filtered_chunks.append(chunk)

            # Add the filtered chunks to the complete list
            all_chunks.extend(filtered_chunks)
            print(f"[INFO] Created {len(filtered_chunks)} meaningful chunks from '{file_name}'.")

            if skipped_chunks > 0:
                print(f"[INFO] Skipped {skipped_chunks} unwanted chunks.")

    # Save all chunks to a JSON file with proper formatting
    output_path = os.path.join(output_dir, "chunked_data.json")
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(all_chunks, out_f, indent=2)

    # Print a summary of the chunking process
    print("\n=== Chunking Summary ===")
    print(f"Total Markdown files processed: {total_files}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Total chunks skipped: {skipped_chunks}")
    print(f"Chunk data saved to: {output_path}")
    print("========================\n")


def main():
    """
    Main function to run the chunking process with default parameters.
    Adjust chunk_size and overlap_sentences if needed.
    """
    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Define input and output directories
    input_directory = "./data/extracted_texts"
    output_directory = "./data/chunks"

    chunk_size = 400         # Approximate word limit per chunk
    overlap_sentences = 1    # Number of overlapping sentences between consecutive chunks

    print("=== Starting Powerful & Qualitative Chunking Process ===")
    # Start the chunking process
    chunk_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        chunk_size=chunk_size,
        overlap_sentences=overlap_sentences
    )
    print("=== Chunking Completed ===")


if __name__ == "__main__":
    main()
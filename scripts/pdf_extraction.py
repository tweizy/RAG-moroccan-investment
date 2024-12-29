import os
import pdfplumber
import logging

def setup_logging(log_file="extraction.log"):
    """
    Configures the logging settings.

    :param log_file: Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

def extract_text_from_page(page):
    """
    Extracts text from a single PDF page.

    :param page: A pdfplumber.page.Page object.
    :return: Extracted text from the page or None if no text is found.
    """
    text = page.extract_text()
    if text:
        return text
    return None

def extract_text_from_pdf(pdf_path):
    """
    Extracts structured text from a PDF file.

    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a single string.
    """
    logging.info(f"Opening PDF: {pdf_path}")
    extracted_text = ""
    try:
        # Open the PDF file using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            # Iterate through each page in the PDF
            for page_number, page in enumerate(pdf.pages, start=1):
                logging.info(f"Extracting text from page {page_number} of {os.path.basename(pdf_path)}")
                page_text = extract_text_from_page(page)
                if page_text:
                    # Add a header for each page to maintain structure
                    extracted_text += f"\n\n--- Page {page_number} ---\n\n"
                    extracted_text += page_text
    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {e}")
    return extracted_text

def extract_text_from_pdfs_in_directory(input_dir, output_dir="./extracted_texts"):
    """
    Extracts text from all PDF files in a specified directory using pdfplumber.

    :param input_dir: Directory containing PDF files to process.
    :param output_dir: Directory to store extracted text/markdown files.
    :return: A dictionary mapping {pdf_filename: extracted_text}.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory at '{output_dir}'.")
    else:
        logging.info(f"Using existing output directory at '{output_dir}'.")

    extracted_data = {}
    total_files = 0
    successful_extractions = 0
    failed_extractions = 0

    # Iterate over all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):  # Process only .pdf files
                total_files += 1
                pdf_path = os.path.join(root, file)
                logging.info(f"\nProcessing file {total_files}: '{file}'")
                try:
                    # Extract text from the current PDF
                    text_content = extract_text_from_pdf(pdf_path)
                    if text_content.strip():
                        # Define the output markdown file path
                        out_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.md")
                        # Save the extracted text to the markdown file
                        with open(out_file, "w", encoding="utf-8") as f:
                            f.write(text_content)
                        logging.info(f"Successfully extracted and saved text to '{out_file}'.")
                        extracted_data[file] = text_content
                        successful_extractions += 1
                    else:
                        logging.warning(f"No text extracted from '{file}'.")
                        extracted_data[file] = ""
                        failed_extractions += 1
                except Exception as e:
                    logging.error(f"Could not extract '{file}': {e}")
                    extracted_data[file] = ""
                    failed_extractions += 1

    # Log a summary of the extraction process
    logging.info("\n=== Extraction Summary ===")
    logging.info(f"Total PDF files processed: {total_files}")
    logging.info(f"Successful extractions: {successful_extractions}")
    logging.info(f"Failed extractions: {failed_extractions}")
    logging.info("==========================\n")

    return extracted_data

def main():
    """
    Main function to run the PDF text extraction process with default parameters.
    """
    # Configure logging to output to both console and file
    setup_logging()

    # Define input and output directories
    input_directory = "./data/pdfs"        # Directory containing original PDF files
    output_directory = "./data/extracted_texts"  # Directory to store extracted Markdown files

    logging.info("=== Starting PDF Text Extraction ===")
    # Extract text from all PDFs in the input directory
    extract_text_from_pdfs_in_directory(input_directory, output_directory)

    logging.info("=== PDF Text Extraction Process Completed ===")

if __name__ == "__main__":
    main()
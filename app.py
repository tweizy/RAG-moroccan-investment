# app.py

import os
import streamlit as st
from markdown import markdown
from rag_cohere import cohere_rag_response

###############################################################################
# Streamlit Application
###############################################################################

def load_tailwind():
    """Load Tailwind CSS and Font Awesome via CDN."""
    tailwind_css = """
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """
    st.markdown(tailwind_css, unsafe_allow_html=True)

def load_custom_styles():
    """Load custom CSS styles for additional customization."""
    custom_css = """
    <style>
    body {
        background-color: #fdf6f0; /* Pastel background */
    }
    .markdown-body {
        color: #2d3748; /* Darker text for contrast */
    }
    .btn-custom {
        background-color: #a3bffa; /* Pastel blue */
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .btn-custom:hover {
        background-color: #90cdf4; /* Slightly darker pastel blue on hover */
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Cohere RAG Demo", layout="centered")
    
    # Load Tailwind and Font Awesome
    load_tailwind()
    load_custom_styles()
    
    # Page Header with Icon
    st.markdown("""
    <div class="flex items-center justify-center mb-6">
        <i class="fas fa-file-alt fa-2x text-pastel-blue mr-3"></i>
        <h1 class="text-3xl font-bold text-pastel-blue">Cohere RAG Demo</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Enter your query about Moroccan investment insights, and retrieve an answer powered by a retrieval-augmented generation pipeline using Cohere.")
    
    # User Input
    user_query = st.text_input(
        label="Your Query",
        value="",
        placeholder="e.g. What incentives does Morocco offer to foreign automotive investors?",
        help="Ask anything related to Moroccan investment insights."
    )
    
    # Generate Button with Custom Styling
    if st.button("Generate", key="generate_button"):
        if user_query.strip() == "":
            st.warning("Please enter a query.")
        else:
            with st.spinner('Generating answer...'):
                final_answer_markdown, retrieved_chunks = cohere_rag_response(user_query)
            
            # Display the Answer
            st.markdown("### Answer:")
            st.markdown(final_answer_markdown, unsafe_allow_html=True)
            
            # Display Retrieved Chunks Individually with Expanders
            if retrieved_chunks:
                st.markdown("### Retrieved Chunks:")
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    with st.expander(f"🔍 Chunk {i}: *{chunk['pdf_name']}*"):
                        st.markdown(chunk['chunk_text'])
                        st.markdown("---")
    
    # Footer with Icons
    st.markdown("""
    <div class="flex items-center justify-center mt-10">
        <a href="https://streamlit.io" target="_blank" class="mx-2">
            <i class="fab fa-streamlit fa-2x text-pastel-blue"></i>
        </a>
        <a href="https://cohere.ai" target="_blank" class="mx-2">
            <i class="fas fa-cloud fa-2x text-pastel-blue"></i>
        </a>
        <a href="https://www.python.org" target="_blank" class="mx-2">
            <i class="fab fa-python fa-2x text-pastel-blue"></i>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional Styling (Pastel Blue Color)
    st.markdown("""
    <style>
    .text-pastel-blue {
        color: #a3bffa;
    }
    </style>
    """, unsafe_allow_html=True)

def run_app():
    main()

if __name__ == "__main__":
    run_app()
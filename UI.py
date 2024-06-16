import streamlit as st
from docx import Document
from PyPDF2 import PdfReader
# from gensim.summarization import summarize
import os
from transformers import pipeline
import base64
import DocumentGeneration, DocumentSummarization
def get_binary_file_downloader_html(content, filename):
    """
    Create a downloader HTML link.

    Args:
        content (str): Content to be downloaded.
        filename (str): Name of the file.

    Returns:
        str: HTML for downloader link.
    """
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def document_summarization(file):
    if file.name.endswith('.txt'):
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file.name.endswith('.pdf'):
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    elif file.name.endswith('.docx'):
        doc = Document(file)
        paragraphs = []
        for para in doc.paragraphs:
            paragraphs.append(para.text)
        text = '\n'.join(paragraphs)
    else:
        return "Unsupported file format. Please upload a .txt, .pdf, or .docx file."

    # summary = summarize(text)

    # summarizer = pipeline("summarization", model="t5-base")
    # summary = summarizer(text, max_length=1000, min_length=30, do_sample=False)
    # print(summary)
    summary = DocumentSummarization.Summarization(text)
    return summary

# def document_generation(document_type,scenario):
def document_generation(document_type, party_a, party_b, location, price, duration, scenario):
    requirements = {
    'document_type': document_type,
    'sections': [
        {
            'title': 'party_a',
            'content': party_a
        },
        {
            'title': 'party_b',
            'content': party_b
        },
        {
            'title': 'location',
            'content': location
        },
        {
            'title': 'price',
            'content': price
        },
        {
            'title': 'duration',
            'content': duration
        },
        {
            'title': 'scenario',
            'content': scenario
        }
    ]
}
    document = DocumentGeneration.Generation(requirements)
    print(document)
    return document

def display_document(document, doc_filename):
    st.markdown(f"## Displaying Document: {doc_filename}")
    # with open(document_path, 'rb') as file:
    #     document_content = file.read().decode(encoding, errors='replace')
    st.write(document)


st.title('AI Powered Legal Documentation Assistant')

option = st.sidebar.radio('Choose an option:', ['Home','Document Summarization', 'Document Generation'])

if option == 'Home':
    st.write("""
    This application is used for document generation and summarization of legal documents. 
    Please consider any misinterpretations as AI may make mistakes.
    
    For document generation, you can choose from 5 types:
    - Rental
    - Lease
    - MOU (Memorandum of Understanding)
    - Employee Contract
    - Loan
    """)

if option == 'Document Summarization':
    st.sidebar.header('Document Summarization')
    uploaded_file = st.sidebar.file_uploader('Upload Document', type=['txt', 'pdf', 'docx'])

    if uploaded_file is not None:
        if st.sidebar.button('Generate Summary'):
            summary = document_summarization(uploaded_file)
            st.subheader('Summary')
            st.write(summary)

elif option == 'Document Generation':
    st.sidebar.header('Document Generation')
    document_type = st.sidebar.selectbox('Document Type', ['Rental', 'Lease','MOU', 'Employee Contract', 'Loan'])
    party_a = st.sidebar.text_input('Parties Involved (A)')
    party_b = st.sidebar.text_input('Parties Involved (B)')
    location = st.sidebar.text_input('Location (If Required)')
    price = st.sidebar.text_input('Price (If Required)')
    duration = st.sidebar.text_input('Duration (If Required)')
    scenario = st.sidebar.text_area('Scenario')

    if st.sidebar.button('Generate Document'):
        # document_path = document_generation(document_type, scenario)
        document, doc_filename = document_generation(document_type, party_a, party_b, location, price, duration, scenario)

        st.write(doc_filename)
        st.subheader('Generated Document')
        display_document(document, doc_filename)
        st.sidebar.markdown(get_binary_file_downloader_html(doc_filename, 'Document'), unsafe_allow_html=True)


# document_type = "Agreement"
# party_a = "John Doe"
# party_b = "Jane Smith"
# location = "New York"
# price = "1000"
# duration = "6 months"
# scenario = "This agreement is for the lease of a commercial property located in downtown New York. The agreement is between John Doe, represented by ABC Corporation, and Jane Smith, represented by XYZ Enterprises. The lease term is for 6 months at a monthly rent of $1000. The property is located at 123 Main Street, New York."

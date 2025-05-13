import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Load text files from a folder into a list of Document objects.
    Each file will be a separate document with metadata containing the file name.
    """
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Giảm kích thước tài liệu để giảm số lượng token
                if len(content) > 1000:
                    content = content[:1000]
                doc = Document(
                    page_content=content,
                    metadata={"source": file_name, "title": file_name.replace('.txt', '')}
                )
                documents.append(doc)
    return documents

def create_vector_store(documents: List[Document], persist_directory: str = None):
    """
    Create a vector store from documents using Google Generative AI embeddings.
    Optionally persist the vector store to disk.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if persist_directory:
        return Chroma.from_documents(
            documents=documents, 
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        return Chroma.from_documents(
            documents=documents, 
            embedding=embeddings
        )

def get_vector_store(persist_directory: str):
    """
    Load a vector store from a persist directory.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

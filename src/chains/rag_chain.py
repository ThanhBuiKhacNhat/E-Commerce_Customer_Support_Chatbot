from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.llm_config import get_llm, create_ecommerce_customer_support_system_prompt
from typing import List, Dict
import os

def format_docs(docs):
    # Giới hạn số lượng tài liệu để tránh vượt quá hạn mức token
    if len(docs) > 2:
        docs = docs[:2]
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, with_sources=False):
    """
    Create a RAG chain for the customer support chatbot.
    
    Args:
        retriever: The retriever component to use for fetching relevant documents
        with_sources: Whether to include source information in the response
    
    Returns:
        A runnable chain that processes queries through the RAG pipeline
    """
    # Get LLM
    llm = get_llm(temperature=0.3)
    
    # System prompt
    system_prompt = create_ecommerce_customer_support_system_prompt()
      # RAG prompt template
    template = """
    Context: {context}
    
    Question: {question}
    
    Answer the question concisely using only information from the context. If the context doesn't have the answer, say so.
    """
    
    if with_sources:
        template += "\n\nSources: {sources}"
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", template)
    ])
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def create_direct_response_chain():
    """
    Create a direct response chain for handling general queries that don't require retrieval.
    
    Returns:
        A runnable chain that processes general queries
    """
    # Get LLM
    llm = get_llm(temperature=0.7)
    
    # System prompt
    system_prompt = create_ecommerce_customer_support_system_prompt()
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # Create the direct response chain
    direct_chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return direct_chain

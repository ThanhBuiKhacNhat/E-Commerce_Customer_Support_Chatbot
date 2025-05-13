from langchain_core.prompts import ChatPromptTemplate
from src.models.llm_config import get_llm
from src.chains.rag_chain import create_rag_chain, create_direct_response_chain
from typing import Dict, Any
from langchain_core.runnables import RunnableBranch
import os
from langsmith import Client
from langsmith.run_helpers import traceable

def create_query_classifier(categories: Dict[str, str]):
    """
    Create a simplified query classifier function using a direct LLM call.
    
    Args:
        categories: Dictionary mapping category names to descriptions
        
    Returns:
        A function that classifies queries
    """
    llm = get_llm(temperature=0)
      # Create a prompt template for classification
    category_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in categories.items()])
    
    template = f"""Classify this query into one of: {", ".join(categories.keys())}
Query: {{query}}
Category:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the classifier chain
    def classify_query(query):
        response = llm.invoke(prompt.format(query=query))
        category = response.content.strip().lower()
        
        # Ensure we return a valid category
        if category not in categories:
            # Default to general if category not recognized
            return "general"
        
        return category
    
    return classify_query

def create_router_chain(retriever):
    """
    Create a router chain that classifies and routes queries to the appropriate handler.
    
    Args:
        retriever: The retriever to use for RAG queries
        
    Returns:
        A chain that routes queries to the appropriate handler
    """
    # Create chains
    rag_chain = create_rag_chain(retriever)
    direct_chain = create_direct_response_chain()
    
    # Create classifier
    categories = {
        "product": "Questions about specific products, including features, prices, availability, or comparisons.",
        "ordering": "Questions about placing orders, payment methods, checkout process, or order modifications.",
        "shipping": "Questions about shipping methods, delivery times, tracking orders, or shipping costs.",
        "returns": "Questions about return policy, return process, refunds, or exchanges.",
        "account": "Questions about user accounts, login issues, password resets, or account settings.",
        "general": "General inquiries, greetings, or questions not falling into other categories."
    }
    
    query_classifier = create_query_classifier(categories)
      # Define routing logic based on classification
    def route_query(query):
        # Phân loại đơn giản dựa trên từ khóa thay vì gọi LLM để giảm số lượng API calls
        query_lower = query.lower()
        
        # Phân loại dựa trên từ khóa
        if any(keyword in query_lower for keyword in ["product", "item", "price", "cost", "available", "stock", "feature"]):
            return {"query": query, "chain": "rag"}
        elif any(keyword in query_lower for keyword in ["order", "buy", "purchase", "checkout", "payment", "pay"]):
            return {"query": query, "chain": "rag"}
        elif any(keyword in query_lower for keyword in ["ship", "delivery", "track", "arrive", "package"]):
            return {"query": query, "chain": "rag"}
        elif any(keyword in query_lower for keyword in ["return", "refund", "exchange", "money back", "cancel"]):
            return {"query": query, "chain": "rag"}
        else:
            # General or account queries can be handled directly
            return {"query": query, "chain": "direct"}    # Create the router chain
    def route_and_execute(query):
        routing = route_query(query)
        
        # Initialize LangSmith client
        client = Client()
        
        # Thực thi chuỗi phù hợp dựa trên phân loại
        try:
            # Trace the execution with LangSmith
            @traceable(project_name="ecommerce_chatbot", name=f"Query: {query[:50]}...")
            def execute_chain():
                if routing["chain"] == "rag":
                    response = rag_chain.invoke(query)
                    print(f"Query: '{query[:50]}...' was routed to RAG chain")
                    return {"response": response, "chain_type": "rag"}
                else:
                    response = direct_chain.invoke(query)
                    print(f"Query: '{query[:50]}...' was routed to direct chain")
                    return {"response": response, "chain_type": "direct"}
            
            result = execute_chain()
            return result["response"]
            
        except Exception as e:
            # Xử lý lỗi một cách nhẹ nhàng
            error_message = f"Sorry, I encountered an error: {str(e)}"
            print(f"Error processing query: {str(e)}")
            return error_message
    
    return route_and_execute

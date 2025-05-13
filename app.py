import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from src.utils.document_loader import load_documents_from_folder, create_vector_store, get_vector_store
from src.chains.router_chain import create_router_chain

# Load environment variables
load_dotenv()

# Set page title and configuration
st.set_page_config(
    page_title="ShopSmart Customer Support Chatbot",
    page_icon="🛒",
    layout="wide"
)

# Set up styling
st.markdown("""
<style>
.big-font {
    font-size:25px !important;
    font-weight: bold;
}
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    width: 80%;
}
.chat-message.user {
    background-color: #2b313e;
    margin-left: auto;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding-left: 1rem;
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<p class='big-font'>ShopSmart Customer Support Chatbot</p>", unsafe_allow_html=True)
st.markdown("📱 Ask any question about our products, ordering, shipping, returns, or other support questions!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize vector store and chains
@st.cache_resource
def initialize_retriever():
    """Initialize or load vector store and create retriever"""
    data_path = os.path.join(os.getcwd(), "data")
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    
    # Check if vector store already exists
    if os.path.exists(persist_directory):
        # Load existing vector store
        vector_store = get_vector_store(persist_directory)
    else:
        # Create new vector store
        documents = load_documents_from_folder(data_path)
        vector_store = create_vector_store(documents, persist_directory)
        vector_store.persist()
      # Create retriever
    return vector_store.as_retriever(
        search_kwargs={"k": 2}  # Giảm từ 3 xuống 2 để giảm số lượng token
    )

# Check if LangSmith API key is available
if os.getenv("LANGCHAIN_API_KEY"):
    try:
        from langsmith import Client
        client = Client()
        st.sidebar.success("✅ Connected to LangSmith - Tracing enabled")
    except Exception as e:
        st.sidebar.warning(f"⚠️ LangSmith connection error: {str(e)}")
else:
    st.sidebar.info("💡 LangSmith tracing is disabled")

# Get the retriever
retriever = initialize_retriever()

# Create the router chain
router_chain = create_router_chain(retriever)

# Chat input and processing
user_input = st.chat_input("Type your question here...")
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        # Get bot response with a spinner to show processing
        with st.spinner("Thinking..."):
            response = router_chain(user_input)
        
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add bot message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        # Handle errors
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with information about the chatbot
with st.sidebar:
    st.header("About this Chatbot")
    st.markdown("""
    This customer support chatbot is built using:
    - Gemini 2.0-flash 
    - LangChain for orchestration
    - Streamlit for UI
    - LangSmith for tracing (if enabled)
    
    It can answer questions about:
    - Products
    - Ordering process
    - Shipping
    - Returns and refunds
    - General inquiries
    """)
    
    # Add clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

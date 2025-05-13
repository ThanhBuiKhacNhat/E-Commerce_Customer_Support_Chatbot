# ShopSmart E-commerce Customer Support Chatbot

A RAG-based customer support chatbot for an e-commerce platform that leverages Google Gemini models, LangChain components, and Streamlit to provide helpful responses to customer queries.

## Features

- Natural language understanding of customer queries
- Context-aware responses based on knowledge base
- Handles queries about products, ordering, shipping, returns, and general inquiries
- Integration with LangSmith for LLM observability and debugging
- Responsive Streamlit UI

## Project Structure

```
E-Commerce-Chatbot/
│
├── data/                      # Knowledge base documents
│   ├── products.txt
│   ├── ordering.txt
│   ├── shipping.txt
│   ├── returns.txt
│   └── common_issues.txt
│
├── src/                       # Source code
│   ├── chains/                # LangChain chains
│   │   ├── rag_chain.py
│   │   └── router_chain.py
│   │
│   ├── models/                # Model configuration
│   │   └── llm_config.py
│   │
│   └── utils/                 # Utility functions
│       └── document_loader.py
│
├── chroma_db/                 # Persisted vector database (created at runtime)
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables (API keys)
└── README.md                  # Documentation
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Google API key for Gemini models
- LangSmith API key (optional, for tracing)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/ThanhBuiKhacNhat/E-Commerce_Customer_Support_Chatbot.git
   cd E-Commerce_Customer_Support_Chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Google API key and LangSmith API key (if available)
   ```
   GOOGLE_API_KEY=your_google_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=ecommerce_chatbot
   ```

### Running the Application

Start the Streamlit app:
```
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Usage

1. When the app starts, it will load the knowledge base and create a vector store (or load an existing one)
2. Type your customer support question in the chat input field
3. The chatbot will:
   - Classify the query type
   - Retrieve relevant information from the knowledge base (if needed)
   - Generate a helpful response
   - Display the response in the chat interface

## LangSmith Integration

This project uses LangSmith for:
- Tracing chains and LLM calls
- Analyzing token usage
- Debugging query processing
- Performance monitoring

To use LangSmith:
1. Sign up at https://smith.langchain.com
2. Get your API key and add it to the `.env` file
3. View traces in the LangSmith dashboard

## Customization

To extend the knowledge base:
1. Add new text files to the `data/` folder
2. Restart the application to rebuild the vector store

To customize model parameters:
1. Edit the `src/models/llm_config.py` file to adjust temperature, model, etc.

## License

MIT

## Contact

NhatThanh - thanhbuikhacnhat162@gmail.com

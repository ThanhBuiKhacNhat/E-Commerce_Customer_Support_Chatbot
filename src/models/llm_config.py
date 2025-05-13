from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm(temperature=0.7, model_name="gemini-2.0-flash"):
    """
    Initialize and return the Gemini model configured with the given parameters.

    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def create_ecommerce_customer_support_system_prompt():
    """
    Create a system prompt for the e-commerce customer support chatbot.
    """
    return """
    You are a customer support agent for ShopSmart e-commerce. Be concise, accurate, and helpful.
    Only use information from the provided context. Be professional and solution-focused.
    """

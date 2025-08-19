from langchain_openai import ChatOpenAI
from typing import Optional, List, Any
from pydantic import SecretStr
import os

def get_llm(with_tools: Optional[List[Any]] = None, temperature: float = 0):
    """
    Factory function to create ChatOpenAI instance with consistent configuration
    
    Args:
        with_tools: List of tools to bind to the LLM
        temperature: Temperature setting for the LLM
        
    Returns:
        ChatOpenAI instance, optionally bound with tools
    """
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",   
        api_key=os.environ.get("GROQ_API_KEY"),        
        model="Qwen/Qwen3-1.7B",      
        temperature=temperature,
    )
    
    if with_tools:
        llm = llm.bind_tools(with_tools)
    
    return llm


# "http://127.0.0.1:1234/v1"
# "https://api.groq.com/openai/v1"
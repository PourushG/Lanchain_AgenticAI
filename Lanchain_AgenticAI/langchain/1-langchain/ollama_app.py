from dotenv import load_dotenv
import streamlit as st
import os

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Set environment variables for LangSmith
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# PROMPT
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Answer user questions in a structured way."),
    ("user", "Question: {question}")
])

st.title("Ollama-based GenAI App 🚀")

# INPUT
user_input = st.text_input("Enter your question:")

# NEW Ollama LLM (Correct Import)
llm = OllamaLLM(model="gemma3:1b")
output_parser = StrOutputParser()

# CHAIN
chain = prompt | llm | output_parser

if user_input:
    response = chain.invoke({"question": user_input})
    st.write(response)

# 1. What is LangServe?
# LangServe is an open-source deployment library built on top of FastAPI (a popular web framework) and Pydantic (for data validation).

# Core Function: It takes a LangChain object (like a conversation chain) and automatically generates a standard API around it.

# Endpoints: It automatically creates endpoints for your chain, including:

# /invoke: Call the chain once.

# /batch: Process multiple inputs in parallel.

# /stream: Stream the output token-by-token (crucial for LLM apps).

# Playground: It provides a built-in graphical interface (at /playground/) where you can test your bot or chain in a browser without writing any frontend code.

# 2. Why is it used?
# Developers use LangServe to avoid writing repetitive "boilerplate" code. Without it, you would have to manually set up a Flask or FastAPI server, write routes, handle streaming protocols, and validate JSON inputs yourself.

# Zero-Setup Streaming: Streaming text from an LLM to a frontend is technically difficult. LangServe handles the complex logic (Server-Sent Events) automatically.

# Input Validation: It automatically looks at your LangChain code to figure out what inputs are required (e.g., "user_question", "chat_history") and validates them.

# Easy Integration: It provides client SDKs (RemoteRunnable) that let you call your API from another Python or JavaScript app as if it were running locally.

# Observability: It integrates well with LangSmith to trace and debug errors in your deployed application.

# 3. Where is it used?
# LangServe is used in the "backend" layer of modern AI application architectures.

# Microservices Architecture: It is often used to run the "AI Brain" as a separate service. For example, your main website (Node.js/React) sends a request to the LangServe microservice (Python) to get an answer.

# Cloud Platforms: It is designed to be containerized (Docker) and deployed on cloud platforms like Google Cloud Run, AWS ECS, Railway, or Hugging Face Spaces.

# Internal Tooling: Companies use it to expose internal LLM tools (e.g., a "Query Database" chain) to other internal teams via a standard API.

# 4. When is it used?
# You should use LangServe at the "Deployment" phase of your project lifecycle.

# Moving from Notebook to App: When you have a working chain in a Jupyter Notebook and want to turn it into a real application accessible by users.

# Separating Frontend from Backend: When you have a frontend team building the UI and an AI team building the logic. LangServe provides the API contract they use to communicate.

# Needing Reliability: When you need production-grade features like retries, heavy concurrency handling, and type safety, which are risky to implement from scratch.

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

groq_api = os.getenv("GROQ_API")

model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are helpful multi-lingual language translator and you need to translate following text in {language}:"),
    ("user","{text}")
])

parser = StrOutputParser()

chain = prompt|model|parser

app = FastAPI(title="Langchain Server", version="1.0", description="A simple API server using langchain runnable interfaces")

add_routes(app, chain, path='/chain')



if __name__=="__main__":
    import uvicorn as uv

    uv.run(app, host="127.0.0.1", port=8080)

    








# Import os to access environment variables
import os

# Load environment variables from the .env file
from dotenv import load_dotenv

# Import ChatGoogleGenerativeAI from LangChain's Google integration
from langchain_google_genai import ChatGoogleGenerativeAI


# Load variables from the .env file
load_dotenv()

# Create a Google Gemini model instance
# The API key is automatically read from GOOGLE_API_KEY
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Send a prompt to the Gemini model
response = llm.invoke(
    "Explain Artificial Intelligence in simple words."
)

# Print only the model's text response
print(response.content)
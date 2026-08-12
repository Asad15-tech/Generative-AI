# Load environment variables from the .env file
from dotenv import load_dotenv

# Import Hugging Face components from LangChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# Load variables from the .env file
load_dotenv()

# Create a Hugging Face endpoint
# The API token is automatically read from HUGGINGFACEHUB_API_TOKEN
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=50,
    temperature=0.7
)

# Wrap the Hugging Face endpoint as a chat model
chat_model = ChatHuggingFace(llm=llm)

# Send a prompt to the model
response = chat_model.invoke(
    "Why Imran Khan is in jail?"
)

# Print only the model's text response
print(response.content)

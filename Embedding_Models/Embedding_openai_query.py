# Import load_dotenv to load environment variables from the .env file
from dotenv import load_dotenv

# Import OpenAIEmbeddings from LangChain's OpenAI integration
from langchain_openai import OpenAIEmbeddings


# Load environment variables from the .env file
load_dotenv()

# Create an OpenAI embedding model
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32
)

# Convert a single query/text into an embedding vector
result = embedding.embed_query(
    "Delhi is the capital of India"
)

# Print the embedding vector
print(result)
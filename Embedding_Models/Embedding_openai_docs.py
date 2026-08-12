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

# Create a list of documents that we want to convert into embeddings
documents = [
    "Python is a popular programming language.",
    "Machine Learning allows computers to learn from data.",
    "Artificial Intelligence is used in many different fields."
]

# Convert all documents into embedding vectors
results = embedding.embed_documents(documents)

# Print each document with its corresponding embedding
for document, vector in zip(documents, results):
    print("\nDocument:", document)
    print("Embedding:", vector)
    print("Dimensions:", len(vector))
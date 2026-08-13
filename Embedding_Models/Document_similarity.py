# Load environment variables from the .env file
from dotenv import load_dotenv

# Import Hugging Face's InferenceClient
from huggingface_hub import InferenceClient

# Import cosine_similarity to compare embedding vectors
from sklearn.metrics.pairwise import cosine_similarity


# Load variables from the .env file
load_dotenv()


# Create a Hugging Face Inference API client
# The API token is read from the HF_TOKEN environment variable
client = InferenceClient(
    provider="hf-inference"
)


# Select a Hugging Face embedding model
model = "sentence-transformers/all-MiniLM-L6-v2"


# Create a collection of documents
documents = [
    "Python is a programming language widely used for AI, machine learning, and data science.",
    "Football is a popular sport played between two teams of eleven players.",
    "The human heart pumps blood throughout the body and supplies oxygen to tissues.",
    "Machine learning allows computers to learn patterns from data and make predictions."
]


# Create a user query
query = "How can I use Python for machine learning?"


# Convert all documents into embedding vectors
document_vectors = client.feature_extraction(
    documents,
    model=model
)


# Convert the user query into an embedding vector
query_vector = client.feature_extraction(
    query,
    model=model
)


# Calculate similarity between the query and every document
similarity_scores = cosine_similarity(
    [query_vector],
    document_vectors
)[0]


# Display the query
print("User Query:")
print(query)


# Display similarity scores for every document
print("\nSimilarity Scores:")

for document, score in zip(documents, similarity_scores):
    print(f"\nScore: {score:.4f}")
    print(f"Document: {document}")


# Find the index of the document with the highest similarity score
best_match_index = similarity_scores.argmax()


# Get the most relevant document
best_document = documents[best_match_index]


# Display the best matching document
print("\n" + "=" * 60)
print("Most Relevant Document:")
print(best_document)

print(f"\nSimilarity Score: {similarity_scores[best_match_index]:.4f}")
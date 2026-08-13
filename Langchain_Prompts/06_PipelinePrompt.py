# Load environment variables from the .env file
from dotenv import load_dotenv

# Import Hugging Face chat model integration
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()


# --------------------------------------------------
# STEP 1: Create the prompt
# --------------------------------------------------

# Create a chat prompt with a dynamic variable
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI teacher. "
            "Explain technical concepts in simple words."
        ),
        (
            "human",
            "Explain {topic} with a simple example."
        ),
    ]
)


# --------------------------------------------------
# STEP 2: Create Hugging Face model
# --------------------------------------------------

# HuggingFaceEndpoint connects LangChain
# with the Hugging Face Inference API
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=150,
    temperature=0.7,
)


# Convert the endpoint into a LangChain chat model
model = ChatHuggingFace(
    llm=llm
)


# --------------------------------------------------
# STEP 3: Create a pipeline
# --------------------------------------------------

# The | operator connects the components:
#
# Input → Prompt → Model
#
chain = prompt | model


# --------------------------------------------------
# STEP 4: Run the complete pipeline
# --------------------------------------------------

response = chain.invoke(
    {
        "topic": "LangChain"
    }
)


# --------------------------------------------------
# STEP 5: Display the response
# --------------------------------------------------

print("=" * 60)
print("AI RESPONSE")
print("=" * 60)

print(response.content)
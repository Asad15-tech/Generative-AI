# Load environment variables from the .env file
from dotenv import load_dotenv

# Import Hugging Face integrations from LangChain
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

# Import the prompt created in prompts.py
from prompts import chat_prompt


# Load variables from .env
load_dotenv()


# --------------------------------------------------
# STEP 1: Create Hugging Face model endpoint
# --------------------------------------------------

# Connect LangChain with the Hugging Face Inference API
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.7,
)


# --------------------------------------------------
# STEP 2: Convert endpoint into a chat model
# --------------------------------------------------

# ChatHuggingFace allows us to work with
# the model using LangChain's chat interface
model = ChatHuggingFace(
    llm=llm
)


# --------------------------------------------------
# STEP 3: Create the LangChain pipeline
# --------------------------------------------------

# Connect:
#
# ChatPromptTemplate → Hugging Face Model
#
chain = chat_prompt | model


# --------------------------------------------------
# STEP 4: Create a function for chatbot responses
# --------------------------------------------------

def get_response(question):

    # Send the user's question through the chain
    response = chain.invoke(
        {
            "question": question
        }
    )

    # Return only the AI-generated text
    return response.content
# Import ChatPromptTemplate from LangChain
from langchain_core.prompts import ChatPromptTemplate


# Create the main chatbot prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI Learning Assistant.

            Your job is to help users understand:
            - Python
            - Artificial Intelligence
            - Machine Learning
            - Deep Learning
            - LangChain
            - Generative AI

            Explain concepts clearly and simply.
            Use examples whenever useful.
            If the user asks a technical question,
            provide a practical explanation.
            """
        ),

        (
            "human",
            "{question}"
        ),
    ]
)
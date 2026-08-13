# Import ChatPromptTemplate from LangChain
from langchain_core.prompts import ChatPromptTemplate


# Create a chat prompt template
# Each tuple represents a message role and its content
prompt = ChatPromptTemplate.from_messages(
    [
        # System message defines how the AI should behave
        (
            "system",
            "You are a helpful AI assistant. Explain concepts in simple words."
        ),

        # Human message contains the user's question
        (
            "human",
            "Explain {topic}."
        ),
    ]
)


# Provide a value for the {topic} variable
messages = prompt.invoke(
    {
        "topic": "LangChain"
    }
)


# Display the generated messages
print(messages)
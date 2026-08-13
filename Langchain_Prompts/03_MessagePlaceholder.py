# Import ChatPromptTemplate and MessagesPlaceholder from LangChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# Import HumanMessage and AIMessage
# These are used to represent previous conversation messages
from langchain_core.messages import HumanMessage, AIMessage


# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        # System message defines the AI's behavior
        (
            "system",
            "You are a helpful AI assistant. "
            "Answer questions in simple and clear language."
        ),

        # This placeholder will dynamically receive chat history
        MessagesPlaceholder(variable_name="chat_history"),

        # Current user question
        (
            "human",
            "{question}"
        ),
    ]
)


# Create previous conversation messages
chat_history = [
    HumanMessage(content="What is LangChain?"),
    AIMessage(
        content="LangChain is a framework for building applications "
                "with language models."
    ),
]


# Provide the chat history and current question
messages = prompt.invoke(
    {
        "chat_history": chat_history,
        "question": "What are its main components?"
    }
)


# Display the generated messages
print(messages)
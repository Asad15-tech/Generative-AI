# ============================================================
# CHATBOT BACKEND
# Phase 2 - Conversation Memory
# ============================================================

import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# HUGGING FACE MODEL
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are LangChain AI Assistant.

            You help users learn:

            - Artificial Intelligence
            - Machine Learning
            - Python
            - Generative AI
            - LangChain

            Answer clearly and simply.

            Use examples when useful.

            Maintain context from the previous
            conversation.
            """
        ),

        (
            "human",
            """
            Previous conversation:

            {history}

            Current question:

            {question}
            """
        )
    ]
)


# ============================================================
# CHAIN
# ============================================================

chain = prompt | llm


# ============================================================
# GET RESPONSE
# ============================================================

def get_response(question):

    # --------------------------------------------------------
    # Convert previous messages into text
    # --------------------------------------------------------

    history_text = ""

    for message in conversation_history:

        if isinstance(message, HumanMessage):

            history_text += (
                f"Human: {message.content}\n"
            )

        elif isinstance(message, AIMessage):

            history_text += (
                f"AI: {message.content}\n"
            )


    # --------------------------------------------------------
    # Send question + history to model
    # --------------------------------------------------------

    response = chain.invoke(
        {
            "history": history_text,
            "question": question
        }
    )


    # --------------------------------------------------------
    # Save conversation
    # --------------------------------------------------------

    conversation_history.append(
        HumanMessage(
            content=question
        )
    )

    conversation_history.append(
        AIMessage(
            content=response
        )
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return response


# ============================================================
# CLEAR MEMORY
# ============================================================

def clear_memory():

    conversation_history.clear()
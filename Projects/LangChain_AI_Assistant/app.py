# Import Streamlit for creating the chatbot UI
import streamlit as st

# Import our chatbot response function
from chatbot import get_response


# --------------------------------------------------
# STEP 1: Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="LangChain AI Assistant",
    page_icon="AI",
    layout="centered"
)


# --------------------------------------------------
# STEP 2: Display application header
# --------------------------------------------------

st.title("LangChain AI Assistant")

st.caption(
    "An AI learning assistant powered by LangChain "
    "and Hugging Face"
)


# --------------------------------------------------
# STEP 3: Initialize chat history
# --------------------------------------------------

# Streamlit reruns the script whenever the user
# interacts with the application.
#
# session_state allows us to keep the conversation
# between these reruns.

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# STEP 4: Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    # Create the appropriate chat message container
    with st.chat_message(message["role"]):

        # Display the message content
        st.markdown(message["content"])


# --------------------------------------------------
# STEP 5: Get new user input
# --------------------------------------------------

user_question = st.chat_input(
    "Ask me anything about AI, ML, Python or LangChain..."
)


# --------------------------------------------------
# STEP 6: Process the user's question
# --------------------------------------------------

if user_question:

    # Store the user's message in chat history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Display the user's message immediately
    with st.chat_message("user"):

        st.markdown(user_question)


    # --------------------------------------------------
    # STEP 7: Generate AI response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        # Show a loading indicator while the model
        # is generating the response
        with st.spinner("Thinking..."):

            response = get_response(
                user_question
            )

        # Display the AI response
        st.markdown(response)


    # --------------------------------------------------
    # STEP 8: Store AI response in chat history
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
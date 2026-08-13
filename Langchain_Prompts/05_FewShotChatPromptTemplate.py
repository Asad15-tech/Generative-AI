# Load environment variables from the .env file
from dotenv import load_dotenv

# Import Hugging Face InferenceClient
from huggingface_hub import InferenceClient

# Import LangChain chat prompt components
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)


# Load environment variables
load_dotenv()


# --------------------------------------------------
# STEP 1: Create Hugging Face API client
# --------------------------------------------------

# InferenceClient automatically reads HF_TOKEN
# from the environment
client = InferenceClient()


# Hugging Face chat model
model = "Qwen/Qwen2.5-7B-Instruct"


# --------------------------------------------------
# STEP 2: Create few-shot examples
# --------------------------------------------------

# These examples show the model how to perform
# the task: informal English -> professional English

examples = [
    {
        "input": "Give me the file.",
        "output": "Could you please share the file with me?"
    },
    {
        "input": "I want this job.",
        "output": "I am very interested in this position."
    },
    {
        "input": "Make this better.",
        "output": "Please improve the quality of this content."
    }
]


# --------------------------------------------------
# STEP 3: Define the structure of each example
# --------------------------------------------------

# Each example contains:
# Human message -> user's informal sentence
# AI message    -> professional version

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)


# --------------------------------------------------
# STEP 4: Create Few-Shot Chat Prompt
# --------------------------------------------------

# This inserts all the examples into our chat prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)


# --------------------------------------------------
# STEP 5: Create the complete chat prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        # System message defines the model's behavior
        (
            "system",
            "You convert informal English into professional English. "
            "Reply only with the professional version of the user's "
            "sentence. Do not ask for more information."
        ),

        # Insert few-shot Human + AI examples
        few_shot_prompt,

        # New user input
        (
            "human",
            "{input}"
        ),
    ]
)


# --------------------------------------------------
# STEP 6: Provide a new input
# --------------------------------------------------

messages = prompt.invoke(
    {
        "input": "Send me your CV."
    }
)


# --------------------------------------------------
# STEP 7: Convert LangChain messages to
# Hugging Face message format
# --------------------------------------------------

hf_messages = []

for message in messages.messages:

    # LangChain uses "human"
    # Hugging Face expects "user"
    if message.type == "human":
        role = "user"

    # LangChain uses "ai"
    # Hugging Face expects "assistant"
    elif message.type == "ai":
        role = "assistant"

    # System role remains "system"
    elif message.type == "system":
        role = "system"

    # Add the converted message
    hf_messages.append(
        {
            "role": role,
            "content": message.content
        }
    )


# --------------------------------------------------
# STEP 8: Display the complete chat prompt
# --------------------------------------------------

print("=" * 60)
print("CHAT PROMPT")
print("=" * 60)

for message in hf_messages:
    print(f"{message['role'].upper()}: {message['content']}")


# --------------------------------------------------
# STEP 9: Send the prompt to Hugging Face
# --------------------------------------------------

response = client.chat.completions.create(
    model=model,
    messages=hf_messages,
    max_tokens=100,
    temperature=0.7
)


# --------------------------------------------------
# STEP 10: Display the model response
# --------------------------------------------------

print("\n" + "=" * 60)
print("HUGGING FACE RESPONSE")
print("=" * 60)

print(response.choices[0].message.content)
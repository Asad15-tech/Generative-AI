# Load environment variables from the .env file
from dotenv import load_dotenv

# Import Hugging Face InferenceClient
from huggingface_hub import InferenceClient

# Import FewShotPromptTemplate and PromptTemplate
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate
)


# Load variables from the .env file
load_dotenv()


# Create Hugging Face API client
client = InferenceClient()


# Select a Hugging Face chat model
model = "Qwen/Qwen2.5-7B-Instruct"


# --------------------------------------------------
# STEP 1: Create examples
# --------------------------------------------------

# These examples teach the model the desired pattern
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
# STEP 2: Define how each example should look
# --------------------------------------------------

example_prompt = PromptTemplate.from_template(
    "Input: {input}\nOutput: {output}"
)


# --------------------------------------------------
# STEP 3: Create Few-Shot Prompt
# --------------------------------------------------

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,

    # Instruction given before the examples
    prefix=(
        "Convert the following informal English into "
        "professional English.\n"
    ),

    # New input given after the examples
    suffix="Input: {input}\nOutput:",

    # Variable required by the template
    input_variables=["input"]
)


# --------------------------------------------------
# STEP 4: Provide a new input
# --------------------------------------------------

final_prompt = prompt.invoke(
    {
        "input": "Send me your CV."
    }
)


# Convert PromptValue into a normal string
prompt_text = final_prompt.to_string()


# Display the final prompt
print("=" * 60)
print("FINAL PROMPT")
print("=" * 60)
print(prompt_text)


# --------------------------------------------------
# STEP 5: Send the prompt to Hugging Face model
# --------------------------------------------------

response = client.chat.completions.create(
    model=model,

    messages=[
        {
            "role": "user",
            "content": prompt_text
        }
    ],

    max_tokens=100,
    temperature=0.7
)


# --------------------------------------------------
# STEP 6: Display model response
# --------------------------------------------------

print("\n" + "=" * 60)
print("HUGGING FACE RESPONSE")
print("=" * 60)

print(response.choices[0].message.content)
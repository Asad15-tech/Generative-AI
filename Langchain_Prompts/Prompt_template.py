# Import PromptTemplate from LangChain
from langchain_core.prompts import PromptTemplate


# Create a prompt template
# {topic} is a dynamic variable that we can change later
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words."
)


# Provide a value for the {topic} variable
final_prompt = prompt.invoke(
    {
        "topic": "LangChain"
    }
)


# Display the generated prompt
print(final_prompt)



# For Multi-Variable Prompt Templates, we can define multiple variables in the template
# Import PromptTemplate from LangChain
from langchain_core.prompts import PromptTemplate


# Create a prompt template
prompt = PromptTemplate.from_template(
    "Explain {topic} for a {audience} using {style}."
)

final_prompt = prompt.invoke(
    {
        "topic": "Artificial Intelligence",
        "audience": "beginner",
        "style": "simple examples"
    }
)

print(final_prompt)
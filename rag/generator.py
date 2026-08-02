import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model_name = os.getenv("GEMINI_MODEL")
prompt = "Explain what an API is in one sentence."

# Ask Gemini a simple question
response = client.models.generate_content(
    model=model_name,
    contents=prompt
)

print(response.text)


# for model in client.models.list():
#     print(model.name)
from transformers import pipeline

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
)

question = input("Question: ")

prompt = f"""<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

result = generator(
    prompt,
    max_new_tokens=50,
    return_full_text=False,
)

print("\nAnswer:")
print(result[0]["generated_text"])
import json
import uuid
from transformers import pipeline
from accelerate import Accelerator

MODEL_PATH = "/opt/extra/avijit/projects/rlof/Ryan_2025/_models/falcon-7b-instruct"

# Initialize the accelerator
accelerator = Accelerator()

# Load text generation model (adjust model name as needed)
generator = pipeline("text-generation", model=MODEL_PATH, trust_remote_code=True, device=accelerator.device)

# Load adversarial prompts
with open("data/adversarial_prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

dataset = []
for prompt in prompts:
    # Generate a response (adjust generation parameters as needed)
    result = generator(prompt, max_length=100, num_return_sequences=1)
    full_text = result[0]["generated_text"]

    # Remove the prompt from the generated response
    response = full_text[len(prompt):].strip()

    dataset.append({
        "id": str(uuid.uuid4()),
        "prompt": prompt,
        "response": response  # Store only the new generated text
    })

# Save the generated dataset for labeling
with open("data/dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset generated with {len(dataset)} examples.")
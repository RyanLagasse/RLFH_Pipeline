# scripts/rlhf_train.py
import json
import torch
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# Configuration
MODEL_NAME = "gpt2"  # Replace with Falcon, Llama, etc., as needed.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3  # Adjust number of epochs as needed.

# Load labeled dataset (each entry should include "prompt" and "label")
with open("data/labeled_dataset.json", "r", encoding="utf-8") as f:
    labeled_data = json.load(f)

# Initialize tokenizer and model (with value head for PPO)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# PPO configuration – adjust hyperparameters as needed.
ppo_config = PPOConfig(
    learning_rate=1e-5,
    mini_batch_size=4,
    ppo_epochs=4,
    log_with=None,  # Set to "tensorboard" if desired
)

ppo_trainer = PPOTrainer(ppo_config, model, tokenizer)

print("Starting RLHF training...")

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for entry in labeled_data:
        prompt = entry["prompt"]
        # Generate a response from the current policy
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        # Note: You can adjust generation parameters here.
        response_ids = model.generate(inputs, max_length=100)
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Define a reward based on the label
        # (Here we give +1 for non-toxic, -1 for toxic responses.)
        reward = 1.0 if entry["label"] == "non-toxic" else -1.0
        
        # Prepare the data in the format expected by PPOTrainer:
        # (The API may require you to provide queries, responses, and rewards.)
        queries = [prompt]
        responses = [response_text]
        rewards = [reward]
        
        # Tokenize the inputs (the PPO step will handle log-probabilities, advantages, etc.)
        query_tensors = tokenizer(queries, return_tensors="pt").input_ids.to(DEVICE)
        # For demonstration, we wrap the generation output as a tensor.
        response_tensors = response_ids
        
        # Perform one PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        print(f"Processed example; PPO stats: {stats}")
    
    print(f"Completed epoch {epoch+1}")

# Save the fine-tuned model (adjust the saving path as needed)
model.save_pretrained("models/rlhf_finetuned_model")
tokenizer.save_pretrained("models/rlhf_finetuned_model")
print("RLHF training complete. Model saved.")

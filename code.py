# Install the Hugging Face Transformers library
!pip install transformers

# Import necessary modules from Hugging Face Transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the GPT-2 model and tokenizer
# 'gpt2' is the basic version. You can also try 'gpt2-medium' or 'gpt2-large' for better results.
model_name = 'gpt2'

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

# Define a function to generate text from a prompt
def generate_text(prompt, max_length=300, temperature=0.7, top_k=50, top_p=0.95):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Use generate with tuned settings
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Prevent truncation warnings
    )

    # Decode and return the full output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Ask the user for a topic
user_input = input("🔹 Enter a topic (e.g., Artificial Intelligence, Climate Change, etc.): ")

# Generate and print the paragraph
output_text = generate_text(user_input)

# Display the output
print("\n📝 Generated Paragraph:\n")
print(output_text)

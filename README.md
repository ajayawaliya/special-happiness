# special-happiness
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
nltk.download('punkt')  # For sentence tokenization

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate reply
def generate_email_reply(input_email, max_length=50):
    # Preprocess: Extract key sentences (simple heuristic)
    sentences = nltk.sent_tokenize(input_email)
    key_points = ' '.join(sentences[:2])  # Use first 2 sentences as context
    
    # Prompt for generation
    prompt = f"Reply to this email: {key_points}\nResponse:"
    
    # Tokenize and generate
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    
    # Decode and clean up
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.replace(prompt, '').strip()
    
    return reply

# Example usage
input_email = "Hi, I wanted to schedule a meeting for next week. What times work for you?"
reply = generate_email_reply(input_email)
print(reply)

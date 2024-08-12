# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
# import torch

# # model = AutoModel.from_pretrained("gpt2_python_d6.bin", local_files_only=True)

# device = torch.device("cuda")

# tokenizer = AutoTokenizer.from_pretrained("gpt2_tokenizer.bin", local_files_only=True)
# model_path = "gpt2_python_d6.bin"

# model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         local_files_only=True,
#         low_cpu_mem_usage=True,
#         device_map="auto",
#         offload_folder="offload/",
#         cache_dir="cache/",
#     )


# def predict(model, tokenizer, query, max_new_tokens, temperature=0.7, top_k=500, top_p=0.3):
#     # ! SEND PREPROCESSED QUERY FOR PREDICTION
#     # ! This will ensure we also save the generated query for ourselves for future preference.
#     input_ids = tokenizer(query, return_tensors="pt").input_ids
#     device = torch.device("cuda")

#     print(f"THE TYPE OF MAX TOKENS - {type(max_new_tokens)}")

#     generation = model.generate(
#         input_ids=input_ids.to(device),
#         max_new_tokens=512,
#         do_sample=True,
#         temperature=temperature,
#         top_k=top_k,
#         top_p=top_p
#         # repetition_penalty=repetition_penalty
#     )
#     result = tokenizer.decode(generation[0])
#     return result

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch

# # Step 1: Load the tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Or use the tokenizer specific to your model

# # Step 2: Load the model
# model = GPT2LMHeadModel.from_pretrained('path_to_your_bin_file', config='path_to_your_config_file.json')
# model.eval()  # Set the model to evaluation mode

# # Step 3: Perform inference (generate text)
# def generate_text(prompt, max_length=50):
#     inputs = tokenizer(prompt, return_tensors='pt')
#     outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Example usage
# if __name__ == "__main__":
#     prompt = input("Enter your prompt: ")
#     generated_text = generate_text(prompt)
#     print("Generated Text:\n", generated_text)

# from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
# import torch

# # Step 1: Define the model configuration manually
# config = GPT2Config(
#     vocab_size=50257,  # Set this to your vocab size
#     n_positions=1024,   # Typically set to the maximum sequence length
#     n_ctx=1024,
#     n_embd=768,         # Set this to the embedding size you used
#     n_layer=12,          # Number of layers in your model
#     n_head=12            # Number of attention heads
# )

# # Step 2: Load the tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Replace with your custom tokenizer if needed
# tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
# # Step 3: Load the model using the manually defined configuration
# model = GPT2LMHeadModel(config)

# # Load the model weights from the .bin file
# # model.load_state_dict(torch.load('gpt2_d12.bin'))
# state_dict = torch.load('gpt2_d12.bin', map_location=torch.device('cuda'))  # Use 'cpu' if you are not using a GPU
# model.load_state_dict(state_dict)

# # Set the model to evaluation mode
# model.eval()

# # Step 4: Perform inference (generate text)
# def generate_text(prompt, max_length=128):
#     inputs = tokenizer(prompt, return_tensors='pt')
#     outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Example usage
# if __name__ == "__main__":
#     prompt = input("Enter your prompt: ")
#     generated_text = generate_text(prompt)
#     print("Generated Text:\n", generated_text)


# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import tiktoken
# import os

# # Define paths
# model_checkpoint_path = "gpt2_d12.bin"  # Update with your path
# tokenizer_path = "gpt2_tokenizer.bin"  # Update with your path

# # Load tokenizer
# # Ensure that you use the same tokenizer as during training
# # If using tiktoken as in your provided script:
# enc = tiktoken.get_encoding("gpt2")

# # Load model
# # Initialize the model architecture you used for training
# # Adjust the model class and configuration accordingly
# from train_gpt2 import GPTConfig, GPT  # Import your model class and config

# # Assuming model was saved in 'float32' format
# def load_model_from_bin(model_path, device):
#     model = GPT.from_pretrained("gpt2")  # Use the correct model architecture
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     return model

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = load_model_from_bin(model_checkpoint_path, device)
# model.to(device)
# model.eval()

# # Inference function
# def infer(model, tokenizer, input_text, max_new_tokens=128, temperature=1.0, top_k=10):
#     # Tokenize input
#     start_ids = [tokenizer.eot_token]
#     input_ids = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

#     # Generate tokens
#     with torch.no_grad():
#         generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

#     # Decode generated tokens
#     output_text = tokenizer.decode(generated_ids[0].tolist())
#     return output_text

# # Example usage
# input_text = "Once upon a time"  # Replace with your input text
# output_text = infer(model, enc, input_text)
# print("Generated Text:")
# print(output_text)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("test_model_slim_1")
model = AutoModelForCausalLM.from_pretrained("test_model_slim_1")
tokens = tokenizer.encode("What the ", return_tensors="pt")
output = model.generate(tokens, max_new_tokens=10, repetition_penalty=1.3)
print(tokenizer.batch_decode(output))
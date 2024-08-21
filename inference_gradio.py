from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr         # ! https://upretihimanshu.medium.com/deploy-your-first-ml-app-using-gradio-1684eec7eb5f

from threading import Thread
from transformers import TextIteratorStreamer


device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("test_model_shaky", device=device)
model = AutoModelForCausalLM.from_pretrained("test_model_shaky")
model.to(device)
streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
print("Model's embedding matrix shape:", model.transformer.wte.weight.shape)
print("Model config vocab_size:", model.config.vocab_size)
print("Model embedding weight shape:", model.transformer.wte.weight.shape)

assert model.config.vocab_size == tokenizer.vocab_size, "Vocabulary size mismatch!"


def generate_text(prompt, max_length, top_p, top_k, temperature, repetition_penalty):
    system_prompt_py = """
    <|endoftext|>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction \n:    """
    system_prompt = "<|endoftext|><|system|> You are a helpful, respectful and honest assistant. Answer query with only HI!<|endoftext|>"
    system_prompt_shakes = "<|endoftext|>"
    full_prompt = system_prompt + prompt + "\n ### Output: \n# Python code\n"
    full_prompt_shakes = system_prompt_shakes + prompt
    # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer([full_prompt_shakes], return_tensors="pt").to(device=device)
    # input_ids = tokenizer.encode(full_prompt_shakes, return_tensors='pt').to(device)
    
    # assert torch.max(inputs) < model.config.vocab_size, "Input contains out-of-bounds token indices!"
    # print("Input tensor shape:", inputs.shape)
    print("Model's embedding matrix shape:", model.transformer.wte.weight.shape)
    print("Model config vocab_size:", model.config.vocab_size)
    print("Model embedding weight shape:", model.transformer.wte.weight.shape)



    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    
    # attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    generate_kwargs = dict(
        inputs,
        # attention_mask=attention_mask,
        max_length=len(inputs) + int(max_length),
        top_p=float(top_p), 
        do_sample=True, 
        top_k=int(top_k), 
        temperature=(temperature),
        streamer=streamer,
        # no_repeat_ngram_size=1,
        repetition_penalty=(repetition_penalty)
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    generated_text=[]

    for text in streamer:
        generated_text.append(text)
        yield "".join(generated_text)


description = """
# Fynder LLM Chat 61M Shit Model
"""
inputs = [
    gr.Textbox(label="Prompt text"),
    gr.Textbox(label="max-lenth generation", value=512),
    gr.Slider(0.0, 1.0, label="top-p value", value=0.95),
    gr.Textbox(label="top-k", value=50),
    gr.Slider(0.0, 1.0, label="temperature", value=0.7),
    gr.Slider(1.0, 10.0, label="repetition penalty", value=1.0),
]
outputs = [gr.Textbox(label="Generated Text")]

my_theme = gr.Theme.from_hub("ParityError/Interstellar")   # NoCrypt/miku

demo = gr.Interface(fn=generate_text, 
                    inputs=inputs, 
                    outputs=outputs, 
                    allow_flagging=False, 
                    description=description,
                    theme=my_theme)

demo.launch(server_name="0.0.0.0", server_port=7861)



# query = "<|endoftext|><|system|> You are a helpful, respectful and honest assistant. Please ensure that your responses are socially unbiased and positive in nature. <|endoftext|> <|prompter|> Hello? <|endoftext|> <|assistant|>"
# # query = "system\n You are a helpful assistant. When a user says Hi, you say Hello.user\nHi!assistant\n"
# input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
# input_len = len(input_ids[0])

# print(input_ids)
# try:
#     print(tokenizer.pad_token_id)
# except:
#     print("pad token not found")

# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# # tokens = tokenizer.encode("<|endoftext|>system\n You are a helpful assistant. When a user says Hi, you say Hello.<|endoftext|>user\nHi!<|endoftext|>assistant\n", return_tensors="pt")


# # ! Attention mask is used to find the padding token. Attention mask differentiates padding tokens from non padding tokens.
# # ! Attention mask
# attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

# # attention_mask = tokens.ne(tokenizer.pad_token_id).long()       # ? .long() converts True/False to 0s and 1s which is the input format for the model
# output = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_new_tokens=64,
#         do_sample=True,
#         temperature=0.9,
#         top_k=5,
#         top_p=0.3,
#         repetition_penalty=1.3,
#     )
# print(output[0][input_len:])
# # output = model.generate(tokens, max_new_tokens=64, repetition_penalty=1.3, attention_mask=attention_mask)
# result = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
# # print(tokenizer.batch_decode(output, skip_special_tokens=True))
# # print(f"Here is the output -- {output}")
# # result = tokenizer.decode(output[0])
# print(result)
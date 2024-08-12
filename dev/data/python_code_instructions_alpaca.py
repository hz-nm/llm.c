"""
Python Code Instruction dataset - An example containing dataset of Simple Python Queries and their responses.
https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca

example doc to highlight the structure of the dataset:
{
  "instruction": "Create a function to calculate the sum of a sequence of integers.",
  "input": "[1, 2, 3, 4, 5]",
  "output": "# Python code def sum_sequence(sequence): sum = 0 for num in sequence: sum += num return sum",
  "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: Create a function to calculate the sum of a sequence of integers. ### Input: [1, 2, 3, 4, 5] ### Output: # Python code def sum_sequence(sequence): sum = 0 for num in sequence: sum += num return sum",
}

Adapted by Yourstruly,
Ameer.
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import argparse

from data_common import write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="Python Code Instruction dataset")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens. Default is 100M")
args = parser.parse_args()

# directories = {
#     ("classic", "10B"): ("fineweb10B", "sample-10BT"),
#     ("classic", "100B"): ("fineweb100B", "sample-100BT"),
#     ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
#     ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
# }
# local_dir, remote_name = directories[(args.type, args.version)]
local_dir = "PythonCodeInstruction-18k"

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
pci_alpaca = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
name = "PythonCodeInstruction-18k"

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["prompt"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    # fw --> The list of strings. i.e. the fine web dataset
    for tokens in pool.imap(tokenize, pci_alpaca, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])

"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinyshakespeare/ folder.
The script prints:

Saved 32768 tokens to tinyshakespeare/tiny_shakespeare_val.bin
Saved 305260 tokens to tinyshakespeare/tiny_shakespeare_train.bin

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of int32 numbers indicating the token ids.
"""

import os
import tiktoken
import numpy as np
from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")

# from mistral_common.tokens.tokenizers.tekken import Tekkenizer
# ! OUR FILE with Changes on par.
from tokenizer.tekken import Tekkenizer

# from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
# from mistral_common.tokens.tokenizers.base import TokenizerVersion

# ! Need the following for implementing the Tekkenizer.
# enc = Tekkenizer(
#     vocab: List[TokenInfo], 
#     pattern: str, FROM OPEN AI for GPT 2---> r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     vocab_size: int, # This is unused so, set it to anything you would like.
#     num_special_tokens: int,  # 12 -> Includes the implemented special tokens.
#     version: TokenizerVersion,    # tokenizer version is v3. You can check it out in the code.
#     *,                            # optional
#     name: str = "tekkenizer",     
#     _path: str | None = None      # optional
#     ) -> Tekkenizer
# enc.encode(
#     s:str,
#     bos: bool,
#     eos: bool
# )

# Read the above parameters from the file directly and then you can pass them into the initializer.
# WHICH FILE?? --> tekken_240718.json
enc = Tekkenizer.from_file("tokenizer/tekken_240718.json")

# ! original
# * enc.encode((text: str, *, allowed_special: AbstractSet[str] | Literal['all'] = set(), disallowed_special: Collection[str] | Literal['all'] = "all") -> list[int])

# ! original
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
# encode = lambda s: enc.encode(s, eos=False, bos=False)

# ! Checking the new Tokenizer!! IT WORKS!!
x = encode("Hello, World!")
print(f"The encoded stuff --> {x}")
print(type(x))
y = enc.decode(x)
print(f"The decoded stuff --> {y}")


def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    # download the TinyShakespeare dataset, unless it's already downloaded
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def tokenize():
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()
    # let's treat every person's statement in the dialog as a separate document
    text = "<|endoftext|>" + text
    text = text.replace('\n\n', '\n\n<|endoftext|>')
    # encode the text
    tokens = encode(text)
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    # val_tokens = tokens[:5]
    # train_tokens = tokens[5:10]
    # save to file
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens)
    write_datafile(train_filename, train_tokens)

if __name__ == "__main__":
    download()
    tokenize()

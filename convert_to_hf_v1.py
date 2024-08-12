from pathlib import Path
import numpy as np
from safetensors import serialize_file
from transformers import GPT2Config, AutoTokenizer

LLMC_HEADER_DTYPE = np.dtype([
    ("magic", "<i4"),       # Little-endian, int32 magic number: 20240326
    ("version", "<i4"),     # Little-endian, int32. fp32 = 3, bf16 = 5.
    ("max_seq_len", "<i4"),
    ("vocab_size", "<i4"),
    ("num_layers", "<i4"),
    ("num_heads", "<i4"),
    ("channels", "<i4"),
    ("padded_vocab_size", "<i4"),
    ("reserved", "248<i4")
])

def make_llmc_parameters_dtype(
        max_seq_len: int = 1024,
        vocab_size: int = 50304,
        padded_vocab_size: int = 0,
        num_layers: int = 12,
        channels: int = 768,
        bf16: bool = True,
    ):

    # we view as uint16 for bf16 since numpy does not support it.
    param_dtype = np.dtype('<u2' if bf16 else '<f4')

    return np.dtype([
        ("wte", param_dtype, (padded_vocab_size, channels)),
        ("wpe", param_dtype, (max_seq_len, channels)),
        ("ln1w", param_dtype, (num_layers, channels)),
        ("ln1b", param_dtype, (num_layers, channels)),
        ("qkvw", param_dtype, (num_layers, 3 * channels, channels)),
        ("qkvb", param_dtype, (num_layers, 3 * channels)),
        ("attprojw", param_dtype, (num_layers, channels, channels)),
        ("attprojb", param_dtype, (num_layers, channels)),
        ("ln2w", param_dtype, (num_layers, channels)),
        ("ln2b", param_dtype, (num_layers, channels)),
        ("fcw", param_dtype, (num_layers, 4 * channels, channels)),
        ("fcb", param_dtype, (num_layers, 4 * channels)),
        ("fcprojw", param_dtype, (num_layers, channels, 4 * channels)),
        ("fcprojb", param_dtype, (num_layers, channels)),
        ("lnfw", param_dtype, channels),
        ("lnfb", param_dtype, channels)
    ])


if __name__ == "__main__":

    llmc_checkpoint = Path("last_ckpt_124M_400B.bin")

    header = np.fromfile(
        llmc_checkpoint,
        dtype=LLMC_HEADER_DTYPE,
        count=1
    )[0]

    print(header)

    llmc_params_dtype = make_llmc_parameters_dtype(
        header["max_seq_len"].item(),
        header["vocab_size"].item(),
        header["padded_vocab_size"].item(),
        header["num_layers"].item(),
        header["channels"].item(),
        header["version"].item() == 5
    )

    # read it
    llmc_params = np.fromfile(
        llmc_checkpoint,
        dtype=llmc_params_dtype,
        offset=1024,
        count=1
    )[0]

    # now let's reconstruct it in HF format
    hf_params = {
        "wte.weight": llmc_params["wte"][:header["vocab_size"]-header["padded_vocab_size"],:],
        "wpe.weight": llmc_params["wpe"],
        "ln_f.weight": llmc_params["lnfw"],
        "ln_f.bias": llmc_params["lnfb"]
    }

    for i in range(header["num_layers"]):
        hf_params[f"h.{i}.ln_1.weight"] = llmc_params["ln1w"][i]
        hf_params[f"h.{i}.ln_1.bias"] = llmc_params["ln1b"][i]
        hf_params[f"h.{i}.attn.c_attn.weight"] = llmc_params["qkvw"][i].transpose()
        hf_params[f"h.{i}.attn.c_attn.bias"] = llmc_params["qkvb"][i]
        hf_params[f"h.{i}.attn.c_proj.weight"] = llmc_params["attprojw"][i].transpose()
        hf_params[f"h.{i}.attn.c_proj.bias"] = llmc_params["attprojb"][i]
        hf_params[f"h.{i}.ln_2.weight"] = llmc_params["ln2w"][i]
        hf_params[f"h.{i}.ln_2.bias"] = llmc_params["ln2b"][i]
        hf_params[f"h.{i}.mlp.c_fc.weight"] = llmc_params["fcw"][i].transpose()
        hf_params[f"h.{i}.mlp.c_fc.bias"] = llmc_params["fcb"][i]
        hf_params[f"h.{i}.mlp.c_proj.weight"] = llmc_params["fcprojw"][i].transpose()
        hf_params[f"h.{i}.mlp.c_proj.bias"] = llmc_params["fcprojb"][i]

    hf_params = {k: {
        "dtype": "bfloat16" if header["version"] == 5 else "float32",
        "shape": v.shape,
        "data": v.tobytes()
    } for k, v in hf_params.items()}

    out_dir = Path("llmc-gpt2-124M-400B")
    out_dir.mkdir(parents=True, exist_ok=True)

    serialize_file(hf_params, out_dir / "model.safetensors", { "format": "pt" })

    config = GPT2Config(
        vocab_size=header["vocab_size"].item(),
        n_positions=header["max_seq_len"].item(),
        n_embd=header["channels"].item(),
        n_layer=header["num_layers"].item(),
        n_head=header["num_heads"].item(),
        activation_function="gelu_new",
        architectures=["GPT2LMHeadModel"],
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True
    ).save_pretrained(out_dir)

    tokenizer = AutoTokenizer.from_pretrained("gpt2").save_pretrained(out_dir)
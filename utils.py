from pathlib import Path
from typing import Tuple, Callable


def get_encoder_decoder(path: Path) -> Tuple[Callable, Callable]:
    with open(path, "r", encoding="utf-8") as f:
        pretrain_text = f.read()

    chars = sorted(list(set(pretrain_text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    return encode, decode

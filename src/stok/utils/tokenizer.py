# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from pathlib import Path

__all__ = [
    "Tokenizer",
    "DEFAULT_VOCAB",
]


class Tokenizer:
    """ESM-2's residue vocabulary without <null_1>: 32 tokens, a `multiple of 8`_.

    One token per residue character; ``vocab_file`` lists one token per line.

    .. _multiple of 8:
        https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/#:~:text=Simply%20padding%20the%20vocabulary%20size,performance%20of%20the%20projection%20layer.
    """

    def __init__(self, vocab_file: str | None = None):
        tokens = (
            [line.strip() for line in Path(vocab_file).read_text().splitlines()]
            if vocab_file is not None
            else DEFAULT_VOCAB
        )
        self.tokens = [token for token in tokens if token]
        self.vocab = {token: i for i, token in enumerate(self.tokens)}
        self.bos_token_id = self.vocab["<cls>"]
        self.pad_token_id = self.vocab["<pad>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.mask_token_id = self.vocab["<mask>"]
        self.all_special_ids = [
            self.bos_token_id,
            self.pad_token_id,
            self.eos_token_id,
            self.unk_token_id,
            self.mask_token_id,
        ]

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self.tokens[int(i)] for i in ids]

    def decode(self, ids) -> str:
        return " ".join(self.convert_ids_to_tokens(ids))


DEFAULT_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "O",
    "Z",
    ".",
    "-",
    "<mask>",
]

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class GroupExample:
    gid: str
    en_doc: str
    zh_doc: str
    en_sum: str


class WikiLinguaGroupDataset(Dataset):
    """
    Loads step-level meaning groups from a JSONL file created in B3/B4.
    Each line is expected to have at least:
      - gid
      - en_doc
      - zh_doc
      - en_sum
    """

    def __init__(self, jsonl_path: str, max_examples: Optional[int] = None) -> None:
        self.jsonl_path = jsonl_path
        self.data: List[GroupExample] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                gid = obj.get("gid")
                en_doc = obj.get("en_doc")
                zh_doc = obj.get("zh_doc")
                en_sum = obj.get("en_sum")

                if not (gid and en_doc is not None and zh_doc is not None and en_sum is not None):
                    raise ValueError(f"Missing required fields in line {i} of {jsonl_path}")

                self.data.append(
                    GroupExample(
                        gid=str(gid),
                        en_doc=str(en_doc),
                        zh_doc=str(zh_doc),
                        en_sum=str(en_sum),
                    )
                )

                if max_examples is not None and len(self.data) >= max_examples:
                    break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> GroupExample:
        return self.data[idx]


def _tokenize_texts(
    tokenizer,
    texts: List[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a list of texts.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _tokenize_labels(
    tokenizer,
    texts: List[str],
    max_length: int,
) -> torch.Tensor:
    """
    Tokenize target texts into labels, with padding tokens masked as -100.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None. Please ensure tokenizer has a pad token.")
    labels = labels.clone()
    labels[labels == pad_id] = -100
    return labels


def make_collate_fn(
    tokenizer,
    max_doc_len: int = 256,
    max_sum_len: int = 64,
    decoder_prefix: Optional[str] = None,
) -> Callable[[List[GroupExample]], Dict[str, Any]]:
    """
    Create a collate_fn bound with tokenizer and truncation lengths.

    decoder_prefix:
      - If provided, we prepend it to the target summary string.
      - For the very first prototype, you can set it to None to keep things simplest.
      - Later when we add multiple renderers, you can use e.g. "<TO_EN_SUM>".
    """

    def collate(batch: List[GroupExample]) -> Dict[str, Any]:
        gids = [ex.gid for ex in batch]
        en_docs = [ex.en_doc for ex in batch]
        zh_docs = [ex.zh_doc for ex in batch]

        if decoder_prefix:
            targets = [f"{decoder_prefix} {ex.en_sum}".strip() for ex in batch]
        else:
            targets = [ex.en_sum for ex in batch]

        en_tok = _tokenize_texts(tokenizer, en_docs, max_length=max_doc_len)
        zh_tok = _tokenize_texts(tokenizer, zh_docs, max_length=max_doc_len)
        labels = _tokenize_labels(tokenizer, targets, max_length=max_sum_len)

        return {
            "gids": gids,
            "en_input_ids": en_tok["input_ids"],
            "en_attention_mask": en_tok["attention_mask"],
            "zh_input_ids": zh_tok["input_ids"],
            "zh_attention_mask": zh_tok["attention_mask"],
            "labels": labels,
        }

    return collate

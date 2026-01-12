from __future__ import annotations

import random
from typing import Dict, Tuple, Any

from datasets import load_dataset


Key = Tuple[str, str]
Val = Dict[str, Any]


def build_english_step_index(ds_en_split) -> tuple[Dict[Key, Val], Dict[str, int]]:
    """
    Build an index:
      key   = (url, section_name)
      value = {
        "document": ...,
        "summary": ...,
        "row_idx": ...,
        "step_idx": ...
      }

    We also keep row_idx and step_idx so we can verify correctness by round-tripping.
    """
    index: Dict[Key, Val] = {}

    stats = {
        "rows_seen": 0,
        "steps_seen": 0,
        "steps_indexed": 0,
        "duplicate_keys": 0,
        "length_mismatch_rows": 0,
        "empty_step_skipped": 0,
    }

    for row_idx, ex in enumerate(ds_en_split):
        stats["rows_seen"] += 1

        url = ex.get("url", "")
        article = ex.get("article", {})

        section_names = article.get("section_name", [])
        documents = article.get("document", [])
        summaries = article.get("summary", [])

        # Defensive: sometimes lengths might mismatch due to data quirks
        n = min(len(section_names), len(documents), len(summaries))
        if not (len(section_names) == len(documents) == len(summaries)):
            stats["length_mismatch_rows"] += 1

        for step_idx in range(n):
            stats["steps_seen"] += 1

            sec = (section_names[step_idx] or "").strip()
            doc = documents[step_idx] or ""
            summ = summaries[step_idx] or ""

            # Skip totally empty steps
            if (not sec) and (not doc) and (not summ):
                stats["empty_step_skipped"] += 1
                continue

            key: Key = (url, sec)
            val: Val = {
                "document": doc,
                "summary": summ,
                "row_idx": row_idx,
                "step_idx": step_idx,
            }

            if key in index:
                # If duplicate, count it, keep the first, but you can also assert identical if you want
                stats["duplicate_keys"] += 1
                # Optional strict check:
                # prev = index[key]
                # if prev["document"] != doc or prev["summary"] != summ:
                #     raise ValueError(f"Duplicate key with different content: {key}")
                continue

            index[key] = val
            stats["steps_indexed"] += 1

    return index, stats


def verify_english_step_index(ds_en_split, index: Dict[Key, Val], n_checks: int = 10) -> None:
    """
    Randomly sample keys, then round-trip back to ds_en_split using row_idx and step_idx.
    If all assertions pass, indexing is correct.
    """
    if not index:
        raise RuntimeError("Index is empty, nothing to verify.")

    keys = list(index.keys())
    n = min(n_checks, len(keys))
    sampled = random.sample(keys, k=n)

    for key in sampled:
        url, sec = key
        val = index[key]
        row_idx = val["row_idx"]
        step_idx = val["step_idx"]

        ex = ds_en_split[row_idx]
        assert ex["url"] == url

        article = ex["article"]
        assert article["section_name"][step_idx].strip() == sec
        assert article["document"][step_idx] == val["document"]
        assert article["summary"][step_idx] == val["summary"]

    print(f"[OK] Verified {n} random keys by round-trip checks.")


if __name__ == "__main__":
    # 1) Load English split (train is enough for building the index)
    ds_en = load_dataset("esdurmus/wiki_lingua", "english", split="train")

    # 2) Build index
    en_index, stats = build_english_step_index(ds_en)

    print("=== English step index stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"unique_keys_in_index: {len(en_index)}")

    # 3) Quick peek at a random entry
    any_key = random.choice(list(en_index.keys()))
    any_val = en_index[any_key]
    print("\n=== Random sample ===")
    print("key:", any_key)
    print("document snippet:", (any_val["document"][:120] + "...") if len(any_val["document"]) > 120 else any_val["document"])
    print("summary:", any_val["summary"])

    # 4) Verify by round-tripping back to dataset
    verify_english_step_index(ds_en, en_index, n_checks=10)

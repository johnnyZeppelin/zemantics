from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

from datasets import load_dataset


Key = Tuple[str, str]  # (url, section_name)


def _safe_strip(x: Any) -> str:
    return (x or "").strip()


def build_english_step_index(ds_en_split) -> Dict[Key, Dict[str, Any]]:
    """
    key   = (url, section_name)
    value = {"document": ..., "summary": ...}
    """
    index: Dict[Key, Dict[str, Any]] = {}
    for ex in ds_en_split:
        url = ex.get("url", "")
        article = ex.get("article", {})
        section_names = article.get("section_name", [])
        documents = article.get("document", [])
        summaries = article.get("summary", [])

        n = min(len(section_names), len(documents), len(summaries))
        for i in range(n):
            sec = _safe_strip(section_names[i])
            doc = documents[i] or ""
            summ = summaries[i] or ""
            key = (url, sec)
            # In B2 you saw no duplicates, so we can store directly.
            if key not in index:
                index[key] = {"document": doc, "summary": summ}
    return index


@dataclass
class AlignFieldSpec:
    """
    Some datasets store alignment fields at top-level, some inside article, some as list per step.
    We'll implement a small resolver to be robust.
    """
    en_url_field: str = "english_url"
    en_sec_field: str = "english_section_name"


def resolve_step_alignment(ex_zh: Dict[str, Any], step_idx: int, spec: AlignFieldSpec) -> Optional[Tuple[str, str]]:
    """
    Try to resolve (english_url, english_section_name) for a given Chinese step.
    Return None if cannot resolve.
    """
    # Case 1: alignment fields at top-level as list aligned with steps
    if spec.en_url_field in ex_zh and spec.en_sec_field in ex_zh:
        en_urls = ex_zh.get(spec.en_url_field)
        en_secs = ex_zh.get(spec.en_sec_field)
        if isinstance(en_urls, list) and isinstance(en_secs, list):
            if step_idx < len(en_urls) and step_idx < len(en_secs):
                return _safe_strip(en_urls[step_idx]), _safe_strip(en_secs[step_idx])
        # Sometimes they are strings (article-level). Not enough for step mapping.
        if isinstance(en_urls, str) and isinstance(en_secs, str):
            return _safe_strip(en_urls), _safe_strip(en_secs)

    # Case 2: alignment fields inside article dict as list
    article = ex_zh.get("article", {})
    if spec.en_url_field in article and spec.en_sec_field in article:
        en_urls = article.get(spec.en_url_field)
        en_secs = article.get(spec.en_sec_field)
        if isinstance(en_urls, list) and isinstance(en_secs, list):
            if step_idx < len(en_urls) and step_idx < len(en_secs):
                return _safe_strip(en_urls[step_idx]), _safe_strip(en_secs[step_idx])
        if isinstance(en_urls, str) and isinstance(en_secs, str):
            return _safe_strip(en_urls), _safe_strip(en_secs)

    # Case 3: unknown layout
    return None


def iter_zh_steps(ex_zh: Dict[str, Any]) -> List[Tuple[int, str, str, str]]:
    """
    Return a list of (step_idx, zh_section_name, zh_document, zh_summary).
    """
    article = ex_zh.get("article", {})
    sec_names = article.get("section_name", [])
    docs = article.get("document", [])
    sums = article.get("summary", [])
    n = min(len(sec_names), len(docs), len(sums))
    steps = []
    for i in range(n):
        steps.append((i, _safe_strip(sec_names[i]), docs[i] or "", sums[i] or ""))
    return steps


def build_groups_from_chinese(
    ds_zh_split,
    en_index: Dict[Key, Dict[str, Any]],
    out_path: str,
    align_spec: AlignFieldSpec,
    max_groups: Optional[int] = None,
) -> Dict[str, int]:
    """
    Create meaning groups by aligning Chinese steps to English steps via english_url + english_section_name.
    Write groups as JSONL to out_path.
    Return stats.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stats = {
        "rows_seen": 0,
        "zh_steps_seen": 0,
        "align_fields_missing": 0,
        "lookup_miss": 0,
        "groups_written": 0,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        for row_idx, ex_zh in enumerate(ds_zh_split):
            stats["rows_seen"] += 1
            zh_url = ex_zh.get("url", "")
            steps = iter_zh_steps(ex_zh)

            for (step_idx, zh_sec, zh_doc, zh_sum) in steps:
                stats["zh_steps_seen"] += 1

                resolved = resolve_step_alignment(ex_zh, step_idx, align_spec)
                if resolved is None:
                    stats["align_fields_missing"] += 1
                    continue
                en_url, en_sec = resolved
                if not en_url or not en_sec:
                    stats["align_fields_missing"] += 1
                    continue

                key = (en_url, en_sec)
                if key not in en_index:
                    stats["lookup_miss"] += 1
                    continue

                en_doc = en_index[key]["document"]
                en_sum = en_index[key]["summary"]

                gid = f"{zh_url}::step{step_idx}"

                group = {
                    "gid": gid,
                    "en_url": en_url,
                    "en_section_name": en_sec,
                    "en_doc": en_doc,
                    "en_sum": en_sum,
                    "zh_url": zh_url,
                    "zh_section_name": zh_sec,
                    "zh_doc": zh_doc,
                    "zh_sum": zh_sum,
                }

                f.write(json.dumps(group, ensure_ascii=False) + "\n")
                stats["groups_written"] += 1

                if max_groups is not None and stats["groups_written"] >= max_groups:
                    return stats

    return stats


def quick_verify_groups(groups_path: str, en_index: Dict[Key, Dict[str, Any]], n_checks: int = 10) -> None:
    """
    Sample groups and verify that en_doc/en_sum exactly match what en_index returns for (en_url, en_section_name).
    """
    # read all offsets quickly (small n_checks so simple read is fine)
    lines = []
    with open(groups_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line)
    if not lines:
        raise RuntimeError(f"No groups found in {groups_path}")

    sampled = random.sample(lines, k=min(n_checks, len(lines)))
    for line in sampled:
        g = json.loads(line)
        key = (g["en_url"], g["en_section_name"])
        assert key in en_index, f"Key not in en_index: {key}"
        assert g["en_doc"] == en_index[key]["document"]
        assert g["en_sum"] == en_index[key]["summary"]

    print(f"[OK] Verified {min(n_checks, len(lines))} groups against en_index.")


if __name__ == "__main__":
    random.seed(42)

    # ===== Load datasets =====
    ds_en = load_dataset("esdurmus/wiki_lingua", "english", split="train")
    ds_zh = load_dataset("esdurmus/wiki_lingua", "chinese", split="train")

    # ===== Build English index =====
    print("Building English step index...")
    en_index = build_english_step_index(ds_en)
    print(f"English index size: {len(en_index)}")

    # ===== Build Chinese-aligned groups =====
    out_path = "data/groups_train.jsonl"
    align_spec = AlignFieldSpec(en_url_field="english_url", en_sec_field="english_section_name")

    print("Building groups from Chinese (train split)...")
    stats = build_groups_from_chinese(
        ds_zh_split=ds_zh,
        en_index=en_index,
        out_path=out_path,
        align_spec=align_spec,
        max_groups=None,  # set to e.g. 2000 for a quick dry run
    )

    print("=== Group build stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # ===== Quick verification =====
    quick_verify_groups(out_path, en_index, n_checks=10)

    # ===== Show a random group =====
    with open(out_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    g = json.loads(random.choice(all_lines))
    print("\n=== Random group sample ===")
    print("gid:", g["gid"])
    print("en key:", (g["en_url"], g["en_section_name"]))
    print("zh key:", (g["zh_url"], g["zh_section_name"]))
    print("zh_doc snippet:", (g["zh_doc"][:120] + "...") if len(g["zh_doc"]) > 120 else g["zh_doc"])
    print("en_sum:", g["en_sum"])


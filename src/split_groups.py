# split_groups

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Tuple


def hash_to_unit_interval(s: str, salt: str = "") -> float:
    """
    Deterministic hash -> [0, 1).
    """
    h = hashlib.sha1((salt + s).encode("utf-8")).hexdigest()
    # take first 8 hex digits -> 32-bit int
    v = int(h[:8], 16)
    return v / float(2**32)


def split_jsonl_by_gid(
    in_path: str,
    out_train: str,
    out_valid: str,
    valid_ratio: float = 0.05,
    salt: str = "wikilingua_v1",
) -> Tuple[int, int, int]:
    """
    Read JSONL with field 'gid', split deterministically by hashing gid.
    Return (n_total, n_train, n_valid).
    """
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    os.makedirs(os.path.dirname(out_valid), exist_ok=True)

    n_total = 0
    n_train = 0
    n_valid = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_train, "w", encoding="utf-8") as ftr, \
         open(out_valid, "w", encoding="utf-8") as fva:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = obj.get("gid")
            if not gid:
                raise ValueError("Missing 'gid' field in an input line.")

            r = hash_to_unit_interval(gid, salt=salt)
            n_total += 1

            if r < valid_ratio:
                fva.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_valid += 1
            else:
                ftr.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_train += 1

    return n_total, n_train, n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL, e.g. data/groups_all.jsonl")
    ap.add_argument("--out_train", default="data/groups_train.jsonl")
    ap.add_argument("--out_valid", default="data/groups_valid.jsonl")
    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--salt", type=str, default="wikilingua_v1")
    args = ap.parse_args()

    n_total, n_train, n_valid = split_jsonl_by_gid(
        in_path=args.in_path,
        out_train=args.out_train,
        out_valid=args.out_valid,
        valid_ratio=args.valid_ratio,
        salt=args.salt,
    )

    print("=== Split stats ===")
    print(f"input_total: {n_total}")
    print(f"train: {n_train}")
    print(f"valid: {n_valid}")
    if n_total > 0:
        print(f"valid_ratio_actual: {n_valid / n_total:.4f}")


if __name__ == "__main__":
    main()

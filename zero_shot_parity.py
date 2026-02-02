#!/usr/bin/env python3
import datasets
import ipdb
from transformers import AutoTokenizer
import itertools
import json
from closed_models import GeminiTokenCounter, GPTTokenCounter


def parity_score(l1: list[int], l2: list[int]) -> float:
    """Compute parity score between two lists of lengths."""
    scores = [i / j if j > 0 else 0 for i, j in zip(l1, l2)]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="Model ID")
    parser.add_argument("--input", "-i", type=str, nargs='+', required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()

    if "gemini" in args.model.lower():
        tokenizer = GeminiTokenCounter(model_name=args.model.split("/")[-1])
    elif "gpt" in args.model.lower():
        tokenizer = GPTTokenCounter(model_name=args.model.split("/")[-1])  
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = datasets.load_dataset("json", data_files=args.input)["train"]
    
    pms_lengths = tokenizer(list(ds['flores_pms']), add_special_tokens=False, return_length=True)[
        "length"
    ]
    ita_lengths = tokenizer(list(ds['flores_ita']), add_special_tokens=False, return_length=True)[
        "length"
    ]
    fra_lengths = tokenizer(list(ds['flores_fra']), add_special_tokens=False, return_length=True)[
        "length"
    ]
    eng_lengths = tokenizer(list(ds['flores_eng']), add_special_tokens=False, return_length=True)[
        "length"
    ]     
    scores = {}
    scores["avg_pms_length"] = sum(pms_lengths) / len(pms_lengths)
    scores["avg_ita_length"] = sum(ita_lengths) / len(ita_lengths)
    scores["avg_fra_length"] = sum(fra_lengths) / len(fra_lengths)
    scores["avg_eng_length"] = sum(eng_lengths) / len(eng_lengths)

    for (lang1, len1), (lang2, len2) in itertools.permutations(
        [
            ("pms", pms_lengths),
            ("ita", ita_lengths),
            ("fra", fra_lengths),
            ("eng", eng_lengths),
        ],
        2,
    ):
        scores[f"parity_{lang1}_vs_{lang2}"] = parity_score(len1, len2)

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)

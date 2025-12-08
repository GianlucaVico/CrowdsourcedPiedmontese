import sentencepiece as spm
import datasets
import itertools
import json

def parity_score(l1: list[int], l2: list[int]) -> float:
    """Compute parity score between two lists of lengths."""
    scores = [i / j if j > 0 else 0 for i, j in zip(l1, l2)]
    return sum(scores) / len(scores)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file=args.model)
    ds = datasets.load_dataset("json", data_files=args.input)["train"]

    pms_lengths = tokenizer.encode_as_pieces(list(ds['flores_pms']))
    ita_lengths = tokenizer.encode_as_pieces(list(ds['flores_ita']))
    fra_lengths = tokenizer.encode_as_pieces(list(ds['flores_fra']))
    eng_lengths = tokenizer.encode_as_pieces(list(ds['flores_eng']))
    pms_lengths = [len(i) for i in pms_lengths]
    ita_lengths = [len(i) for i in ita_lengths]
    fra_lengths = [len(i) for i in fra_lengths]
    eng_lengths = [len(i) for i in eng_lengths]

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

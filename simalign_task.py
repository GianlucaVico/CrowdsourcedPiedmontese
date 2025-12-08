#!/usr/bin/env python3
from simalign import SentenceAligner
import argparse

# From simalign
METHODS = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help="Source language file")
    parser.add_argument('target', type=str, help="Target language file")
    parser.add_argument('output', type=str, help="Output alignment file")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (e.g., 'cpu' or 'cuda')")
    parser.add_argument('--model', type=str, default='xlmr', help="Model to use for alignment") 
    parser.add_argument('--method', type=str, default='i')

    args = parser.parse_args()
    if args.method not in METHODS.keys():
        raise ValueError(f"Unknown method {args.method}, choose from {list(METHODS.keys())}")
    
    aligner = SentenceAligner(model=args.model, device=args.device, matching_methods=args.method)
    with open(args.source) as src, open(args.target) as tgt, open(args.output, 'w') as out:
        for src_line, tgt_line in zip(src, tgt):
            alignments = aligner.get_word_aligns(src_line, tgt_line)[METHODS[args.method]]
            alignments_str = ' '.join([f"{src_idx}-{tgt_idx}" for src_idx, tgt_idx in alignments])
            out.write(alignments_str + '\n')
    

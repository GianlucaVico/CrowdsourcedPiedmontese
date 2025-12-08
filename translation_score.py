#!/usr/bin/env python3
import evaluate
import json
import ipdb
import pandas as pd
import sacrebleu
import numpy as np
from scipy.stats import bootstrap


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="Input file prefix")
    parser.add_argument('output', type=str, help="Output score file")
    args = parser.parse_args()    

    df = pd.read_json(args.input, lines=True)
    max_refs = max(len(refs) if isinstance(refs, list) else 1 for refs in df["reference"]) 
    flat_preds = []
    flat_refs = []
    flat_sources = []
    all_refs = [[] for _ in range(max_refs)]  
    all_preds = []  
    
    for item in df.iloc:
        pred = item['predicted']
        refs = item['reference']
        source = item['sentence']
        n_refs = len(refs) if isinstance(refs, list) else 1
        n_preds = len(pred) if isinstance(pred, list) else 1
        if n_preds == 1 and isinstance(pred, str):
            pred = [pred]

        for p in pred:
            all_preds.append(p)
            for i in range(max_refs):
                if i < n_refs:
                    tmp = refs[i] if isinstance(refs, list) else refs
                    if isinstance(tmp, list):
                        tmp = tmp[0]
                    all_refs[i].append(tmp)
                else:
                    all_refs[i].append(None)
            if n_refs == 1:
                flat_preds.append(refs[0] if isinstance(refs, list) else refs)
                flat_refs.append(refs[0] if isinstance(refs, list) else refs)
                flat_sources.append(source)
            else:
                for i in range(n_refs):
                    flat_preds.append(p)
                    flat_refs.append(refs[i])
                    flat_sources.append(source)
    scores = {}
    bleu = sacrebleu.BLEU(lowercase=True)
    chrf = sacrebleu.CHRF(lowercase=True, word_order=2)
    ter = sacrebleu.TER(case_sensitive=False, no_punct=True, normalized=True)
    
    bleu_scores = bleu.corpus_score(hypotheses=all_preds, references=all_refs, n_bootstrap=1000)
    chrf_scores = chrf.corpus_score(hypotheses=all_preds, references=all_refs, n_bootstrap=1000)
    ter_scores = ter.corpus_score(hypotheses=all_preds, references=all_refs, n_bootstrap=1000)
    bleu_sign = bleu.get_signature().format()
    chrf_sign = chrf.get_signature().format()
    ter_sign = ter.get_signature().format()

    scores['bleu'] = json.loads(bleu_scores.format(signature=bleu_sign, is_json=True))
    scores['chrf++'] = json.loads(chrf_scores.format(signature=chrf_sign, is_json=True))
    scores['ter'] = json.loads(ter_scores.format(signature=ter_sign, is_json=True))
    comet = evaluate.load('comet')
    comet_scores = comet.compute(predictions=flat_preds, references=flat_refs, sources=flat_sources)
    scores['comet'] = {"score": comet_scores['mean_score']}
    bs = bootstrap(data=(comet_scores['scores'],), statistic=np.mean, vectorized=True, rng=42, n_resamples=1000)
    std = bs.standard_error
    ci_low, ci_high = bs.confidence_interval
    scores['comet'].update({"std": std, "ci_low": ci_low, "ci_high": ci_high})

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
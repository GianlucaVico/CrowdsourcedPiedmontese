#!/usr/bin/env python3
import evaluate
import json
import pandas as pd
import numpy as np
from scipy.stats import bootstrap

CATEGORIES = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
CAT_TO_ID = {cat: idx for idx, cat in enumerate(CATEGORIES)}

def bootstrap_confidence_interval(metric, key, predictions, references, num_resamples=1000, confidence_level=0.95, seed=42, **kargs):
    def call_metric(*args):
        ref = args[0]
        pred = args[1]

        result = metric.compute(predictions=pred, references=ref, **kargs)
        return result[key]

    bs = bootstrap(
        data=(references, predictions),
        statistic=call_metric,
        vectorized=False,
        n_resamples=num_resamples,
        confidence_level=confidence_level,
        paired=True,
        rng=seed,
    )
    std = bs.standard_error
    ci_low, ci_high = bs.confidence_interval
    return {"std": std, "ci_low": ci_low, "ci_high": ci_high}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, nargs='+', required=True, help="Input file prefix")
    parser.add_argument('--output', '-o', type=str, required=True, help="Output score file")
    args = parser.parse_args()

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1") # labels=CATEGORIES, average='micro'
    precision = evaluate.load("precision") # labels=CATEGORIES, average='micro'
    recall = evaluate.load("recall") # labels=CATEGORIES, average='micro'

    df = pd.concat([pd.read_json(f, lines=True) for f in args.input])
    answers = list(df['predicted_category'])
    labels = list(df['true_category'])

    scores = {}
    answers_idx = [CAT_TO_ID.get(cat, -1) for cat in answers]
    labels_idx = [CAT_TO_ID.get(cat, -1) for cat in labels]

    scores['accuracy'] = {"score": accuracy.compute(predictions=answers_idx, references=labels_idx)['accuracy']}
    scores['f1'] = {"score": f1.compute(predictions=answers_idx, references=labels_idx, average='micro', labels=list(CAT_TO_ID.values()))['f1']}
    scores['precision'] = {"score": precision.compute(predictions=answers_idx, references=labels_idx, average='micro', labels=list(CAT_TO_ID.values()))['precision']}
    scores['recall'] = {"score": recall.compute(predictions=answers_idx, references=labels_idx, average='micro', labels=list(CAT_TO_ID.values()))['recall']}

    kargs = {"average": 'micro', "labels": list(CAT_TO_ID.values())}
    for metric_name, metric in zip(["f1", "precision", "recall"], [f1, precision, recall]):
        scores[metric_name].update(bootstrap_confidence_interval(metric, metric_name, answers_idx.copy(), labels_idx.copy(), **kargs))
    scores['accuracy'].update(bootstrap_confidence_interval(accuracy, key='accuracy', predictions=answers_idx.copy(), references=labels_idx.copy()))

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)

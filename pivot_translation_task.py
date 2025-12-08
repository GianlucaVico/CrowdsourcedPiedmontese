#!/usr/bin/env python3
import transformers
import datasets
import torch
import pandas as pd
import itertools
import tqdm
from closed_models import ClosedModel, CLOSED_MODELS

LANG_NAMES = {
    "ita": "Italian",
    "pms": "Piedmontese",
    "fra": "French",
    "eng": "English",
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help="Model ID")
    parser.add_argument('--output_file', '-o', type=str, required=True, help="Output file prefix")
    parser.add_argument('--pivot_lang', '-p', type=str, default="ita", help="Pivot language code (e.g., 'eng')") 
    args = parser.parse_args()


    ds = datasets.load_dataset("json", data_files={"dev": "data/pms_dev.jsonl", "devtest": "data/pms_devtest.jsonl"})

    model_id = args.model
    output_file = args.output_file
    pivot_lang = args.pivot_lang

    if model_id in CLOSED_MODELS:
        model = ClosedModel(model_id)
    else:
        model = transformers.pipeline("text-generation", model=model_id, device_map="auto", dtype=torch.bfloat16)
    
    def fmt(item_ds, system_prompt, user_prompt):        
        for item in item_ds:       
            if system_prompt is None:
                messages = [
                    {"role": "user", "content": user_prompt.format(item)}
                ]                
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(item)}
                ]
            yield messages

    langs = list(LANG_NAMES.keys())
    langs.remove(pivot_lang)
    for from_, to_ in tqdm.tqdm(itertools.permutations(LANG_NAMES.keys(), 2), total=len(LANG_NAMES)*(len(LANG_NAMES)-1)):
        from_lang = LANG_NAMES[from_]
        to_lang = LANG_NAMES[to_]

        system_prompt_to_pivot = (
            f"You are a helpful assistant that translates the following sentence from {from_lang} to {pivot_lang}. "
            "Do not add any explanations."
        )
        user_prompt_to_pivot = f"Translate the following {from_lang} source text to {pivot_lang}:\n{from_lang}: {{}}\n{pivot_lang}: "

        system_prompt_from_pivot = (
            f"You are a helpful assistant that translates the following sentence from {pivot_lang} to {to_lang}. "
            "Do not add any explanations."
        )
        user_prompt_from_pivot = f"Translate the following {pivot_lang} source text to {to_lang}:\n{pivot_lang}: {{}}\n{to_lang}: "

        if "tower" in model_id.lower():
            system_prompt_to_pivot = None
            system_prompt_from_pivot = None
        
        for split in ["dev", "devtest"]:
            sentences = list(ds[split][f'flores_{from_}'])
            answers = []
            pivots = []
            references = list(ds[split][f'flores_{to_}'])
            references = [[i] for i in references]
            splits = [split] * len(sentences)
            ids = ds[split]['flores_id']
            for answer in model(fmt(ds[split][f'flores_{from_}'], system_prompt_to_pivot, user_prompt_to_pivot), do_sample=False, max_new_tokens=256):
                tmp = answer[-1]['generated_text'][-1]['content'].strip()
                pivots.append(tmp)
            for answer in model(fmt(pivots, system_prompt_from_pivot, user_prompt_from_pivot), do_sample=False, max_new_tokens=256):
                tmp = answer[-1]['generated_text'][-1]['content'].strip()
                answers.append(tmp)
            df = pd.DataFrame({'split': splits, 'id': ids, 'sentence': sentences, 'reference': references, 'predicted': answers, 'pivot': pivots})

            if to_ == "pms":
                df = df.groupby('id').aggregate({i: lambda x: x if i != 'reference' else list for i in df.columns})
            df.to_json(f"{output_file}.{split}.{from_}_{to_}.pivot_{pivot_lang}.jsonl", orient='records', lines=True)

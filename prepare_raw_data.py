#!/usr/bin/env python3
import json
import os
import glob
import datasets
import collections
import pandas as pd
import tqdm
import ipdb

FOLDER = "data/raw/submissions_20251118_000001/pms"
OUT_DEV = "data/pms_dev.jsonl"
OUT_DEVTEST = "data/pms_devtest.jsonl"

EVAL_DICT = {  
    "Interamente corretta o quasi": 4,
    "Probabilmente corretta, l'avrei scritta in altro modo": 3,
    "Parzialmente corretta": 2,
    "Totalmente sbagliata o quasi": 1,
    "Risposta mancante, offensiva o non pertinente": 0,
    "Non lo so": -1,    
}

FIX = "data/fix.jsonl"  
REMOVE = "data/remove.jsonl"
UNCATEGORIZED = "uncategorized"

# columns: split: str, flores id: int, flores ita: str, flores pms: str, review scores: list of int, category: str,  flores_fra: str, flores_eng: str

def clean_text(text):
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    return text.strip()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str, default=FOLDER)
    parser.add_argument('--output-dev', type=str, default=OUT_DEV)
    parser.add_argument('--output-devtest', type=str, default=OUT_DEVTEST)
    args = parser.parse_args()

    if args.input_folder is not None:
        FOLDER = args.input_folder
    if args.output_dev is not None:
        OUT_DEV = args.output_dev
    if args.output_devtest is not None:
        OUT_DEVTEST = args.output_devtest

    sib_ds = datasets.load_dataset('Davlan/sib200', 'ita_Latn') 
    sib_train = sib_ds['train'].to_pandas()
    sib_val = sib_ds['validation'].to_pandas()
    sib_test = sib_ds['test'].to_pandas()
    sib_ds = pd.concat([sib_train, sib_val, sib_test], ignore_index=True)    

    fra = datasets.load_dataset("openlanguagedata/flores_plus", "fra_Latn")
    eng = datasets.load_dataset("openlanguagedata/flores_plus", "eng_Latn")

    ds = collections.defaultdict(dict)
    for filepath in glob.glob(os.path.join(FOLDER, f"*-*.json")):        
        with open(filepath, "r") as f:
            sample = json.load(f)
        
        file = os.path.basename(filepath)
        split = sample['sample_data']['sample_split']
        ds[file]['split'] = split
        
        flores_id = sample['sample_data']['sample_flores_id']
        ds[file]['flores_id'] = flores_id
        if sample['data']['valid'] and not sample['data']['empty'] and sample['sample_data']['valid']:
            ds[file]['flores_ita'] = sample['sample_data']['sample_ita'].strip()
            ds[file]['flores_pms'] = clean_text(sample['data']['translation'])

        if sample['data']['valid'] and sample['review_data']['valid']:
            evaluation = sample['data']['translation-evaluation']
            evaluation = EVAL_DICT[evaluation]
            evaluation_file = os.path.basename(sample['review_data']['selected_path'])
            if ds[evaluation_file].get('review_scores') is None:
                ds[evaluation_file]['review_scores'] = []
            ds[evaluation_file]['review_scores'].append(evaluation)
        if flores_id is not None:
            sib_row = sib_ds[sib_ds['text'] == sample['sample_data']['sample_ita']]
            if len(sib_row) > 1:
                print(f"Warning: multiple SIB rows for flores_id {flores_id}")
            if not sib_row.empty:
                ds[file]['category'] = sib_row.iloc[0]['category']
            else:
                ds[file]['category'] = UNCATEGORIZED
            ds[file]['flores_fra'] = fra[split][flores_id]['text']
            ds[file]['flores_eng'] = eng[split][flores_id]['text']
    
    df = pd.DataFrame.from_records(list(ds.values()))
    df = df.dropna(subset=['flores_pms', 'split'], ignore_index=True)
    bad_samples = df[df['review_scores'].apply(lambda x: (isinstance(x, list)) and (0 in x))]
    print("Samples with review score 0:", len(bad_samples))    
    if os.path.exists(FIX):
        fixes = pd.read_json(FIX, lines=True)
        for _, row in fixes.iterrows():
            mask = (df['split'] == row['split']) & \
                (df['flores_id'] == row['flores_id']) & \
                (clean_text(df['flores_pms']) == clean_text(row['old_flores_pms']))
            if len(df[mask]) == 0:
                print("Warning: no matching sample found for fix:", row)
                continue
            df.loc[mask, 'flores_pms'] = clean_text(row['flores_pms'])
            df.loc[mask, 'review_scores'].apply(lambda x: x if x != 0 else 1)
        # Recheck bad samples
        bad_samples = df[df['review_scores'].apply(lambda x: (isinstance(x, list)) and (0 in x))]
        print("Samples with review score 0 after applying fixes (to be discated):", len(bad_samples))
        # Discard bad samples
        df = df[~df.index.isin(bad_samples.index)]
    
    if os.path.exists(REMOVE):
        removes = pd.read_json(REMOVE, lines=True)
        for _, row in removes.iterrows():
            mask = (df['split'] == row['split']) & \
                (df['flores_id'] == row['flores_id']) & \
                (clean_text(df['flores_pms']) == clean_text(row['flores_pms']))
            df = df[~mask]

    dev = df[df['split'] == 'dev']
    devtest = df[df['split'] == 'devtest']
    dev.to_json(OUT_DEV, orient='records', lines=True)
    devtest.to_json(OUT_DEVTEST, orient='records', lines=True)

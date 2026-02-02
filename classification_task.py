#!/usr/bin/env python3
import transformers
import datasets
import torch
import pandas as pd
from closed_models import ClosedModel, CLOSED_MODELS

CATEGORIES = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
KEYWORDS_MAP = {
    "science": "science/technology",
    "technology": "science/technology",
    "travel": "travel",
    "politic": "politics", # dont check final s
    "sport": "sports", # dont check final s
    "health": "health",
    "entertainment": "entertainment",
    "geography": "geography"
}

def extract_category(prediction: str, category_map: dict[str, str]) -> str:
    for keyword, category in category_map.items():
        if keyword in prediction.lower():
            return category
    return "unknown"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help="Model ID")
    parser.add_argument('--lang', '-l', type=str, choices=['ita', 'pms', 'fra', 'eng'], required=True, help="Language to classify")
    parser.add_argument('--output_file', '-o', type=str, required=True, help="Output file prefix")
    args = parser.parse_args()


    ds = datasets.load_dataset(
        "json", 
        data_files={"dev": "data/pms_dev.jsonl", "devtest": "data/pms_devtest.jsonl"},
    )
    ds = ds.filter(lambda x: x['category'] != 'uncategorized')
    
    model_id = args.model
    lang = args.lang
    output_file = args.output_file

    if model_id in CLOSED_MODELS:
        model = ClosedModel(model_id)
    else:
        model = transformers.pipeline("text-generation", model=model_id, device_map="auto", dtype=torch.bfloat16)
    
    system_prompt = (
        "You are a helpful assistant that classifies the following sentence into one of the following categories:"
        "science/technology, travel, politics, sports, health, entertainment, geography." 
        "Do not add any explanations."
    )
    # user_prompt = "Classify the following sentence: '{}'\nCategory: "
    user_prompt = "Is this a piece of news regarding {{\"science, technology, travel, politics, sports, health, entertainment, or geography\"}}? {}."

    def fmt(item_ds):
        # if "qwen" in model_id.lower():
        #     for item in item_ds:            
        #         messages = [
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": user_prompt.format(item.lower())}
        #         ]
        #         yield messages
        # else:
        #     for item in item_ds:            
        #         messages = [
        #             {"role": "user", "content": user_prompt.format(item.lower())}
        #         ]
        #         yield messages        
        for item in item_ds:            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(item.lower())}
            ]
            yield messages

    
    for split in ["dev", "devtest"]:
        sentences = list(ds[split][f'flores_{lang}'])
        answers = []
        labels = list(ds[split]['category'])
        full_outputs = []
        for answer in model(fmt(ds[split][f'flores_{lang}']), do_sample=False, max_new_tokens=100):
            generated = answer[-1]['generated_text'][-1]['content'].strip()
            full_outputs.append(generated)
            answer = extract_category(generated, KEYWORDS_MAP)
            answers.append(answer)
            
        df = pd.DataFrame({'sentence': sentences, 'predicted_category': answers, 'true_category': labels, 'answer': full_outputs})
        df.to_json(f"{output_file}.{lang}.{split}.jsonl", orient='records', lines=True)
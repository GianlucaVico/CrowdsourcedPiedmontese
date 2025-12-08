#!/usr/bin/env python3
"""
Remove private information from raw data before sharing.
"""
import argparse
import json
import glob
import os

LANG_MAP = {
    "Piemontese": "Piedmontese",
    "Italiano": "Italian",
    "Inglese": "English",
    "Altro": "Other",
    "Islandese,Inglese,Italiano": "Icelandic,English,Italian",
}

PROFICIENCY_MAP = {
    "Perfettamente o quasi, riesco a esprimere praticamente tutto": "Perfectly or almost",
    "Abbastanza, ma a volte lo mischio con l'italiano (o la lingua che uso principalmente)": "Fairly",
    "Poco, conosco alcune espressioni, ma faccio fatica a esprimere rasi nuove": "Little",    
    "Niente o quasi, solo qualche parola": "Nothing or almost",
}

AGREEMENT_MAP = {
    "D'accordo": "Agree",
    "In disaccordo": "Disagree",
    "Neutrale": "Neutral",
}

LANGUAGE_SOURCE_MAP = {
    'Genitori': 'Parents',
    'Nonni': 'Grandparents',
    'Parenti': 'Relatives',
    'Amici o Colleghi': 'Friends or Colleagues',
    'Altro': 'Other',
    'Padrone di casa, marito, suoceri…': 'Landlord, husband, parents-in-law…',
    'Lavoro': 'Work',
    'Ciro': 'Ciro',
    'Persone anziane con cui ho lavorato': 'Elderly people I worked with',
    'Persone de paese dove vivo': 'People from the village where I live',
    'Da solo, koiné ed astesano ': 'Alone, koiné and local speech',
    'Compagnia teatrale dialettale': 'Dialect theatre company',
    'Clienti ': 'Clients',
    'Da autodidatta ': 'Self-taught',
    'Sentendolo parlare in paese': 'Hearing it spoken in town',
    'Lavorando nel sociale con gli anziani': 'Working in social care with the elderly',
}   

AGE_MAP = {
    "Tra 20 e 30": "20-30",
    "Tra 30 e 40": "30-40",
    "Tra 40 e 50": "40-50",
    "Tra 50 e 60": "50-60",
    "Meno di 20": "<20",
    "Più di 60": ">60",
}

EVALUATION_MAP = {
    "Interamente corretta o quasi": "Entirely correct or almost",
    "Probabilmente corretta, l'avrei scritta in altro modo": "Probably correct",    
    "Parzialmente corretta": "Partially correct",
    "Totalmente sbagliata o quasi": "Totally wrong or almost",
    "Risposta mancante, offensiva o non pertinente": "Missing, offensive or not relevant",
    "Non lo so": "I don't know",
}

EVALUATION_TO_SCORE = {  
    "Interamente corretta o quasi": 4,
    "Probabilmente corretta, l'avrei scritta in altro modo": 3,
    "Parzialmente corretta": 2,
    "Totalmente sbagliata o quasi": 1,
    "Risposta mancante, offensiva o non pertinente": 0,
    "Non lo so": -1,    
}

def inv_map(d):
    return {v: k for k, v in d.items()}

def clean_file(file_path, out_dir):
    with open(file_path) as f:
        data = json.load(f)
    del data['group']
    del data['data']['feedback']
    data['data']['daily-language'] = LANG_MAP.get(data['data']['daily-language'], data['data']['daily-language'])
    data['data']['proficiency'] = PROFICIENCY_MAP.get(data['data']['proficiency'], data['data']['proficiency'])
    data['data']['grammar'] = AGREEMENT_MAP.get(data['data']['grammar'], data['data']['grammar'])
    data['data']['use'] = AGREEMENT_MAP.get(data['data']['use'], data['data']['use'])
    data['data']['language-source'] = [LANGUAGE_SOURCE_MAP.get(item, item) for item in data['data']['language-source']]
    data['data']['other-language-source'] = LANGUAGE_SOURCE_MAP.get(data['data']['other-language-source'], data['data']['other-language-source'])
    data['data']['age-group'] = AGE_MAP.get(data['data']['age-group'], data['data']['age-group'])
    data['data']['translation-evaluation-score'] = EVALUATION_TO_SCORE.get(data['data']['translation-evaluation'], -1)
    data['data']['translation-evaluation'] = EVALUATION_MAP.get(data['data']['translation-evaluation'], data['data']['translation-evaluation'])
    data['review_data']['selected_path'] = os.path.basename(data['review_data']['selected_path'])
    
    out_path = os.path.join(out_dir, os.path.basename(file_path))
    if os.path.exists(out_path):
        raise ValueError(f"Output file {out_path} already exists")
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help='Input folder with raw JSON files')
    parser.add_argument('output_folder', type=str, help='Output folder for cleaned JSON files')
    parser.add_argument('--map-json', type=str, default=None, help='Optional JSON file with custom mapping for cleaning')
    args = parser.parse_args()

    if args.input_folder == args.output_folder:
        raise ValueError("Input and output folders must be different")
    
    os.makedirs(args.output_folder, exist_ok=True)

    files = glob.glob(os.path.join(args.input_folder, '*.json'))
    for file_path in files:
        clean_file(file_path, args.output_folder)

    if args.map_json is not None:
        maps = {
            'LANG_MAP': LANG_MAP,
            'PROFICIENCY_MAP': PROFICIENCY_MAP,
            'AGREEMENT_MAP': AGREEMENT_MAP,
            'LANGUAGE_SOURCE_MAP': LANGUAGE_SOURCE_MAP,
            'AGE_MAP': AGE_MAP,
            'EVALUATION_MAP': EVALUATION_MAP,
            'EVALUATION_TO_SCORE': EVALUATION_TO_SCORE,
            'LANG_MAP_INV': inv_map(LANG_MAP),
            'PROFICIENCY_MAP_INV': inv_map(PROFICIENCY_MAP),
            'AGREEMENT_MAP_INV': inv_map(AGREEMENT_MAP),
            'LANGUAGE_SOURCE_MAP_INV': inv_map(LANGUAGE_SOURCE_MAP),
            'AGE_MAP_INV': inv_map(AGE_MAP),
            'EVALUATION_MAP_INV': inv_map(EVALUATION_MAP),
            'EVALUATION_TO_SCORE_INV': inv_map(EVALUATION_TO_SCORE),
        }
        with open(args.map_json, 'w') as f:
            json.dump(maps, f, indent=2)

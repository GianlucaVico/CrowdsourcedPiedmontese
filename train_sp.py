import datasets
import sentencepiece as spm
import random

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('vocab', type=int)
    parser.add_argument('samples', type=int)
    args = parser.parse_args()  

    samples = []

    for lang in ['pms_Latn', "ita_Latn", "fra_Latn", "eng_Latn"]:           
        ds = datasets.load_dataset("cis-lmu/Glot500", lang, split="train", streaming=True)
        samples += [sample['text'] for _, sample in zip(range(args.samples), ds)]
    
    rng = random.Random(42)
    rng.shuffle(samples)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(samples), 
        model_prefix=f"{args.model}.unigram", 
        vocab_size=args.vocab,
        model_type='unigram',
        byte_fallback=True
    )
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(samples), 
        model_prefix=f"{args.model}.bpe", 
        vocab_size=args.vocab,
        model_type='bpe',
        byte_fallback=True
    )

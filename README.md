# Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography

This is the repository for the paper [Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography](https://arxiv.org/abs/2602.14675v1) presented at the Thirteenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial) 2026, colocated with EACL 2026 in Rabat, Morocco.

The data is available here: [http://hdl.handle.net/11372/LRT-6086](http://hdl.handle.net/11372/LRT-6086) and [https://huggingface.co/datasets/ufal/CrowdsourcingPiedmontese](https://huggingface.co/datasets/ufal/CrowdsourcingPiedmontese).

We use Snakemake to run the experiments, but each step can be run manually. The scripts assume that the datasets are in the `data` folder, without subfolders. The notebook `data_analysis.ipynb` and `plots.ipynb` are used mostly for the figures in the paper.

## Tasks

- Tokenizer parity
- Machine translation
- Topic classification
- Word alignment

## Related repositories

The repository for the questionnaire can be found here: [CrowdTranslation](https://github.com/GianlucaVico/CrowdTranslation).
The tool for annotating the word alignments can be found here: [CrowdTranslationAnnotator](https://github.com/GianlucaVico/CrowdTranslationAnnotator).

## Cite

```bibtex
@inproceedings{vico-libovicky-2026-crowdsourcing,
    title = "Crowdsourcing {P}iedmontese to Test {LLM}s on Non-Standard Orthography",
    author = "Vico, Gianluca  and
      Libovick{\'y}, Jind{\v{r}}ich",
    editor = {Scherrer, Yves  and
      Aepli, No{\"e}mi  and
      Blaschke, Verena  and
      Jauhiainen, Tommi  and
      Ljube{\v{s}}i{\'c}, Nikola  and
      Nakov, Preslav  and
      Tiedemann, J{\"o}rg  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the 13th Workshop on {NLP} for Similar Languages, Varieties and Dialects",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.vardial-1.6/",
    doi = "10.18653/v1/2026.vardial-1.6",
    pages = "70--86",
    abstract = "We present a crowdsourced dataset for Piedmontese, an endangered Romance language of northwestern Italy. The dataset comprises 145 Italian{--}Piedmontese parallel sentences derived from Flores+, with translations produced by speakers writing in their natural orthographic style rather than adhering to standardized conventions, along with manual word alignment. We use this resource to benchmark several large language models on tokenization parity, topic classification, and machine translation. Our analysis reveals that Piedmontese incurs a tokenization penalty relative to higher-resource Romance languages, yet LLMs achieve classification performance approaching that of Italian, French, and English. Machine translation results are asymmetric: models translate adequately from Piedmontese into high-resource languages, but generation into Piedmontese remains challenging. The dataset and code are publicly released."
}
```

**License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
    

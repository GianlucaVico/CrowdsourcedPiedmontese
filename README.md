# Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography

This is the repository for the paper "Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography" presented at the Thirteenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial) 2026, colocated with EACL 2026 in Rabat, Morocco.

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
@misc{vico2026crowdsourcingpiedmontesetestllms,
      title={Crowdsourcing Piedmontese to Test LLMs on Non-Standard Orthography}, 
      author={Gianluca Vico and Jindřich Libovický},
      year={2026},
      eprint={2602.14675},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.14675}, 
}
```

**License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
    

import itertools

# Replace '...' with your setup
SLURM_CPU_PARTITION = "..."
SLURM_GPU_PARTITION = "..."
SLURM_GPU_EXTRA="-G 1 -C '...'"
SLURM_GPU_EXTRA_SMALL="-G 1 -C '...'"


models = {
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma": "google/gemma-3-27b-it",
    "qwen": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "eurollm": "utter-project/EuroLLM-9B-Instruct",
    "tower": "Unbabel/Tower-Plus-9B",
    "gemini": "google/gemini-2.5-flash-preview-09-2025",
    "gpt": "openai/gpt-4o-mini",
}

langs = ["ita", "pms", "fra", "eng"]
directions = [f"{src}_{tgt}" for src, tgt in itertools.permutations(langs, 2)]
pivot_directions = [f"{src}_{tgt}" for src, tgt in itertools.permutations(["pms", "fra", "eng"], 2)]

splits = ["dev", "devtest"]

def get_gpus(wc):
    if wc.model in ["llama", "qwen"]:
        return "-G 2"
    elif wc.model in ["gemini", "gpt"]:
        return ""
    else:
        return "-G 1"

def get_partition(wc):
    if wc.model in ["gemini", "gpt"]:
        return SLURM_CPU_PARTITION
    else:
        return SLURM_GPU_PARTITION

def get_constraint(wc):
    if wc.model in ["gemini", "gpt"]:
        return ""
    elif wc.model in ["llama", "qwen"]:
        return "..."
    elif wc.model in ["eurollm", "tower", "gemma"]:
        return "..."
    else:
        return "..."

def get_mem(wc):
    if wc.model in ["gemini", "gpt"]:
        return "15G"
    else:
        return "60G"

wildcard_constraints:
    model="|".join(models.keys()),
    lang="|".join(langs),
    split="|".join(splits),
    alignment_method="eflomal|simalign",
    sp="bpe|unigram",
    direction="|".join(directions)

localrules: all, prepare_alignment_data

rule all:
    input:
        expand("results/classification/{model}.{lang}.{split}.jsonl.scores", model=models.keys(), lang=langs, split=splits),
        expand("results/translation/{model}.{split}.{direction}.jsonl.scores", model=models.keys(), split=splits, direction=directions),
        expand("results/translation/{model}.{split}.{direction}.pivot_ita.jsonl.scores", model=models.keys(), split=splits, direction=pivot_directions),
        expand("results/parity/{model}.{split}.jsonl", model=list(models.keys()) + ["bpe", "unigram"], split=splits),
        expand("results/align/ita_pms.{alignment_method}.scores", alignment_method=["eflomal", "simalign"]),
        expand("results/parity/{model}.jsonl", model=list(models.keys()) + ["bpe", "unigram"]),
        expand("results/classification/{model}.{lang}.jsonl.scores", model=models.keys(), lang=langs),
        expand("results/translation/{model}.{direction}.jsonl.scores", model=models.keys(), direction=directions),
        expand("results/translation/{model}.{direction}.pivot_ita.jsonl.scores", model=models.keys(), direction=pivot_directions),
        "results/translation/baseline.ita_pms.jsonl.scores",
        "results/translation/baseline.pms_ita.jsonl.scores",
        "results/translation/baseline.pms_fra.jsonl.scores",
        "results/translation/baseline.ita_fra.jsonl.scores",
        "results/translation/baseline.pms_eng.jsonl.scores",
        "results/translation/baseline.ita_eng.jsonl.scores"

rule classification:
    input:
        "data/pms_dev.jsonl",
        "data/pms_devtest.jsonl",      
    output:
        "checkpoints/classification/{model}.{lang}.dev.jsonl",
        "checkpoints/classification/{model}.{lang}.devtest.jsonl",
    params:
        model=lambda wildcards: models[wildcards.model],
        output_file=lambda wildcards: f"checkpoints/classification/{wildcards.model}"
    resources:
        mem=get_mem,
        slurm_partition=get_partition,
        constraint=get_constraint,
        cpus_per_task=1,
        tasks=1,
        nodes=1,        
        slurm_extra=get_gpus,       
    shell:
        """
        ./classification_task.py -m {params.model} -o {params.output_file} -l {wildcards.lang}
        """


rule classification_eval:
    input:
        "checkpoints/classification/{model}.{lang}.{split}.jsonl",
    output:
        "results/classification/{model}.{lang}.{split}.jsonl.scores",
    resources:
        mem="5G",
        slurm_partition=SLURM_CPU_PARTITION,
        cpus_per_task=1
    shell:
        """
        ./classification_score.py -i {input} -o {output}
        """

use rule classification_eval as classification_join_eval with:
    input:
        expand("checkpoints/classification/{{model}}.{{lang}}.{split}.jsonl", split=splits),
    output:
        "results/classification/{model}.{lang}.jsonl.scores",

rule parity:
    input:
        "data/pms_{split}.jsonl",
    output:
        "results/parity/{model}.{split}.jsonl",
    params:
        model=lambda wildcards: models[wildcards.model],
    resources:
        mem="5G",
        slurm_partition=SLURM_CPU_PARTITION,
        cpus_per_task=1
    shell:
        """
        ./zero_shot_parity.py -m {params.model} -i {input} -o {output}
        """

use rule parity as parity_join with:
    input:
        expand("data/pms_{split}.jsonl", split=splits),
    output:
        "results/parity/{model}.jsonl",

rule translation:
    input:
        "data/pms_dev.jsonl",
        "data/pms_devtest.jsonl",      
    output:
        expand("checkpoints/translation/{{model}}.{split}.{direction}.jsonl", split=splits, direction=directions),
    params:
        model=lambda wildcards: models[wildcards.model],
        output_file=lambda wildcards: f"checkpoints/translation/{wildcards.model}"
    resources:
        mem=get_mem,
        slurm_partition=get_partition,
        constraint=get_constraint,
        cpus_per_task=1,
        tasks=1,
        nodes=1,        
        slurm_extra=get_gpus,
    shell:
        """
        ./translation_task.py -m {params.model} -o {params.output_file}
        """

rule translation_eval:
    input:
        "checkpoints/translation/{model}.{split}.{direction}.jsonl",
    output:
        "results/translation/{model}.{split}.{direction}.jsonl.scores",
    resources:
        mem="15G",
        slurm_partition=SLURM_GPU_PARTITION,
        cpus_per_task=1,
        constraint="...",
        tasks=1,
        nodes=1,        
        slurm_extra="-G 1",
    shell:
        """
        ./translation_score.py -i {input} -o {output}
        """

use rule translation_eval as translation_join_eval with:
    input:
        expand("checkpoints/translation/{{model}}.{split}.{{direction}}.jsonl", split=splits),
    output:
        "results/translation/{model}.{direction}.jsonl.scores",

rule translation_pivot:
    input:
        "data/pms_dev.jsonl",
        "data/pms_devtest.jsonl",      
    output:
        expand("checkpoints/translation/{{model}}.{split}.{direction}.pivot_ita.jsonl", split=splits, direction=pivot_directions),
    params:
        model=lambda wildcards: models[wildcards.model],
        output_file=lambda wildcards: f"checkpoints/translation/{wildcards.model}",
        pivot_lang="ita"
    resources:
        mem=get_mem,
        slurm_partition=get_partition,
        constraint=get_constraint,
        cpus_per_task=1,
        tasks=1,
        nodes=1,        
        slurm_extra=get_gpus,
    shell:
        """
        ./pivot_translation_task.py -m {params.model} -o {params.output_file} -p {params.pivot_lang}
        """

use rule translation_eval as translation_pivot_eval with:
    input:
        "checkpoints/translation/{model}.{split}.{direction}.pivot_ita.jsonl",
    output:
        "results/translation/{model}.{split}.{direction}.pivot_ita.jsonl.scores",

use rule translation_pivot_eval as translation_pivot_join_eval with:
    input:
        expand("checkpoints/translation/{{model}}.{split}.{{direction}}.pivot_ita.jsonl", split=splits),
    output:
        "results/translation/{model}.{direction}.pivot_ita.jsonl.scores",

rule train_sp: # Generally run manually once
    output:
        "checkpoints/sentencepiece.unigram.model",
        "checkpoints/sentencepiece.unigram.vocab",
        "checkpoints/sentencepiece.bpe.model",
        "checkpoints/sentencepiece.bpe.vocab",
    resources:
        mem="5G",
        slurm_partition=SLURM_CPU_PARTITION,
        cpus_per_task=1
    shell:
        """
        python3 train_sp.py checkpoints/sentencepiece 32000 100000
        """

rule sp_parity:
    input:
        data="data/pms_{split}.jsonl",
        model="checkpoints/sentencepiece.{sp}.model",
    output:
        "results/parity/{sp}.{split}.jsonl",
    resources:
        mem="5G",
        slurm_partition=SLURM_CPU_PARTITION,
        cpus_per_task=1
    shell:
        """
        python3 sp_parity.py -m {input.model} -i {input.data} -o {output}
        """

use rule sp_parity as sp_parity_join with:
    input:
        data=expand("data/pms_{split}.jsonl", split=splits),
        model="checkpoints/sentencepiece.{sp}.model",
    output:
        "results/parity/{sp}.jsonl",
    
rule prepare_alignment_data: # Generally run manually once
    output:
        ita="data/alignment.ita",
        ita_dev="data/alignment.ita.dev",
        ita_devtest="data/alignment.ita.devtest",
        pms="data/alignment.pms",
        pms_dev="data/alignment.pms.dev",
        pms_devtest="data/alignment.pms.devtest",
        align="data/ita_pms.align",
        align_dev="data/ita_pms.dev.align",
        align_devtest="data/ita_pms.devtest.align"
    params:
        "data/span_list.json"
    shell:
        """
        ./prepare_for_alignment.py -s {params} -oa {output.align} -os {output.ita} -ot {output.pms}
        ./prepare_for_alignment.py -s {params} -oa {output.align_dev} -os {output.ita_dev} -ot {output.pms_dev} --split dev
        ./prepare_for_alignment.py -s {params} -oa {output.align_devtest} -os {output.ita_devtest} -ot {output.pms_devtest} --split devtest
        """

rule prepare_raw_data: # Generally run manually once
    output:
        dev="data/pms_dev.jsonl",
        devtest="data/pms_devtest.jsonl"
    shell:
        """
        ./prepare_raw_data.py
        """

rule eflomal_align:
    input:
        ita="data/alignment.ita",
        pms="data/alignment.pms",
    output:
        align="results/align/ita_pms.eflomal.align",
        fwd="results/align/ita_pms.eflomal.fwd",
        rev="results/align/ita_pms.eflomal.rev"
    resources:
        mem="10G",
        slurm_partition=SLURM_CPU_PARTITION,
        cpus_per_task=1
    shell:
        """
        eflomal-align -s {input.ita} -t {input.pms} -f {output.fwd} -r {output.rev}
        atools -c grow-diag-final-and -i {output.fwd} -j {output.rev} > {output.align}
        """

rule simalign_align:
    input:
        ita="data/alignment.ita",
        pms="data/alignment.pms",
    output:
        align="results/align/ita_pms.simalign.align"
    resources:
        mem="10G",
        slurm_partition=SLURM_GPU_PARTITION,
        constraint="gpuram24G|gpuram40G|gpuram48G",
        cpus_per_task=1,
        tasks=1,
        nodes=1,        
        slurm_extra="-G 1",
    params:
        method='i',
        model='xlmr'
    shell:
        """
        ./simalign_task.py {input.ita} {input.pms} {output.align} --model {params.model} --method {params.method}
        """

rule alignment_score:
    input:
        gold="data/ita_pms.align",
        input="results/align/ita_pms.{alignment_method}.align"
    output:
        "results/align/ita_pms.{alignment_method}.scores"
    resources:
        mem="5G",
        slurm_partition=SLURM_CPU_PARTITION,
        cpus_per_task=1
    shell:
        """
        ./alignment_score.py {input.gold} {input.input} {output}
        """
rule translation_score_baseline:
    output:
        "results/translation/baseline.{eval}_{trg}.jsonl.scores", # use eval when evaluating x->trg               
    params:
        source_files=expand("checkpoints/translation/gemma.{split}.{{trg}}_{{eval}}.jsonl", split=splits), # the actual model does not matter, we only use the reference field
        target_files=expand("checkpoints/translation/gemma.{split}.{{eval}}_{{trg}}.jsonl", split=splits),
    resources:
        mem="15G",
        slurm_partition=SLURM_GPU_PARTITION,
        cpus_per_task=1,
        constraint="gpuram24G|gpuram40G|gpuram48G",
        tasks=1,
        nodes=1,        
        slurm_extra="-G 1",
    shell:
        """
        ./translation_score_baseline.py --source {params.source_files} --target {params.target_files} --output {output}
        """

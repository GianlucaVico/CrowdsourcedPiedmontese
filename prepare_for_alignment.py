#!/usr/bin/env python3
"""
Use the span list file to create the ground truth alignment files and the tokenized test files.
"""

# Preprocessing:
# remove punctuation
# lowercase
# remove multiple spaces
# space after apostrophes at the end of words ('d: no space, d': space)

# Tokenization:
# split on spaces
# remove punctuation tokens except apostrophes

# Algorithm:
# Replace each span with §# where # is the span index
# Split by span: sequence of spans and parts that are not spans
# Clean each part separately 
# Generate the sequence of token indexes 
# Use the spans to generate the alignment
# Reconstruct the tokenized text

# Span list: list of dictionaries with keys: 
# split: 'dev' or 'devtest'
# flores_id: int
# spans: list of [src str, trg str] pairs
# span_indexes: list of pairs of [start, end] indexes in ita and pms texts
# pms: original pms text
# ita: original ita text

import argparse
import json
import re
import itertools

import ipdb

def span_argsort(span_indexes: list[tuple[int, int]]) -> list[int]:
    """Return the indices that would sort the spans by their start positions."""
    return sorted(range(len(span_indexes)), key=lambda i: span_indexes[i][1], reverse=True)

def replace_tokens(text: str, span_indexes: list[tuple[int, int], tuple[int, int]], is_source: bool)-> tuple[str, list[str]]:
    """Replace spans with special tokens §0, §1, ..."""
    i = 0 if is_source else 1
    spans_indexes = [pair[i] for pair in span_indexes] # list of (start, end) tuples
    spans = []
    mask = [-1] * len(text)
    for i, (start, end) in enumerate(spans_indexes): # start from the end to not mess up indexes
        span_text = text[start:end]
        spans.append(span_text)
        # token = f"  §{i}  "
        for j in range(start, end):
            mask[j] = i
    chars = [] # [-1, -1, 0, 0, 0, -1, -1, 1, 1, -1] -> " xx §0xx§1x"
    last = -1
    for i, m in enumerate(mask):
        if m == -1:
            chars.append(text[i])
        elif m != last:
            chars.append(f" §{m} ")
            last = m
    text = ''.join(chars)
    return text, spans

def to_parts(text: str)-> list[str]:
    """"Split the text into aligned (special tokens) and unaligned parts (strings)."""
    regex = re.compile(r'( ?§[0-9]+ ?)')
    parts = regex.split(text)
    parts = [part.strip() for part in parts]
    parts = [part for part in parts if part != '']
    return parts

def special_to_index(part: str)-> int:
    """§# -> #"""
    return int(part.strip()[1:])

def tokenize(part: str) -> list[str]:
    """Preprocess the text part."""
    # remove punctuation
    part = re.sub(r'[.,;:!?]', '', part)
    # space around -
    part = re.sub(r'\s*-\s*', ' - ', part)
    # lowercase
    part = part.lower()
    # space after apostrophes at the end of words
    part = re.sub(r"(\w)'", r"\1' ", part)
    # remove multiple spaces
    part = re.sub(r'\s+', ' ', part)
    part = part.strip()

    tokens = part.split(' ') 
    tokens = map(lambda x: x.strip(), tokens)
    tokens = [token for token in tokens if token != '']
    return tokens

def token_offset(parts: list[list[str]], spans: list[list[str]]) -> list[int]:
    """Compute the token index offsets for each part."""
    offsets = []
    current_offset = 0
    for part in parts:
        offsets.append(current_offset)
        if len(part) == 0:
            continue
        if part[0].lstrip().startswith('§'):
            span_index = special_to_index(part[0])
            span_tokens = spans[span_index]
            current_offset += len(span_tokens)
        else:
            current_offset += len(part)
    return offsets

def make_alignments(aligned_parts: list[tuple[int, int]], 
    src_offsets: list[int], trg_offsets: list[int],
    src_parts: list[list[str]], trg_parts: list[list[str]],
    src_spans: list[list[str]], trg_spans: list[list[str]]) -> list[tuple[int, int]]:
    """Generate token-level alignments from part-level alignments."""
    alignments = []
    for src_index, trg_index in aligned_parts:
        src_start = src_offsets[src_index] # How many tokens before this part
        trg_start = trg_offsets[trg_index]
        src_special = src_parts[src_index][0] # §#
        trg_special = trg_parts[trg_index][0]
        src_span = src_spans[special_to_index(src_special)]
        trg_span = trg_spans[special_to_index(trg_special)]
        for i, j in itertools.product(range(len(src_span)), range(len(trg_span))):
            pair = (src_start + i, trg_start + j)
            alignments.append(pair)
    return alignments    

def combine_tokens(parts: list[list[str]], aligned_spans: list[list[str]]) -> str:
    """Reconstruct the tokenized text from parts and aligned spans."""
    tokens = []
    for part in parts:
        if len(part) == 0:
            continue
        elif part[0].lstrip().startswith('§'):
            span_index = special_to_index(part[0])
            span_tokens = aligned_spans[span_index]
            tokens.extend(span_tokens)
        else:
            tokens.extend(part)
    tokens = ' '.join(tokens)
    tokens = re.sub(r'\s+', ' ', tokens)
    return tokens.strip()

def process_sample(src_text: str, trg_text: str, span_indexes: list[tuple[tuple[int, int], tuple[int, int]]]):
    # Replace aligned spans
    src_replace, src_aligned_spans = replace_tokens(src_text, span_indexes, is_source=True)
    trg_replace, trg_aligned_spans = replace_tokens(trg_text, span_indexes, is_source=False)
    assert len(src_aligned_spans) == len(trg_aligned_spans), "Source and target parts length mismatch"
    
    # Split aligned and unaligned parts
    src_parts = to_parts(src_replace)
    trg_parts = to_parts(trg_replace)
    

    aligned_part_indexes = [] # list of tuples with indexes of (src §#, trg §#)
    for i, src_part in enumerate(src_parts):
        if src_part.strip().startswith('§'):
            j = trg_parts.index(src_part)
            aligned_part_indexes.append((i,j))
    assert len(src_aligned_spans) == len(aligned_part_indexes), "Missing aligned tokens"
    
    # Clean and tokenize
    src_aligned_spans = [tokenize(span) for span in src_aligned_spans]
    trg_aligned_spans = [tokenize(span) for span in trg_aligned_spans]
    src_parts = [tokenize(part) if not part.strip().startswith('§') else [part] for part in src_parts]
    trg_parts = [tokenize(part) if not part.strip().startswith('§') else [part] for part in trg_parts]
    
    # From part indexes to token indexes
    src_offsets = token_offset(src_parts, src_aligned_spans)
    trg_offsets = token_offset(trg_parts, trg_aligned_spans)
    
    # Alignments
    alignments = make_alignments(aligned_part_indexes, src_offsets, trg_offsets, src_parts, trg_parts, src_aligned_spans, trg_aligned_spans)
    
    # Tokenized texts
    src_tokenized = combine_tokens(src_parts, src_aligned_spans)
    trg_tokenized = combine_tokens(trg_parts, trg_aligned_spans)
    return alignments, src_tokenized, trg_tokenized

def print_aligned(alignments, src_tokenized, trg_tokenized):
    src = src_tokenized.split(' ')
    trg = trg_tokenized.split(' ')
    for (i, j) in alignments:
        print(f"{i}-{j}: {src[i]} -> {trg[j]}")
    

def process_pairs(items: list[dict[str, str|int|tuple[str, str], list[tuple[tuple[int, int], tuple[int, int]]]]], split: str=None):
    alignments = []
    src_texts = []
    trg_texts = []

    for item in items:
        src_text = item['ita']
        trg_text = item['pms']
        span_indexes = item['spans_index']
        if split is not None and item['split'] != split:
            continue
        a, s, t = process_sample(src_text, trg_text, span_indexes)
        alignments.append(a)
        src_texts.append(s)
        trg_texts.append(t)
    return alignments, src_texts, trg_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--span_list', '-s', type=str, help='JSON file with span lists')
    parser.add_argument('--output_alignments', '-oa', type=str, help='Output alignment file')
    parser.add_argument('--output_src', '-os', type=str, help='Output tokenized test file')
    parser.add_argument('--output_trg', '-ot', type=str, help='Output tokenized test file')
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()

    with open(args.span_list, 'r') as f:
        span_data = json.load(f)
    alignments, src_texts, trg_texts = process_pairs(span_data, args.split)
    with open(args.output_alignments, 'w') as f:
        for alignment in alignments:
            alignment_str = ' '.join([f"{i}-{j}" for i, j in alignment])
            f.write(alignment_str + '\n')
    with open(args.output_src, 'w') as f:
        for src_text in src_texts:
            f.write(src_text + '\n')
    with open(args.output_trg, 'w') as f:
        for trg_text in trg_texts:
            f.write(trg_text + '\n')
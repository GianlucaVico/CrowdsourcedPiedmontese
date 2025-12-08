#!/usr/bin/env python3

# Copied from rom simalign, with minor modifications (removed line numbers)
import argparse
import json
import os.path


def load_gold(g_path):
	gold_f = open(g_path, "r")
	pros = {}
	surs = {}
	all_count = 0.
	surs_count = 0.

	for n, line in enumerate(gold_f):
		line = line.strip().split()

		pros[n] = set([x.replace("p", "-") for x in line])
		surs[n] = set([x for x in line if "p" not in x])

		all_count += len(pros[n])
		surs_count += len(surs[n])

	return pros, surs, surs_count

def calc_score(input_path, probs, surs, surs_count):
	total_hit = 0.
	p_hit = 0.
	s_hit = 0.
	target_f = open(input_path, "r")

	for n, line in enumerate(target_f):
		line = line.strip().split()

		if len(line[0].split("-")) > 2:
			line = ["-".join(x.split("-")[:2]) for x in line]

		p_hit += len(set(line) & set(probs[n]))
		s_hit += len(set(line) & set(surs[n]))
		total_hit += len(set(line))
	target_f.close()

	y_prec = round(p_hit / max(total_hit, 1.), 3)
	y_rec = round(s_hit / max(surs_count, 1.), 3)
	y_f1 = round(2. * y_prec * y_rec / max((y_prec + y_rec), 0.01), 3)
	aer = round(1 - (s_hit + p_hit) / (total_hit + surs_count), 3)

	return y_prec, y_rec, y_f1, aer


if __name__ == "__main__":
	'''
	Calculate alignment quality scores based on the gold standard.
	The output contains Precision, Recall, F1, and AER.
	The gold annotated file should be selected by "gold_path".
	The generated alignment file should be selected by "input_path".
	Both gold file and input file are in the FastAlign format with sentence number at the start of line separated with TAB.

	usage: python calc_align_score.py gold_file generated_file output_file
	'''

	parser = argparse.ArgumentParser(description="Calculate alignment quality scores based on the gold standard.", epilog="example: python calc_align_score.py gold_path input_path output_path")
	parser.add_argument("gold_path")
	parser.add_argument("input_path")
	parser.add_argument("output_path")
	args = parser.parse_args()

	if not os.path.isfile(args.input_path):
		print("The input file does not exist:\n", args.input_path)
		exit()

	probs, surs, surs_count = load_gold(args.gold_path)
	y_prec, y_rec, y_f1, aer = calc_score(args.input_path, probs, surs, surs_count)

	print("Prec: {}\tRec: {}\tF1: {}\tAER: {}".format(y_prec, y_rec, y_f1, aer))
	scores = {
		"Precision": y_prec,
		"Recall": y_rec,
		"F1": y_f1,
		"AER": aer
	}
	with open(args.output_path, "w") as out_f:
		json.dump(scores, out_f, indent=2)

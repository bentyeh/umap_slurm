#!/bin/bash

# Generate a chromosome sizes file (2-column tab-delimited file of names of each
# sequence and the length of each sequence) given an input FASTA file.
# Usage: ./make_chrom_sizes.sh input.fasta output.chrom.sizes

awk '/^>/ {if (seq) print name, seq; split($0, a, " "); name=substr(a[1], 2); seq=0; next} {seq+=length($0)} END {if (seq) print name, seq}' OFS='\t' "$1" > "$2"

#!/bin/bash

# This script replaces underscores in the header lines of a FASTA file with dashes.
# Usage: ./rename_chroms.sh input.fasta output.fasta
awk '{ if ($0 ~ /^>/) { gsub("_", "-", $0) } print $0 }' "$1" > "$2"

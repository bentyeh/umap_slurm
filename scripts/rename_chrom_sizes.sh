#!/bin/bash

# This script replaces underscores in chromosome names file with hyphens.
# Usage: ./rename_chrom_sizes.sh input.chrom.sizes output.chrom.sizes
awk -F'\t' -v OFS='\t' '{ gsub("_", "-", $1); print $1,$2 }' "$1" > "$2"

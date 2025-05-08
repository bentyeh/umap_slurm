#!/bin/bash

# This script replaces hyphens in chromosome names in BED files with underscores,
# removes any track header lines, and coordinate-sorts the file.
# Usage: ./rename_chroms_bed.sh input.bed[.gz] output.bed

set -euo pipefail
path_in="$1"
path_out="$2"

case "$path_in" in
  *.gz) cmd_open="unpigz -c" ;;
  *) cmd_open="cat" ;;
esac

export LC_ALL=C

$cmd_open "$path_in" |
awk -F'\t' 'BEGIN {OFS=FS} $1 !~ /^track/ {gsub(/-/, "_", $1); print}' |
sort -k1,1 -k2,2n -k3,3n > "$path_out"

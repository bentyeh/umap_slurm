#!/bin/bash

# This script replaces hyphens in chromosome names in wiggle files with underscores.
# It also checks that the start positions are increasing for the same chromosome.
# Usage: ./rename_chroms_wig.sh input.wig[.gz] output.wig

set -euo pipefail
path_in="$1"
path_out="$2"

case "$path_in" in
  *.gz) cmd_open="unpigz -c" ;;
  *) cmd_open="cat" ;;
esac

$cmd_open "$path_in" |
awk '
/^(fixedStep|variableStep)/ {
    # Replace hyphens in chromosome name with underscores
    match($0, /chrom=[^ \t]+/)
    if (RSTART > 0) {
        chrom_field = substr($0, RSTART, RLENGTH)
        gsub("-", "_", chrom_field)
        $0 = substr($0, 1, RSTART - 1) chrom_field substr($0, RSTART + RLENGTH)

        # Extract chromosome name
        match(chrom_field, /chrom=([^ \t]+)/, chrom_arr)
        chrom = chrom_arr[1]

        # Extract start position
        match($0, /start=([0-9]+)/, start_arr)
        if (start_arr[1] != "") {
            start = start_arr[1] + 0

            # Check that start is increasing for the same chromosome
            if (chrom == prev_chrom && start <= prev_start) {
                print "Error: Start position " start " not greater than previous start " prev_start " on chromosome " chrom > "/dev/stderr"
                exit 1
            }

            prev_chrom = chrom
            prev_start = start
        }
    }
}
{ print }
' > "$path_out"

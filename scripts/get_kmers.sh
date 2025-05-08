#!/bin/bash
#
# Args
# $1: conda environment prefix
# $2: Umap output directory
# $3: Umap script directory
# $4: kmer size

CONDA_PREFIX="$1"
DIR_OUT="$2"
DIR_UMAP_SCRIPTS="$3"
K="$4"

if [ -z "$CONDA_PREFIX" ] || [ -z "$DIR_OUT" ] || [ -z "$DIR_UMAP_SCRIPTS" ] || [ -z "$K" ]; then
    echo "Usage: $0 <conda_prefix> <dir_out> <dir_umap_scripts> <k>"
    exit 1
fi

source ~/.bashrc
conda activate "$CONDA_PREFIX"
set -euo pipefail
trap 'echo "Error on line $LINENO: $BASH_COMMAND" >&2' ERR
export var_id=$((SLURM_ARRAY_TASK_ID + OFFSET))

python "${DIR_UMAP_SCRIPTS}/get_kmers.py" \
    "${DIR_OUT}/chrsize.tsv" \
    "${DIR_OUT}/kmers/k${K}" \
    "${DIR_OUT}/chrs" \
    "${DIR_OUT}/chrsize_index.tsv" \
    --var_id var_id \
    --kmer "k${K}"

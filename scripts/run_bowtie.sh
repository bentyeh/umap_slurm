#!/bin/bash
#
# Args
# $1: conda environment prefix
# $2: Umap output directory
# $3: Umap script directory
# $4: kmer size
# $5: Bowtie installation directory
# $6: Bowtie index prefix

CONDA_PREFIX="$1"
DIR_OUT="$2"
DIR_UMAP_SCRIPTS="$3"
K="$4"
DIR_BOWTIE="$5"
BOWTIE_INDEX_PREFIX="$6"

if [ -z "$CONDA_PREFIX" ] || [ -z "$DIR_OUT" ] || [ -z "$DIR_UMAP_SCRIPTS" ] || [ -z "$K" ] || [ -z "$DIR_BOWTIE" ] || [ -z "$BOWTIE_INDEX_PREFIX" ]; then
    echo "Usage: $0 <conda_prefix> <dir_out> <dir_umap_scripts> <k> <dir_bowtie> <bowtie_index_prefix>"
    exit 1
fi

source ~/.bashrc
conda activate "$CONDA_PREFIX"
set -euo pipefail
trap 'echo "Error on line $LINENO: $BASH_COMMAND" >&2' ERR
export var_id=$((SLURM_ARRAY_TASK_ID + OFFSET))

python "${DIR_UMAP_SCRIPTS}/run_bowtie.py" \
    "${DIR_OUT}/kmers/k${K}" \
    "$DIR_BOWTIE" \
    "$(dirname "$BOWTIE_INDEX_PREFIX")" \
    "$(basename "$BOWTIE_INDEX_PREFIX")" \
    -var_id var_id

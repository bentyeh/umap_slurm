#!/bin/bash
#
# Download Umap and install dependencies (Bowtie 1.1.0; create a conda environment with Python 2.7)
#
# Args
# $1: Directory to store temporary files
# $2: Directory to store UMAP scripts
# $3: Directory to install Bowtie 1.1.0
# $4: Path to conda environment prefix (where Python 2.7 will be installed)
# $5: Whether to overwrite existing files; set to True to overwrite

DIR_TEMP="$1"
DIR_UMAP="$2"
DIR_BOWTIE="$3"
CONDA_PREFIX="$4"
OVERWRITE="$5"

if [ -z "$DIR_TEMP" ] || [ -z "$DIR_UMAP" ] || [ -z "$DIR_BOWTIE" ] || [ -z "$CONDA_PREFIX" ]; then
    echo "Usage: $0 <temp_dir> <umap_dir> <bowtie_dir> <conda_prefix> [overwrite]"
    exit 1
fi

set -euo pipefail

#######################
# Download Umap
#######################

if [ -d "$DIR_UMAP" ] && [ -f "${DIR_UMAP}/ubismap.py" ] && [ "$OVERWRITE" != "True" ]; then
    echo "Umap scripts already exist at $DIR_UMAP. Will not re-install again."
else
    path_download="$DIR_TEMP/umap.tar.gz"
    dir_extracted="$DIR_TEMP/umap_extracted"

    wget -O "$path_download" 'https://github.com/hoffmangroup/umap/archive/refs/tags/1.2.1.tar.gz'
    mkdir -p "$dir_extracted"
    tar -xzf "$path_download" -C "$dir_extracted" --strip-components=1
    if [ -d "$DIR_UMAP" ]; then
        if [ "$OVERWRITE" != "True" ]; then
            echo "Directory $DIR_UMAP already exists. Please remove it or set OVERWRITE to True to overwrite."
            exit 1
        fi
        rm -rf "$DIR_UMAP"
    fi
    mv "$dir_extracted/umap" "$DIR_UMAP"
    rm -rf "$dir_extracted"
    rm "$path_download"

    while true; do
        echo "Modify uint8_to_bed_parallel.py:371 and uint8_to_bed_parallel.py:376 to be the following:"
        echo '    poses_end = np.append(poses_end, [len(uniquely_mappable) - 1])'
        read -p 'Enter "done" when completed.' DONE
        if [ "$DONE" == "done" ]; then
            break
        fi
    done
fi

#######################
# Download Bowtie 1.1.0
#######################

if [ -d "$DIR_BOWTIE" ] && [ -f "${DIR_BOWTIE}/bowtie-build" ] && [ "$OVERWRITE" != "True" ]; then
    echo "Bowtie appears to already be installed at $DIR_BOWTIE. Will not re-install again."
else
    path_download="${DIR_TEMP}/bowtie.zip"
    dir_bowtie_unzipped="${DIR_TEMP}/bowtie-1.1.0"

    # download the Bowtie zip file
    wget -O "$path_download" 'https://master.dl.sourceforge.net/project/bowtie-bio/bowtie/1.1.0/bowtie-1.1.0-linux-x86_64.zip'
    # unzipping it creates a new folder "bowtie-1.1.0"
    unzip -d "$DIR_TEMP" "$path_download"
    # move it to the desired location
    if [ -d "$DIR_BOWTIE" ]; then
        if [ "$OVERWRITE" != "True" ]; then
            echo "Directory $DIR_BOWTIE already exists. Please remove it or set OVERWRITE to True to overwrite."
            exit 1
        fi
        rm -rf "$DIR_BOWTIE"
    fi
    mv "$dir_bowtie_unzipped" "$DIR_BOWTIE"
    rm "$path_download"
fi

#######################
# Create conda environment with Python 2.7
#######################

if [ -d "$CONDA_PREFIX" ] && [ -f "${CONDA_PREFIX}/bin/python2" ] && [ "$OVERWRITE" != "True" ]; then
    echo "Conda environment appears to already be installed at $CONDA_PREFIX. Will not re-install again."
else
    if [ -d "$CONDA_PREFIX" ]; then
        if [ "$OVERWRITE" != "True" ]; then
            echo "Directory $CONDA_PREFIX already exists. Please remove it or set OVERWRITE to True to overwrite."
            exit 1
        fi
        rm -rf "$CONDA_PREFIX"
    fi

    conda create --prefix "$CONDA_PREFIX" -c defaults -c conda-forge -y python=2.7.18 pandas
fi
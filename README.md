Scripts to run [Umap](https://github.com/hoffmangroup/umap) on a SLURM cluster

The [existing suggested way of running Umap on a SLURM cluster](https://github.com/hoffmangroup/umap) was buggy, so I made a new wrapper/runner script myself.

Key assumptions
1. Chromosome names do not have hyphens in them.
   - This pipeline first renames all chromosomes, replacing underscores with hyphens. From the Umap documentation:
     > Important: The chromosome names in the fasta file should not contain underscore. Underscore is used in Bismap to differentiate reverse complement chromosomes.
   - At the end of the pipeline, before converting wiggle and BED files to their binary format equivalents (bigWig, bigBed), the chromosome are renamed again, replacing hyphens with underscores.
2. The scripts assume that the following programs are installed and available from PATH. Versions shown are what I used; other versions may work, but I have not checked or tested.
   - [GNU bash](https://www.gnu.org/software/bash/) 5.1.8
   - [Python](https://www.python.org/) 3.10 or 3.11
   - [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 24.11.3
   - [pigz](https://zlib.net/pigz/) 2.7
   - [GNU awk](https://www.gnu.org/software/gawk/) 5.1.0
   - [bedToBigBed](https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/) 2.10
   - [wigToBigWig](https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/) 2.9


Usage
1. Install Umap

   ```bash
   scripts/install_umap.sh DIR_TEMP DIR_UMAP DIR_BOWTIE CONDA_PREFIX [overwrite]
   ```

2. Run Umap

    ```bash
    python run_umap.py [options] DIR_TEMP DIR_FINAL PATH_GENOME_FASTA DIR_SCRIPTS PATH_CONDA_SBATCH CONDA_PREFIX --KMERS KMERS [KMERS ...]
    ```

import argparse
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd

DTYPE_JOB_STATES = pd.CategoricalDtype(
    [
        'FAILED',
        'TIMEOUT',
        'DEADLINE',
        'OUT_OF_MEMORY',
        'BOOT_FAIL',
        'NODE_FAIL',
        'CANCELLED',
        'PREEMPTED',
        'SUSPENDED',
        'RUNNING',
        'PENDING',
        'COMPLETED'
    ],
    ordered=True
)
FAILURE_STATES = ('FAILED', 'TIMEOUT', 'DEADLINE', 'OUT_OF_MEMORY', 'BOOT_FAIL', 'NODE_FAIL', 'CANCELLED')

def parse_args():
    parser = argparse.ArgumentParser(description='Run Umap on a genome.')
    parser.add_argument(
        'DIR_TEMP',
        help='Path to temporary directory. Subdirectories include logs and Umap output folder; optionally install Bowtie and Umap in this directory.'
    )
    parser.add_argument(
        'DIR_FINAL',
        help='Directory to save the final multi-read mappability bigWigs and single-read mappability bigBed files.'
    )
    parser.add_argument('PATH_GENOME_FASTA', help='Path to the genome FASTA file.')
    parser.add_argument('DIR_SCRIPTS', help='Directory to SLURM submission scripts.')
    parser.add_argument('PATH_CONDA_SBATCH', help='Path to the conda sbatch script.')
    parser.add_argument('CONDA_PREFIX', help='Path to the conda environment prefix.')
    parser.add_argument(
        '--KMERS',
        required=True,
        nargs='+',
        type=int,
        help='Kmer sizes to use for Umap. Must be between 1 and 255.'
    )
    parser.add_argument('--chrom_sizes', help='Chromosome sizes file.')
    parser.add_argument('--dir_bowtie', help='Bowtie installation directory.')
    parser.add_argument('--dir_umap', help='Umap installation directory.')
    parser.add_argument('--dir_out', help='Umap output directory.')
    parser.add_argument('--dir_logs', help='Log directory.')
    parser.add_argument('--prefix_bowtie_index', help='Bowtie index prefix.')
    parser.add_argument('--reprocess', action='store_true')
    parser.add_argument('--num_jobs', type=int, default=40, help='Number of jobs to run in parallel.')
    return parser.parse_args()


def preprocess_dependencies(
    dependencies: list[str | int | None],
    jobname: str = "",
    verbose: bool = True,
    wait: bool = True
) -> list[str]:
    """
    Preprocess job dependencies to remove None values and convert to strings.
    1. If any of the dependencies failed, raise an error.
    2. If any of the dependencies are not found, wait JobAcctGatherFrequency seconds. If they are still not found,
       raise a ValueError.
    3. If any of the dependencies are completed, remove them from the list.

    Args
    - dependencies: List of job IDs to check.
    - jobname: Name of the job being submitted.
    - verbose: Whether to print the full submission command to standard error.
    - wait: Whether to wait for JobAcctGatherFrequency seconds before checking for missing dependencies.

    Returns: List of job IDs that are still pending or running.
    """
    dependencies = [str(jobid) for jobid in dependencies if jobid is not None]
    # Instead of directly passing these Job IDs to sbatch --dependency, we preprocess them first.
    # This prevents job submission failure if those jobs finished more than MinJobAge seconds ago.
    # See sbatch documententation. Use scontrol show config | grep MinJobAge to get the MinJobAge
    # SLURM configuration parameter value.
    if len(dependencies) == 0:
        return []
    states = get_job_state(dependencies, collapse_arrays_and_steps=True)
    if any(state in FAILURE_STATES for state in states.values()):
        raise ValueError(f"One or more dependencies failed: {states}")

    # If any dependencies are not found by sacct, wait JobAcctGatherFrequency seconds and check again.
    for jobid in dependencies.copy():
        if jobid not in states:
            if wait is True:
                JobAcctGatherFrequency = int(get_slurm_config()['JobAcctGatherFrequency'])
                time.sleep(JobAcctGatherFrequency)
                return preprocess_dependencies(dependencies, jobname=jobname, verbose=verbose, wait=False)
            raise ValueError(f"Trying to submit job {jobname}: dependency {jobid} not found in SLURM job status.")

    if verbose:
        print(f"Trying to submit job {jobname}: dependency states =", states, file=sys.stderr, flush=True)

    for jobid in dependencies.copy():
        if states[jobid] == 'COMPLETED':
            dependencies.remove(jobid)
    return dependencies


def submit_job(
    *args,
    submit_command: str = "sbatch",
    dependencies: list[str | int | None] | None = None,
    jobname: str = "",
    export: str = "",
    array: str = "",
    c: int = 1,
    time: str = "12:00:00",
    mem: str = "4G",
    output: str = "",
    error: str = "",
    verbose: bool = True
) -> int:
    """
    Submit SLURM job and return job ID.

    Args
    - *args: program and arguments to run.
    - submit_command: Command to submit the job ("sbatch" or "srun").
    - dependencies: List of job IDs that this job depends on.
    - jobname, export, array, c, time, mem, output, error: SLURM job parameters.
    - verbose: Whether to print the full submission command to standard error.
    """
    assert submit_command in ["sbatch", "srun"], f"submit_command {submit_command} not 'sbatch' or 'srun'."
    command = [
        submit_command,
        "-c",
        f"{c}",
        f"--time={time}",
        f"--mem={mem}",
    ]
    if submit_command == "sbatch":
        command.append("--parsable")
    if jobname != "":
        command.append(f"--job-name={jobname}")
    if array != "":
        command.append(f"--array={array}")
    if export != "":
        command.append(f"--export={export}")
    if output != "":
        command.append(f"--output={output}")
    if error != "":
        command.append(f"--error={error}")
    if dependencies is not None:
        dependencies = preprocess_dependencies(dependencies, jobname=jobname, verbose=verbose)
        if len(dependencies) > 0:
            command.append("--dependency=afterok:" + ",".join(dependencies))

    command.extend(args)
    if verbose:
        print(" ".join(command), file=sys.stderr, flush=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if submit_command == 'srun':
            match = re.match(r'srun: job (\d+) has been allocated resources', result.stderr.splitlines()[1])
            if match is None:
                raise ValueError(f"Failed to parse job ID from srun output: {result.stdout}; {result.stderr}")
            jobid = int(match.group(1))
        else:
            jobid = int(result.stdout.strip())
        if verbose:
            print(f"Submitted job {jobname} with {'array' if array != '' else 'job'} ID {jobid}", file=sys.stderr, flush=True)
        return jobid
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}.", file=sys.stderr)
        print("stderr:", e.stderr, file=sys.stderr)
        raise


def get_job_state(
    jobids: list[str] | None = None,
    user: str | None = os.environ['USER'],
    collapse_arrays_and_steps: bool = False
) -> dict[str, str]:
    """
    Get the status of SLURM jobs.

    Args:
    - jobids: List of job IDs to check.
    - user: User name to check for running jobs. If None, does not filter by user.
    - collapse_arrays_and_steps: If True, collapse arrays and steps into a single job ID.

    Returns: dictionary mapping job IDs to their status.
    - If a requested job ID is not found, it will not be included in the output.
    """
    command = ["sacct", "--parsable2"]
    if jobids is not None:
        command.extend(["-j", ','.join(jobids)])
    if user is not None:
        command.extend(["-u", user])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    df_status = pd.read_csv(io.StringIO(result.stdout), sep='|', header=0).astype(dict(JobID=str))
    df_status = df_status.loc[~df_status['JobID'].str.endswith(('.batch', '.extern'))].copy()
    df_status['JobID_base'] = df_status['JobID'].str.extract(r'^(\d+)', expand=False)
    if collapse_arrays_and_steps:
        return (
            df_status
            .astype(dict(State=DTYPE_JOB_STATES))
            .sort_values(['JobID_base', 'State'])
            .groupby('JobID_base')['State']
            .first()
            .to_dict()
        )
    else:
        return df_status.set_index('JobID')['State'].to_dict()


def wait_to_submit(max_jobs: int = 1000, max_wait: int = 1200, user: str | None = None) -> None:
    """
    Wait until the number of running jobs is less than max_jobs.

    Args:
    - max_jobs: Maximum number of pending or running jobs to allow before submitting new jobs.
    - max_wait: Maximum time to wait in seconds.
    - user: User name to check for running jobs. If None, use the current user.
    """
    if user is None:
        user = os.environ['USER']
    start = time.time()
    while time.time() - start < max_wait:
        command = f'squeue -u {user} -t PENDING,RUNNING --array -h -o "%A.%a.%i" | wc -l'
        n_running_jobs = int(subprocess.run(command, shell=True, capture_output=True, text=True, check=True).stdout.strip())
        if n_running_jobs < max_jobs:
            break
        max_wait_remaining = max_wait - int(time.time() - start)
        print(
            f"Waiting for {n_running_jobs} jobs to drop below {max_jobs} or a maximum of {max_wait_remaining} seconds.",
            file=sys.stderr
        )
        time.sleep(20)


def get_slurm_config() -> dict[str, str]:
    """
    Get SLURM configuration parameters via scontrol show config.
    """
    result = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True, check=True)
    regex_option = re.compile(r'^(?P<key>\S+)\s+=\s+(?P<value>.*)$')
    config = dict()
    for line in result.stdout.splitlines():
        match = regex_option.match(line.rstrip())
        if match:
            config[match.group('key')] = match.group('value')
    return config


def main():
    args = parse_args()

    #############################
    # validate required arguments
    #############################
    PATH_GENOME_FASTA = Path(args.PATH_GENOME_FASTA)
    assert PATH_GENOME_FASTA.is_file()

    DIR_TEMP = Path(args.DIR_TEMP)
    assert not DIR_TEMP.is_file()
    DIR_TEMP.mkdir(parents=True, exist_ok=True)

    DIR_FINAL = Path(args.DIR_FINAL)
    assert not DIR_FINAL.is_file()
    DIR_FINAL.mkdir(parents=True, exist_ok=True)

    DIR_SCRIPTS = Path(args.DIR_SCRIPTS)
    assert DIR_SCRIPTS.is_dir()

    KMERS = args.KMERS
    assert all(1 <= k <= 255 for k in KMERS), "Kmers must be between 1 and 255."

    PATH_CONDA_SBATCH = Path(args.PATH_CONDA_SBATCH)
    assert PATH_CONDA_SBATCH.is_file() and os.access(str(PATH_CONDA_SBATCH), os.X_OK)

    CONDA_PREFIX = Path(args.CONDA_PREFIX)
    assert CONDA_PREFIX.is_dir() and CONDA_PREFIX.joinpath('bin', 'python').is_file()

    num_jobs = args.num_jobs
    assert num_jobs > 0, "Number of jobs must be greater than 0."

    #############################
    # Optional arguments
    #############################
    path_chrom_sizes = Path(args.chrom_sizes) if args.chrom_sizes else DIR_TEMP.joinpath("genome.chrom.sizes")

    dir_bowtie = Path(args.dir_bowtie) if args.dir_bowtie else DIR_TEMP.joinpath('bowtie')
    assert dir_bowtie.is_dir()

    dir_umap = Path(args.dir_umap) if args.dir_umap else DIR_TEMP.joinpath('umap')
    assert dir_umap.is_dir()

    dir_out = Path(args.dir_out) if args.dir_out else DIR_TEMP.joinpath('umap_out')
    assert not dir_out.is_file()
    dir_out.mkdir(parents=True, exist_ok=True)

    dir_logs = Path(args.dir_logs) if args.dir_logs else DIR_TEMP.joinpath('logs')
    assert not dir_logs.is_file()
    dir_logs.mkdir(parents=True, exist_ok=True)

    prefix_bowtie_index = args.prefix_bowtie_index if args.prefix_bowtie_index else str(DIR_TEMP.joinpath('bowtie_index'))

    reprocess = args.reprocess

    #############################
    # Derived paths and parameters
    #############################

    path_bowtie = dir_bowtie.joinpath('bowtie')
    path_bowtie_build = dir_bowtie.joinpath("bowtie-build")

    path_genome_fasta_renamed = DIR_TEMP.joinpath("genome_renamed.fa")
    path_chrom_sizes_renamed = DIR_TEMP.joinpath("genome_renamed.chrom.sizes")

    path_segment_indices = dir_out.joinpath('chrsize_index.tsv')

    # Umap scripts
    py_ubismap = dir_umap.joinpath("ubismap.py")
    py_unify_bowtie = dir_umap.joinpath("unify_bowtie.py")
    py_combine_umaps = dir_umap.joinpath("combine_umaps.py")
    py_uint8_to_bed_parallel = dir_umap.joinpath("uint8_to_bed_parallel.py")
    py_combine_wigs_or_beds = dir_umap.joinpath("combine_wigs_or_beds.py")

    # My scripts
    script_rename_chroms = DIR_SCRIPTS.joinpath("rename_chroms.sh")
    script_rename_chrom_sizes = DIR_SCRIPTS.joinpath("rename_chrom_sizes.sh")
    script_make_chrom_sizes = DIR_SCRIPTS.joinpath("make_chrom_sizes.sh")
    script_get_kmers = DIR_SCRIPTS.joinpath("get_kmers.sh")
    script_run_bowtie = DIR_SCRIPTS.joinpath("run_bowtie.sh")
    script_rename_chroms_wig = DIR_SCRIPTS.joinpath("rename_chroms_wig.sh")
    script_rename_chroms_bed = DIR_SCRIPTS.joinpath("rename_chroms_bed.sh")

    ############################
    # Run pipeline
    ############################

    jobids = dict()
    if not path_chrom_sizes.is_file():
        jobids['make_chrom_sizes'] = submit_job(
            str(script_make_chrom_sizes),
            str(PATH_GENOME_FASTA),
            str(path_chrom_sizes),
            jobname="make_chrom_sizes",
            time="01:00:00", 
            output=dir_logs.joinpath("make_chrom_sizes.log")
        )

    if not path_chrom_sizes_renamed.is_file():
        jobids['rename_chrom_sizes'] = submit_job(
            str(script_rename_chrom_sizes),
            str(path_chrom_sizes),
            str(path_chrom_sizes_renamed),
            jobname="rename_chrom_sizes",
            dependencies=[jobids.get('make_chrom_sizes')],
            time="01:00:00",
            output=dir_logs.joinpath("rename_chrom_sizes.log")
        )

    if not path_genome_fasta_renamed.is_file():
        jobids['rename_chroms'] = submit_job(
            str(script_rename_chroms),
            str(PATH_GENOME_FASTA),
            str(path_genome_fasta_renamed),
            jobname="rename_chroms",
            output=dir_logs.joinpath("rename_chroms.log")
        )

    path_bowtie_index_files = [Path(str(prefix_bowtie_index) + suffix) for suffix in \
                               ('.1.ebwt', '.2.ebwt', '.3.ebwt', '.4.ebwt', '.rev.1.ebwt', '.rev.2.ebwt')]
    if not all(path.is_file() for path in path_bowtie_index_files):
        jobids['bowtie_index'] = submit_job(
            str(path_bowtie_build),
            str(path_genome_fasta_renamed),
            str(prefix_bowtie_index),
            jobname="bowtie_index",
            submit_command="srun",
            dependencies=[jobids.get('rename_chroms')],
            time="12:00:00",
            mem="32G",
            output=dir_logs.joinpath("bowtie_index.log")
        )

    if not (dir_out.joinpath('genome', 'genome.fa').exists() and path_segment_indices.exists()) or reprocess:
        jobids['initialize_umap'] = submit_job(
            str(PATH_CONDA_SBATCH),
            str(CONDA_PREFIX),
            "python",
            str(py_ubismap),
            "--kmers", *list(map(str, KMERS)),
            "-SimultaneousJobs", str(num_jobs),
            "-write_script", "/dev/null",
            "-var_id", "SLURM_ARRAY_TASK_ID",
            str(path_genome_fasta_renamed),
            str(path_chrom_sizes_renamed),
            str(dir_out),
            'ubismap', # queue name for qsub job submission
            str(path_bowtie),
            submit_command="srun",
            jobname="initialize_umap",
            dependencies=[jobids.get(x) for x in ('rename_chrom_sizes', 'rename_chroms') if x in jobids],
            output=dir_logs.joinpath("ubismap.log"),
        )

    df_segment_indices = pd.read_csv(path_segment_indices, sep='\t', header=0)
    n_chroms = len(df_segment_indices['Chromosome'].unique())
    n_segments = len(df_segment_indices)
    SLURM_CONFIG = get_slurm_config()
    max_array_size = int(SLURM_CONFIG['MaxArraySize']) - 1

    print('Number of segments:', n_segments, flush=True)
    print('max_array_size:', max_array_size, flush=True)

    for kmer in KMERS:
        # shard genome into kmers
        jobids[('get_kmers', kmer)] = []
        for i, index_start in enumerate(range(1, n_segments+1, max_array_size)):
            wait_to_submit()
            offset = i * max_array_size
            end = min(index_start+max_array_size-1, n_segments) - offset
            array=f"{index_start - offset}-{end}%{num_jobs}"
            export=f"ALL,OFFSET={offset}"
            jobname = f"get_kmers_{kmer}_{index_start - offset}-{end}"
            jobids[('get_kmers', kmer)].append(submit_job(
                str(script_get_kmers),
                str(CONDA_PREFIX),
                str(dir_out),
                str(dir_umap),
                str(kmer),
                jobname=jobname,
                array=array,
                export=export,
                output=dir_logs.joinpath(f"get_kmers_{kmer}_{offset}-%A_%a.log"),
            ))

        # align kmers to genome
        jobids[('align_kmers', kmer)] = []
        for i, index_start in enumerate(range(1, n_segments+1, max_array_size)):
            wait_to_submit()
            offset = i * max_array_size
            end = min(index_start+max_array_size-1, n_segments) - offset
            array=f"{index_start - offset}-{end}%{num_jobs}"
            export=f"ALL,OFFSET={offset}"
            jobname = f"align_kmers_{kmer}_{index_start - offset}-{end}"
            jobids[('align_kmers', kmer)].append(submit_job(
                str(script_run_bowtie),
                str(CONDA_PREFIX),
                str(dir_out),
                str(dir_umap),
                str(kmer),
                str(dir_bowtie),
                str(prefix_bowtie_index),
                dependencies=[jobids.get('bowtie_index')] + jobids[('get_kmers', kmer)],
                jobname=jobname,
                array=array,
                export=export,
                output=dir_logs.joinpath(f"align_kmers_{kmer}_{offset}-%A_%a.log"),
            ))

        # merge Bowtie outputs into a uint8 vector for each chromosome
        wait_to_submit()
        jobname = f"unify_bowtie_{kmer}"
        jobids[('unify_bowtie', kmer)] = submit_job(
            str(PATH_CONDA_SBATCH),
            str(CONDA_PREFIX),
            "python",
            str(py_unify_bowtie),
            str(dir_out.joinpath('kmers', f'k{kmer}')),
            str(dir_out.joinpath('chrsize.tsv')),
            "-var_id", "SLURM_ARRAY_TASK_ID",
            jobname=jobname,
            dependencies=jobids.get(('align_kmers', kmer)),
            mem="16G",
            array=f"1-{n_chroms}%{num_jobs}",
            output=dir_logs.joinpath(f"unify_bowtie_{kmer}-%A_%a.log"),
        )

    # Combines mappability uint8 vectors of differet kmer values into 1 uint8 vector per chromosome.
    # make the output directory first to prevent race conditions (each of the jobs will also try to
    # create the directory if it does not exist)
    dir_out.joinpath('kmers', 'globalmap').mkdir(exist_ok=True)
    time.sleep(5)
    jobids['combine_umaps'] = submit_job(
        str(PATH_CONDA_SBATCH),
        str(CONDA_PREFIX),
        "python",
        str(py_combine_umaps),
        str(dir_out.joinpath('kmers')),
        str(dir_out.joinpath('chrsize.tsv')),
        "-out_dir", str(dir_out.joinpath('kmers', 'globalmap')),
        "-var_id", "SLURM_ARRAY_TASK_ID",
        jobname="combine_umaps",
        dependencies=[jobids.get(('unify_bowtie', kmer)) for kmer in KMERS],
        mem="16G",
        array=f"1-{n_chroms}%{num_jobs}",
        output=dir_logs.joinpath(f"combine_umaps-%A_%a.log"),
    )

    for kmer in KMERS:
        # generate BED files and wiggle files
        # - output files: {DIR_OUT}/kmers/bedFiles/{chrom}.k{k}.genome_umap.bed.gz
        # - output files: {DIR_OUT}/kmers/wigFiles/genome_umap.{chrom}.k{k}.MultiReadMappability.wg.gz
        # make the directories first to prevent race conditions (each of the jobs will also try to create the directories
        # if it does not exist)
        dir_out.joinpath('kmers', 'wigFiles').mkdir(exist_ok=True)
        dir_out.joinpath('kmers', 'bedFiles').mkdir(exist_ok=True)
        time.sleep(5)
        wait_to_submit()
        jobids[('make_bed', kmer)] = submit_job(
            str(PATH_CONDA_SBATCH),
            str(CONDA_PREFIX),
            "python",
            str(py_uint8_to_bed_parallel),
            str(dir_out.joinpath('kmers', 'globalmap')),
            str(dir_out.joinpath('kmers', 'bedFiles')),
            'genome_umap',
            "-chrsize_path", str(dir_out.joinpath('chrsize.tsv')),
            "-bed",
            "-kmers", f"k{kmer}",
            "-var_id", "SLURM_ARRAY_TASK_ID",
            jobname=f"make_bed_{kmer}",
            dependencies=[jobids.get('combine_umaps')],
            mem="16G",
            array=f"1-{n_chroms}%{num_jobs}",
            output=dir_logs.joinpath(f"make_bed_{kmer}-%A_%a.log"),
        )
        wait_to_submit()
        jobids[('make_wiggle', kmer)] = submit_job(
            str(PATH_CONDA_SBATCH),
            str(CONDA_PREFIX),
            "python",
            str(py_uint8_to_bed_parallel),
            str(dir_out.joinpath('kmers', 'globalmap')),
            str(dir_out.joinpath('kmers', 'wigFiles')),
            'genome_umap',
            "-chrsize_path", str(dir_out.joinpath('chrsize.tsv')),
            "-wiggle",
            "-kmers", f"k{kmer}",
            "-var_id", "SLURM_ARRAY_TASK_ID",
            jobname=f"make_wiggle_{kmer}",
            dependencies=[jobids.get('combine_umaps')],
            mem="16G",
            array=f"1-{n_chroms}%{num_jobs}",
            output=dir_logs.joinpath(f"make_wiggle_{kmer}-%A_%a.log"),
        )

        # merge the BED files and wiggle files of different chromosomes
        # - output files: "{DIR_OUT}/bedFiles/k{kmer}.merged.genome_umap.bed.gz"
        # - output files: "{DIR_OUT}/wigFiles/k{kmer}.merged.genome_umap.MultiReadMappability.wg.gz"
        # make the directories first to prevent race conditions (each of the jobs will also try to create the directories
        # if it does not exist)
        dir_out.joinpath('wigFiles').mkdir(exist_ok=True)
        dir_out.joinpath('bedFiles').mkdir(exist_ok=True)
        time.sleep(5)
        jobids[('merge_bed', kmer)] = submit_job(
            str(PATH_CONDA_SBATCH),
            str(CONDA_PREFIX),
            "python",
            str(py_combine_wigs_or_beds),
            str(dir_out.joinpath('kmers', 'bedFiles')),
            str(dir_out.joinpath('bedFiles')),
            "--kmers", f"k{kmer}",
            dependencies=[jobids.get(('make_bed', kmer))],
            jobname=f"merge_bed_{kmer}",
            mem="16G",
            output=dir_logs.joinpath(f"merge_bed_{kmer}.log"),
        )
        jobids[('merge_wiggle', kmer)] = submit_job(
            str(PATH_CONDA_SBATCH),
            str(CONDA_PREFIX),
            "python",
            str(py_combine_wigs_or_beds),
            str(dir_out.joinpath('kmers', 'wigFiles')),
            str(dir_out.joinpath('wigFiles')),
            "--kmers", f"k{kmer}",
            dependencies=[jobids.get(('make_wiggle', kmer))],
            jobname=f"merge_wiggle_{kmer}",
            mem="16G",
            output=dir_logs.joinpath(f"merge_wiggle_{kmer}.log"),
        )

        # rename chromosomes - replace hyphens with underscores
        path_cleaned_bed = dir_out.joinpath('bedFiles', f"k{kmer}.merged.genome_umap.bed")
        path_cleaned_wig = dir_out.joinpath('wigFiles', f"k{kmer}.merged.genome_umap.MultiReadMappability.wg")
        jobids[('rename_chroms_bed', kmer)] = submit_job(
            str(script_rename_chroms_bed),
            str(dir_out.joinpath('bedFiles', f"k{kmer}.merged.genome_umap.bed.gz")),
            str(path_cleaned_bed),
            dependencies=[jobids.get(('merge_bed', kmer))],
            jobname=f"rename_chroms_bed_{kmer}",
            output=dir_logs.joinpath(f"rename_chroms_bed_{kmer}.log"),
        )
        jobids[('rename_chroms_wig', kmer)] = submit_job(
            str(script_rename_chroms_wig),
            str(dir_out.joinpath('wigFiles', f"k{kmer}.merged.genome_umap.MultiReadMappability.wg.gz")),
            str(path_cleaned_wig),
            dependencies=[jobids.get(('merge_wiggle', kmer))],
            jobname=f"rename_chroms_wig_{kmer}",
            output=dir_logs.joinpath(f"rename_chroms_wig_{kmer}.log"),
        )

        # convert bed to bigBed
        path_bigbed = DIR_FINAL.joinpath(f"k{kmer}.SingleReadMappability.bb")
        jobids[('bed_to_bigbed', kmer)] = submit_job(
            str(PATH_CONDA_SBATCH),
            "py3",
            "bedToBigBed",
            "-type=bed6",
            "-tab",
            str(path_cleaned_bed),
            str(path_chrom_sizes),
            str(path_bigbed),
            dependencies=[jobids.get(('rename_chroms_bed', kmer))],
            jobname=f"bed_to_bigbed_{kmer}",
            output=dir_logs.joinpath(f"bed_to_bigbed_{kmer}.log"),
            mem="100G"
        )

        # convert wiggle to bigWig
        path_bigwig = DIR_FINAL.joinpath(f"k{kmer}.MultiReadMappability.bw")
        jobids[('wig_to_bigwig', kmer)] = submit_job(
            str(PATH_CONDA_SBATCH),
            "py3",
            "wigToBigWig",
            str(path_cleaned_wig),
            str(path_chrom_sizes),
            str(path_bigwig),
            dependencies=[jobids.get(('rename_chroms_wig', kmer))],
            jobname=f"wig_to_bigwig_{kmer}",
            output=dir_logs.joinpath(f"wig_to_bigwig_{kmer}.log"),
            mem="100G"
        )


    print(jobids)


if __name__ == '__main__':
    main()

'''
This Python script writes the Bash scripts for reproducing the results of the hyperparameter and benchmark study.
Before executing this script.
Original code from https://github.com/ies-research/multi-annotator-machine-learning/blob/main/empirical_evaluation/python_scripts/write_bash_scripts.py
'''
import os
from itertools import product


def write_commands(
    config_combs: list,
    path_python_file: str = ".",
    directory: str = ".",
    use_slurm: bool = True,
    mem: str = "20gb",
    max_n_parallel_jobs: int = 12,
    cpus_per_task: int = 4,
    slurm_logs_path: str = "slurm_logs",
    job_name: str = "experiments"
):
    """
    Writes Bash scripts for the experiments.

    Parameters
    ----------
    config_combs : list
        A list of dictionaries defining the configurations of the experiments.
    path_python_file : str, default="."
        Absolute path to the Python file to be executed.
    directory : str
        Path to the directory where the Bash scripts are to be saved.
    use_slurm : bool
        Flag whether SLURM shall be used.
    mem : str
        RAM size allocated for each experiment. Only used if `use_slurm=True`.
    max_n_parallel_jobs : int
        Maximum number of experiments executed in parallel. Only used if `use_slurm=True`.
    cpus_per_task : int
        Number of CPUs allocated for each experiment. Only used if `use_slurm=True`.
    use_gpu : bool
        Flag whether to use a GPU. Only used if `use_slurm=True`.
    slurm_logs_path : str
        Path to the directory where the SLURM logs are to saved. Only used if `use_slurm=True`.
    """
    filename = os.path.join(directory, f"{job_name}.sh")
    commands = []
    n_jobs = 0
    for cfg_dict in config_combs:
        keys, values = zip(*cfg_dict["params"].items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        n_jobs += len(permutations_dicts)
        python_command = f"srun python"
        if not use_slurm:
            commands = [commands[0]]
            python_command = f"python"
        for param_dict in permutations_dicts:
            commands.append(
                f"{python_command} "
                f"{path_python_file} "
                f"dataset={cfg_dict['dataset']} "
                # f"model={cfg_dict['model']} "
                # f"query_strategy={cfg_dict['query_strategy']} "
                # f"seed={cfg_dict['seed']} "
                # f"batch_size={cfg_dict['batch_size']} "
            )
            if cfg_dict["backbone"] is not None:
                commands[-1] += f"backbone={cfg_dict['backbone']} "
            if hasattr(cfg_dict, "cache"):
                commands[-1] += f"cache={cfg_dict['cache']}"
            for k, v in param_dict.items():
                commands[-1] += f"{k}={v} "
            if not use_slurm:
                commands.append("wait")
    
    if max_n_parallel_jobs > n_jobs:
        max_n_parallel_jobs = n_jobs
    commands_to_save = [
        f"#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=1-{n_jobs}%{max_n_parallel_jobs}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --ntasks=1",
        f"#SBATCH --get-user-env",
        f"#SBATCH --time=12:00:00",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --partition=main",
        f"#SBATCH --output={slurm_logs_path}/{job_name}_%A_%a.log",
        f'eval "$(sed -n "$(($SLURM_ARRAY_TASK_ID+{12})) p" {filename})"',
        f"exit 0",
    ]
    commands_to_save.extend(commands)
    print(filename)
    with open(filename, "w") as f:
        for item in commands_to_save:
            f.write("%s\n" % item)


if __name__ == "__main__":
    # TODO: Update the default arguments of the `write_commands` function below to fit your machine.
    path_python_file = "hydra_experiment.py"# "your/absolute/path/to/perform_experiment.py"
    directory = "./bash_scripts/"# "your/absolute/path/to/bash_scripts/"
    use_slurm = True
    mem = "10gb"
    max_n_parallel_jobs = 110
    cpus_per_task = 4
    accelerator = "cpu"
    slurm_logs_path = ""

    # List of seeds to ensure reproducibility.
    seeds = list(range(10))

    dataset_with_backbone = [
        ("cifar10", "dino_head"), ("trec6", "bert_base"), ("letter", None),
    ]
    # ================================= Create bash scripts for downloading datasets. ================================
    # Note: The experiment will throw an error after downloading the dataset and caching it.
    config_combs = []
    job_name = "download"
    for dataset, backbone in dataset_with_backbone:
        config_combs.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "cache": True,
                "params": {
                    "seed": [0],
                    "batch_size": [0],
                },
            },
        )
    write_commands(
        path_python_file=path_python_file,
        directory=directory,
        config_combs=config_combs,
        slurm_logs_path=slurm_logs_path,
        max_n_parallel_jobs=max_n_parallel_jobs,
        mem=mem,
        cpus_per_task=cpus_per_task,
        use_slurm=use_slurm,
        job_name=job_name,
    )

    # ================================= Create bash scripts for benchmark. ============================================
    classifiers = [
        "random_sampling",
        "uncertainty_sampling",
    ]
    dataset_with_backbone = [
        ("cifar10", "dino_head"), ("trec6", "bert_base"), ("letter", None),
    ]
    models = [
        "logistic_regression", "random_forest"
    ]
    query_strategies = [
        "random_sampling", "uncertainty_sampling"
    ]
    batch_sizes = [1]
    
    config_combs = []
    job_name = "experiments"
    for dataset, backbone in dataset_with_backbone:
        config_combs.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "params": {
                    "model": models,
                    "query_strategy": query_strategies,
                    "seed": seeds,
                    "batch_size": batch_sizes,
                },
            },
        )
    write_commands(
        path_python_file=path_python_file,
        directory=directory,
        config_combs=config_combs,
        slurm_logs_path=slurm_logs_path,
        max_n_parallel_jobs=max_n_parallel_jobs,
        mem=mem,
        cpus_per_task=cpus_per_task,
        use_slurm=use_slurm,
        job_name=job_name,
    )
import pandas as pd
import os
import sys
sys.path.append("../..")

files_dict = {
    'dataset': [],
    'model': [],
    'qs_strategy': [],
    'batch_size': [],
    'seed': [],
}

experiment_folder = "experiments/experiments_results/"

def get_file(files_dict, path="experiments/"):
    datasets = os.listdir(path)
    for dataset in datasets:
        model_folders = os.path.join(path, dataset)
        models = os.listdir(model_folders)
        for model in models:
            qs_folders = os.path.join(model_folders, model)
            qs_strategies = os.listdir(qs_folders)
            for qs_strategy in qs_strategies:
                batch_size_folders = os.path.join(qs_folders, qs_strategy)
                batch_sizes = os.listdir(batch_size_folders)
                for batch_size in batch_sizes:
                    seed_files = os.path.join(batch_size_folders, batch_size)
                    files = os.listdir(seed_files)
                    for file in files:
                        if ".csv" in file:
                            files_dict["dataset"].append(dataset)
                            files_dict["model"].append(model)
                            files_dict["qs_strategy"].append(qs_strategy)
                            files_dict["batch_size"].append(batch_size)
                            files_dict["seed"].append(file)

get_file(files_dict, experiment_folder)
df = pd.DataFrame.from_dict(data=files_dict)
df.to_csv("src_shiny_app/experiments.csv", index=False)

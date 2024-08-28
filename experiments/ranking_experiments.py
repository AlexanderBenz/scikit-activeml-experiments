import pandas as pd
import numpy as np
import os
import argparse
"""
This script allows users to generate Tables displaying the mean and std for all
selected experiments. It allows to calculate the rankings. Note that at least
two params of () in combination with the index and column need to be given in
order to generat sensible table.

Example
`python ranking_experiments.py --model=LogisticRegression --bs=50 --min=0 --max=20`
Will generate a table for the LogisticRegression model with batch_size 50 
between the cycles 0-20
"""

parser = argparse.ArgumentParser(description='Select ranking options')
parser.add_argument('--dataset', metavar='dataset', required=False, type=str,
                    help='Select dataset if None selects all availabel. Connect multiple using +')
parser.add_argument('--model', metavar='model', required=False, type=str,
                    help='Select Model if None selects all availabel. Connect multiple using +')
parser.add_argument('--bs', metavar='batch_size', required=False, type=str,
                    help='Select batch_size if None selects all availabel. Connect multiple using +')
parser.add_argument('--qs', metavar='qs_strategy', required=False, type=str,
                    help='Select QS strategy if None selects all availabel. Connect multiple using +')
parser.add_argument('--index', metavar='index', default="qs_strategy", required=False, type=str,
                    help='Desides which variable is used for the index to display the results')
parser.add_argument('--column', metavar='column', default="dataset", required=False, type=str,
                    help='Desides which variable is used for the columns to display the results')
parser.add_argument('--metric', metavar='metric', default="accuracy", required=False, type=str,
                    help='Select metric used for the results')
parser.add_argument('--min', metavar='min', default=0, required=False, type=int,
                    help='The minimal index for the cycles')
parser.add_argument('--max', metavar='max', default=5, required=False, type=int,
                    help='The maximal index for the cycles')
parser.add_argument('--decimals', metavar='decimals', default=4, required=False, type=int,
                    help='The number of decimals after 0.')
parser.add_argument('--seeds', metavar='seeds', default=10, required=False, type=int,
                    help='The minimum number of seeds selected for the ranking results. ' +
                    'All seeds above will be reduced to the number while experiments with less seeds ' +
                    'will be excluded from the ranking.')
args_ = parser.parse_args()
args = vars(args_)

results_path = "metric_rankings/"
os.makedirs(results_path, exist_ok=True)
experiment_dir = "../src_shiny_app/"
filepath = experiment_dir + "experiments.csv"
df = pd.read_csv(filepath, dtype=str)

# Could be used instead if no dataset should be excluded
if args["dataset"] is None:
    selected_datasets_list = sorted(df['dataset'].unique())
else:
    selected_datasets_list = args["dataset"].split("+")
if args["model"] is None:
    selected_models_list = sorted(df['model'].unique())
else:
    selected_models_list = args["model"].split("+")
if args["qs"] is None:
    selected_qs_list = sorted(df['qs_strategy'].unique())
else:
    selected_qs_list = args["qs"].split("+")
if args["bs"] is None:
    selected_batch_sizes_list = sorted(df['batch_size'].unique())
else:
    selected_batch_sizes_list = args["bs"].split("+")

# Use the code below if you want to deselect specific experiments without listing all
# selected_datasets_list = [
#     "ag_news_bert-base-uncased",
#     "aloi",
#     "banking77_bert-base-uncased", 
#     "cat_and_dog_dinov2_vits14",
#     "cifar100_dinov2_vits14", 
#     "cifar10_dinov2_vits14",
#     "dbpedia_bert-base-uncased",
#     "dtd_dinov2_vits14",
#     "iris",
#     "letter",
#     "pendigits",
#     "trec6_bert-base-uncased",
#     ]
# selected_qs_list = [
#     "Alce", 
#     "Badge", 
#     "Clue", 
#     "ContrastiveAL", 
#     "CoreSet", 
#     "GreedySamplingX", 
#     "ProbCover", 
#     "ProbabilisticAL", 
#     "RandomSampling", 
#     "TypiClust",
#     "USEntropy",
#     "USMargin", 
#     "USLeastConfident",
#     ]
# selected_models_list = [
#     "LogisticRegression",
#     "ParzenWindowClassifier",
#     "RandomForestClassifier"
#     ]
# selected_batch_sizes_list = [
#     "1",
#     "10",
#     "50",
#     ]
metric_str = args["metric"] 
min_index = args["min"] 
max_index = args["max"] 
decimals = args["decimals"]
# all lists need to be the same lengh as all the others
min_n_seeds = args["seeds"]

experiment_dict = {
    "dataset": selected_datasets_list,
    "model": selected_models_list,
    "qs_strategy": selected_qs_list,
    "batch_size": selected_batch_sizes_list,
}

# Available options ("dataset", "qs_strategy", "model", "batch_size")
display_types = (args["column"], args["index"])
filename_types = [key for key in experiment_dict.keys() if key not in display_types]

for i, x in enumerate(["dataset", "model", "qs_strategy", "batch_size"]):
    if x == display_types[0]:
        column = i
    if x == display_types[1]:
        index = i
    


# Select and load the correct experiments
df = pd.read_csv(filepath, dtype=str)
df = df.loc[df['dataset'].isin(selected_datasets_list)]
df = df.loc[df['model'].isin(selected_models_list)]
df = df.loc[df['qs_strategy'].isin(selected_qs_list)]
df = df.loc[df['batch_size'].isin(selected_batch_sizes_list)]
selected_dataframes_list = df.values
df = df[["dataset", "model", "qs_strategy", "batch_size"]].drop_duplicates()
old_path = ""
df_metrics = []
for selected_dataframe in selected_dataframes_list:
    current_path = "/".join(selected_dataframe[:4])
    complete_path = "experiments_results/" + "/".join(selected_dataframe)
    if old_path != current_path:
        df_metrics.append([])
    df_metrics[-1].append(pd.read_csv(complete_path, index_col="step"))
    old_path = current_path
# create the results dataframes
results_df = pd.DataFrame(columns=experiment_dict[display_types[0]], index=experiment_dict[display_types[1]])
ranking_results_df = pd.DataFrame(columns=experiment_dict[display_types[0]], index=experiment_dict[display_types[1]])
for (df_list, sel) in zip(df_metrics,df.values):
    # Save all metrics for each df as most experiments are run on multiple seeds
    metric = []
    for df_ in df_list:
        values = df_[metric_str].values
        if len(metric) > 0:
            values = values[:len(metric[-1])]
        metric.append(values)
    # calculate the error bars 
    reshaped_result = np.array(metric).reshape((-1, len(metric[0])))
    max_posible_index = min(max_index, reshaped_result.shape[1]-1)
    min_possible_index = max_posible_index - (max_index - min_index)
    errorbar_mean = np.mean(reshaped_result, axis=0)
    errorbar_std =np.std(reshaped_result, axis=0)
    # calculate the mean for the given intervall
    mean_interval = np.round(np.mean(errorbar_mean[min_possible_index:max_posible_index]), decimals=decimals)
    std_interval =  np.round(np.std(errorbar_std[min_possible_index:max_posible_index]), decimals=decimals)
    results_df.loc[sel[index],sel[column]] = str(mean_interval) + u"\u00B1" + str(std_interval)
    # see if the number of runs is greater than seeds to calculate a fair score
    
    if len(reshaped_result[:,max_posible_index]) >= min_n_seeds:
        
        ranking_results_df.loc[sel[index],sel[column]] = np.mean(reshaped_result[:min_n_seeds,min_possible_index:max_posible_index],axis=0)
    
# Rank each dataset seperatly for the last metric values over all seeds
for key in ranking_results_df.keys():
    print(key)
    # seperate missing values
    mask = ranking_results_df[key].notna()
    mask_values = mask.values
    # if no results exist for the dataset ignore this column
    if np.sum(mask_values) == 0:
        continue
    replace_list = np.full(shape=len(mask_values), fill_value=np.nan)
    seed_metric = ranking_results_df[key].values
    seed_metric = np.array([ s for s in seed_metric[mask_values]])
    print(seed_metric)
    # if only one experiment exists ranking is not viable
    if len(seed_metric.shape) < 2:
        continue
    # calculate rank for each seed and replace the lists
    ranking_seeds = len(seed_metric) - np.argsort(seed_metric, axis=0).argsort(axis=0)
    ranking_mean = np.mean(ranking_seeds, axis=1)
    replace_list[mask] = ranking_mean
    ranking_results_df[key] = replace_list
    print(ranking_seeds)

print(results_df)
print(ranking_results_df)    

# Save the results
results_df.to_csv(results_path+f"{experiment_dict[filename_types[0]][0]}+{experiment_dict[filename_types[1]][0]}+{min_index}-{max_index}_means.csv")
ranking_results_df.to_csv(results_path+f"{experiment_dict[filename_types[0]][0]}+{experiment_dict[filename_types[1]][0]}+{min_index}-{max_index}_ranking.csv")
Active learning Benchmark
##########################
This is an implementation for Benchmarking active learning using **scikit-activeml**.
This repository includes experiments scripts, results and the code for the
interactive website displaying the experiment results.

This repository includes the results for the following datasets, query strategies
models, and batch sizes.

Datasets
=================
+------------+---------+---------+--------------+---------------------+
| Dataset    | Samples | Classes | General Task |           Dimension |
+============+=========+=========+==============+=====================+
| CAT_vs_DOG |  23 410 |    2    |     Image    |     4x4px-500x500px |
+------------+---------+---------+--------------+---------------------+
| CIFAR10    |  60 000 |    10   |     Image    |             32x32px |
+------------+---------+---------+--------------+---------------------+
| CIFAR100   |  60 000 |   100   |     Image    |             32x32px |
+------------+---------+---------+--------------+---------------------+
| DTD        |  5 640  |    47   |     Image    | 300x300px-640x640px |
+------------+---------+---------+--------------+---------------------+
| AG-NEWS    | 127 600 |    4    |     Text     |      100–1000 words |
+------------+---------+---------+--------------+---------------------+
| BANK77     |  13083  |    77   |     Text     |        13–433 words |
+------------+---------+---------+--------------+---------------------+
| DBpedia    | 630 000 |    14   |     Text     |       3-13600 words |
+------------+---------+---------+--------------+---------------------+
| TREC6      |   5952  |    6    |     Text     |       Not Available |
+------------+---------+---------+--------------+---------------------+
| ALOI       | 108 000 |   1000  |    Tabular   |        128 features |
+------------+---------+---------+--------------+---------------------+
| IRIS       |   150   |    3    |    Tabular   |          4 features |
+------------+---------+---------+--------------+---------------------+
| LETTER     |  20 000 |    26   |    Tabular   |         16 features |
+------------+---------+---------+--------------+---------------------+
| PENDIGITS  |  10992  |    10   |    Tabular   |         16 features |
+------------+---------+---------+--------------+---------------------+

Query Strategies
=================
+----+--------------------------+----------------------------------+-------------+
|    | Active learning strategy | Classifiers                      | Batch sizes |
+====+==========================+==================================+=============+
| 1  | ALCE                     | LogisticRegressionClassification |      1      |
+----+--------------------------+----------------------------------+-------------+
| 2  | BADGE                    | RandomForestClassification       |      10     |
+----+--------------------------+----------------------------------+-------------+
| 3  | CLUE                     | ParzenWindowClassifier           |      50     |
+----+--------------------------+----------------------------------+-------------+
| 4  | ContrastiveAL            |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 5  | CoreSet                  |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 6  | GreedySamplingX          |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 7  | ProbabilisticAL          |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 8  | ProbCover                |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 9  | RandomSampling           |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 10 | TypiClust                |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 11 | USEntropy                |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 12 | USLeastConfident         |                                  |             |
+----+--------------------------+----------------------------------+-------------+
| 13 | USMargin                 |                                  |             |
+----+--------------------------+----------------------------------+-------------+


Quick start
##########################
All uploaded experiment results can be seen at `GitHub pages <https://alexanderbenz.github.io/scikit-activeml-experiments/>`_.
The plot generations may take some time when selecting multiple results.
The experiment results can be found in the **experiment** folder in the 
subfolder **experiments-results**.
It is recommanded to run experiments on a server or seperate machine as the 
current combinations result in 1170 experiment conficuration for each dataset.

Installation guide
==================

``git clone https://github.com/AlexanderBenz/scikit-activeml-experiments.git``
``pip install -r requirements.txt`` (Note: Please install `pytorch <https://pytorch.org/get-started/locally/>`_ using the linked installation guide.)
(optional) ``pip install -r requirements_experiments.txt -r requirements_extra.txt``

Execute the experiments locally 
===============================
All experiment files are in the experiments' directory.
To select wich experiments to run, comment the experiment you want to exclude in 
the **write_bash_script.py** file. 
In there also select if you want to run the experiments
slurm using the **use_slurm** variable. Please note that running the experiments 
locally may take a very long time.
All experiments are run for Python version 3.11. If error incure please submit
a Issue using the **BUG** Template.
To directly seperate the new experiments available without moving them to change the
variable **experiment_dir_path** in **hydra_experiment.py**

Move to the experiment folder and run 
``cd experiments``
``python write_bash_script.py``
Now you can download and calculate embedding files for each dataset using
``bash bash_scripts/download.sh``
This will result in stopping the experiment after the download displaying an 
error message that shows the batch size is not the right size.

To start the experiments, use the command
``bash bash_scripts/(selected_dataset).sh``

Add new experiments
====================
To provide new experiments, there are three options.

1. New dataset/backbone: 
    - Add a new config file to **experiment/config/dataset** with the filename
      ending on **.yaml**. (optional) Add a backbone file in **experiment/config/backbone** 

    - Use eather **HuggingFace**, **openml** (scikit-learn), or **torch** to load the dataset
      or make a file **experimets/data/{dataset_name}_{X/y}.npy** or when using
      an embedding model **experimets/data/{dataset_name}_{backbone}_{X/y}.npy**.

    - Update the **experimets/write_bash_script.py** by adding the new dataset
2. new Strategy:
    - Add a new config file to **experiment/config/query-strategy** with the filename
      ending on **.yaml**.

    - If the strategy is not from **scikit-activeml** please implement your strategy 
      following the `user guide <https://scikit-activeml.github.io/scikit-activeml-docs/latest/contributing.html#contributing-code>`_.

2. new Strategy:
    - Add a new config file to **experiment/config/model** with the filename 
      ending on **.yaml**.

    - Use a **scikit-learn**/ **scikit-activeml** model or convert the model to 
      a **BaseEstimator** from **scikit-learn**


Running the website locally
============================

To run the website locally to display the results locally, there are two 
different methods. Either open the app using:
``shiny run src_shiny_app/app.py``

If you want to run the server locally, use the following commands:
``shiny static-assets remove``
``shinylive export src_shiny_app docs``
``python -m http.server --directory docs --bind localhost 8008``


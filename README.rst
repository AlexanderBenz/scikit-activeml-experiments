`Active learning Benchmark <https://alexanderbenz.github.io/scikit-activeml-experiments/>`_
############################################################################################
This is an implementation for Benchmarking active learning using **scikit-activeml**.
This repository includes experiments scripts, results and the code for the
interactive website displaying the experiment results.

This repository includes the results for the following datasets, query strategies
models, and batch sizes.

Datasets
=================
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| Dataset                                                                                                                    | Samples | Classes | General Task |           Dimension |
+============================================================================================================================+=========+=========+==============+=====================+
| `CAT_vs_DOG <https://huggingface.co/datasets/microsoft/cats_vs_dogs>`_                                                     |  23410  |      2  |     Image    |     4x4px-500x500px |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `CIFAR10 <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10>`_    |  60000  |     10  |     Image    |             32x32px |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `CIFAR100 <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100>`_ |  60000  |    100  |     Image    |             32x32px |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `DTD <https://pytorch.org/vision/stable/generated/torchvision.datasets.DTD.html#torchvision.datasets.DTD>`_                |   5640  |     47  |     Image    | 300x300px-640x640px |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `AG-NEWS <https://huggingface.co/datasets/fancyzhx/ag_news>`_                                                              | 127600  |      4  |     Text     |      100–1000 words |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `BANK77 <https://huggingface.co/datasets/PolyAI/banking77>`_                                                               |  13083  |     77  |     Text     |        13–433 words |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `DBpedia <https://huggingface.co/datasets/fancyzhx/dbpedia_14>`_                                                           | 630000  |     14  |     Text     |       3-13600 words |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `TREC6 <https://www.openml.org/search?type=data&status=active&id=6>`_                                                      |   5952  |      6  |     Text     |       10 words avg. |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `ALOI <https://www.openml.org/search?type=data&status=active&id=42396>`_                                                   | 108000  |   1000  |    Tabular   |        128 features |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `IRIS <https://www.openml.org/search?type=data&status=active&id=61>`_                                                      |    150  |      3  |    Tabular   |          4 features |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `LETTER <https://www.openml.org/search?type=data&status=active&id=32>`_                                                    |  20000  |     26  |    Tabular   |         16 features |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+
| `PENDIGITS <https://huggingface.co/datasets/CogComp/trec>`_                                                                |  10992  |     10  |    Tabular   |         16 features |
+----------------------------------------------------------------------------------------------------------------------------+---------+---------+--------------+---------------------+

Query Strategies, Classifiers
=============================
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
|    | Active learning strategy                                                                                                                                                                                  | Classifiers                                                                                                                                                                                                | Batch sizes |
+====+===========================================================================================================================================================================================================+============================================================================================================================================================================================================+=============+
| 1  | `ALCE <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-CostEmbeddingAL-Active_Learning_with_Cost_Embedding_%28ALCE%29.html>`_                   | `LogisticRegressionClassification <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_                                                                       |      1      |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 2  | `BADGE <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-Badge-Batch_Active_Learning_by_Diverse_Gradient_Embedding_%28BADGE%29.html>`_           | `RandomForestClassification <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/api/skactiveml.classifier.ParzenWindowClassifier.html#skactiveml.classifier.ParzenWindowClassifier>`_ |      10     |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 3  | `CLUE <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-Clue-Clustering_Uncertainty-weighted_Embeddings_%28CLUE%29.html>`_                       | `ParzenWindowClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_                                                                                 |      50     |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 4  | `ContrastiveAL <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-ContrastiveAL-Contrastive_Active_Learning_%28CAL%29.html>`_                     |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 5  | `CoreSet <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-CoreSet-Core_Set.html>`_                                                              |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 6  | `GreedySamplingX <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-GreedySamplingX-Greedy_Sampling_on_the_Feature_Space_%28GSx%29.html>`_        |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 7  | `ProbabilisticAL <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-ProbabilisticAL-Multi-class_Probabilistic_Active_Learning_%28McPAL%29.html>`_ |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 8  | `ProbCover <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-ProbCover-Probability_Coverage_%28ProbCover%29.html>`_                              |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 9  | `RandomSampling <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-RandomSampling-Random_Sampling.html>`_                                         |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 10 | `TypiClust <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-TypiClust-Typical_Clustering_%28TypiClust%29.html>`_                                |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 11 | `USEntropy <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-UncertaintySampling-Uncertainty_Sampling_with_Entropy.html>`_                       |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 12 | `USLeastConfident <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-UncertaintySampling-Uncertainty_Sampling_with_Margin.html>`_                 |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+
| 13 | `USMargin <https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-UncertaintySampling-Uncertainty_Sampling_with_Least-Confidence.html>`_               |                                                                                                                                                                                                            |             |
+----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------+

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

``git clone https://github.com/AlexanderBenz/scikit-activeml-experiments.git`` \
``pip install -r requirements.txt`` (Note: Please install `pytorch <https://pytorch.org/get-started/locally/>`_ using the linked installation guide.) \
(optional) ``pip install -r requirements_experiments.txt -r requirements_extra.txt``\

Execute the experiments locally 
===============================
All experiment files are in the experiments' directory.\
To select wich experiments to run, comment the experiment you want to exclude in \
the **write_bash_script.py** file. \
In there also select if you want to run the experiments \
slurm using the **use_slurm** variable. Please note that running the experiments \
locally may take a very long time. \
All experiments are run for Python version 3.11. If error incure please submit \
a Issue using the **BUG** Template. \
To directly seperate the new experiments available without moving them to change the \
variable **experiment_dir_path** in **hydra_experiment.py**

Move to the experiment folder and run \
``cd experiments`` \
``python write_bash_script.py`` \
Now you can download and calculate embedding files for each dataset using \
``bash bash_scripts/download.sh`` \
This will result in stopping the experiment after the download displaying an 
error message that shows the batch size is not the right size.

To start the experiments, use the command \
``bash bash_scripts/(selected_dataset).sh`` \

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

2. new model:
    - Add a new config file to **experiment/config/model** with the filename 
      ending on **.yaml**.

    - Use a **scikit-learn**/ **scikit-activeml** model or convert the model to 
      a **BaseEstimator** from **scikit-learn**


Running the website locally
============================

Before running the website locally it is important to change the code
as the download path for the experiment results needs to be changed or new
results won't be shown. The changes need to be made in: \
**src_shiny_app/app.py** for the method **load_experiment**. \
To run the website locally to display the results locally, there are two 
different methods. Either open the app using: \
``shiny run src_shiny_app/app.py`` \

If you want to run the server locally, use the following commands: \
``shiny static-assets remove`` \
``shinylive export src_shiny_app docs`` \
``python -m http.server --directory docs --bind localhost 8008`` \


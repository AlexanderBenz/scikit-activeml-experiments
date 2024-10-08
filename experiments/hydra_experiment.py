from omegaconf import DictConfig

import hydra
from hydra.utils import instantiate
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, log_loss, average_precision_score, balanced_accuracy_score, accuracy_score
from skactiveml.utils import call_func, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import SubSamplingWrapper
from util import get_transformer_by_name
import cpuinfo

import torch
from torchvision.transforms import v2
from torch.utils.data import  DataLoader

# experiment_dir_path = "experiments"
experiment_dir_path = "experiments_results"

def load_dataset_X_y(filepath):
    """
    Load X and y from the file.

    Parameters:
    - filepath (str) : Filepath to the embedded files.
    """
    X = np.load(f'{filepath}_X.npy', allow_pickle=True)
    y = np.load(f'{filepath}_y.npy', allow_pickle=True)
    return X, y

class DataloaderClass():
    """
    Used to add a a transformer for image datasets. Also transforms grayscaled
    images into RGB.

    Parameters:
    - transformer (torchvision.transforms) : Transformer of the backbone model.
    - features_name (str): Dictonary name for the features.
    - label_name (str): Dictonary name for the labels.
    """
    def __init__(self, transformer, features_name, label_name):
        self.transformer = transformer
        self.features_name = features_name
        self.label_name = label_name
        self.transform_to_rgb = v2.RGB()

    def collate_fn(self, batch):
        """
        Call function used when itterating through the dataloader. Adds
        transformer bevore returning the batch.

        Parameters:
        - batch (dataset): Dictonary of the dataset of the given dataloader for 
        the batch size.

        Returns:
        - dataset (torch.ndarray): The transformed batch from the dataloader.
        """
        return {
            self.features_name: torch.stack([self.transformer(x[self.features_name]) for x in batch]),
            self.label_name: torch.tensor([x[self.label_name] for x in batch])
        }
    
    def to_rgb_transformer(self, batch):
        """
        Transformer function used when returning the . Adds
        transformer bevore returning the batch.

        Parameters:
        - batch (dataset): Dictonary of the dataset for the batch size.

        Returns:
        - dataset (dataset): The batch dataset RGB transformed.
        """
        return {
                self.features_name: self.transform_to_rgb(batch[self.features_name]),
                self.label_name: batch[self.label_name]
            }

def load_torch_dataset(loader, root_dir, split_dict, backbone_name, download=True):
    """
    Load and process a given dataset for training or validation.

    Parameters:
    - loader (torchvision.datasets) : Torch function to load the dataset.
    - root_dir (str) : Root directory where the dataset will be stored.
    - split_dict (dict) : Dictonary of additinal parameters to download the selected split.
    - backbone_name (str) : Name for which transformer should be used for the dataloader.
    - download (bool) : Boolean indicating whether the dataset should be saved localy.

    Returns:
    - dataset (torch.ndarray): The dataset.
    """
    transformer = get_transformer_by_name(backbone_name)
    dataset = instantiate(loader, root=root_dir, download=download, transform=transformer, **split_dict)
    return dataset

def process_dataset(dataset, is_train, model_emb, batch_size=4, num_workers=4, device="cpu", tokenizer=None, features_name=None, label_name=None, collate_fn=None):
    """
    Load and process a given dataset for training or validation.

    Parameters:
    - dataset (torch.ndarray|dataset): The dataset.
    - is_train (bool|string|None) : Value indicating whether the dataset is for training (True) or validation (False).
    If None the dataloader will is given as a Dictionary or dataset object.
    - model_emb (transformers| callable) : Embeddings model used to transform the dataset into a embeddings featurespace
    - batch_size (int) : The batch_size used for the DataLoader.
    - num_workers (int) : The num_workers used for the DataLoader.
    - device (str) : The deviced used for model_emb
    - tokenizer (transformers| None) : Tokenizer is used for text models in combination with the model_emb. 
    If None dataset is seen as transformed.
    - collate_fn (callable) : The collate_fn used for the DataLoader.
    - features_name (str): Dictonary name for the features.
    - label_name (str): Dictonary name for the labels.
    

    Returns:
    - X (numpy.ndarray): Concatenated embeddings of the dataset.
    - y_true (numpy.ndarray): Concatenated true labels of the dataset.
    """
        
    dataloaders = []
    # Create DataLoader
    if is_train is None:
        if hasattr(dataset, "keys"):
            for split in dataset.keys():
                dataloaders.append(DataLoader(dataset[split], batch_size=batch_size, num_workers=num_workers))
        else:
            dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, collate_fn=collate_fn))
    else:
        dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers))
    embedding_list = []
    label_list = []
    
    # Iterate through the DataLoader and extract embeddings
    model_emb.eval()
    with torch.no_grad():
        for dataloader in dataloaders:
            single_embedding_list = []
            single_label_list = []
            if tokenizer is None:
                for i, data in enumerate(dataloader):
                    if i % 10==0:
                        print(len(dataloader), i)
                    if features_name is None:
                        image, label = data
                    else:
                        image = data[features_name]
                        label = data[label_name]
                    embeddings = model_emb(image.to(device)).cpu()
                    single_embedding_list.append(embeddings)
                    single_label_list.append(label)
            else:
                for data in dataloader:
                    # datafeatures und labels in config speichern 
                    texts = data[features_name]
                    labels = data[label_name]
                    embeddings = get_text_embeddings(texts, tokenizer, model_emb, device)
                    single_embedding_list.append(embeddings)
                    single_label_list.append(labels)

            # Concatenate embeddings and labels
            embedding_list.append(torch.cat(single_embedding_list, dim=0))
            label_list.append(torch.cat(single_label_list, dim=0))
        X = torch.cat(embedding_list, dim=0).numpy()
        y_true = torch.cat(label_list, dim=0).numpy()

    return X, y_true

# Define a function to process the text and obtain embeddings
def get_text_embeddings(text, tokenizer, model_emb, device):
    """
    Function that trnsforms the texts using the given Tokenizer to obtain embeddings.

    Parameters:
    - text (list) : The batch texts form the dataloader.
    - tokenizer (transformers) : Tokenizer is used for text models in combination with the model_emb. 
    - model_emb (transformers) : Embeddings model used to transform the dataset into a embeddings featurespace
    - device (str) : The deviced used for model_emb

    Returns:
    - embeddings (torch.ndarray): The embeddings of the batch.

    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs.to(device)
    with torch.no_grad():
        outputs = model_emb(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings.cpu()

def gen_seed(random_state:np.random.RandomState):
    """
    Function used to generate a consistent random seed
    random_state
    """
    return random_state.randint(0, 2**31)

def gen_random_state(random_state:np.random.RandomState):
    return np.random.RandomState(gen_seed(random_state))

def neg_brier_score_(p_class_pred, y_true):
    """
    Calculates the brier score for the muli-class case as scikit-learn only
    implemented the binary class case.

    Parameters:
    - p_class_pred (numpy.ndarray) : Predict_proba of the classifier
    - y_true (numpy.ndarray) : True label of all samples

    Returns:
    - neg_brier_score (float): Returns the brier score.
    """
    y_one_hot = np.eye(p_class_pred.shape[1])[y_true]
    return np.mean(np.sum((p_class_pred - y_one_hot)**2, axis=1))

def create_folders(params):
    """
    Function used to generate the experiment folder structer to safe the 
    experiment results.
    """
    exclude = "seed"
    dirs = {k:str(v) for k,v in params.items() if k not in exclude}
    filepath = f"{experiment_dir_path}/"+"/".join(dirs.values())
    os.makedirs(filepath, exist_ok=True)
    return filepath

decimals = 4

@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # initialize experiment variables
    dataset_name = cfg.dataset.name
    classes = cfg.dataset.classes
    qs_name = cfg.query_strategy.name
    qs_params = cfg.query_strategy.params
    model_name = cfg.model.name
    model_params = cfg.model.params

    batch_size = cfg.batch_size
    n_cycles = cfg.n_cycles
    seed = cfg.seed
    master_random_state = np.random.RandomState(seed)

    data_dir = cfg.cache_dir
    cache = cfg.cache
    # create experiment path where the datasets are saved.    
    embeddings_model = ""
    if hasattr(cfg, "backbone"):
        embeddings_model = cfg.backbone.name
        embeddings_model_path = cfg.backbone.path
        experiment_base_path = f'{data_dir}/{dataset_name}_{embeddings_model}'
    else:
        experiment_base_path = f'{data_dir}/{dataset_name}'

    X = None
    y = None
    dataset_train = None
    dataset_eval = None
    dataset = None

    # ======== Load experiment data=========

    # Load X and y if cache is available
    if os.path.exists(experiment_base_path + "_X.npy"):
        print("Cache found loading : " + dataset_name )
        X, y = load_dataset_X_y(experiment_base_path)
    else:
        print("No cache found for : " + dataset_name )

        features_name = None
        label_name = None
        collate_fn = None
        if hasattr(cfg.dataset, "features_name"):
            features_name = cfg.dataset.features_name
            label_name = cfg.dataset.label_name

        # Load dataset using the method saved in the dataset config
        # Note using openml load_dataset may leeds to errors
        data_loader_str = str(cfg.dataset.class_definition._target_).split(".")[0]
        if data_loader_str == "sklearn" or data_loader_str == "openml":
            if data_loader_str == "openml":
                dataset_tmp = instantiate(cfg.dataset.class_definition, download_data=cache)
                X_df, y, _, _ = dataset_tmp.get_data(target=dataset_tmp.default_target_attribute)
            else:
                dataset_tmp = instantiate(cfg.dataset.class_definition, cache=cache, return_X_y=True)
                X_df, y = dataset_tmp
            X = X_df.values
        elif data_loader_str == "datasets":
            dataset = instantiate(cfg.dataset.class_definition)
            # As hugging face dosn't include transformers in the load function
            # create a collate_fn for the dataloader to transform the batch.
            transformer = get_transformer_by_name(embeddings_model)
            if transformer is not None:
                dl_class = DataloaderClass(transformer, features_name, label_name)
                collate_fn = dl_class.collate_fn
                dataset.set_transform(dl_class.to_rgb_transformer)
            
        elif data_loader_str == "torchvision":
            dataset_train = load_torch_dataset(loader=cfg.dataset.class_definition, root_dir=data_dir, split_dict=cfg.dataset.train_params, backbone_name=embeddings_model, download=cache)
            dataset_eval = load_torch_dataset(loader=cfg.dataset.class_definition, root_dir=data_dir, split_dict=cfg.dataset.eval_params, backbone_name=embeddings_model, download=cache)     
            if hasattr(cfg.dataset, "extra_split_params"):
                dataset_extra = load_torch_dataset(loader=cfg.dataset.class_definition, root_dir=data_dir, split_dict=cfg.dataset.extra_split_params, backbone_name=embeddings_model, download=cache) 
                dataset_eval = torch.utils.data.ConcatDataset([dataset_eval, dataset_extra])

        # If a backbone is given load and use backbone on dataset_train and dataset_eval
        if hasattr(cfg, "backbone"):
            num_workers = cfg.dataset.num_workers
            dataloader_batch_size = cfg.dataset.dataloader_batch_size

            torch.hub.set_dir('cache/')
            tokenizer = None
            if hasattr(cfg.backbone, "tokenizer"):
                tokenizer = instantiate(cfg.backbone.tokenizer)
                model_emb = instantiate(cfg.backbone.class_definition)
            else:
                model_emb = instantiate(cfg.backbone.class_definition, repo_or_dir=embeddings_model_path, model=embeddings_model)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model_emb.to(device)
            

            # Use the embedings model to generate X and y depending on the loading method used
            if dataset is None:
                X_train, y_train_true = process_dataset(dataset=dataset_train, is_train=True, model_emb=model_emb, batch_size=dataloader_batch_size, num_workers=num_workers, device=device, tokenizer=tokenizer, features_name=features_name, label_name=label_name)
                print(len(X_train))
                X_test, y_test_true = process_dataset(dataset=dataset_eval, is_train=False, model_emb=model_emb, batch_size=dataloader_batch_size, num_workers=num_workers, device=device, tokenizer=tokenizer, features_name=features_name, label_name=label_name)
                print(len(X_test))
                X = np.append(X_train, X_test, axis=0)
                y = np.append(y_train_true, y_test_true, axis=0)
            else:
                X, y = process_dataset(dataset=dataset, is_train=None, model_emb=model_emb, batch_size=dataloader_batch_size, num_workers=num_workers, device=device, tokenizer=tokenizer, features_name=features_name, label_name=label_name, collate_fn=collate_fn)
        else:
            # If no embeding is needed and X and y are not saved using the loading method
            if dataset_eval is not None:
                dataset = np.append(dataset_train, dataset_eval, axis=0)

            if dataset is not None:
                X = dataset[features_name]
                y = dataset[label_name]
        # Save X and y
        if cache:
            os.makedirs(data_dir, exist_ok=True)
            print("Saving files to cache " + experiment_base_path)
            np.save(f'{experiment_base_path}_X.npy', X)
            np.save(f'{experiment_base_path}_y.npy', y)

    if model_params is None:
        model_params = {}
    if qs_params is None:
        qs_params = {}
    
    # ======== Generate experiment methodes =========

    # Encode labels to use sklearn functions
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data into training and test/eval data to use for the experiment
    test_size = cfg.dataset.params.test_size
    train_size = 1 - test_size 
    train_indices, test_indices = train_test_split(np.arange(0, X.shape[0]), test_size=test_size, train_size=train_size, stratify=y, random_state=master_random_state)
    X_train = X[train_indices]
    y_train_true = y[train_indices]
    X_test = X[test_indices]
    y_test_true = y[test_indices]

    # Standartize if no backbone is given
    if not hasattr(cfg, "backbone"):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"Starting experiment for Dataset={dataset_name}, qs_name={qs_name}, model={model_name}, batch_size={batch_size}", flush=True)
    # Generate model and query strategy
    model_class_str = str(cfg.model.class_definition._target_).split(".")[0]
    if "skactiveml" != model_class_str:
        clf = SklearnClassifier(instantiate(cfg.model.class_definition, **model_params), random_state=gen_seed(master_random_state), classes=np.arange(classes))
    else:
        clf = instantiate(cfg.model.class_definition, random_state=gen_seed(master_random_state), classes=np.arange(classes), **model_params)
    if hasattr(qs_params, "classes"):
        qs_params["classes"]= np.arange(classes).tolist()
    qs = SubSamplingWrapper(instantiate(cfg.query_strategy.class_definition, random_state=gen_seed(master_random_state), **qs_params), max_candidates=cfg.n_max_candidates, random_state=gen_seed(master_random_state))
    y_train = np.full(shape=y_train_true.shape, fill_value=MISSING_LABEL)
    clf.fit(X_train, y_train)
    processor_name = cpuinfo.get_cpu_info()["brand_raw"]
    dataset_params_name = dataset_name
    if hasattr(cfg, "backbone"):
        dataset_params_name += "_" + embeddings_model
    params = {
        'dataset': dataset_params_name,
        'model': model_name,
        'qs': qs_name,
        'batch_size': batch_size,
        'seed': seed,
    }

    metric_dict = {
        'accuracy': [],
        'auroc': [],
        'f1_micro': [],
        'f1_macro': [],
        'neg_brier_score': [],
        'neg_log_loss': [],
        'average_precision': [],
        'balanced_accuracy': [],
        'time': [],
    }
    query_indices = []

    # ======== Begin experiment =========
    for c in range(n_cycles):
        if c %10==0:
            print(f"number of cycles: {c}", flush=True)
        # stop learning cycle if all samples are aquired
        if len(X_train) <= c*batch_size:
            break
        # active learning cycle
        start = time.time()
        query_idx = call_func(qs.query, X=X_train, y=y_train, batch_size=batch_size, clf=clf, discriminator=clf)
        query_indices.append(query_idx)
        end = time.time()
        y_train[query_idx] = y_train_true[query_idx]
        clf.fit(X_train, y_train)

        # generate metrics
        y_test_pred = clf.predict(X_test)
        y_test_proba = clf.predict_proba(X_test)
        
        score = accuracy_score(y_test_true,y_test_pred) 
        # as the scores function differently depending on binary or multi-class
        # classification seperate each use case.
        if classes <= 2:
            auroc = roc_auc_score(y_test_true, y_test_proba[:,1])
            y_true_proba = np.array([y_p[y_t] for y_p, y_t in zip(y_test_proba, y_test_true)])
            average_precision = average_precision_score(y_test_true, y_true_proba)
        else:
            auroc = roc_auc_score(y_test_true, y_test_proba, multi_class='ovr')
            average_precision = average_precision_score(y_test_true, y_test_proba)
        f1_micro = f1_score(y_test_true, y_test_pred, average='micro')
        f1_macro = f1_score(y_test_true, y_test_pred, average='macro')
        neg_brier_score = neg_brier_score_(y_test_proba, y_test_true)
        neg_log_loss = log_loss(y_test_true, y_test_proba)
        balanced_accuracy = balanced_accuracy_score(y_test_true, y_test_pred)
        # save metrics
        metric_dict['accuracy'].append(np.round(score, decimals=decimals))
        metric_dict['auroc'].append(np.round(auroc, decimals=decimals))
        metric_dict['f1_micro'].append(np.round(f1_micro, decimals=decimals))
        metric_dict['f1_macro'].append(np.round(f1_macro, decimals=decimals))
        metric_dict['neg_brier_score'].append(np.round(neg_brier_score, decimals=decimals))
        metric_dict['neg_log_loss'].append(np.round(neg_log_loss,decimals=decimals))
        metric_dict['average_precision'].append(np.round(average_precision, decimals=decimals))
        metric_dict['balanced_accuracy'].append(np.round(balanced_accuracy, decimals=decimals))
        metric_dict['time'].append(np.round(end - start, decimals=decimals))

    # ======== Save experiment results =========
    # Save metric_dict using the experiments parameters
    filepath = create_folders(params=params)
    df = pd.DataFrame.from_dict(data=metric_dict)
    outpath = os.path.join(filepath, f"{params['seed']}.csv")
    df.to_csv(outpath, index=True, index_label="step")
    np.save(os.path.join(filepath, f"{params['seed']}_query_indices_with_{processor_name}.npy"), query_indices)
    print("Experiment Done")


if __name__ == "__main__":
    my_app()
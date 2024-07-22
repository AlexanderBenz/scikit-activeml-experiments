from omegaconf import DictConfig

import hydra
from hydra.utils import instantiate
import numpy as np
import time
import platform
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, log_loss, average_precision_score, balanced_accuracy_score, accuracy_score
from skactiveml.utils import call_func, MISSING_LABEL
from skactiveml.classifier import SklearnClassifier

import torch
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader

import warnings
warnings.filterwarnings("ignore")

def load_dataset_X_y(filepath):
    X = np.load(f'{filepath}_X.npy', allow_pickle=True)
    y = np.load(f'{filepath}_y.npy', allow_pickle=True)
    return X, y


def load_torch_dataset(loader, root_dir, split_dict, download=True):
    """
    Load and process a given dataset for training or validation.

    Parameters:
    - root_dir (str) : Root directory where the dataset will be stored.
    - is_train (bool) : Boolean indicating whether the dataset is for training (True) or validation (False).
    - batch_size (int) : The batch_size used for the DataLoader.

    Returns:
    - dataset (torch.ndarray): The dataset.
    """
    transformer = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
    dataset = instantiate(loader, root=root_dir, download=download, transform=transformer, **split_dict)
    return dataset

def process_dataset(dataset, is_train, model_emb, batch_size=4, num_workers=4, device="cpu", tokenizer=None):
    """
    Load and process a given dataset for training or validation.

    Parameters:
    - root_dir (str) : Root directory where the dataset will be stored.
    - is_train (bool) : Boolean indicating whether the dataset is for training (True) or validation (False).
    - batch_size (int) : The batch_size used for the DataLoader.

    Returns:
    - X (numpy.ndarray): Concatenated embeddings of the dataset.
    - y_true (numpy.ndarray): Concatenated true labels of the dataset.
    """
        
    dataloaders = []
    # Create DataLoader
    if is_train is None:
        for split in dataset.keys():
            dataloaders.append(DataLoader(dataset[split], batch_size=batch_size, num_workers=num_workers))
    else:
        dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers))
    embedding_list = []
    label_list = []
    
    # Iterate through the DataLoader and extract embeddings
    with torch.no_grad():
        for dataloader in dataloaders:
            single_embedding_list = []
            single_label_list = []
            if tokenizer is None:
                for _, data in enumerate(dataloader):
                    image, label = data
                    embeddings = model_emb(image.to(device)).cpu()
                    single_embedding_list.append(embeddings)
                    single_label_list.append(label)
            else:
                for data in dataloader:
                    texts = data['text']
                    labels = data["label"]
                    embeddings = get_bert_embeddings(texts, tokenizer, model_emb, device)
                    single_embedding_list.append(embeddings)
                    single_label_list.extend(labels)

            # Concatenate embeddings and labels
            embedding_list = torch.cat(single_embedding_list, dim=0)
            label_list = torch.cat(single_label_list, dim=0)
        X = torch.cat(embedding_list, dim=0).numpy()
        y_true = torch.cat(label_list, dim=0).numpy()

    return X, y_true

# Define a function to process the text and obtain embeddings
def get_bert_embeddings(text, tokenizer, model_emb, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs.to(device)
    with torch.no_grad():
        outputs = model_emb(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings.cpu()

def gen_seed(random_state:np.random.RandomState):
    return random_state.randint(0, 2**31)

def gen_random_state(random_state:np.random.RandomState):
    return np.random.RandomState(gen_seed(random_state))

def neg_brier_score_(p_class_pred, y_true):
    y_one_hot = np.eye(p_class_pred.shape[1])[y_true]
    return np.mean(np.sum((p_class_pred - y_one_hot)**2, axis=1))

def create_folders(params):
    exclude = "seed"
    dirs = {k:str(v) for k,v in params.items() if k not in exclude}
    filepath = "experiments/"+"/".join(dirs.values())
    os.makedirs(filepath, exist_ok=True)
    return filepath

decimals = 4

@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:

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
    dataloader_batch_size = cfg.dataset.dataloader_batch_size
    num_workers = cfg.dataset.num_workers
    # TODO: X,y zusammen abspeichern, dann train_test_split anwenden mit dem randomseed, stratify=y, train = 2/3 test 1/3,
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
    if os.path.exists(experiment_base_path + "_X.npy"):
        print("Cache found loading : " + dataset_name )
        X, y = load_dataset_X_y(experiment_base_path)
    else:
        print("No cache found for : " + dataset_name )
        data_loader_str = str(cfg.dataset.class_definition._target_).split(".")[0]
        if data_loader_str == "openml":
            dataset = instantiate(cfg.dataset.class_definition, download_data=cache)
            X_df, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            X = X_df.values
        elif data_loader_str == "datasets":
            dataset_params = cfg.dataset.params
            dataset = instantiate(cfg.dataset.class_definition, **dataset_params)
        elif data_loader_str == "torchvision":
            dataset_train = load_torch_dataset(loader=cfg.dataset.class_definition, root_dir=data_dir, split_dict=cfg.dataset.train_params, download=cache)
            dataset_eval = load_torch_dataset(loader=cfg.dataset.class_definition, root_dir=data_dir, split_dict=cfg.dataset.eval_params, download=cache)

        # If a backbone is given load and use backbone on dataset_train and dataset_eval
        if hasattr(cfg, "backbone"):
            torch.hub.set_dir('cache/')
            tokenizer = None
            if hasattr(cfg.backbone, "tokenizer"):
                tokenizer = instantiate(cfg.backbone.tokenizer)
                model_emb = instantiate(cfg.backbone.class_definition)
            else:
                model_emb = instantiate(cfg.backbone.class_definition, repo_or_dir=embeddings_model_path, model=embeddings_model)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model_emb.to(device)

            # Loaded Datasets
            if dataset is None:
                X_train, y_train_true = process_dataset(dataset=dataset_train, is_train=True, model_emb=model_emb, batch_size=dataloader_batch_size, num_workers=num_workers, device=device, tokenizer=tokenizer)
                X_train, y_train_true = process_dataset(dataset=dataset_eval, is_train=False, model_emb=model_emb, batch_size=dataloader_batch_size, num_workers=num_workers, device=device, tokenizer=tokenizer)
                X = np.append(X_train, X_test, axis=0)
                y = np.append(y_train_true, y_test_true, axis=0)
            else:
                X, y = process_dataset(dataset=dataset, is_train=None, model_emb=model_emb, batch_size=dataloader_batch_size, num_workers=num_workers, device=device, tokenizer=tokenizer)
            

        if cache:
            print("Saving files to cache " + experiment_base_path)
            np.save(f'{experiment_base_path}_X.npy', X)
            np.save(f'{experiment_base_path}_y.npy', y)

    if model_params is None:
        model_params = {}
    if qs_params is None:
        qs_params = {}

    train_size = 0
    test_size = cfg.dataset.params.test_size
    if hasattr(cfg.dataset.params, "train_size"):
        train_size = cfg.dataset.params.test_size
    else:
        train_size = 1 - test_size 
    le = LabelEncoder()
    y = le.fit_transform(y)
    train_indices, test_indices = train_test_split(np.arange(0, X.shape[0]), test_size=test_size, train_size=train_size, stratify=y, random_state=master_random_state)
    X_train = X[train_indices]
    y_train_true = y[train_indices]
    X_test = X[test_indices]
    y_test_true = y[test_indices]

    print(f"Starting experiment for Dataset={dataset_name}, qs_name={qs_name}, model={model_name}, batch_size={batch_size}")
    model_class_str = str(cfg.model.class_definition._target_).split(".")[0]
    if "skactiveml" != model_class_str:
        clf = SklearnClassifier(instantiate(cfg.model.class_definition, **model_params), random_state=gen_seed(master_random_state), classes=np.arange(classes))
    else:
        clf = instantiate(cfg.model.class_definition, random_state=gen_seed(master_random_state), classes=np.arange(classes), **model_params)
    
    qs = instantiate(cfg.query_strategy.class_definition, random_state=gen_seed(master_random_state), **qs_params)
    y_train = np.full(shape=y_train_true.shape, fill_value=MISSING_LABEL)
    clf.fit(X_train, y_train)
    processor_name = platform.processor()
    params = {
        'dataset': dataset_name + embeddings_model,
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
    filepath = create_folders(params=params)
    for c in range(n_cycles):
        start = time.time()
        query_idx = call_func(qs.query, X=X_train, y=y_train, batch_size=batch_size, clf=clf, discriminator=clf)
        query_indices.append(query_idx)
        end = time.time()
        y_train[query_idx] = y_train_true[query_idx]
        clf.fit(X_train, y_train)

        y_test_pred = clf.predict(X_test)
        y_test_proba = clf.predict_proba(X_test)

        score = accuracy_score(y_test_true,y_test_pred) 
        auroc = roc_auc_score(y_test_true, y_test_proba, multi_class='ovr')
        f1_micro = f1_score(y_test_true, y_test_pred, average='micro')
        f1_macro = f1_score(y_test_true, y_test_pred, average='macro')
        neg_brier_score = neg_brier_score_(y_test_proba, y_test_true)
        neg_log_loss = log_loss(y_test_true, y_test_proba)
        average_precision = average_precision_score(y_test_true, y_test_proba)
        balanced_accuracy = balanced_accuracy_score(y_test_true, y_test_pred)

        metric_dict['accuracy'].append(np.round(score, decimals=decimals))
        metric_dict['auroc'].append(np.round(auroc, decimals=decimals))
        metric_dict['f1_micro'].append(np.round(f1_micro, decimals=decimals))
        metric_dict['f1_macro'].append(np.round(f1_macro, decimals=decimals))
        metric_dict['neg_brier_score'].append(np.round(neg_brier_score, decimals=decimals))
        metric_dict['neg_log_loss'].append(np.round(neg_log_loss,decimals=decimals))
        metric_dict['average_precision'].append(np.round(average_precision, decimals=decimals))
        metric_dict['balanced_accuracy'].append(np.round(balanced_accuracy, decimals=decimals))
        metric_dict['time'].append(np.round(end - start, decimals=decimals))
    df = pd.DataFrame.from_dict(data=metric_dict)

    outpath = os.path.join(filepath, f"{params['seed']}.csv")
    df.to_csv(outpath, index=True, index_label="step")
    np.save(os.path.join(filepath, f"{params['seed']}_query_indices_with_{processor_name}.npy"), query_indices)
    print("Experiment Done")


if __name__ == "__main__":
    my_app()
import torch
import torch.nn as nn
import os, random

from dataclasses import dataclass
import numpy.typing as npt
import pickle as pkl
import argparse

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')


def interpolate_pt(x, new_len):
    x = F.interpolate(x, new_len, mode='linear')
    return x.float()


def uea_normalize(x, sequence_length):
    x = rearrange(x, 'B T M -> B M T')
    x = interpolate_pt(x, sequence_length)
    x = rearrange(x, 'B M T -> (B M) T')
    # x = (x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + 1e-8)
    return x


@dataclass
class ClassificationEmbeddingDataset:
    train_token: npt.NDArray = None
    train_histogram: npt.NDArray = None
    test_token: npt.NDArray = None
    test_histogram: npt.NDArray = None
    train_labels: npt.NDArray = None
    test_labels: npt.NDArray = None
    dataset_name: str = None


def get_embedding(lit_model, loader, desc="Embedding"):
    '''
    Encode the time series data into token and histogram representations.
    Input:
        lit_model: LitVQShape model
        loader: PyTorch DataLoader
    Output:
        rep_tokens: Token representations of the time series data
        rep_histograms: Histogram representations of the time series data
        labels: Labels of the dataset
    '''

    torch.set_float32_matmul_precision('medium')
    device = lit_model.device
    model = lit_model.model
    sequence_length = 512
    
    rep_tokens = []
    rep_histograms = []
    labels = []

    with torch.no_grad():
        for _, x, y, _ in tqdm(loader, desc=desc):
            x = uea_normalize(x, sequence_length)
            representations, _ = model(x.to(device), mode='tokenize')
            rep_tokens.append(representations['token'].cpu())
            rep_histograms.append(representations['histogram'].cpu())
            labels.append(y)
    
    rep_tokens = torch.cat(rep_tokens, dim=0)
    rep_histograms = torch.cat(rep_histograms, dim=0)
    labels = torch.cat(labels, dim=0)

    rep_tokens = rearrange(rep_tokens, '(B M) N T -> B M N T', M=loader.dataset.feature_df.shape[-1])
    rep_histograms = rearrange(rep_histograms, '(B M) T -> B M T', M=loader.dataset.feature_df.shape[-1])

    return rep_tokens, rep_histograms, labels


def run_embedding(lit_model, dataset, root, max_uts_per_batch=1024):
    '''
    Embedding the train and test datasets.
    Input:
        lit_model: LitVQShape model
        dataset: str, name of the dataset
        root: str, root directory of the dataset
        max_uts_per_batch: int, maximum number of univariate time series in a batch, adjust based on GPU memory size
    Output:
        embedding_dataset: embedded dataset in numpy arrays
    '''
    from data_provider.data_loader import UEAloader
    from data_provider.uea import collate_fn

    train_dataset = UEAloader(root_path=f"{root}/{dataset}", flag='TRAIN')
    test_dataset = UEAloader(root_path=f"{root}/{dataset}", flag='TEST')
    max_seq_len = max(train_dataset.max_seq_len, test_dataset.max_seq_len)
    batch_size = max(1, int(max_uts_per_batch/train_dataset.feature_df.shape[-1]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, max_len=max_seq_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, max_len=max_seq_len))

    train_tokens, train_histograms, train_labels = get_embedding(lit_model, train_loader, desc=f"{dataset} Train")
    test_tokens, test_histograms, test_labels = get_embedding(lit_model, test_loader, desc=f"{dataset} Test")

    print(f"[{dataset}] Train: tokens {train_tokens.shape}, histograms {train_histograms.shape}")
    print(f"[{dataset}] Test: tokens {test_tokens.shape}, histograms {test_histograms.shape}")

    embedding_dataset = ClassificationEmbeddingDataset(
        train_token=train_tokens.numpy(),
        train_histogram=train_histograms.numpy(),
        test_token=test_tokens.numpy(),
        test_histogram=test_histograms.numpy(),
        train_labels=train_labels.numpy(),
        test_labels=test_labels.numpy(),
        dataset_name=dataset
    )

    return embedding_dataset


class Classifier(nn.Module):
    '''
    A linear classifier with dropout on input features.
    '''
    def __init__(self, input_dim, output_dim, dropout=0):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc(self.dropout(x))


def seed_everything(seed=42):
    '''
    Set the seed for reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_validate(
        train_features, train_labels, 
        val_features, val_labels, 
        feature_type,
        num_epoch=100,
        weight_decay=0,
        dropout=0,
        seed=0
    ):
    '''
    Main training and validation loop.
    Input:
        train_features: numpy array, train features
        train_labels: numpy array, train labels
        val_features: numpy array, validation features
        val_labels: numpy array, validation labels
        feature_type: str, token or histogram
    Output:
        best_val_accuracy: float, best validation accuracy
    '''

    train_features = train_features.reshape(train_features.shape[0], -1)
    val_features = val_features.reshape(val_features.shape[0], -1)

    scaler = StandardScaler() if feature_type == "token" else MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    train_features = torch.from_numpy(train_features).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_features = torch.from_numpy(val_features).float()
    val_labels = torch.from_numpy(val_labels).long()
    device = torch.device("cuda")

    features = torch.cat([train_features, val_features], dim=0)
    labels = torch.cat([train_labels, val_labels], dim=0)
    
    seed_everything(seed)
    train_loader = DataLoader(list(zip(train_features, train_labels)), batch_size=64, shuffle=True)
    val_loader = DataLoader(list(zip(val_features, val_labels)), batch_size=64, shuffle=False)

    classifier = Classifier(
        input_dim=features.shape[1], 
        output_dim=len(np.unique(labels)), 
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005, weight_decay=weight_decay)

    best_val_accuracy = 0
    for epoch in range(num_epoch):
        # Train
        classifier.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(batch_features.to(device))
            loss = torch.nn.functional.cross_entropy(outputs, batch_labels.squeeze().to(device))
            loss.backward()
            optimizer.step()

        # Validation
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = classifier(batch_features.to(device))
                val_loss += torch.nn.functional.cross_entropy(outputs, batch_labels.squeeze().to(device)).item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels.squeeze().to(device)).sum().item()
        val_accuracy = correct / total
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
    
    return best_val_accuracy


def cross_validation(features, labels, feature_type, n_folds=5, weight_decay=0, dropout=0, num_epoch=100, verbose=True):
    '''
    5-fold cross-validation for hyperparameter tuning.
    Tuned hyperparameters:
        weight_decay: 0, 0.01, 0.1, 1, 10, 100
        dropout: 0, 0.2, 0.5, 0.8
    '''
    seed_list = [0, 1234, 42, 8237, 2024]
    sample_indices = np.arange(len(features))
    np.random.shuffle(sample_indices)

    fold_val_accuracies = []
    for fold in range(n_folds):
        val_indices = sample_indices[fold * len(sample_indices) // n_folds:(fold + 1) * len(sample_indices) // n_folds]
        train_indices = np.setdiff1d(sample_indices, val_indices)

        train_features = features[train_indices]
        train_labels = labels[train_indices]
        val_features = features[val_indices]
        val_labels = labels[val_indices]

        best_val_accuracy = train_and_validate(
            train_features, train_labels, 
            val_features, val_labels,
            feature_type=feature_type,
            dropout=dropout,
            weight_decay=weight_decay,
            num_epoch=num_epoch,
            seed=seed_list[fold]
        )

        fold_val_accuracies.append(best_val_accuracy)
        if verbose:
            print(f'Fold {fold + 1}, Validation Accuracy: {best_val_accuracy:.4f}')

    return np.mean(fold_val_accuracies), np.std(fold_val_accuracies)


def run_classification(dataset_path, feature_type="token", method="default"):
    '''
    Main classification loop.
    '''

    # Make hyperparameter list
    l2_list = [0, 0.01, 0.1, 1, 10, 100]
    dropout_list = [0, 0.2, 0.5, 0.8]
    hparam_list = []
    for l2 in l2_list:
        for dropout in dropout_list:
            hparam_list.append((l2, dropout))

    seed_list = [0, 1234, 42, 8237, 2024]

    # Load embedding dataset
    dataset_name = dataset_path.split("_")[1][:-4]
    full_path = os.path.join(embed_save_path, dataset_path)
    print(f"Loading {full_path}")
    with open(full_path, "rb") as f:
        r = pkl.load(f)

    if method == "cv":
        # Cross-validation
        cross_val_accuracies = []
        for weight_decay, dropout in hparam_list:
            accu, std = cross_validation(
                features=r.train_token if feature_type == "token" else r.train_histogram,
                labels=r.train_labels,
                feature_type=feature_type,
                dropout=dropout,
                weight_decay=weight_decay,
                num_epoch=100,
                verbose=False
            )
            cross_val_accuracies.append(accu)
            print(f"{dataset_name} weight_decay = {weight_decay:.2f}, dropout = {dropout:.2f} | cv accuracy = {accu:.4f}", flush=True)

        # Select the best hyperparameters
        best_hparam = hparam_list[np.argmax(cross_val_accuracies)]
        best_accuracy = np.max(cross_val_accuracies)

    elif method == "default":
        best_hparam = None
        best_test_accuracy = 0
        best_test_std = 0

        # Hyperparameter tuning with the default validation set
        validation_accuracies = []
        for weight_decay, dropout in hparam_list:
            run_test_accuracies = []
            for run in range(len(seed_list)):
                test_accu = train_and_validate(
                    train_features=r.train_token if feature_type == "token" else r.train_histogram,
                    train_labels=r.train_labels,
                    val_features=r.test_token if feature_type == "token" else r.test_histogram,
                    val_labels=r.test_labels,
                    feature_type=feature_type,
                    weight_decay=weight_decay,
                    dropout=dropout,
                    num_epoch=100,
                    seed=seed_list[run]
                )
                run_test_accuracies.append(test_accu)

            mean_test_accuracy = np.mean(run_test_accuracies)
            validation_accuracies.append(mean_test_accuracy)
            print(f"{dataset_name} weight_decay = {weight_decay:.2f}, dropout = {dropout:.2f} | val accuracy = {mean_test_accuracy:.4f}", flush=True)

        # Select the best hyperparameters
        best_hparam = hparam_list[np.argmax(validation_accuracies)]
        best_accuracy = np.max(validation_accuracies)

    # Test with the best hyperparameters
    best_weight_decay, best_dropout = best_hparam
    print(f"{dataset_name}: Best weight_decay = {best_weight_decay:.2f}, dropout = {best_dropout:.2f}, validation accuracy = {best_accuracy:.4f}")

    run_test_accuracies = []
    results = []
    for run in range(len(seed_list)):
        test_accu = train_and_validate(
            train_features=r.train_token if feature_type == "token" else r.train_histogram,
            train_labels=r.train_labels,
            val_features=r.test_token if feature_type == "token" else r.test_histogram,
            val_labels=r.test_labels,
            dropout=best_dropout,
            weight_decay=best_weight_decay,
            num_epoch=100,
            feature_type=feature_type,
            seed=seed_list[run]
        )
        run_test_accuracies.append(test_accu)

    results.append({
        'weight_decay': best_weight_decay,
        'dropout': best_dropout,
        'mean': np.mean(run_test_accuracies),
        'std': np.std(run_test_accuracies)
    })
    
    print(f"{dataset_name}: Test accuracy = {np.mean(run_test_accuracies):.4f} ± {np.std(run_test_accuracies):.4f}\n", flush=True)
    return np.mean(run_test_accuracies), np.std(run_test_accuracies), best_hparam, results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--checkpoint_name', type=str, default='uea_dim512_codebook512')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'embed', 'cls'])
    parser.add_argument('--validation', type=str, default='default', choices=['default', 'cv'])
    parser.add_argument('--feature_type', type=str, default='all', choices=['all', 'token', 'histogram'])
    
    args = parser.parse_args()

    datasets=[
        'ArticularyWordRecognition', 
        'AtrialFibrillation', 
        'BasicMotions', 
        # 'CharacterTrajectories', 
        # 'Cricket', 
        # 'DuckDuckGeese', 
        # 'ERing', 
        # 'EigenWorms', 
        # 'Epilepsy', 
        # 'EthanolConcentration', 
        # 'FaceDetection', 
        # 'FingerMovements', 
        # 'HandMovementDirection', 
        # 'Handwriting', 
        # 'Heartbeat', 
        # # 'InsectWingbeat', 
        # 'JapaneseVowels', 
        # 'LSST', 
        # 'Libras', 
        # 'MotorImagery', 
        # 'NATOPS', 
        # 'PEMS-SF', 
        # 'PenDigits', 
        # 'PhonemeSpectra', 
        # 'RacketSports', 
        # 'SelfRegulationSCP1', 
        # 'SelfRegulationSCP2', 
        # 'SpokenArabicDigits', 
        # 'StandWalkJump', 
        # 'UWaveGestureLibrary'
    ]

    # Use VQShape to encode the TS data and save the embedding dataset
    if args.mode in ['all', 'embed']:
        from vqshape.pretrain import LitVQShape
        checkpoint_path = f"{args.checkpoint_dir}/{args.checkpoint_name}/VQShape.ckpt"
        lit_model = LitVQShape.load_from_checkpoint(checkpoint_path, 'cuda')
        print(f"Model loaded from {checkpoint_path}. Device: {lit_model.device}")

        embed_save_path = f"{args.save_dir}/mtsc/{args.checkpoint_name}"
        if not os.path.exists(embed_save_path):
            os.makedirs(embed_save_path)

        for i, dataset in enumerate(datasets):
            print(f"Processing {dataset} ({i+1}/{len(datasets)})")
            embedding_dataset = run_embedding(lit_model, dataset, root=args.dataset_dir)
            with open(os.path.join(embed_save_path, f"embedding_{dataset}.pkl"), "wb") as f:
                pkl.dump(embedding_dataset, f)
            print()

    # Train the classifier
    if args.mode in ['all', 'cls']:
        embed_save_path = os.path.join(f"{args.save_dir}/mtsc", args.checkpoint_name)
        print(f"Results path: {embed_save_path}")

        dataset_with_results = sorted([i for i in os.listdir(embed_save_path) if 'embedding' in i])
        feature_type_list = ["token", "histogram"] if args.feature_type == "all" else [args.feature_type]
        
        for feature_type in feature_type_list:
            master_df = []

            print(f"Classification with {feature_type} features", flush=True)
            for dataset in tqdm(dataset_with_results, total=len(dataset_with_results), disable=True):
                dataset_name = dataset.split("_")[1][:-4]
                mean_test_accu, std_test_accu, best_hparam, full_results = run_classification(dataset, feature_type, method=args.validation)

                for d in full_results:
                    d['dataset'] = dataset_name
                    d['feature_type'] = feature_type
                master_df.extend(full_results)

            master_df = pd.DataFrame(master_df)
            master_df.to_csv(f"{args.save_dir}/mtsc/{args.checkpoint_name}-{feature_type}.csv", index=False)

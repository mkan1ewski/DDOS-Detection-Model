import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes network-specific socket features and junk columns
    to prevent model overfitting.
    """
    df.columns = df.columns.str.strip()
    columns_to_drop: list[str] = [
        'Unnamed: 0',
        'Flow ID',
        'Source IP',
        'Source Port',
        'Destination IP',
        'Destination Port',
        'Timestamp',
        'Fwd Header Length.1',
        'SimillarHTTP',
        'Inbound'
    ]
    existing_columns_to_drop: list[str] = [
        col for col in columns_to_drop if col in df.columns]
    filtered_df: pd.DataFrame = df.drop(existing_columns_to_drop, axis=1)
    return filtered_df


def clean_missing_and_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    return df


def encode_multiclass_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encodes text labels to integers (0, 1, 2...).
    Returns the updated DataFrame and the class mapping dictionary.
    """
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])
    encoded_keys = label_encoder.transform(label_encoder.classes_)
    class_names = label_encoder.classes_
    mapping = {int(k): str(v) for k, v in zip(encoded_keys, class_names)}
    return df, mapping


def build_balanced_multiclass_dataset(directory_path: str, global_max_per_class: int = 20_000) -> tuple[pd.DataFrame, dict]:
    """
    Creates multi-class dataset, taking as a whole underrepresented classes.
    Returns the master DataFrame and the class mapping dictionary.
    """
    processed_dfs = []

    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".csv") and filename != "master_clean_dataset.csv":
            file_path = os.path.join(directory_path, filename)
            print(f"Loading: {filename}...")

            temp_df = load_dataset(file_path)
            temp_df = filter_features(temp_df)
            temp_df = clean_missing_and_infinite_values(temp_df)
            temp_df["Label"] = temp_df["Label"].astype(
                str).str.strip().str.upper()

            for _, group_df in temp_df.groupby("Label"):
                if len(group_df) > global_max_per_class:
                    processed_dfs.append(group_df.sample(
                        n=global_max_per_class, random_state=SEED))
                else:
                    processed_dfs.append(group_df)

    print("\nMerging into global dataset...")
    master_df = pd.concat(processed_dfs, ignore_index=True)
    master_df = master_df.sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)

    print("Applying label encoding...")
    master_df, class_mapping = encode_multiclass_labels(master_df)

    return master_df, class_mapping


def split_dataframe(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X: np.ndarray = df.drop("Label", axis=1).values
    y: np.ndarray = df["Label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0-train_frac), random_state=SEED, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled


class DDOSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, is_binary: bool = False):
        self.X = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y)
        if is_binary:
            # 0 was mapped to BENIGN, everything else is an attack.
            self.y = (y_tensor != 0).long()
        else:
            self.y = y_tensor.long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def get_dataloaders(data_dir: str, is_binary: bool = False, batch_size: int = 32):
    """
    Main function to retrieve dataloaders for the project.
    If 'is_binary' is True, all attack classes will be grouped to 1, and BENIGN to 0.
    """
    master_dataset_path = "../data/master_clean_dataset.csv"
    mapping_path = "../data/class_mapping.json"

    # Cache handling
    if os.path.exists(master_dataset_path) and os.path.exists(mapping_path):
        print("Loading cached dataset and mapping...")
        master_df = pd.read_csv(master_dataset_path)
        with open(mapping_path, "r") as f:
            loaded_mapping = json.load(f)
            class_mapping = {int(k): v for k, v in loaded_mapping.items()}
    else:
        print("Building dataset...")
        os.makedirs(os.path.dirname(master_dataset_path), exist_ok=True)
        master_df, class_mapping = build_balanced_multiclass_dataset(
            data_dir, global_max_per_class=20_000)
        master_df.to_csv(master_dataset_path, index=False)
        with open(mapping_path, "w") as f:
            json.dump(class_mapping, f, indent=4)

    if not is_binary:
        print("\n--- FINAL MULTI-CLASS DISTRIBUTION ---")
        print(master_df["Label"].value_counts().sort_index())
        print("\n--- CLASS MAPPING ---")
        for encoded_val, original_name in class_mapping.items():
            print(f"Class {encoded_val} -> {original_name}")
        print("\nDataset shape:", master_df.shape)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataframe(master_df)
    X_train, X_val, X_test = normalize(X_train, X_val, X_test)

    input_dim = X_train.shape[-1]

    train_dataset = DDOSDataset(X_train, y_train, is_binary=is_binary)
    val_dataset = DDOSDataset(X_val, y_val, is_binary=is_binary)
    test_dataset = DDOSDataset(X_test, y_test, is_binary=is_binary)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    if is_binary:
        class_mapping = {0: "BENIGN", 1: "ATTACK"}

    return train_loader, val_loader, test_loader, input_dim, class_mapping


def plot_dataset_distribution(y_tensor: torch.Tensor, class_mapping: dict, dataset_name: str):
    """
    Generates a bar chart showing class distribution with original class names.
    """
    counts = pd.Series(y_tensor.numpy()).value_counts().sort_index()
    counts.index = counts.index.map(class_mapping)

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x=counts.index, y=counts.values,
                     hue=counts.index, palette="viridis", legend=False)

    plt.title(
        f"Class distribution in {dataset_name}", fontsize=15, fontweight='bold')
    plt.xlabel("Class name", fontsize=12)
    plt.ylabel("Number of samples", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    xytext=(0, 3),
                    textcoords='offset points',
                    fontsize=10)

    plt.tight_layout()
    plt.show()
